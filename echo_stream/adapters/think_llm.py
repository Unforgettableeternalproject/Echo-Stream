"""ThinkStage adapter：LLM 串流（Phase 2）。

## 為什麼用 SessionControl 的 backend 而不是自己接 LLM

`echo_thought_core`（SessionControl）本來就是 U.E.P 的 **LLM 存取層**，
有 provider 註冊表（gemini / ollama / openai）與四層漸進式壓縮。
Echo Stream 是整合層，在這裡再接一次 LLM 只會多一個要維護的地方。

它原本只有 ``query()``（同步、回完整字串），拿來做串流對話會讓整條管線
退化成「等整段生成完」——也就是我們一開始要解決的問題。所以 2026-08-20
在 SessionControl 補上了 ``stream_query()``，Echo Stream 消費它。

## 這一層只要求「有 stream_query」

:class:`StreamingBackend` Protocol 刻意只描述這一個方法。
SessionControl 的 ``AbstractBackend`` 天然符合，測試可以注入假物件，
而 Echo Stream 不需要知道 provider 是誰——換 LLM 只換 backend 設定。

## Phase 2 的範圍

只接 LLM 串流。對話歷史用最樸素的 list，**Phase 4 才換成
ContextManager 的四層壓縮**——半套的壓縮比沒有壓縮更難除錯。

⚠️ 屆時要注意壓縮的競態陷阱：絕不能「先清空原始內容再等 LLM 回摘要」
（Claude Code #40352、Codex #13946 都踩過，API 失敗時原始對話被永久吞掉）。
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .. import config
from ..contracts.cancellation import CancellationToken, CancelledError
from ..contracts.style import SpeechStyle
from ..contracts.types import Sentence, Utterance
from ..core.emotion import MARKER_PROMPT, extract_style
from ..core.splitter import SentenceSplitter, SplitPolicy
from ..core.tracer import (
    MARK_MEMORY_DONE,
    MARK_MEMORY_START,
    MARK_THINK_FIRST_TOKEN,
    LatencyTracer,
)


@runtime_checkable
class StreamingBackend(Protocol):
    """任何能逐段吐出文字的 LLM 後端。

    SessionControl 的 ``AbstractBackend`` 天然符合這個形狀。
    """

    def stream_query(
        self, messages: Sequence[Any], system_instruction: str | None = None, **kwargs
    ) -> AsyncIterator[str]: ...


def build_backend(
    provider: str | None = None,
    session_repo: str | None = None,
    **overrides: Any,
) -> Any:
    """從 SessionControl 的 registry 建一個 backend。

    **這是唯一 import echo_thought_core 的地方**，其餘程式碼只認
    :class:`StreamingBackend` 這個形狀。
    """
    repo = session_repo or config.get("ECHO_STREAM_SESSION_REPO")
    config.ensure_importable(
        __import__("pathlib").Path(repo) if repo else config.subsystem_path("SESSION")
    )

    from echo_thought_core.backends.registry import BackendRegistry

    name = provider or config.get("ECHO_STREAM_LLM_PROVIDER", "openai")
    cfg: dict[str, Any] = {
        "model": config.get("ECHO_STREAM_LLM_MODEL", "gpt-5.6-luna"),
        "api_key": config.get("OPENAI_API_KEY"),
        "project": config.get("OPENAI_PROJECT"),
        "organization": config.get("OPENAI_ORG"),
        "base_url": config.get("OPENAI_BASE_URL"),
        "api": config.get("ECHO_STREAM_LLM_API", "chat"),
        "reasoning_effort": config.get("ECHO_STREAM_LLM_REASONING_EFFORT", "high"),
        "max_output_tokens": int(config.get("ECHO_STREAM_LLM_MAX_TOKENS", "1024")),
    }
    temperature = config.get("ECHO_STREAM_LLM_TEMPERATURE")
    if temperature:
        cfg["temperature"] = float(temperature)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    cfg.update(overrides)

    return BackendRegistry.default().create(name, cfg)


def to_messages(pairs: Sequence[dict[str, str]]) -> list[Any]:
    """把 ``{"role", "content"}`` 轉成 SessionControl 的 Message。

    轉換放在這裡而不是讓呼叫端處理，是為了讓 Echo Stream 的其餘部分
    不必知道 pydantic schema 的存在。
    """
    try:
        from echo_thought_core.schemas.message import Message as SessionMessage
    except ImportError:  # 測試用假 backend 時不需要真的 schema
        return list(pairs)
    return [SessionMessage(role=p["role"], content=p["content"]) for p in pairs]


class EchoBackend:
    """把使用者的話原樣回傳的假 backend。

    給「不想燒 token 但要驗管線」的場合用——走的是與真 backend
    完全相同的 :class:`StreamingBackend` 路徑。
    """

    def __init__(self, delay_s: float = 0.0, prefix: str = "你說的是：") -> None:
        self.delay_s = delay_s
        self.prefix = prefix

    async def stream_query(
        self, messages: Sequence[Any], system_instruction: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        last_user = ""
        for msg in reversed(list(messages)):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                last_user = (
                    msg.get("content") if isinstance(msg, dict) else msg.content
                )
                break
        for ch in f"{self.prefix}{last_user}":
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield ch


_LIST_MARKER_RE = re.compile(r"^[\s\-–—•·・*]+\s*")
"""句首的 markdown 列表符號。LLM 慣性輸出「- 《文明 VII》：…」這種列表，
符號會被 TTS 唸出來、也讓字幕難看。只剝句首——句中的連字號可能是
"well-known" 這種合法用法。"""


class LLMThinkStage:
    """ThinkStage 實作：對話歷史 + LLM 串流 + 切句。"""

    name = "llm"

    def __init__(
        self,
        backend: StreamingBackend | None = None,
        *,
        system_prompt: str = "",
        split_policy: SplitPolicy | None = None,
        tracer: LatencyTracer | None = None,
        history_limit: int = 20,
        default_style: SpeechStyle | None = None,
        emotion_markers: bool = False,
        memory: Any = None,
    ) -> None:
        self._backend = backend
        self.system_prompt = system_prompt
        self.split_policy = split_policy or SplitPolicy()
        self.tracer = tracer
        self.history_limit = history_limit
        """保留幾則歷史訊息。**佔位方案**——Phase 4 換成 ContextManager
        的四層漸進式壓縮後就不需要粗暴截斷。"""

        self.default_style = default_style
        self.emotion_markers = emotion_markers
        """讓 LLM 夾帶 ``[開心]`` 這類標記、逐句轉成 SpeechStyle。
        開啟時 system prompt 會自動補上標記指示（接在人設之後）。"""

        self.memory = memory
        """Phase 4 的 EchoMemory。平行 retrieve 的邏輯已經寫好，接上就生效。"""

        self.history: list[dict[str, str]] = []
        self._pending_user_text = ""
        self._pending_memory: str | None = None
        """上一輪 retrieve 的結果，這一輪注入。§7.4：late-context 注入
        結構上不可能（KV cache 無法換前綴），所以是「下一輪生效」。"""
        self.last_injected_memory: str | None = None
        """這一輪實際注入了什麼——Web 除錯顯示用，不參與邏輯。"""

    @property
    def backend(self) -> StreamingBackend:
        if self._backend is None:
            self._backend = build_backend()
        return self._backend

    async def prepare(self) -> None:
        _ = self.backend  # 提早建立，讓設定錯誤在暖機階段就爆而不是第一句話

    async def aclose(self) -> None:
        close = getattr(self._backend, "aclose", None)
        if callable(close):
            with contextlib.suppress(Exception):
                await close()

    # --- 串流 ---

    async def stream(
        self, utterance: Utterance, token: CancellationToken
    ) -> AsyncIterator[Sentence]:
        self._pending_user_text = utterance.text
        messages = to_messages(self._build_messages(utterance.text))

        # §7.4：Memory retrieve 與 LLM prefill 平行，首句不等它
        memory_task = (
            asyncio.ensure_future(self._retrieve(utterance))
            if self.memory is not None
            else None
        )

        try:
            splitter = SentenceSplitter(self.split_policy)
            async for sentence in splitter.split(
                self._tokens(messages, utterance.turn_id, token),
                utterance.turn_id,
                token,
            ):
                if sentence.text:
                    sentence.text = _LIST_MARKER_RE.sub("", sentence.text)
                if self.emotion_markers and sentence.text:
                    # 先解析再交給下游——sanitize_for_tts 只會剝括號符號、
                    # 保留內容，順序反了標記就會被唸出來
                    cleaned, style = extract_style(sentence.text, self.default_style)
                    sentence.text = cleaned
                    if style is not None:
                        sentence.style = style
                if self.default_style is not None and sentence.style is None:
                    sentence.style = self.default_style
                yield sentence
        finally:
            if memory_task is not None:
                if not memory_task.done():
                    memory_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await memory_task

    async def _tokens(
        self, messages: Sequence[Any], turn_id: str, token: CancellationToken
    ) -> AsyncIterator[str]:
        first = True
        stream = self.backend.stream_query(
            messages, system_instruction=self._system_instruction()
        )
        async for piece in stream:
            token.raise_if_cancelled()
            if first:
                if self.tracer is not None:
                    self.tracer.mark(turn_id, MARK_THINK_FIRST_TOKEN)
                first = False
            yield piece

    def _system_instruction(self) -> str | None:
        """組出這一輪的 system prompt。

        情緒標記指示**接在人設之後**——人設是基底，功能指示是補充，
        順序反了會讓角色語氣被格式說明稀釋。
        """
        parts = [self.system_prompt or ""]
        if self.emotion_markers:
            parts.append(MARKER_PROMPT)
        combined = "\n\n".join(p for p in parts if p)
        return combined or None

    async def _retrieve(self, utterance: Utterance) -> None:
        """與 LLM 平行跑，**不擋首句**。結果存起來，**下一輪**注入。

        retrieve 回來的內容不會注入到已經在生成的回應裡——那在技術上不可能
        （KV cache 一旦建立就無法替換前綴，所有 provider 皆然）。
        正解是首句用不含記憶斷言的 filler，retrieve 完成後開新請求續寫；
        起步版先驗證「下一輪生效」的體感夠不夠（§4 可選工項）。
        """
        if self.tracer is not None:
            self.tracer.mark(utterance.turn_id, MARK_MEMORY_START)
        try:
            result = await self.memory.retrieve(utterance.text)
            if result:
                self._pending_memory = result
        except (CancelledError, Exception):  # noqa: B014 - 旁路，失敗不該炸管線
            return
        finally:
            if self.tracer is not None:
                self.tracer.mark(utterance.turn_id, MARK_MEMORY_DONE)

    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        """組出這一輪的 messages。記憶附掛在**最後一則 user message 內**。

        ⚠️ 不放 system prompt——system + history 是 prompt cache 的穩定前綴
        （「越講越快」的來源），記憶每輪變、放前綴會讓 cache 全滅。

        history 存的是**不含記憶的原文**：注入文字不進歷史，
        代價是下一輪 prefill 會在「上一則 user message」處 cache 分歧、
        重算最後一組對答（便宜）；換來的是 context 不被逐輪的記憶
        blob 塞胖、過期記憶也不會殘留在歷史裡繼續影響後續輪次。
        """
        messages = list(self.history[-self.history_limit :])
        content = user_text
        if self._pending_memory:
            content = (
                f"{user_text}\n\n"
                f"（系統檢索到的相關記憶，供參考，與問題無關時忽略：\n"
                f"{self._pending_memory}）"
            )
            self.last_injected_memory = self._pending_memory
            self._pending_memory = None
        else:
            self.last_injected_memory = None
        messages.append({"role": "user", "content": content})
        return messages

    # --- 善後 ---

    async def commit(self, turn_id: str, spoken_text: str, generated_text: str) -> None:
        """把這一輪寫回歷史。

        **寫 spoken_text 不是 generated_text**——被插話中止時兩者不同，
        寫錯會讓 LLM 以為自己講完了整段，下一輪的指代全錯。

        使用者的話也是在這裡才進歷史（不是在 stream 開始時），
        這樣一輪對話的兩邊要嘛都進、要嘛都不進，不會出現
        「使用者問了但機器人的回答不見了」的半套狀態。
        """
        if self._pending_user_text:
            self.history.append({"role": "user", "content": self._pending_user_text})
        if spoken_text:
            self.history.append({"role": "assistant", "content": spoken_text})
        self._pending_user_text = ""


__all__ = [
    "StreamingBackend",
    "LLMThinkStage",
    "EchoBackend",
    "build_backend",
    "to_messages",
]
