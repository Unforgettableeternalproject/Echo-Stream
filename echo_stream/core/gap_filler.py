"""GapFiller——使用者講完到第一段真音訊之間的「墊片音」（Phase 4b D 段）。

## 為什麼

TTFA 的兩個大頭（LLM 純網路往返、TTS 每次合成的固定開銷）都是程式碼救不了的，
真實場景 8-15s、新會話更糟。**絕對延遲壓不下去，就只剩感知延遲可以動。**
Vapi / Retell 那類 voice agent 的標配：STT 一完就先出「嗯…」，使用者感覺對方
已經在回應，而不是乾等。

## 三個設計決策（艾斯維爾 2026-08-23 裁決）

1. **排隊，不打斷**。墊片音寫進 sink 之後，真音訊接在它後面——它就是這輪
   TTS 輸出的一部分。中途切掉反而突兀。代價是真音訊可能晚墊片音的長度才播，
   所以墊片要短（「嗯…」一兩秒）。
2. **預先合成一個語音庫，不即時合成**。使用者講話時讓 TTS 開工會跟 STT 搶 GPU
   （已知陷阱 #2：STT 從 ~1s 惡化到 10s）。暖機時用當前聲線合成好放記憶體，
   面板改了語速/情緒就背景重合成。多變體 + 不重複上一個，避免「預錄顯機械」。
3. **一輪最多一聲**。墊片播完真音訊還沒接上，也不補第二聲——艾斯維爾的經驗是
   補了反而怪（「嗯…嗯……」像結巴）。寧可留一段安靜。
4. **只排隊、不搶位**：真音訊到達前就決定出不出聲。追加與真音訊寫入之間用
   :meth:`GapFiller.stop` 收斂——runner 在寫第一段真音訊**之前**等它結束，
   所以 sink 裡絕不會出現「真音訊 → 墊片音」的倒序。

## 邊界

這裡不決定「說什麼」——文案全部來自設定檔；也不合成——合成走 SpeakStage
（fake / IndexTTS2 都行，它看不到差別）。core 不 import 任何子系統。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..contracts.cancellation import CancellationToken
from ..contracts.style import SpeechStyle
from ..contracts.types import AudioChunk, Sentence

logger = logging.getLogger(__name__)

GAP_SENTENCE_INDEX = -1
"""墊片音的 chunk 掛在這個 sentence_index 上——它不對應任何句子，
barge-in 推算 spoken_text 時要把它排除。"""


@dataclass(slots=True)
class GapFillerPolicy:
    """墊片音策略。預設值見 ``echo_stream.toml`` 的 ``[gap_filler]``。"""

    enabled: bool = False
    start_delay_s: float = 0.4
    """STT 完成後先停多久才出聲。人類接話也不是零間隔，太快反而假。"""
    phrases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "zh": ["嗯…", "嗯，讓我想想。"],
            "en": ["Hmm…", "Let me think."],
            "ja": ["えっと…", "うーん…"],
        }
    )
    """候選文案，**按語言分組**（key 是 STT 給的語言碼：zh / en / ja…）。
    使用者講英文就不能「嗯」中文——艾斯維爾 2026-08-23。每輪從該語言的池子
    隨機挑一句、不連續重複；該語言沒池子就退回 ``default_language``。
    **零語意**——「對」「是啊」這種在問句後面會變成回答，不能用；
    「我不記得了」是可被推翻的記憶斷言，也不能用。"""
    default_language: str = "zh"
    """STT 沒給語言（fake / 偵測失敗）或該語言沒池子時用哪一組。"""


class GapFiller:
    """持有語音庫、在 turn 的空檔期把墊片音排進 sink。

    生命週期：:meth:`build`（暖機 / 設定變更後）→ 每個 turn :meth:`start` →
    真音訊到達前 :meth:`stop`。
    """

    def __init__(self, policy: GapFillerPolicy | None = None) -> None:
        self.policy = policy or GapFillerPolicy()
        self._bank: dict[str, list[AudioChunk]] = {}
        self._last_pick: str | None = None
        """上次挑到的文案——不連續重複。"""
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self.emitted_duration_s = 0.0
        """這一輪排進 sink 的墊片音總長。runner 推算 spoken_text 時要扣掉。"""
        self.last_emitted: str | None = None
        """這一輪出了哪一聲——turn meta / session log 用。"""
        self.build_seconds: float | None = None
        self.dirty = False
        """語音庫與當前聲線不符（設定改了但正在講話，不能重合成）。turn 結束後重建。"""

    # --- 語音庫 ---

    @property
    def ready(self) -> bool:
        return bool(self._bank)

    @property
    def phrases(self) -> list[str]:
        """所有語言的文案攤平（語音庫要全部合成）。"""
        return list(
            dict.fromkeys(
                t for pool in self.policy.phrases.values() for t in pool if t.strip()
            )
        )

    def phrases_for(self, language: str | None) -> list[str]:
        """某語言的池子。``zh-TW`` → ``zh``；沒有就退回預設語言。"""
        code = (language or "").split("-")[0].split("_")[0].lower()
        pools = self.policy.phrases
        pool = pools.get(code) if code else None
        if not pool:
            pool = pools.get(self.policy.default_language) or next(iter(pools.values()), [])
        return [t for t in pool if t.strip()]

    async def build(self, speak_stage: Any, style: SpeechStyle | None = None) -> None:
        """用 SpeakStage 把所有文案合成好。**要在沒人講話時跑**——它會佔 TTS。

        每句獨立一次 stream（各自一個 turn_id / token），失敗的句子跳過，
        不讓一句壞文案拖垮整個庫。
        """
        t0 = time.perf_counter()
        bank: dict[str, list[AudioChunk]] = {}
        for text in self.phrases:
            try:
                bank[text] = await self._synthesize(speak_stage, text, style)
            except Exception as exc:  # noqa: BLE001 - 墊片是加分項
                logger.warning(
                    "gap filler 合成失敗，跳過 %r：%s: %s", text, type(exc).__name__, exc
                )
        self._bank = {k: v for k, v in bank.items() if v}
        self.dirty = False
        self.build_seconds = time.perf_counter() - t0
        logger.info("gap filler 語音庫：%d 句，%.1fs", len(self._bank), self.build_seconds)

    async def _synthesize(
        self, speak_stage: Any, text: str, style: SpeechStyle | None
    ) -> list[AudioChunk]:
        turn_id = f"gap-{abs(hash(text)) % 10**8:08x}"

        async def one() -> AsyncIterator[Sentence]:
            yield Sentence(
                text=text, turn_id=turn_id, index=0, is_first=True, is_last=True,
                split_reason="gap", style=style,
            )

        chunks: list[AudioChunk] = []
        async for chunk in speak_stage.stream(one(), CancellationToken()):
            chunks.append(chunk)
        return chunks

    def _pick(self, language: str | None) -> list[AudioChunk] | None:
        pool = [c for c in self.phrases_for(language) if c in self._bank]
        if not pool:
            return None
        if len(pool) > 1 and self._last_pick in pool:
            pool = [c for c in pool if c != self._last_pick]
        text = random.choice(pool)
        self._last_pick = text
        self.last_emitted = text
        return self._bank[text]

    # --- 每輪 ---

    def start(
        self,
        sink: Any,
        turn_id: str,
        token: CancellationToken,
        on_first_audio=None,
        language: str | None = None,
    ) -> None:
        """開始排墊片音（背景 task）。``language`` 是 STT 偵測到的語言碼，決定用哪組
        文案；``on_first_audio`` 在墊片寫入 sink 時呼叫（打點用）。"""
        self.emitted_duration_s = 0.0
        self.last_emitted = None
        self._stopped = asyncio.Event()
        self._task = None
        if not self.policy.enabled or not self.ready or sink is None:
            return
        self._task = asyncio.ensure_future(
            self._run(sink, turn_id, token, on_first_audio, language)
        )

    async def stop(self) -> None:
        """真音訊來了（或 turn 結束）：不再追加，並**等正在寫的那一段寫完**。

        runner 必須在寫第一段真音訊之前 await 這個——否則會倒序。
        """
        self._stopped.set()
        task, self._task = self._task, None
        if task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task

    async def _run(
        self,
        sink: Any,
        turn_id: str,
        token: CancellationToken,
        on_first_audio,
        language: str | None,
    ) -> None:
        if await self._wait(self.policy.start_delay_s):
            return  # 真音訊比墊片還快——不出聲，這是最好的情況
        if self._stopped.is_set() or token.is_cancelled:
            return
        chunks = self._pick(language)
        if not chunks:
            return
        for i, src in enumerate(chunks):
            chunk = AudioChunk(
                pcm=src.pcm, sample_rate=src.sample_rate, turn_id=turn_id,
                sentence_index=GAP_SENTENCE_INDEX, chunk_index=i, channels=src.channels,
            )
            await sink.write(chunk)
            self.emitted_duration_s += chunk.duration_s
        if on_first_audio is not None:
            on_first_audio()

    async def _wait(self, seconds: float) -> bool:
        """等 seconds；期間被 stop 就回 True。"""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopped.wait(), timeout=max(0.0, seconds))
            return True
        return False


__all__ = ["GapFiller", "GapFillerPolicy", "GAP_SENTENCE_INDEX"]
