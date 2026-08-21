"""LLMThinkStage 的測試。

用假 backend——不打真 API、不需要 echo_thought_core。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from echo_stream.adapters.think_llm import EchoBackend, LLMThinkStage
from echo_stream.contracts.cancellation import (
    CancellationToken,
    CancelledError,
    CancelReason,
)
from echo_stream.contracts.style import SpeechStyle
from echo_stream.contracts.types import Utterance
from echo_stream.core.splitter import SplitPolicy
from echo_stream.core.tracer import MARK_THINK_FIRST_TOKEN, LatencyTracer


class FakeBackend:
    """逐字吐出固定回應，記錄收到的參數。"""

    def __init__(self, response: str = "第一句很短。第二句也不長。", delay_s: float = 0.0):
        self.response = response
        self.delay_s = delay_s
        self.last_messages: list[Any] = []
        self.last_system: str | None = None
        self.call_count = 0
        self.delivered = 0
        """實際吐出的字數。取消後應該停止增加。"""

    async def stream_query(
        self, messages: Sequence[Any], system_instruction: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        self.last_system = system_instruction
        self.call_count += 1
        for ch in self.response:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            self.delivered += 1
            yield ch


async def collect(stage: LLMThinkStage, text: str = "你好", token=None):
    token = token or CancellationToken()
    return [
        s async for s in stage.stream(Utterance(text=text), token) if s.text
    ]


# --- 基本串流 ---


async def test_切成句子():
    stage = LLMThinkStage(
        FakeBackend(), split_policy=SplitPolicy(first_min_weight=1.0, min_weight=1.0)
    )
    sentences = await collect(stage)
    assert [s.text for s in sentences] == ["第一句很短。", "第二句也不長。"]


async def test_首句有標記():
    stage = LLMThinkStage(
        FakeBackend(), split_policy=SplitPolicy(first_min_weight=1.0, min_weight=1.0)
    )
    sentences = await collect(stage)
    assert sentences[0].is_first
    assert not sentences[1].is_first


async def test_system_prompt_傳給_backend():
    backend = FakeBackend()
    stage = LLMThinkStage(backend, system_prompt="你是諾薇亞")
    await collect(stage)
    assert backend.last_system == "你是諾薇亞"


async def test_沒有_system_prompt_時傳_None():
    """空字串與 None 對 API 是不同的東西。"""
    backend = FakeBackend()
    await collect(LLMThinkStage(backend))
    assert backend.last_system is None


# --- 對話歷史 ---


async def test_歷史在_commit_後才寫入():
    """stream 期間不寫——被中止時才有機會只寫實際播出的部分。"""
    stage = LLMThinkStage(FakeBackend())
    await collect(stage)
    assert stage.history == []

    await stage.commit("t1", "第一句很短。", "第一句很短。第二句也不長。")
    assert len(stage.history) == 2


async def test_commit_寫的是_spoken_不是_generated():
    """被插話中止時兩者不同，寫錯會讓 LLM 以為自己講完了整段。"""
    stage = LLMThinkStage(FakeBackend())
    await collect(stage)
    await stage.commit("t1", "只播出這句。", "生成了很多很多內容。")

    assert stage.history[-1] == {"role": "assistant", "content": "只播出這句。"}


async def test_使用者的話與回應成對進歷史():
    """不會出現「使用者問了但機器人的回答不見了」的半套狀態。"""
    stage = LLMThinkStage(FakeBackend())
    await collect(stage, "今天天氣如何")
    await stage.commit("t1", "很好。", "很好。")

    assert stage.history[0] == {"role": "user", "content": "今天天氣如何"}
    assert stage.history[1]["role"] == "assistant"


async def test_什麼都沒播出時不寫_assistant():
    stage = LLMThinkStage(FakeBackend())
    await collect(stage)
    await stage.commit("t1", "", "生成了但沒播出")

    roles = [m["role"] for m in stage.history]
    assert "assistant" not in roles


async def test_歷史會送給_backend():
    backend = FakeBackend()
    stage = LLMThinkStage(backend)
    stage.history = [
        {"role": "user", "content": "第一輪"},
        {"role": "assistant", "content": "第一輪回應"},
    ]
    await collect(stage, "第二輪")

    contents = [
        m["content"] if isinstance(m, dict) else m.content for m in backend.last_messages
    ]
    assert contents == ["第一輪", "第一輪回應", "第二輪"]


async def test_歷史超過上限會截斷():
    backend = FakeBackend()
    stage = LLMThinkStage(backend, history_limit=2)
    stage.history = [
        {"role": "user", "content": f"第{i}輪"} for i in range(10)
    ]
    await collect(stage, "最新")

    assert len(backend.last_messages) == 3  # 2 則歷史 + 1 則新的


# --- 取消 ---


async def test_取消會停止消費_token():
    backend = FakeBackend(response="一二三四五六七八九十。" * 5, delay_s=0.002)
    stage = LLMThinkStage(backend)
    token = CancellationToken()

    got = 0
    with pytest.raises(CancelledError):
        async for _ in stage.stream(Utterance(text="你好"), token):
            got += 1
            token.cancel(CancelReason.BARGE_IN)

    delivered_at_cancel = backend.delivered
    await asyncio.sleep(0.1)
    assert backend.delivered == delivered_at_cancel, "取消後不該繼續拉 token"


# --- 打點 ---


async def test_首_token_有打點():
    tracer = LatencyTracer()
    stage = LLMThinkStage(FakeBackend(), tracer=tracer)
    tracer.start("t1")

    async for _ in stage.stream(Utterance(text="你好", turn_id="t1"), CancellationToken()):
        pass

    assert MARK_THINK_FIRST_TOKEN in tracer.get("t1").marks


# --- 語音風格 ---


async def test_預設風格會套到每一句():
    style = SpeechStyle(emotion={"calm": 0.8})
    stage = LLMThinkStage(FakeBackend(), default_style=style)
    sentences = await collect(stage)

    assert all(s.style is style for s in sentences)


async def test_沒設風格時句子不帶_style():
    stage = LLMThinkStage(FakeBackend())
    sentences = await collect(stage)
    assert all(s.style is None for s in sentences)


# --- Memory 平行 ---


async def test_memory_retrieve_不擋首句():
    """§7.4 定案：retrieve 與 LLM prefill 平行。"""

    class SlowMemory:
        def __init__(self):
            self.done = False

        async def retrieve(self, text):
            await asyncio.sleep(0.3)
            self.done = True

    memory = SlowMemory()
    stage = LLMThinkStage(FakeBackend(), memory=memory)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    first = None
    async for sentence in stage.stream(Utterance(text="你好"), CancellationToken()):
        if sentence.text:
            first = loop.time() - t0
            break

    assert first is not None
    assert first < 0.2, "首句不該等 memory 的 0.3 秒"


async def test_memory_爆炸不會炸掉管線():
    """retrieve 是旁路，失敗不該讓整輪對話掛掉。"""

    class BoomMemory:
        async def retrieve(self, text):
            raise RuntimeError("記憶庫爆了")

    stage = LLMThinkStage(FakeBackend(), memory=BoomMemory())
    sentences = await collect(stage)
    assert sentences


# --- EchoBackend ---


async def test_echo_backend_回傳使用者的話():
    stage = LLMThinkStage(EchoBackend(prefix=""), split_policy=SplitPolicy())
    sentences = await collect(stage, "測試一下。")
    assert "".join(s.text for s in sentences) == "測試一下。"


# --- 與 PipelineRunner 整合 ---


async def test_接進_PipelineRunner():
    from echo_stream.contracts.types import TurnPhase
    from echo_stream.core.pipeline import PipelineRunner
    from echo_stream.fakes import FakeAudioSink, FakeSpeakStage

    stage = LLMThinkStage(FakeBackend())
    runner = PipelineRunner(
        think_stage=stage,
        speak_stage=FakeSpeakStage(first_chunk_extra_s=0.0, rtf=0.0),
        sink=FakeAudioSink(realtime=False),
    )
    result = await runner.run_turn(Utterance(text="你好"))

    assert result.phase is TurnPhase.DONE
    assert result.spoken_text == result.generated_text
    assert len(stage.history) == 2


# --- 逐句情緒標記 ---


async def test_情緒標記轉成_style_並從文字剔除():
    backend = FakeBackend(response="[開心]太好了，這下成功了！接下來繼續處理剩的。")
    stage = LLMThinkStage(backend, emotion_markers=True)
    sentences = await collect(stage)
    assert all("[" not in s.text for s in sentences)
    assert sentences[0].style is not None
    assert sentences[0].style.emotion.get("happy", 0) > 0


async def test_標記開啟時_system_prompt_接上指示():
    from echo_stream.core.emotion import MARKER_PROMPT

    backend = FakeBackend()
    stage = LLMThinkStage(backend, system_prompt="人設在前", emotion_markers=True)
    await collect(stage)
    assert backend.last_system.startswith("人設在前")
    assert MARKER_PROMPT in backend.last_system


async def test_標記關閉時不動_prompt_也不解析():
    backend = FakeBackend(response="[開心]照舊輸出，不要動我的方括號內容。")
    stage = LLMThinkStage(backend, system_prompt="人設", emotion_markers=False)
    sentences = await collect(stage)
    assert backend.last_system == "人設"
    assert any("[開心]" in s.text for s in sentences)


async def test_無標記的句子退回_default_style():
    default = SpeechStyle(emotion={"calm": 0.5}, speed=1.1)
    backend = FakeBackend(response="[開心]第一句有標記喔！後面這句就沒有標記了。")
    stage = LLMThinkStage(backend, emotion_markers=True, default_style=default)
    sentences = await collect(stage)
    assert sentences[0].style.emotion.get("happy", 0) > 0
    assert sentences[0].style.speed == 1.1  # base 的語速被保留
    assert sentences[-1].style is default


async def test_列表符號從句首剝除():
    backend = FakeBackend(
        response="我推薦幾款遊戲喔，你聽聽看。- 《文明七》：每次都想著只玩一回合就好。"
    )
    stage = LLMThinkStage(backend)
    sentences = await collect(stage)
    assert all(not s.text.lstrip().startswith("-") for s in sentences)
