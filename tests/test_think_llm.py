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


# --- 記憶注入（Phase 4a）---


class FakeMemory:
    """回固定文字的假記憶。記錄查詢，可設定失敗。"""

    def __init__(self, result: str | None = "他養的貓叫毛球", fail: bool = False):
        self.result = result
        self.fail = fail
        self.queries: list[str] = []

    async def retrieve(self, text: str) -> str | None:
        if self.fail:
            raise RuntimeError("記憶引擎壞了")
        self.queries.append(text)
        return self.result


def _last_user_content(backend: FakeBackend) -> str:
    for msg in reversed(backend.last_messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "user":
            return msg.get("content") if isinstance(msg, dict) else msg.content
    return ""


async def test_記憶下一輪才注入():
    # delay 給事件迴圈讓 retrieve task 跑完——真 backend 的網路 await 天然如此
    backend = FakeBackend(delay_s=0.001)
    stage = LLMThinkStage(backend, memory=FakeMemory())

    await collect(stage, "我家的貓很可愛")
    # 本輪：retrieve 與生成平行，結果不進這一輪的 prompt
    assert "毛球" not in _last_user_content(backend)
    await stage.commit("t1", "好可愛。", "好可愛。")

    await collect(stage, "牠叫什麼名字？")
    content = _last_user_content(backend)
    assert content.startswith("牠叫什麼名字？")
    assert "毛球" in content


async def test_記憶不寫進歷史():
    backend = FakeBackend(delay_s=0.001)
    stage = LLMThinkStage(backend, memory=FakeMemory())
    await collect(stage, "第一輪")
    await stage.commit("t1", "回一", "回一")
    await collect(stage, "第二輪")
    await stage.commit("t2", "回二", "回二")

    assert all("毛球" not in m["content"] for m in stage.history)


async def test_記憶注入後清空_不重複注入():
    backend = FakeBackend(delay_s=0.001)
    memory = FakeMemory()
    stage = LLMThinkStage(backend, memory=memory)
    await collect(stage, "第一輪")
    await stage.commit("t1", "回一", "回一")

    memory.result = None  # 第二輪 retrieve 撈不到東西
    await collect(stage, "第二輪")
    await stage.commit("t2", "回二", "回二")
    assert "毛球" in _last_user_content(backend)  # 第一輪的結果在這裡用掉

    await collect(stage, "第三輪")
    assert "毛球" not in _last_user_content(backend)  # 不能再出現


async def test_retrieve_失敗不炸管線():
    backend = FakeBackend()
    stage = LLMThinkStage(backend, memory=FakeMemory(fail=True))
    sentences = await collect(stage, "你好")
    assert sentences  # 生成照常


async def test_沒接記憶時行為不變():
    backend = FakeBackend()
    stage = LLMThinkStage(backend)
    await collect(stage, "你好")
    assert _last_user_content(backend) == "你好"


async def test_profile_接在人設之後_標記之前():
    from echo_stream.core.emotion import MARKER_PROMPT

    backend = FakeBackend()
    stage = LLMThinkStage(backend, system_prompt="人設在前", emotion_markers=True)
    stage.profile = "- 妹妹：使用者的妹妹即將就讀筑波大學"
    await collect(stage)
    sys_ = backend.last_system
    assert sys_.startswith("人設在前")
    assert "關於這位使用者" in sys_ and "筑波大學" in sys_
    assert sys_.index("筑波大學") < sys_.index(MARKER_PROMPT), "profile 在情緒標記指示之前"


async def test_profile_空字串不加區塊():
    backend = FakeBackend()
    stage = LLMThinkStage(backend, system_prompt="人設")
    stage.profile = ""
    await collect(stage)
    assert backend.last_system == "人設"


# --- 工具呼叫（C 段）---


class _Call:
    """模仿 SessionControl 的 ToolCall 事件——think stage 只看形狀。"""

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


def _d(m: Any) -> dict[str, Any]:
    """messages 在有 echo_thought_core 時是 pydantic Message、沒有時是 dict——測試兩邊都要過。"""
    if isinstance(m, dict):
        return m
    out = {"role": m.role, "content": m.content}
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
        out["tool_name"] = m.tool_name
    if m.metadata.get("tool_calls"):
        out["tool_calls"] = m.metadata["tool_calls"]
    return out


class ToolBackend:
    """第一段：（可選文字後）提出工具呼叫；收到 tool 結果後第二段吐續寫。"""

    def __init__(
        self,
        call: _Call | None = None,
        preface: str = "",
        final: str = "你上次說你養了一隻叫小黑的貓。",
        call_again: bool = False,
    ) -> None:
        self.call = call or _Call("call_1", "deep_recall", '{"query": "貓"}')
        self.preface = preface
        self.final = final
        self.call_again = call_again
        self.requests: list[dict[str, Any]] = []

    async def stream_query(
        self, messages: Sequence[Any], system_instruction: str | None = None, **kwargs
    ) -> AsyncIterator[Any]:
        self.requests.append({"messages": list(messages), **kwargs})
        has_tool_result = any(_d(m)["role"] == "tool" for m in messages)
        if "tools" in kwargs and (not has_tool_result or self.call_again):
            for ch in self.preface:
                yield ch
            yield self.call
            return
        for ch in self.final:
            yield ch


def _executor(log: list, result: str = "找到：使用者養了一隻貓叫小黑"):
    async def run(name: str, args: dict) -> dict:
        log.append((name, args))
        return {"success": True, "result": result}

    return run


async def test_工具往返_先filler後續寫_結果接回第二段():
    log: list = []
    backend = ToolBackend()
    stage = LLMThinkStage(
        backend, tools=[{"type": "function", "function": {"name": "deep_recall"}}],
        tool_executor=_executor(log), tool_filler="嗯，讓我想一下。",
    )
    texts = [s.text for s in await collect(stage, "我的貓叫什麼")]

    assert texts[0] == "嗯，讓我想一下。", "開口前就查工具 → filler 先出"
    assert "".join(texts[1:]) == "你上次說你養了一隻叫小黑的貓。"
    assert log == [("deep_recall", {"query": "貓"})]

    assert "tools" in backend.requests[0]
    second = [_d(m) for m in backend.requests[1]["messages"]]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"] == "嗯，讓我想一下。", "filler 進 assistant 訊息，模型知道自己說過"
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1] == {
        "role": "tool", "content": "找到：使用者養了一隻貓叫小黑",
        "tool_call_id": "call_1", "tool_name": "deep_recall",
    }
    assert stage.last_tool_calls[0]["name"] == "deep_recall"
    assert stage.last_tool_calls[0]["ok"] is True
    assert stage.last_tool_calls[0]["args"] == {"query": "貓"}


async def test_模型已開口則不插filler():
    backend = ToolBackend(preface="好的，")
    stage = LLMThinkStage(
        backend, tools=[{"x": 1}], tool_executor=_executor([]),
    )
    texts = "".join(s.text for s in await collect(stage))
    assert texts.startswith("好的，")
    assert "讓我想一下" not in texts
    assert _d(backend.requests[1]["messages"][-2])["content"] == "好的，"


async def test_沒給工具時請求不帶tools():
    backend = FakeBackend()
    stage = LLMThinkStage(backend)
    await collect(stage)
    assert stage.last_tool_calls == []


async def test_超過最大往返就收回tools():
    backend = ToolBackend(call_again=True)
    stage = LLMThinkStage(
        backend, tools=[{"x": 1}], tool_executor=_executor([]), max_tool_rounds=2,
    )
    texts = "".join(s.text for s in await collect(stage))
    assert texts.endswith("你上次說你養了一隻叫小黑的貓。")
    assert len(backend.requests) == 3
    assert "tools" in backend.requests[0] and "tools" in backend.requests[1]
    assert "tools" not in backend.requests[2], "第三段不帶 tools，逼模型回答"
    assert len(stage.last_tool_calls) == 2


async def test_executor炸了回錯誤結果不斷鏈():
    async def boom(name, args):
        raise RuntimeError("engine down")

    backend = ToolBackend()
    stage = LLMThinkStage(backend, tools=[{"x": 1}], tool_executor=boom)
    texts = "".join(s.text for s in await collect(stage))
    assert "小黑" in texts
    tool_msg = _d(backend.requests[1]["messages"][-1])
    assert tool_msg["role"] == "tool" and "engine down" in tool_msg["content"]
    assert stage.last_tool_calls[0]["ok"] is False


async def test_壞掉的arguments退成空dict():
    log: list = []
    backend = ToolBackend(call=_Call("c", "deep_recall", "{not json"))
    stage = LLMThinkStage(backend, tools=[{"x": 1}], tool_executor=_executor(log))
    await collect(stage)
    assert log == [("deep_recall", {})]


async def test_工具往返不進跨輪歷史():
    backend = ToolBackend()
    stage = LLMThinkStage(backend, tools=[{"x": 1}], tool_executor=_executor([]))
    await collect(stage, "我的貓")
    await stage.commit("t", "嗯，讓我想一下。你上次說你養了一隻叫小黑的貓。", "")
    assert [m["role"] for m in stage.history] == ["user", "assistant"]


async def test_工具區間有打點():
    from echo_stream.core.tracer import MARK_TOOL_DONE, MARK_TOOL_START

    tracer = LatencyTracer()
    backend = ToolBackend()
    stage = LLMThinkStage(
        backend, tools=[{"x": 1}], tool_executor=_executor([]), tracer=tracer,
    )
    utt = Utterance(text="貓")
    tracer.start(utt.turn_id)
    _ = [s async for s in stage.stream(utt, CancellationToken())]
    trace = tracer.get(utt.turn_id)
    assert MARK_TOOL_START in trace.marks and MARK_TOOL_DONE in trace.marks
    assert trace.segments_ms()["tool"] is not None
    assert MARK_THINK_FIRST_TOKEN in trace.marks, "filler 算首 token"
