"""IndexTTS2 adapter 的測試。

用**假 backend**——不需要 GPU、不需要 echo_tts。
Phase 1 的多數 bug 會出在 push→pull 橋接而不是模型，
橋接邏輯必須能在秒級迭代下驗證。

真模型的驗收走 `python -m echo_stream tts-check`（需要 GPU）。
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import AsyncIterator

import pytest

from echo_stream.adapters.speak_indextts import IndexTTS2SpeakStage, tensor_to_pcm16
from echo_stream.contracts.cancellation import CancellationToken, CancelReason
from echo_stream.contracts.types import Sentence

np = pytest.importorskip("numpy", reason="adapter 的音訊轉換需要 numpy")


class FakeBackend:
    """模擬 IndexTTS2Backend 的同步 push callback 行為。"""

    def __init__(
        self,
        segments_per_sentence: int = 2,
        segment_seconds: float = 0.5,
        synth_delay_s: float = 0.0,
        sample_rate: int = 22050,
    ) -> None:
        self.segments_per_sentence = segments_per_sentence
        self.segment_seconds = segment_seconds
        self.synth_delay_s = synth_delay_s
        self.sample_rate = sample_rate
        self.calls: list[str] = []
        self.loaded = False
        self.completed: list[str] = []
        """完整跑完 generate 的句子。被取消時不會進來——用來驗證上游確實中止。"""

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        text,
        output_path,
        on_segment_audio=None,
        language=None,
        emotion_overrides=None,
    ):
        self.calls.append(text)
        total = self.segments_per_sentence
        samples = int(self.sample_rate * self.segment_seconds)
        for idx in range(total):
            if self.synth_delay_s:
                time.sleep(self.synth_delay_s)
            wave = np.sin(
                2 * math.pi * 220.0 * np.arange(samples) / self.sample_rate
            ).astype("float32") * 0.2
            if on_segment_audio is not None:
                on_segment_audio(wave, idx, total)
        self.completed.append(text)
        return output_path


async def sentence_stream(texts: list[str], turn_id: str = "t1") -> AsyncIterator[Sentence]:
    for i, text in enumerate(texts):
        yield Sentence(
            text=text,
            turn_id=turn_id,
            index=i,
            is_first=i == 0,
            is_last=i == len(texts) - 1,
        )


# --- 音訊轉換 ---


def test_轉成_int16_pcm():
    arr = np.array([0.0, 1.0, -1.0], dtype="float32")
    pcm = tensor_to_pcm16(arr)
    assert len(pcm) == 6  # 3 samples × 2 bytes
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767]


def test_超出範圍會被夾住():
    """引擎偶爾會吐出略超過 ±1.0 的值，不夾會 wrap around 變成爆音。"""
    arr = np.array([2.5, -2.5], dtype="float32")
    assert np.frombuffer(tensor_to_pcm16(arr), dtype="<i2").tolist() == [32767, -32767]


def test_多聲道取第一軌():
    arr = np.zeros((2, 4), dtype="float32")
    arr[0] = 1.0
    assert np.frombuffer(tensor_to_pcm16(arr), dtype="<i2").tolist() == [32767] * 4


# --- 基本串流 ---


async def test_產出音訊塊():
    backend = FakeBackend(segments_per_sentence=2)
    stage = IndexTTS2SpeakStage(backend=backend)
    token = CancellationToken()

    chunks = [
        c async for c in stage.stream(sentence_stream(["第一句。", "第二句。"]), token)
    ]

    assert len(chunks) == 4  # 2 句 × 2 段
    assert backend.calls == ["第一句。", "第二句。"]


async def test_索引對得上():
    stage = IndexTTS2SpeakStage(backend=FakeBackend(segments_per_sentence=2))
    chunks = [
        c async for c in stage.stream(sentence_stream(["一。", "二。"]), CancellationToken())
    ]

    assert [c.sentence_index for c in chunks] == [0, 0, 1, 1]
    assert [c.chunk_index for c in chunks] == [0, 1, 0, 1]


async def test_最後一段標記_is_last():
    stage = IndexTTS2SpeakStage(backend=FakeBackend(segments_per_sentence=2))
    chunks = [
        c async for c in stage.stream(sentence_stream(["一。", "二。"]), CancellationToken())
    ]
    assert [c.is_last for c in chunks] == [False, False, False, True]


async def test_空句子被跳過():
    backend = FakeBackend()
    stage = IndexTTS2SpeakStage(backend=backend)

    async def stream_with_empty():
        yield Sentence(text="有內容。", turn_id="t", index=0, is_first=True)
        yield Sentence(text="", turn_id="t", index=1, is_last=True)

    _ = [c async for c in stage.stream(stream_with_empty(), CancellationToken())]
    assert backend.calls == ["有內容。"]


async def test_音訊時長正確():
    stage = IndexTTS2SpeakStage(
        backend=FakeBackend(segments_per_sentence=1, segment_seconds=0.5)
    )
    chunks = [c async for c in stage.stream(sentence_stream(["一。"]), CancellationToken())]
    assert abs(chunks[0].duration_s - 0.5) < 0.01


async def test_prepare_會載入_backend():
    backend = FakeBackend()
    stage = IndexTTS2SpeakStage(backend=backend)
    await stage.prepare()
    assert backend.loaded


# --- 背壓 ---


async def test_有界通道把背壓傳到推理執行緒():
    """無界的話 TTS 會一路合成到底，barge-in 就省不到 GPU 時間。"""
    backend = FakeBackend(segments_per_sentence=6, segment_seconds=0.05)
    stage = IndexTTS2SpeakStage(backend=backend, queue_size=1)
    token = CancellationToken()

    stream = stage.stream(sentence_stream(["一句話。"]), token)
    first = await stream.__anext__()
    assert first is not None

    # 只拉一段就停手，讓通道塞住
    await asyncio.sleep(0.2)
    assert len(backend.calls) == 1

    with pytest.raises(Exception):  # noqa: B017 - 取消或關閉都可接受
        token.cancel(CancelReason.BARGE_IN)
        async for _ in stream:
            pass


# --- 取消 ---


async def test_取消會中止後續句子的合成():
    backend = FakeBackend(segments_per_sentence=1, synth_delay_s=0.05)
    stage = IndexTTS2SpeakStage(backend=backend, queue_size=2)
    token = CancellationToken()

    got = 0
    with pytest.raises(Exception):  # noqa: B017
        async for _ in stage.stream(
            sentence_stream(["一。", "二。", "三。", "四。", "五。"]), token
        ):
            got += 1
            if got == 1:
                token.cancel(CancelReason.BARGE_IN)

    await asyncio.sleep(0.2)
    assert len(backend.calls) < 5, "取消後不該把剩下的句子全部合成完"


async def test_取消能穿透到執行緒中的推理():
    """callback 是唯一的協作點——同步推理不在 asyncio 的取消範圍內。"""
    started = threading.Event()
    backend = FakeBackend(segments_per_sentence=20, synth_delay_s=0.02)
    original = backend.generate

    def tracked(*args, **kwargs):
        started.set()
        return original(*args, **kwargs)

    backend.generate = tracked
    stage = IndexTTS2SpeakStage(backend=backend, queue_size=1)
    token = CancellationToken()

    stream = stage.stream(sentence_stream(["很長的一句話。"]), token)
    await stream.__anext__()
    assert started.is_set()

    token.cancel(CancelReason.BARGE_IN)
    with pytest.raises(Exception):  # noqa: B017
        async for _ in stream:
            pass

    await asyncio.sleep(0.3)
    assert not backend.completed, "generate 應該被 callback 拋出的例外中止，而不是跑完"


# --- 錯誤傳播 ---


async def test_backend_爆炸會傳給消費端():
    class BoomBackend(FakeBackend):
        def generate(self, *args, **kwargs):
            raise RuntimeError("引擎爆了")

    stage = IndexTTS2SpeakStage(backend=BoomBackend())
    with pytest.raises(RuntimeError, match="引擎爆了"):
        async for _ in stage.stream(sentence_stream(["一。"]), CancellationToken()):
            pass


# --- 與 PipelineRunner 整合 ---


async def test_接進_PipelineRunner():
    from echo_stream.contracts.types import TurnPhase, Utterance
    from echo_stream.core.pipeline import PipelineRunner
    from echo_stream.core.splitter import SplitPolicy
    from echo_stream.fakes import FakeAudioSink, FakeThinkStage

    backend = FakeBackend(segments_per_sentence=1, segment_seconds=0.1)
    think = FakeThinkStage(
        responses=["第一句很短。第二句也不長。"],
        first_token_delay_s=0.0,
        token_interval_s=0.0,
        memory_delay_s=0.0,
        split_policy=SplitPolicy(),
    )
    runner = PipelineRunner(
        think_stage=think,
        speak_stage=IndexTTS2SpeakStage(backend=backend),
        sink=FakeAudioSink(realtime=False),
    )
    result = await runner.run_turn(Utterance(text="你好"))

    assert result.phase is TurnPhase.DONE
    assert result.spoken_text == result.generated_text
    assert backend.calls
