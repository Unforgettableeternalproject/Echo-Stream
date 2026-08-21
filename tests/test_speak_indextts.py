"""IndexTTS2 adapter 的測試。

用**假 backend**——不需要 GPU、不需要 echo_tts。
Phase 1 的多數 bug 會出在 push→pull 橋接而不是模型，
橋接邏輯必須能在秒級迭代下驗證。

真模型的驗收走 `python -m echo_stream tts-check`（需要 GPU）。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import pytest

from echo_stream.adapters.speak_indextts import IndexTTS2SpeakStage, tensor_to_pcm16
from echo_stream.contracts.cancellation import CancellationToken, CancelReason
from echo_stream.contracts.types import Sentence
from tests.conftest import FakeTTSBackend as FakeBackend, sentence_stream

np = pytest.importorskip("numpy", reason="adapter 的音訊轉換需要 numpy")


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
    # 關掉段緣後處理——這裡驗的是原始時長換算，靜音墊另有專屬測試
    stage = IndexTTS2SpeakStage(
        backend=FakeBackend(segments_per_sentence=1, segment_seconds=0.5),
        edge_fade_ms=0,
        segment_silence_ms=0,
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


# --- 段緣平滑（消爆音） ---


def test_段緣淡入淡出從零開始():
    from echo_stream.adapters.speak_indextts import smooth_segment_edges

    pcm = b"\x00\x40" * 1000  # 恆定振幅 16384
    out = smooth_segment_edges(pcm, 22050, fade_ms=4.0, tail_silence_ms=0)
    import struct

    samples = struct.unpack(f"<{len(out) // 2}h", out)
    assert samples[0] == 0  # 段首歸零——邊界不連續就是爆音來源
    assert samples[-1] == 0
    assert samples[500] == 16384  # 中段不受影響


def test_段尾靜音墊長度正確():
    from echo_stream.adapters.speak_indextts import smooth_segment_edges

    pcm = b"\x00\x40" * 1000
    out = smooth_segment_edges(pcm, 22050, fade_ms=0, tail_silence_ms=20.0)
    pad_samples = int(22050 * 20 / 1000)
    assert len(out) == len(pcm) + pad_samples * 2
    assert out[len(pcm):] == b"\x00\x00" * pad_samples


def test_過短的段不會因淡化壞掉():
    from echo_stream.adapters.speak_indextts import smooth_segment_edges

    pcm = b"\x00\x40" * 4  # 4 個樣本，比淡化窗還短
    out = smooth_segment_edges(pcm, 22050, fade_ms=4.0, tail_silence_ms=0)
    assert len(out) == len(pcm)


# --- 逐句 style 套用 ---


async def test_style_的情緒向量乘上強度():
    from echo_stream.contracts.style import SpeechStyle
    from echo_stream.contracts.types import Sentence

    backend = FakeBackend(segments_per_sentence=1)
    stage = IndexTTS2SpeakStage(backend=backend)
    await stage.prepare()

    style = SpeechStyle(emotion={"happy": 0.8}, intensity=0.5)
    stage._apply_style(style)
    assert backend.emotion_vector[0] == pytest.approx(0.4)  # happy 是第 0 維


async def test_中性句還原角色檔預設情緒():
    """一句帶情緒之後的中性句**不能殘留**上一句的情緒。"""
    from echo_stream.contracts.style import SpeechStyle

    backend = FakeBackend(segments_per_sentence=1)
    backend.emotion_vector = [0.0] * 7 + [0.5]  # 角色檔預設：calm 0.5
    stage = IndexTTS2SpeakStage(backend=backend)
    await stage.prepare()

    stage._apply_style(SpeechStyle(emotion={"angry": 0.8}))
    assert backend.emotion_vector[1] > 0  # angry 已套上

    stage._apply_style(None)
    assert backend.emotion_vector == [0.0] * 7 + [0.5]  # 還原，不是殘留


async def test_style_語速退場時回到基準值():
    from echo_stream.contracts.style import SpeechStyle

    backend = FakeBackend(segments_per_sentence=1)
    stage = IndexTTS2SpeakStage(backend=backend, speed=0.2)
    await stage.prepare()
    assert backend.speed_offset == 0.2  # 啟動基準

    stage._apply_style(SpeechStyle(speed=1.3))
    assert backend.speed_offset != 0.2

    stage._apply_style(None)
    assert backend.speed_offset == 0.2


def test_sanitize_把列表破折號換成頓點():
    from echo_stream.adapters.speak_indextts import sanitize_for_tts

    assert sanitize_for_tts("小說 - 死亡擱淺") == "小說，死亡擱淺"
    # 連字號緊貼單字（well-known、2-3）不受影響
    assert sanitize_for_tts("a well-known game") == "a well-known game"
