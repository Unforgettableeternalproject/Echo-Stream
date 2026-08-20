"""測試共用元件。"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator

import pytest

from echo_stream.contracts.types import Sentence


class FakeTTSBackend:
    """模擬 ``IndexTTS2Backend`` 的同步 push callback 行為。

    存在的理由：Phase 1 的多數 bug 會出在 push→pull 橋接，不是模型。
    橋接邏輯必須能在沒有 GPU、沒有 echo_tts 的環境下秒級迭代。
    """

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
        """完整跑完 generate 的句子。被取消時不會進來——
        用來驗證取消確實中止了上游，而不是只丟棄了結果。"""

        self.emotion_vector: list[float] | None = None
        self.speed_offset: float | None = None

    def load(self) -> None:
        self.loaded = True

    def set_speed(self, value: float) -> float:
        self.speed_offset = value
        return value

    def unload(self) -> None:
        self.loaded = False

    def generate(
        self,
        text,
        output_path,
        on_segment_audio=None,
        language=None,
        emotion_overrides=None,
    ):
        import numpy as np

        self.calls.append(text)
        total = self.segments_per_sentence
        samples = int(self.sample_rate * self.segment_seconds)
        for idx in range(total):
            if self.synth_delay_s:
                time.sleep(self.synth_delay_s)
            wave = (
                np.sin(2 * math.pi * 220.0 * np.arange(samples) / self.sample_rate).astype(
                    "float32"
                )
                * 0.2
            )
            if on_segment_audio is not None:
                on_segment_audio(wave, idx, total)
        self.completed.append(text)
        return output_path


@pytest.fixture
def fake_tts_backend():
    """預設設定的假 TTS backend。需要自訂參數時直接用 :class:`FakeTTSBackend`。"""
    return FakeTTSBackend()


async def sentence_stream(
    texts: list[str], turn_id: str = "t1", style=None
) -> AsyncIterator[Sentence]:
    """把字串列表變成 Sentence 流。"""
    for i, text in enumerate(texts):
        yield Sentence(
            text=text,
            turn_id=turn_id,
            index=i,
            is_first=i == 0,
            is_last=i == len(texts) - 1,
            style=style,
        )
