"""GapFiller（D 段）——墊片音的排隊語意、一輪一聲、按語言挑、打點。"""

from __future__ import annotations

import asyncio

import pytest

from echo_stream import defaults
from echo_stream.contracts.cancellation import CancellationToken
from echo_stream.contracts.types import AudioChunk, Utterance
from echo_stream.core.gap_filler import GAP_SENTENCE_INDEX, GapFiller, GapFillerPolicy
from echo_stream.core.pipeline import PipelineRunner
from echo_stream.core.tracer import MARK_GAP_FIRST_AUDIO, LatencyTracer
from echo_stream.fakes import FakeSpeakStage, FakeThinkStage


class ListSink:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.played_duration_s = 0.0

    async def write(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)
        self.played_duration_s += chunk.duration_s

    def stop(self) -> None:
        pass

    async def drain(self) -> None:
        pass


def _policy(**kw) -> GapFillerPolicy:
    base = dict(
        enabled=True,
        start_delay_s=0.0,
        phrases={"zh": ["嗯…"], "en": ["Hmm…"]},
    )
    base.update(kw)
    return GapFillerPolicy(**base)


async def _built(policy: GapFillerPolicy | None = None) -> GapFiller:
    gap = GapFiller(policy or _policy())
    await gap.build(FakeSpeakStage(first_chunk_extra_s=0.0, rtf=0.0))
    return gap


async def test_build_把所有語言的文案都合成進語音庫():
    gap = await _built()
    assert gap.ready
    assert set(gap._bank) == {"嗯…", "Hmm…"}
    assert gap.build_seconds is not None


async def test_按語言挑_沒對應池子退回預設語言():
    gap = await _built(_policy(phrases={"zh": ["嗯…"], "en": ["Hmm…"]}, default_language="zh"))
    assert gap.phrases_for("en") == ["Hmm…"]
    assert gap.phrases_for("en-US") == ["Hmm…"]
    assert gap.phrases_for("ja") == ["嗯…"], "沒日文池子 → 預設語言"
    assert gap.phrases_for(None) == ["嗯…"]


async def test_不連續重複():
    gap = await _built(_policy(phrases={"zh": ["A", "B"]}))
    picks = []
    for _ in range(10):
        gap._pick("zh")
        picks.append(gap.last_emitted)
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))


async def test_墊片排在真音訊前面_且只有一聲():
    """核心語意：排隊、不打斷、一輪最多一聲。"""
    sink = ListSink()
    gap = await _built()
    token = CancellationToken()
    gap.start(sink, "t", token, language="zh")
    await asyncio.sleep(0.05)
    await gap.stop()  # 真音訊來了
    await sink.write(
        AudioChunk(pcm=b"\x00\x00" * 100, sample_rate=100, turn_id="t", sentence_index=0)
    )
    await asyncio.sleep(0.05)

    idx = [c.sentence_index for c in sink.chunks]
    assert idx[0] == GAP_SENTENCE_INDEX
    assert idx[-1] == 0
    assert idx.count(GAP_SENTENCE_INDEX) == len(sink.chunks) - 1
    assert gap.last_emitted == "嗯…"
    assert gap.emitted_duration_s > 0


async def test_真音訊比起播延遲還快_就不出聲():
    sink = ListSink()
    gap = await _built(_policy(start_delay_s=0.5))
    gap.start(sink, "t", CancellationToken())
    await gap.stop()  # 立刻就有真音訊
    assert sink.chunks == []
    assert gap.last_emitted is None
    assert gap.emitted_duration_s == 0.0


async def test_關閉或沒語音庫就什麼都不做():
    sink = ListSink()
    gap = GapFiller(_policy(enabled=False))
    gap.start(sink, "t", CancellationToken())
    await asyncio.sleep(0.02)
    assert sink.chunks == []
    gap2 = GapFiller(_policy())  # enabled 但沒 build
    gap2.start(sink, "t", CancellationToken())
    await asyncio.sleep(0.02)
    assert sink.chunks == []


async def test_runner_整合_墊片先於真音訊_打點_感知TTFA():
    tracer = LatencyTracer()
    sink = ListSink()
    gap = await _built()
    runner = PipelineRunner(
        think_stage=FakeThinkStage(first_token_delay_s=0.3, memory_delay_s=0.0, tracer=tracer),
        speak_stage=FakeSpeakStage(first_chunk_extra_s=0.0, rtf=0.0),
        sink=sink,
        tracer=tracer,
        gap_filler=gap,
    )
    await runner.prepare()
    result = await runner.run_turn(Utterance(text="你好嗎", language="zh"))

    assert result.phase.value == "done"
    idx = [c.sentence_index for c in sink.chunks]
    assert idx[0] == GAP_SENTENCE_INDEX
    assert GAP_SENTENCE_INDEX not in idx[idx.index(0):], "真音訊之後不得再有墊片"

    trace = tracer.get(result.turn_id)
    assert MARK_GAP_FIRST_AUDIO in trace.marks
    segs = trace.segments_ms()
    assert segs["gap_audio"] is not None
    assert segs["perceived_ttfa"] <= segs["ttfa"]
    assert "感知 TTFA" in tracer.report(result.turn_id)
    # spoken_text 推算要扣掉墊片時長——否則「嗯…」那段會被算成真句子已播出
    assert result.spoken_text == result.generated_text
    assert result.spoken_duration_s == pytest.approx(
        sum(c.duration_s for c in sink.chunks if c.sentence_index != GAP_SENTENCE_INDEX), abs=1e-6
    )


async def test_runner_沒給gap_filler行為不變():
    sink = ListSink()
    runner = PipelineRunner(
        think_stage=FakeThinkStage(first_token_delay_s=0.0, memory_delay_s=0.0),
        speak_stage=FakeSpeakStage(first_chunk_extra_s=0.0, rtf=0.0),
        sink=sink,
    )
    await runner.prepare()
    result = await runner.run_turn(Utterance(text="你好"))
    assert result.phase.value == "done"
    assert all(c.sentence_index >= 0 for c in sink.chunks)


# --- echo_stream.toml ---


def test_defaults_讀toml_缺檔回空(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHO_STREAM_CONFIG", str(tmp_path / "nope.toml"))
    defaults.load_defaults.cache_clear()
    assert defaults.load_defaults() == {}
    assert defaults.section("gap_filler") == {}

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[gap_filler]\nenabled = true\nstart_delay_s = 0.3\n'
        '[gap_filler.phrases]\nzh = ["嗯…"]\nen = ["Hmm…"]\n'
        '[split]\nfirst_min = 9.0\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ECHO_STREAM_CONFIG", str(cfg))
    defaults.load_defaults.cache_clear()
    sec = defaults.section("gap_filler")
    assert sec["phrases"] == {"zh": ["嗯…"], "en": ["Hmm…"]}
    assert defaults.section("split") == {"first_min": 9.0}


def test_defaults_壞檔不擋啟動(monkeypatch, tmp_path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is = = not toml", encoding="utf-8")
    monkeypatch.setenv("ECHO_STREAM_CONFIG", str(cfg))
    defaults.load_defaults.cache_clear()
    assert defaults.load_defaults() == {}


def test_server_從toml建policy(monkeypatch, tmp_path):
    from echo_stream.web.server import _gap_policy_from_defaults, _split_policy_from_defaults

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[split]\nfirst_min = 9.0\nmax = 30.0\n'
        '[gap_filler]\nenabled = true\nstart_delay_s = 0.25\ndefault_language = "en"\n'
        '[gap_filler.phrases]\nen = ["Hmm…"]\nja = ["えっと…"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ECHO_STREAM_CONFIG", str(cfg))
    defaults.load_defaults.cache_clear()
    split = _split_policy_from_defaults()
    assert split.first_min_weight == 9.0 and split.max_weight == 30.0
    assert split.min_weight == 20.0, "沒寫的鍵走程式碼預設"
    gap = _gap_policy_from_defaults()
    assert gap.enabled and gap.start_delay_s == 0.25 and gap.default_language == "en"
    assert gap.phrases == {"en": ["Hmm…"], "ja": ["えっと…"]}
