"""Discord STT adapter：backend 過濾與後處理、動態熱詞、bot 入口。

golden 案例移植自 bot repo tests/test_stt_hotwords.py（假模型、english_guard、
hotword_leak）與 cogs/stt.py 的過濾規則；不載任何真模型，opencc 以假模組替身。
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from echo_stream.adapters import stt_discord as sd
from echo_stream.adapters.input_stt import TranscriberBackend, TranscriptionResult
from echo_stream.adapters.stt_discord import (
    DiscordAudioSource,
    DiscordSttProfile,
    DiscordTranscriberBackend,
)

# ---------------------------------------------------------------------------
# 假模型與假 opencc
# ---------------------------------------------------------------------------


class _Seg:
    def __init__(self, text, lp=-0.3, nsp=0.0, cr=1.0, start=0.0, end=1.0):
        self.text, self.start, self.end = text, start, end
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = lp, nsp, cr


class _Info:
    language, language_probability = "zh", 1.0


class _FakeModel:
    def __init__(self, *segs):
        self.calls: list[dict] = []
        self.segs = segs

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(self.segs), _Info()


def _model(text="你好", lp=-0.3, **kw):
    return _FakeModel(_Seg(text, lp, **kw))


def _pcm(std=3000.0, seconds=1.0):
    n = int(16000 * seconds)
    return np.clip(np.random.RandomState(0).randn(n) * std, -32768, 32767).astype("<i2").tobytes()


def _backend(model, **over):
    over.setdefault("convert_simplified", False)
    return DiscordTranscriberBackend(model=model, **over)


@pytest.fixture
def fake_opencc(monkeypatch):
    """假 OpenCC：s2t 只改簡體字；s2twp 另外把「软件」轉「軟體」。"""
    simp = {"们": "們", "说": "說", "软": "軟", "件": "件", "么": "麼", "台": "台"}

    class OpenCC:
        def __init__(self, scheme):
            self.scheme = scheme

        def convert(self, text):
            out = "".join(simp.get(c, c) for c in text)
            if self.scheme == "s2twp":
                out = out.replace("軟件", "軟體")
                out = out.replace("演算法", "演演算法")  # 模擬 s2twp 誤傷繁體
            return out

    monkeypatch.setitem(sys.modules, "opencc", types.SimpleNamespace(OpenCC=OpenCC))


# ---------------------------------------------------------------------------
# 契約與基本回傳
# ---------------------------------------------------------------------------


def test_implements_protocol_and_result_fields():
    b = _backend(_model("你好", -0.3))
    assert isinstance(b, TranscriberBackend)
    r = b.transcribe(_pcm(), 16000)
    assert isinstance(r, TranscriptionResult)
    assert r.text == "你好" and r.language == "zh"
    assert r.confidence == pytest.approx(0.7408, abs=1e-3)  # exp(-0.3)


def test_rejects_wrong_sample_rate():
    with pytest.raises(ValueError):
        _backend(_model()).transcribe(_pcm(), 48000)


def test_language_param_passed_through():
    m = _model()
    _backend(m).transcribe(_pcm(), 16000)
    assert m.calls[0]["language"] == "zh" and m.calls[0]["task"] == "transcribe"
    m2 = _model()
    _backend(m2, language=None).transcribe(_pcm(), 16000)
    assert m2.calls[0]["language"] is None
    assert "initial_prompt" not in m.calls[0]  # Whisper 預設不用 initial_prompt


# ---------------------------------------------------------------------------
# RMS 門檻
# ---------------------------------------------------------------------------


def test_rms_gate_blocks_silence_without_calling_model():
    m = _model()
    b = _backend(m)
    assert b.transcribe(b"\x00\x00" * 16000, 16000) is None
    assert b.transcribe(_pcm(std=50), 16000) is None  # rms ~0.0015 < 0.005
    assert m.calls == []
    assert b.transcribe(_pcm(std=3000), 16000) is not None


def test_rms_gate_threshold_configurable():
    m = _model()
    quiet = _pcm(std=300)  # rms ~0.009
    assert _backend(m).transcribe(quiet, 16000) is not None
    assert _backend(_model(), min_rms=0.02).transcribe(quiet, 16000) is None
    assert _backend(_model(), min_rms=0.0).transcribe(b"\x00\x00" * 160, 16000) is not None


# ---------------------------------------------------------------------------
# 簡體才轉繁體
# ---------------------------------------------------------------------------


def test_simplified_converted_only_when_detected(fake_opencc):
    b = _backend(_model("我们说软件"), convert_simplified=True)
    assert b.transcribe(_pcm(), 16000).text == "我們說軟體"


def test_traditional_not_touched_by_s2twp(fake_opencc):
    # 純繁體輸入：s2twp 會誤傷成「演演算法」，必須完全不轉
    b = _backend(_model("這個演算法很好"), convert_simplified=True)
    assert b.transcribe(_pcm(), 16000).text == "這個演算法很好"


def test_shared_glyphs_are_not_simplified_evidence(fake_opencc):
    # 「台」繁簡同形，不構成簡體證據
    b = _backend(_model("台灣的演算法"), convert_simplified=True)
    assert b.transcribe(_pcm(), 16000).text == "台灣的演算法"


def test_convert_disabled_keeps_simplified(fake_opencc):
    b = _backend(_model("我们说"), convert_simplified=False)
    assert b.transcribe(_pcm(), 16000).text == "我们说"


def test_missing_opencc_degrades_gracefully(monkeypatch):
    monkeypatch.setitem(sys.modules, "opencc", None)  # import 會丟 ImportError
    b = _backend(_model("我们说"), convert_simplified=True)
    assert b.transcribe(_pcm(), 16000).text == "我们说"


def test_real_opencc_golden_if_installed():
    pytest.importorskip("opencc")
    b = _backend(_model("这个演算法很好"), convert_simplified=True)
    assert "個" in b.transcribe(_pcm(), 16000).text
    b2 = _backend(_model("這個演算法很好"), convert_simplified=True)
    assert b2.transcribe(_pcm(), 16000).text == "這個演算法很好"


# ---------------------------------------------------------------------------
# credit / 垃圾 / 幻覺過濾（golden：bot cogs/stt.py 規則）
# ---------------------------------------------------------------------------

CREDITS_DROP = [
    "詞曲 李宗盛",
    "作詞 作曲 周杰倫",
    "字幕由 Amara 提供",
    "某某 字幕組",
    "編曲：王力宏",
]
CREDITS_KEEP = [
    "我覺得李宗盛的歌很好聽",
    "曲 這首歌很好聽",
    "我們來討論字幕的問題吧",
]


@pytest.mark.parametrize("text", CREDITS_DROP)
def test_credit_dropped(text):
    assert sd.credit_hallucination(text)
    assert _backend(_model(text)).transcribe(_pcm(), 16000) is None


@pytest.mark.parametrize("text", CREDITS_KEEP)
def test_credit_kept(text):
    assert sd.credit_hallucination(text) is None
    assert _backend(_model(text)).transcribe(_pcm(), 16000) is not None


def test_credit_filter_can_be_disabled():
    # filter_hallucination 也關掉，才能單獨看 credit 開關
    b = _backend(_model("作詞 作曲 周杰倫"), filter_credits=False, filter_hallucination=False)
    assert b.transcribe(_pcm(), 16000).text == "作詞 作曲 周杰倫"


def test_garbage_cleaned_only_with_cjk():
    assert sd.clean_whisper_garbage("你好Ｂ我是﹑小明�") == "你好，我是、小明"
    assert sd.clean_whisper_garbage("Ｂ") == "Ｂ"  # 無中文：不動
    assert sd.clean_whisper_garbage("ＡＢＣ你好") == "ＡＢＣ你好"  # Ｂ 夾在英數中不是獨立
    b = _backend(_model("你好Ｂ世界"), clean_garbage=True)
    assert b.transcribe(_pcm(), 16000).text == "你好，世界"
    b2 = _backend(_model("你好Ｂ世界"), clean_garbage=False)
    assert b2.transcribe(_pcm(), 16000).text == "你好Ｂ世界"


@pytest.mark.parametrize(
    "text", ["謝謝", "Thank you.", "請不吝點贊訂閱", "Subtitles by the Amara.org community"]
)
def test_hallucination_blacklist_dropped(text):
    assert sd.is_hallucination(text)
    assert _backend(_model(text)).transcribe(_pcm(), 16000) is None


@pytest.mark.parametrize("text", ["Thanks for listening.", "Thank you for listening!"])
def test_listening_outro_dropped_ungated(text):
    assert sd.is_hallucination(text)


@pytest.mark.parametrize(
    "text", ["See you later.", "See you late", "Have a good day.", "The end.", "See you next time!"]
)
def test_outro_dropped_when_short_and_low_logprob(text):
    assert _backend(_model(text, lp=-0.8, end=0.8)).transcribe(_pcm(), 16000) is None


@pytest.mark.parametrize(
    "text, lp, end",
    [
        ("See you later.", -0.4, 0.8),  # 信心夠高：使用者真的在道別
        ("See you later.", -0.8, 2.5),  # 太長：不是典型短幻聽
        ("OK, see you.", -0.9, 0.8),  # 非套語
        ("Sorry", -0.9, 0.6),  # 短而低信心但不是套語
        ("好, bye", -0.9, 0.6),  # 中英混合不屬於套語
    ],
)
def test_outro_filter_does_not_kill_real_speech(text, lp, end):
    assert sd.outro_hallucination(text, lp, end, -0.6, 1.5) is None
    b = _backend(_model(text, lp=lp, end=end), filter_hallucination=False)
    assert b.transcribe(_pcm(), 16000) is not None


def test_outro_filter_tail_segment_only():
    # 中文本體保留，只丟掉結尾那段低信心套語
    m = _FakeModel(
        _Seg("我先去睡了", lp=-0.3, end=1.0), _Seg("Have a good day.", lp=-0.9, start=1.0, end=1.6)
    )
    r = _backend(m).transcribe(_pcm(), 16000)
    assert r is not None and r.text == "我先去睡了"


def test_outro_filter_missing_logprob_is_conservative():
    assert sd.outro_hallucination("The end.", None, 0.5, -0.6, 1.5) is None


def test_outro_filter_can_be_disabled():
    b = _backend(_model("The end.", lp=-0.9, end=0.8), filter_outro=False)
    assert b.transcribe(_pcm(), 16000) is not None


def test_hallucination_filter_exact_not_substring():
    assert sd.is_hallucination("謝謝你幫我修好電腦") is None
    assert _backend(_model("謝謝你幫我修好電腦")).transcribe(_pcm(), 16000) is not None


def test_segment_level_thresholds():
    assert _backend(_model("你好", nsp=0.9)).transcribe(_pcm(), 16000) is None
    assert _backend(_model("你好", lp=-2.0)).transcribe(_pcm(), 16000) is None
    assert _backend(_model("你好", cr=3.0)).transcribe(_pcm(), 16000) is None


def test_only_kept_segments_are_joined():
    m = _FakeModel(_Seg("你好"), _Seg("垃圾", nsp=0.95), _Seg("世界"))
    assert _backend(m).transcribe(_pcm(), 16000).text == "你好 世界"


def test_empty_segments_or_text_returns_none():
    assert _backend(_FakeModel()).transcribe(_pcm(), 16000) is None
    assert _backend(_model("   ")).transcribe(_pcm(), 16000) is None


# english_guard golden（bot tests/test_stt_hotwords.py:121-144）
def test_english_guard_off_keeps_short_english():
    r = _backend(_model("nutty", -0.8)).transcribe(_pcm(std=300), 16000)
    assert r is not None and r.text == "nutty"


def test_english_guard_on_drops_quiet_unsure_english():
    b = _backend(_model("nutty", -0.8), english_guard=True)
    assert b.transcribe(_pcm(std=300), 16000) is None


def test_english_guard_on_keeps_confident_loud_long_or_chinese():
    quiet, loud = _pcm(std=300), _pcm(std=3000)
    cases = [
        (_model("nutty", -0.2), quiet),
        (_model("more dangerous than I thought", -0.8), quiet),
        (_model("nutty", -0.8), loud),
        (_model("好 OK", -0.8), quiet),
    ]
    for m, pcm in cases:
        assert _backend(m, english_guard=True).transcribe(pcm, 16000) is not None


# hotword_leak golden（bot tests/test_stt_hotwords.py:147-150）
def test_hotword_leak_filter():
    hw = ["Victor", "Calico", "Haze", "Dynamo"]
    assert sd.hotword_leak_reason("Victor Calico Haze Dynamo", hw)
    assert sd.hotword_leak_reason("Victor Victor 小心喔", hw) is None


def test_hotword_leak_drops_segment_in_backend():
    b = _backend(_model("Victor Calico Haze Dynamo"))
    b.set_hotwords(["Victor", "Calico", "Haze", "Dynamo"])
    assert b.transcribe(_pcm(), 16000) is None


def test_prompt_leak():
    assert sd.prompt_leak_reason("對話", "以下是對話", 0.5)
    assert sd.prompt_leak_reason("我們去吃飯", "以下是對話", 0.5) is None
    b = _backend(_model("對話"))
    b.set_initial_prompt("以下是對話")
    assert b.transcribe(_pcm(), 16000) is None


# ---------------------------------------------------------------------------
# 動態熱詞
# ---------------------------------------------------------------------------


def test_hotwords_passed_joined_and_default_empty():
    m = _model()
    b = _backend(m)
    b.transcribe(_pcm(), 16000)
    assert "hotwords" not in m.calls[0]
    b.set_hotwords(["成員甲", "Bernie"])
    b.transcribe(_pcm(), 16000)
    assert m.calls[1]["hotwords"] == "成員甲, Bernie"  # bot 同樣用 ", ".join


def test_per_speaker_overrides_global_then_falls_back():
    m = _model()
    b = _backend(m)
    b.set_hotwords(["全域"])
    b.set_hotwords(["Victor"], speaker_id="u1")
    b.transcribe_for("u1", _pcm(), 16000)
    b.transcribe_for("u2", _pcm(), 16000)
    b.transcribe(_pcm(), 16000)
    assert [c.get("hotwords") for c in m.calls] == ["Victor", "全域", "全域"]
    b.set_hotwords([], speaker_id="u1")  # 清掉逐聲道 → 退回全域
    b.transcribe_for("u1", _pcm(), 16000)
    assert m.calls[-1]["hotwords"] == "全域"
    b.set_hotwords(None)  # 清全域
    b.transcribe_for("u1", _pcm(), 16000)
    assert "hotwords" not in m.calls[-1]


def test_hotwords_dedup_and_truncate(caplog):
    b = _backend(_model(), hotwords_max=3)
    b.set_hotwords(["A", "a", " B ", "", "C", "D", "E"])
    assert b.get_hotwords() == ("A", "B", "C")
    assert any("hotwords_max" in r.message for r in caplog.records)


def test_profile_initial_hotwords_and_clear_speaker():
    b = _backend(_model(), hotwords=("X", "Y"))
    assert b.get_hotwords() == ("X", "Y")
    b.set_hotwords(["S"], speaker_id="u1")
    b.clear_speaker("u1")
    assert b.get_hotwords("u1") == ("X", "Y")


def test_initial_prompt_only_when_set():
    m = _model()
    b = _backend(m)
    b.set_initial_prompt("短提示", speaker_id="u1")
    b.transcribe_for("u1", _pcm(), 16000)
    b.transcribe(_pcm(), 16000)
    assert m.calls[0]["initial_prompt"] == "短提示" and "initial_prompt" not in m.calls[1]


def test_set_hotwords_thread_safe_under_concurrent_transcribe():
    m = _model()
    b = _backend(m)
    pcm = _pcm()
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(200):
                b.set_hotwords([f"w{i}", f"v{i}"], speaker_id="u1" if i % 2 else None)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                b.transcribe_for("u1", pcm, 16000)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=f) for f in (writer, reader, reader)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    for c in m.calls:  # 每次拿到的都是完整的一組（不會一半新一半舊）
        hw = c.get("hotwords")
        if hw:
            a, v = hw.split(", ")
            assert a[1:] == v[1:]


# ---------------------------------------------------------------------------
# 模型載入：只載一次、注入不重載、import 不載模型
# ---------------------------------------------------------------------------


def test_injected_model_never_loads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("不該載入模型")

    monkeypatch.setattr(sd, "_load_custom_whisper", boom)
    monkeypatch.setattr(sd, "_load_echo_stt_model", boom)
    b = DiscordTranscriberBackend(model=_model(), model_path="/x")
    b.preload()
    for _ in range(3):
        b.transcribe(_pcm(), 16000)


def test_lazy_custom_model_loaded_once(monkeypatch):
    loads = []
    m = _model()
    monkeypatch.setattr(sd, "_load_custom_whisper", lambda p: loads.append(p.model_path) or m)
    b = DiscordTranscriberBackend(model_path="C:/models/breeze", convert_simplified=False)
    assert loads == []  # 建構不載
    for _ in range(3):
        b.transcribe(_pcm(), 16000)
    assert loads == ["C:/models/breeze"] and len(m.calls) == 3


def test_default_path_uses_echo_stt_loader_once(monkeypatch):
    loads = []
    m = _model()
    monkeypatch.setattr(sd, "_load_echo_stt_model", lambda: loads.append(1) or m)
    b = DiscordTranscriberBackend(convert_simplified=False)
    b.transcribe(_pcm(), 16000)
    b.transcribe(_pcm(), 16000)
    assert loads == [1]


def test_module_has_no_toplevel_heavy_imports():
    """重依賴只能在函式內延遲 import（頂層 import 沒有縮排）。"""
    heavy = ("faster_whisper", "opencc", "echo_stt", "torch", "scipy")
    with open(sd.__file__, encoding="utf-8") as f:
        for line in f:
            if line.startswith(("import ", "from ")):
                assert not any(h in line for h in heavy), line


def test_profile_defaults_match_bot_and_overrides():
    p = DiscordSttProfile()
    assert (p.language, p.min_rms, p.convert_simplified, p.convert_scheme) == (
        "zh", 0.005, True, "s2twp")
    assert (p.max_no_speech_prob, p.min_avg_logprob, p.max_compression_ratio) == (0.7, -1.2, 2.4)
    b = DiscordTranscriberBackend(p, model=_model(), min_rms=0.1)
    assert b.profile.min_rms == 0.1 and p.min_rms == 0.005  # 不改動傳入的 profile


# ---------------------------------------------------------------------------
# DiscordAudioSource
# ---------------------------------------------------------------------------


class _FakeStage:
    def __init__(self):
        self.events: list[tuple] = []
        self.lock = threading.Lock()

    def feed(self, speaker_id, pcm, ts=None):
        with self.lock:
            self.events.append(("feed", speaker_id, pcm, ts))

    def join(self, speaker_id):
        self.events.append(("join", speaker_id))

    def leave(self, speaker_id, flush=True):
        self.events.append(("leave", speaker_id, flush))

    def discard(self, speaker_id=None):
        self.events.append(("discard", speaker_id))


def _stereo48(n=48000, seed=1):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, 2) * 3000).astype("<i2").tobytes()


def _fed(stage, who):
    return b"".join(e[2] for e in stage.events if e[0] == "feed" and e[1] == who)


def test_source_resamples_per_member_and_forwards_ts():
    stage = _FakeStage()
    src = DiscordAudioSource(stage, clock=lambda: 42.0)
    pcm = _stereo48()
    for i in range(0, len(pcm), 3840):
        src.feed("a", pcm[i : i + 3840], ts=7.5)
    src.feed("b", pcm[:3840])
    assert all(e[3] == 7.5 for e in stage.events if e[1] == "a")
    assert [e[3] for e in stage.events if e[1] == "b"] == [42.0]
    # 輸出為 16k mono int16：約 1/3 樣本、每樣本 2 bytes
    assert abs(len(_fed(stage, "a")) // 2 - 16000) <= 30


def test_source_member_state_independent_and_equals_whole_pass():
    from echo_stream.adapters.discord_resample import StreamingResampler48to16

    stage = _FakeStage()
    src = DiscordAudioSource(stage)
    a, b = _stereo48(seed=1), _stereo48(seed=2)
    for i in range(0, len(a), 3840):  # 交錯餵兩個成員
        src.feed("a", a[i : i + 3840])
        src.feed("b", b[i : i + 3840])
    src.leave("a")
    src.leave("b")
    for who, raw in (("a", a), ("b", b)):
        r = StreamingResampler48to16()
        assert _fed(stage, who) == r.process(raw) + r.flush()


def test_source_leave_flushes_tail_before_stage_leave_and_removes_state():
    stage = _FakeStage()
    src = DiscordAudioSource(stage)
    src.feed("a", _stereo48(4800))
    src.leave("a", flush=True)
    kinds = [e[0] for e in stage.events]
    assert kinds[-1] == "leave" and stage.events[-1] == ("leave", "a", True)
    assert "a" not in src._members
    n = len(stage.events)
    src.leave("a", flush=False)  # 未知成員也轉接給 stage，不丟例外
    assert stage.events[n] == ("leave", "a", False)


def test_source_leave_noflush_drops_tail():
    stage = _FakeStage()
    src = DiscordAudioSource(stage)
    src.feed("a", _stereo48(4800))
    before = len(_fed(stage, "a"))
    src.leave("a", flush=False)
    assert len(_fed(stage, "a")) == before


def test_source_join_resets_and_discard_forwards():
    stage = _FakeStage()
    src = DiscordAudioSource(stage)
    pcm = _stereo48(9600)
    src.feed("a", pcm[:1000])  # 留下半個 frame 之外的歷史
    src.join("a")
    assert stage.events[-1] == ("join", "a")
    src.feed("b", pcm)
    src.discard("a")
    assert stage.events[-1] == ("discard", "a")
    src.discard(None)
    assert stage.events[-1] == ("discard", None)
    # discard 後該成員的重取樣器已重置：再餵同一段，輸出與全新實例相同
    from echo_stream.adapters.discord_resample import StreamingResampler48to16

    stage.events.clear()
    src.feed("b", pcm)
    assert _fed(stage, "b") == StreamingResampler48to16().process(pcm)


def test_source_on_leave_hook_clears_backend_speaker():
    b = _backend(_model())
    b.set_hotwords(["S"], speaker_id="a")
    src = DiscordAudioSource(_FakeStage(), on_leave=b.clear_speaker)
    src.leave("a")
    assert b.get_hotwords("a") == ()


def test_source_feed_from_multiple_threads():
    stage = _FakeStage()
    src = DiscordAudioSource(stage)
    pcm = _stereo48(24000)

    def run(who):
        for i in range(0, len(pcm), 3840):
            src.feed(who, pcm[i : i + 3840])

    ts = [threading.Thread(target=run, args=(f"m{i}",)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    outs = {_fed(stage, f"m{i}") for i in range(4)}
    assert len(outs) == 1  # 同樣輸入 → 各成員輸出相同，沒有互相污染


def test_解碼品質取保留段落的最差值():
    a = _Seg("第一句", lp=-0.3, cr=1.2, start=0.0, end=1.0)
    b = _Seg("第二句", lp=-0.8, cr=1.8, start=1.0, end=2.0)
    a.temperature, b.temperature = 0.0, 0.4
    r = _backend(_FakeModel(a, b)).transcribe(_pcm(seconds=2.0), 16000)
    assert r.temperature == 0.4
    assert r.avg_logprob == pytest.approx(-0.8)
    assert r.compression_ratio == pytest.approx(1.8)


def test_解碼品質_段落沒有_temperature_時為_None():
    r = _backend(_model("你好", -0.3)).transcribe(_pcm(), 16000)
    assert r.temperature is None
    assert r.avg_logprob == pytest.approx(-0.3) and r.compression_ratio == pytest.approx(1.0)


def test_溫度fallback預設收斂為兩階():
    m = _model()
    _backend(m).transcribe(_pcm(), 16000)
    assert m.calls[0]["temperature"] == [0.0, 0.2]


def test_溫度設None時交給faster_whisper預設():
    m = _model()
    _backend(m, temperature=None).transcribe(_pcm(), 16000)
    assert "temperature" not in m.calls[0]
