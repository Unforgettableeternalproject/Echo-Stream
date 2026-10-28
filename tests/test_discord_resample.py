"""Discord 串流重取樣器：與整段處理等價、與 bot 演算法比對。"""

from __future__ import annotations

import numpy as np
import pytest

from echo_stream.adapters.discord_resample import StreamingResampler48to16, design_lowpass

RATE = 48000


def _stereo(n: int, seed: int = 0) -> bytes:
    """隨機雜訊 + 1kHz / 5kHz 正弦（左右聲道略有差異）。"""
    rng = np.random.RandomState(seed)
    t = np.arange(n) / RATE
    left = 8000 * np.sin(2 * np.pi * 1000 * t) + 3000 * rng.randn(n)
    right = 6000 * np.sin(2 * np.pi * 5000 * t) + 3000 * rng.randn(n)
    pcm = np.stack([left, right], axis=1)
    return np.clip(pcm, -32768, 32767).astype("<i2").tobytes()


def _mono(pcm: bytes) -> np.ndarray:
    st = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).astype(np.float64)
    return st.mean(axis=1)


def _stream(pcm: bytes, sizes: list[int]) -> np.ndarray:
    r = StreamingResampler48to16()
    out, pos, i = [], 0, 0
    while pos < len(pcm):
        step = sizes[i % len(sizes)]
        out.append(r.process_float(pcm[pos : pos + step]))
        pos += step
        i += 1
    out.append(r.flush_float())
    return np.concatenate(out)


def _bot_numpy_fir(mono: np.ndarray) -> np.ndarray:
    """bot cogs/stt.py:163 的 numpy 備援（193 階 7.6kHz Kaiser sinc、same 卷積後 [::3]）。"""
    numtaps = 193
    fc = 7600.0 / RATE
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.kaiser(numtaps, 8.0)
    h = (h / h.sum()).astype(np.float32)
    return np.convolve(mono.astype(np.float32), h, mode="same")[::3]


CHUNKINGS = {
    "discord_20ms": [3840],
    "irregular": [4, 7, 1000, 3841, 13, 99999, 2],  # 含非 4 倍數（半個 frame）
    "tiny": [1, 3, 5],
}


@pytest.mark.parametrize("name", list(CHUNKINGS))
def test_streaming_matches_whole_pass(name):
    pcm = _stereo(RATE * 2 + 37)  # 2 秒多一點，長度不是 3 的倍數
    whole = _stream(pcm, [len(pcm)])
    got = _stream(pcm, CHUNKINGS[name])
    assert got.shape == whole.shape
    # 同一濾波器、同一累加順序近似：差異應在浮點誤差等級
    assert np.max(np.abs(got - whole)) < 1e-6


def test_matches_scipy_resample_poly():
    sig = pytest.importorskip("scipy.signal")
    pcm = _stereo(RATE + 11)
    ref = sig.resample_poly(_mono(pcm), 1, 3)
    got = _stream(pcm, CHUNKINGS["irregular"])
    assert got.shape == ref.shape  # ceil(n/3)
    assert np.max(np.abs(got - ref)) < 1e-6


def test_filter_design_is_unity_gain_lowpass():
    h = design_lowpass()
    assert len(h) == 61
    assert h.sum() == pytest.approx(1.0)
    resp = np.abs(np.fft.rfft(h, 4096))
    freqs = np.fft.rfftfreq(4096, 1 / RATE)
    assert resp[freqs < 6000].min() > 0.95  # 通帶平坦
    assert resp[freqs > 10000].max() < 0.02  # 抗混疊：8kHz 以上被壓


def test_close_to_bot_numpy_fir_in_band():
    """與 bot 的 numpy FIR 備援（不同濾波器）比對：通帶訊號應近似一致。"""
    n = RATE * 2
    t = np.arange(n) / RATE
    tone = (10000 * np.sin(2 * np.pi * 1000 * t)).astype("<i2")
    pcm = np.stack([tone, tone], axis=1).tobytes()
    got = _stream(pcm, [3840])
    ref = _bot_numpy_fir(_mono(pcm))
    m = min(len(got), len(ref))
    sl = slice(200, m - 200)  # 排除兩端邊界
    err = np.sqrt(np.mean((got[sl] - ref[sl]) ** 2)) / np.sqrt(np.mean(ref[sl] ** 2))
    assert err < 0.01


def test_aliasing_tone_suppressed():
    """12kHz 在 16k 取樣下會混疊成 4kHz；有抗混疊濾波就該被壓掉。"""
    t = np.arange(RATE) / RATE
    tone = (10000 * np.sin(2 * np.pi * 12000 * t)).astype("<i2")
    pcm = np.stack([tone, tone], axis=1).tobytes()
    out = _stream(pcm, [3840])
    assert np.sqrt(np.mean(out[100:-100] ** 2)) < 10000 * 0.02


def test_no_chunk_boundary_artifacts_vs_stateless():
    """對照組：無狀態逐 chunk 處理在邊界會有明顯誤差，有狀態版沒有。"""
    sig = pytest.importorskip("scipy.signal")
    pcm = _stereo(RATE)
    whole = _stream(pcm, [len(pcm)])
    stateless = np.concatenate(
        [sig.resample_poly(_mono(pcm[i : i + 3840]), 1, 3) for i in range(0, len(pcm), 3840)]
    )
    k = min(len(stateless), len(whole))
    assert np.max(np.abs(stateless[:k] - whole[:k])) > 100  # 無狀態確實有瑕疵
    assert np.max(np.abs(_stream(pcm, [3840]) - whole)) < 1e-6


def test_int16_output_length_and_reset():
    pcm = _stereo(RATE // 2)
    r = StreamingResampler48to16()
    a = r.process(pcm) + r.flush()
    assert len(a) // 2 == -(-(RATE // 2) // 3)  # ceil(n/3) 個 int16
    # flush 之後狀態歸零，再處理同一段結果相同
    b = r.process(pcm) + r.flush()
    assert a == b
    r.process(pcm[:5000])
    r.reset()
    assert r.process(pcm) + r.flush() == a


def test_empty_and_partial_frame():
    r = StreamingResampler48to16()
    assert r.process(b"") == b""
    assert r.process(b"\x01\x02\x03") == b""  # 不足一個 frame
    assert r.flush() == b"\x00\x00" or r.flush() == b""
