"""Discord 音訊串流重取樣：48kHz stereo int16 → 16kHz mono int16。

bot 端（cogs/stt.py ``discord_pcm_to_float16k``）要求「對整段一次轉換」，因為
逐封包做無狀態的 ``resample_poly`` 會在 chunk 邊界產生濾波瑕疵。這裡把同一個
濾波器改寫成**有狀態**的串流版本：跨 chunk 保留 FIR 歷史，輸出與整段處理
在數學上等價（只差 int16 量化）。

濾波器設計與 ``scipy.signal.resample_poly(x, 1, 3)`` 的預設相同——
61 階 Kaiser(beta=5) 窗 sinc 低通、截止 = 新 Nyquist（8kHz）、零相位置中。
設計用純 numpy 重現，不依賴 scipy（有 scipy 時測試會拿它當對照組）。
抗混疊低通永遠存在，不會直接 ``[::3]``。

每個聲道（成員）一個實例，實例本身**不是** thread-safe。
"""

from __future__ import annotations

import math

import numpy as np

DISCORD_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
_DECIMATION = DISCORD_SAMPLE_RATE // TARGET_SAMPLE_RATE  # 3
_FRAME_BYTES = 4  # stereo int16
_HALF_LEN = 10 * _DECIMATION  # 與 scipy resample_poly 相同：半長 = 10 * max(up, down)


def design_lowpass(decimation: int = _DECIMATION) -> np.ndarray:
    """重現 scipy ``resample_poly`` 預設濾波器（firwin + Kaiser beta=5，DC 增益 1）。"""
    half = 10 * decimation
    numtaps = 2 * half + 1
    cutoff = 1.0 / decimation  # 相對 Nyquist
    n = np.arange(numtaps) - half
    h = cutoff * np.sinc(cutoff * n) * np.kaiser(numtaps, 5.0)
    return h / h.sum()


class StreamingResampler48to16:
    """有狀態的 48k stereo int16 → 16k mono int16 重取樣器。

    ``process`` 吃任意大小的 chunk（含不是 4 bytes 倍數的），不足一個 frame 的
    尾巴會留到下一次。``flush`` 補零沖出濾波器尾端，使總輸出長度等於整段處理
    的 ``ceil(n / 3)``，之後狀態歸零。
    """

    def __init__(self) -> None:
        self._h = design_lowpass()
        self.reset()

    def reset(self) -> None:
        """丟棄所有狀態（濾波歷史、半個 frame、取樣計數）。"""
        # 起始以 _HALF_LEN 個零當左側補零，與整段處理的邊界行為一致
        self._x = np.zeros(_HALF_LEN, dtype=np.float64)
        self._next = _HALF_LEN  # 下一個輸出中心在 _x 內的索引
        self._rest = b""
        self._n_in = 0
        self._n_out = 0

    def _to_mono(self, pcm: bytes) -> np.ndarray:
        data = self._rest + pcm
        usable = len(data) - len(data) % _FRAME_BYTES
        self._rest = data[usable:]
        if usable == 0:
            return np.zeros(0, dtype=np.float64)
        stereo = np.frombuffer(data[:usable], dtype="<i2").reshape(-1, 2)
        return stereo.astype(np.float64).mean(axis=1)

    def _run(self, mono: np.ndarray) -> np.ndarray:
        """把新樣本接進歷史，吐出所有「前後 30 點都已到齊」的輸出（int16 刻度的浮點）。"""
        self._n_in += mono.size
        x = np.concatenate((self._x, mono)) if mono.size else self._x
        # valid 卷積第 j 項 = 以 x[j + HALF] 為中心的濾波結果（核對稱，卷積 = 相關）
        if x.size < 2 * _HALF_LEN + 1:
            self._x = x
            return np.zeros(0, dtype=np.float64)
        y = np.convolve(x, self._h, mode="valid")
        out = y[self._next - _HALF_LEN :: _DECIMATION]
        self._next += _DECIMATION * out.size
        keep_from = self._next - _HALF_LEN
        self._x = x[keep_from:]
        self._next -= keep_from
        self._n_out += out.size
        return out

    @staticmethod
    def _to_int16(y: np.ndarray) -> bytes:
        return np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes()

    def process_float(self, pcm: bytes) -> np.ndarray:
        """同 ``process``，但回傳 float64（int16 刻度、未量化），給等價性測試用。"""
        return self._run(self._to_mono(pcm))

    def process(self, pcm: bytes) -> bytes:
        """處理一個 chunk，回傳 16k mono int16 bytes（可能為空）。"""
        return self._to_int16(self.process_float(pcm))

    def flush_float(self) -> np.ndarray:
        target = math.ceil(self._n_in / _DECIMATION)
        missing = target - self._n_out
        out = np.zeros(0, dtype=np.float64)
        if missing > 0:
            pad = np.zeros(_HALF_LEN + _DECIMATION, dtype=np.float64)
            out = self._run(pad)[:missing]
        self.reset()
        return out

    def flush(self) -> bytes:
        """沖出尾端並重置狀態（成員離開、段落結束時呼叫）。"""
        return self._to_int16(self.flush_float())


__all__ = [
    "DISCORD_SAMPLE_RATE",
    "TARGET_SAMPLE_RATE",
    "StreamingResampler48to16",
    "design_lowpass",
]
