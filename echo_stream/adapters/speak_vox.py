"""SpeakStage adapter：VOX TTS。

對外介面與 :class:`~echo_stream.adapters.speak_indextts.IndexTTS2SpeakStage`
相同（prepare / stream / aclose），bot 只換 builder 就能切換。

## 與 IndexTTS2 adapter 的差異

* VoxBackend 的 ``stream(text, lang, speed)`` 是**同步 generator**，
  逐段 yield 約 160ms、48kHz 的單聲道 float32 numpy chunk；
  不是 push callback，所以不需要靠例外中止，直接 ``gen.close()`` 即可。
* backend 同時只能處理一個請求：stage 內部句子本來就序列合成，
  另外加一把 lock，連「上一個 turn 被取消、執行緒還在收尾」時
  新 turn 也不會並行呼叫 backend。
* 輸出取樣率與 IndexTTS2 一致（預設 22050、int16），48kHz 用 soxr
  的串流重採樣器轉換（狀態跨 chunk 延續，chunk 邊界不會有不連續）。

## 取消

取消檢查與 ``gen.close()`` 都在**工作執行緒內**做：generator 綁在
建立它的執行緒上，跨執行緒 close 會拋 ValueError。正在跑的那一步
（幾十 ms）會跑完，之後不再繼續。通道有界，背壓同樣傳到 backend。

## 段緣平滑

句內 chunk 邊界是連續訊號，不能每個 chunk 都淡入淡出（會每 160ms 凹一次）。
所以只在句首淡入、句尾淡出 + 靜音墊。為了在不知道哪個 chunk 是最後一個的
情況下仍能淡出，會把輸出尾端 ``edge_fade_ms`` 的樣本留在 carry buffer
延後一個 chunk 送出——只多延遲幾毫秒，不影響 TTFA。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .. import config
from ..contracts.cancellation import CancellationToken, CancelledError, CancelReason
from ..contracts.style import SpeechStyle
from ..contracts.types import TTS_SAMPLE_RATE, AudioChunk, Sentence
from ..core.channel import ChannelClosed, StreamChannel
from .speak_indextts import sanitize_for_tts, smooth_segment_edges, tensor_to_pcm16

VOX_SAMPLE_RATE = 48_000
"""VoxBackend 固定的輸出取樣率。"""

VOX_BACKEND_MODULE = "echo_tts.runtime.vox_backend"
"""VoxBackend 所在模組（TTS repo 內）。路徑是假設，TTS 端落地後以實際為準。"""


class VoxSpeakStage:
    """把 VoxBackend 包成 SpeakStage。

    ``backend`` 可注入（測試用假 backend）；None 時在 :meth:`prepare`
    才延遲 import TTS repo 的 ``VoxBackend``，模組頂層不碰 TTS repo。
    """

    name = "vox"

    def __init__(
        self,
        *,
        backend: Any | None = None,
        tts_repo: str | Path | None = None,
        backend_config: dict | None = None,
        sample_rate: int = TTS_SAMPLE_RATE,
        language: str | None = None,
        speed: float | None = None,
        sentence_buffer: int = 2,
        queue_size: int | None = None,
        edge_fade_ms: float = 4.0,
        segment_silence_ms: float = 15.0,
    ) -> None:
        self._backend = backend
        self._tts_repo = Path(tts_repo) if tts_repo else None
        self.backend_config = backend_config
        self.sample_rate = sample_rate
        self.language = language
        self.speed = speed
        """stage 預設語速；句子的 ``SpeechStyle.speed`` 非 1.0 時優先。
        都沒有就傳 None，由 backend 決定。"""
        self.sentence_buffer = sentence_buffer
        self.queue_size = queue_size or int(config.get("ECHO_STREAM_TTS_QUEUE", "2"))
        self.edge_fade_ms = edge_fade_ms
        self.segment_silence_ms = segment_silence_ms
        self._backend_lock = threading.Lock()
        """backend 同時只能處理一個請求；跨 turn 也要串行。"""
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vox-tts"
        )
        """所有 backend 呼叫（建立、prepare、合成、unload）都走這條**同一個**執行緒。

        torch inductor 的 CUDA graph tree manager 是執行緒區域（TLS）的，只在做
        compile／暖機的那條執行緒初始化；合成落到別條執行緒會在
        ``cudagraph_trees.get_obj`` 斷言失敗。預設執行緒池不保證同一條。"""
        self._prepared = False

    # --- 生命週期 ---

    async def prepare(self) -> None:
        """載入模型與暖機。很慢（數十秒起），在 executor 跑且只做一次。"""
        if self._prepared:
            return
        loop = asyncio.get_running_loop()
        if self._backend is None:
            self._backend = await loop.run_in_executor(self._executor, self._build_backend)
        prepare = getattr(self._backend, "prepare", None)
        if callable(prepare):
            await loop.run_in_executor(self._executor, prepare)
        self._prepared = True

    def _build_backend(self) -> Any:
        """建立真的 VoxBackend。**只有這裡 import TTS repo。**"""
        import importlib

        repo = self._tts_repo or config.subsystem_path("TTS")
        config.ensure_importable(repo)
        module = importlib.import_module(VOX_BACKEND_MODULE)
        return module.VoxBackend(self.backend_config or {})

    async def aclose(self) -> None:
        backend = self._backend
        unload = getattr(backend, "unload", None)
        if callable(unload):
            loop = asyncio.get_running_loop()
            # 在 executor 內取得 lock 再 unload，不會踩到正在推理的執行緒
            await loop.run_in_executor(self._executor, self._unload_locked, unload)
        self._prepared = False

    def _unload_locked(self, unload: Any) -> None:
        with self._backend_lock, contextlib.suppress(Exception):
            unload()

    # --- 串流 ---

    async def stream(
        self, sentences: AsyncIterator[Sentence], token: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        await self.prepare()
        loop = asyncio.get_running_loop()

        # 句子緩衝：合成期間 SentenceSplitter 要能繼續跑，維持層間重疊
        sentence_ch: StreamChannel[Sentence] = StreamChannel(
            maxsize=self.sentence_buffer, token=token
        )
        audio_ch: StreamChannel[AudioChunk] = StreamChannel(
            maxsize=self.queue_size, token=token
        )

        # per-call 停止訊號：消費端提早離開（break / aclose）但 token 沒取消時，
        # 工作執行緒要靠它從阻塞的 put 與 lock 等待中脫身，否則會永久握住 lock
        stop = threading.Event()

        feeder = asyncio.ensure_future(self._feed(sentences, sentence_ch))
        producer = asyncio.ensure_future(
            self._produce(sentence_ch, audio_ch, loop, token, stop)
        )

        try:
            async for chunk in audio_ch:
                yield chunk
        finally:
            stop.set()
            for task in (feeder, producer):
                if not task.done():
                    task.cancel()
            for task in (feeder, producer):
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task

    async def _feed(
        self, sentences: AsyncIterator[Sentence], channel: StreamChannel[Sentence]
    ) -> None:
        try:
            async for sentence in sentences:
                if not sentence.text:
                    continue
                await channel.put(sentence)
        except CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - 傳給消費端而不是消失在 task 裡
            channel.fail(exc)
            return
        channel.close()

    async def _produce(
        self,
        sentence_ch: StreamChannel[Sentence],
        audio_ch: StreamChannel[AudioChunk],
        loop: asyncio.AbstractEventLoop,
        token: CancellationToken,
        stop: threading.Event,
    ) -> None:
        try:
            async for sentence in sentence_ch:
                token.raise_if_cancelled()
                await loop.run_in_executor(
                    self._executor, self._synthesize_one, sentence, audio_ch, loop, token, stop
                )
        except CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            audio_ch.fail(exc)
            return
        audio_ch.close()

    def _resolve_speed(self, style: SpeechStyle | None) -> float | None:
        """只取 speed。style 非 1.0 優先，否則用 stage 預設，都沒有回 None。"""
        if style is not None and style.speed != 1.0:
            return style.speed
        return self.speed

    def _synthesize_one(
        self,
        sentence: Sentence,
        audio_ch: StreamChannel[AudioChunk],
        loop: asyncio.AbstractEventLoop,
        token: CancellationToken,
        stop: threading.Event,
    ) -> None:
        """在 executor 執行緒裡消費同步 generator。**這個函式不在 event loop 上。**"""
        import numpy as np

        speak_text = sanitize_for_tts(sentence.text)
        if not speak_text:
            return  # 整句都是符號——沒東西可唸

        speed = self._resolve_speed(sentence.style)
        n_hold = int(self.sample_rate * self.edge_fade_ms / 1000)
        resampler = self._make_resampler()
        carry = np.zeros(0, dtype="float32")
        chunk_index = 0

        def emit(samples: Any, *, first: bool, final: bool) -> None:
            nonlocal chunk_index
            pcm = tensor_to_pcm16(samples) if len(samples) else b""
            if self.edge_fade_ms > 0 or self.segment_silence_ms > 0:
                pcm = smooth_segment_edges(
                    pcm,
                    self.sample_rate,
                    fade_ms=self.edge_fade_ms,
                    tail_silence_ms=self.segment_silence_ms if final else 0.0,
                    fade_in=first,
                    fade_out=final,
                )
            is_last = final and sentence.is_last
            if not pcm and not is_last:
                return
            chunk = AudioChunk(
                pcm=pcm,
                sample_rate=self.sample_rate,
                turn_id=sentence.turn_id,
                sentence_index=sentence.index,
                chunk_index=chunk_index,
                is_last=is_last,
            )
            chunk_index += 1
            # 通道滿會阻塞此執行緒 = 背壓；token 取消或 stop 設起時脫身
            self._put(audio_ch, chunk, loop, stop)

        if not self._acquire_backend(token, stop):
            return
        try:
            gen = None
            try:
                gen = self._backend.stream(speak_text, lang=self.language, speed=speed)
                for raw in gen:
                    if token.is_cancelled or stop.is_set():
                        return
                    buf = np.concatenate([carry, self._resample(resampler, raw, last=False)])
                    if n_hold > 0:
                        # 尾端留 n_hold 個樣本，等句尾時才能正確淡出
                        if len(buf) <= n_hold:
                            carry = buf
                            continue
                        carry, buf = buf[-n_hold:], buf[:-n_hold]
                    else:
                        carry = buf[:0]
                    emit(buf, first=chunk_index == 0, final=False)
                if token.is_cancelled or stop.is_set():
                    return
                tail = np.concatenate([carry, self._resample(resampler, None, last=True)])
                emit(tail, first=chunk_index == 0, final=True)
            except (CancelledError, ChannelClosed, concurrent.futures.CancelledError):
                return
            finally:
                # 取消 / 例外時關閉 generator，必須在建立它的這條執行緒
                if gen is not None:
                    with contextlib.suppress(Exception):
                        gen.close()
        finally:
            self._backend_lock.release()

    def _acquire_backend(self, token: CancellationToken, stop: threading.Event) -> bool:
        """取得 backend lock。等待期間 token 取消或 stop 設起就放棄，不啟動推理。"""
        while not self._backend_lock.acquire(timeout=0.05):
            if token.is_cancelled or stop.is_set():
                return False
        if token.is_cancelled or stop.is_set():
            self._backend_lock.release()
            return False
        return True

    @staticmethod
    def _put(
        channel: StreamChannel[AudioChunk],
        chunk: AudioChunk,
        loop: asyncio.AbstractEventLoop,
        stop: threading.Event,
    ) -> None:
        """從工作執行緒寫入通道；滿時阻塞（背壓），但 stop 設起時會脫身。"""
        future = asyncio.run_coroutine_threadsafe(channel.put(chunk), loop)
        while True:
            try:
                future.result(timeout=0.05)
                return
            except concurrent.futures.TimeoutError:
                if stop.is_set():
                    future.cancel()
                    raise CancelledError(CancelReason.SHUTDOWN) from None

    # --- 重採樣 ---

    def _make_resampler(self) -> Any | None:
        """48kHz → 輸出取樣率的串流重採樣器（soxr）；取樣率相同時不需要。"""
        if self.sample_rate == VOX_SAMPLE_RATE:
            return None
        import soxr

        return soxr.ResampleStream(
            VOX_SAMPLE_RATE, self.sample_rate, 1, dtype="float32", quality="HQ"
        )

    @staticmethod
    def _to_mono_f32(raw: Any) -> Any:
        import numpy as np

        arr = np.asarray(raw, dtype="float32")
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr[0]
        return np.atleast_1d(arr)

    def _resample(self, resampler: Any | None, raw: Any | None, *, last: bool) -> Any:
        import numpy as np

        arr = np.zeros(0, dtype="float32") if raw is None else self._to_mono_f32(raw)
        if resampler is None:
            return arr
        return resampler.resample_chunk(arr, last=last).astype("float32", copy=False)


__all__ = ["VoxSpeakStage", "VOX_SAMPLE_RATE"]
