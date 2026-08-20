"""SpeakStage adapter：IndexTTS2（Phase 1）。

## 這一層在解什麼問題

後端與契約的方向相反：

* `IndexTTS2Backend.generate()` 是**同步阻塞**呼叫，透過
  ``on_segment_audio(tensor, seg_idx, total)`` **push** 音訊出來
* :class:`~echo_stream.contracts.stages.SpeakStage` 是 **async pull**

所以要做 push→pull 橋接：在 executor 執行緒跑同步推理，callback 用
:meth:`StreamChannel.put_threadsafe` 把音訊塞進**有界**通道，
event loop 這側從通道拉。

## 有界是重點，不是效能調參

無界通道會讓 TTS 一路合成到底。使用者在第 2 句插話時，後面 8 句的 GPU
時間已經花掉了——取消的「上游傳播」變成空話，因為上游早就跑完了。

有界通道滿溢時 ``put_threadsafe`` 會**阻塞推理執行緒**，這正是要的效果：
**背壓要能傳到 GPU**。

## 取消怎麼穿透到 GPU

``generate()` 在執行緒裡跑，asyncio 的取消碰不到它。唯一的協作點是
``on_segment_audio`` callback：

1. token 取消 → 阻塞中的 ``put_threadsafe`` 拿到 ``CancelledError``
2. 例外從 callback 拋出 → 傳播出 ``engine.synthesize`` → ``generate()`` 中止
3. 剩餘的段不會被合成

**已經開始的那一段仍會算完**——段內無法中斷。這是段要切短的另一個理由，
不只是為了 TTFA。

## 為什麼句子要另外緩衝

合成期間我們卡在 ``run_in_executor``，不會去消費上游的句子流。
如果不緩衝，SentenceSplitter 就被 TTS 擋住了，
「LLM 生成第二句的時間被 TTS 合成第一句吃掉」這個重疊也就沒了——
整條管線退化回序列相加。

所以有個 feeder task 持續把句子拉進有界緩衝。
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .. import config
from ..contracts.cancellation import CancellationToken, CancelledError, CancelReason
from ..contracts.types import TTS_SAMPLE_RATE, AudioChunk, Sentence
from ..core.channel import StreamChannel

DEFAULT_SAMPLE_RATE = TTS_SAMPLE_RATE
"""22050Hz，對齊 IndexTTS2 config 的 ``s2mel.preprocess_params.sr``。"""


class _Aborted(Exception):
    """內部訊號：從 callback 拋出以中止 ``generate()``。"""


def tensor_to_pcm16(tensor: Any) -> bytes:
    """把引擎吐出的音訊轉成 int16 little-endian PCM bytes。

    接受 torch tensor 或 numpy array——測試用假 backend 時是後者。
    contracts 層一律用 bytes，torch/numpy 只活在 adapter 邊界。
    """
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        import numpy as np

        arr = np.asarray(tensor, dtype="float32")

    import numpy as np

    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr[0]
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class IndexTTS2SpeakStage:
    """把 IndexTTS2 包成 SpeakStage。

    ``backend`` 可注入——`IndexTTS2Backend` 本身也接受 ``engine_factory``，
    但注入整個 backend 讓 adapter 的橋接邏輯能在**沒有 GPU、沒有 echo_tts**
    的環境下測試。Phase 1 的多數 bug 會出在橋接而不是模型。
    """

    name = "indextts2"

    def __init__(
        self,
        *,
        backend: Any | None = None,
        tts_repo: str | Path | None = None,
        device: str | None = None,
        use_fp16: bool | None = None,
        queue_size: int | None = None,
        sentence_buffer: int = 2,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str | None = None,
        emotion_vector: list[float] | None = None,
        emotion_overrides: dict | None = None,
        speed: float | None = None,
        output_dir: str | Path | None = None,
        keep_wav: bool = False,
        warmup: bool = False,
        warmup_text: str = "嗯。",
    ) -> None:
        self._backend = backend
        self._tts_repo = Path(tts_repo) if tts_repo else None
        self.device = device or config.get("ECHO_STREAM_TTS_DEVICE", "cuda")
        self.use_fp16 = (
            use_fp16 if use_fp16 is not None else config.get_bool("ECHO_STREAM_TTS_FP16", True)
        )
        self.queue_size = queue_size or int(config.get("ECHO_STREAM_TTS_QUEUE", "2"))
        self.sentence_buffer = sentence_buffer
        self.sample_rate = sample_rate
        self.language = language
        self.emotion_vector = emotion_vector
        self.emotion_overrides = emotion_overrides
        self.speed = (
            speed if speed is not None else config.get_float("ECHO_STREAM_TTS_SPEED", 0.0)
        )
        """語速。引擎裡是 ``stretch_ratio = config_stretch_ratio - speed × 0.5``
        （config 預設 1.72，stretch 越大越慢），範圍 -1.0 ~ 1.0。

        預設 0.0 是**引擎原本的預設值**，與 webapp 一致——adapter 不該偷改
        既有的音色特性。要調快就設這個參數，別讓管線層自己猜。"""
        self.keep_wav = keep_wav
        self.warmup = warmup
        self.warmup_text = warmup_text
        self._output_dir = Path(output_dir) if output_dir else None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self._prepared = False

    # --- 生命週期 ---

    async def prepare(self) -> None:
        """載入模型。很慢（數十秒），所以在 executor 跑並且只做一次。"""
        if self._prepared:
            return
        loop = asyncio.get_running_loop()
        if self._backend is None:
            self._backend = await loop.run_in_executor(None, self._build_backend)
        load = getattr(self._backend, "load", None)
        if callable(load):
            await loop.run_in_executor(None, load)

        # 語速要在載入後才能設（會動到引擎狀態）
        if self.speed:
            set_speed = getattr(self._backend, "set_speed", None)
            if callable(set_speed):
                await loop.run_in_executor(None, set_speed, self.speed)

        if self.warmup:
            await loop.run_in_executor(None, self._run_warmup)
        self._prepared = True

    def _run_warmup(self) -> None:
        """跑一次丟棄的合成。

        **預設關閉，因為 IndexTTS2 引擎自己已經做了。**

        `checkpoints_v25/config.yaml` 裡 ``enable_warmup: true``，
        引擎在 ``_init_models`` 階段就會跑 ``_warmup_models()``
        （預熱文本「你好，這是預熱測試。」），觸發 torch.compile 與
        CUDA kernel 編譯。載入回報的時間本來就含這一段。

        所以 adapter 再暖一次是**重複勞動**：實測多花 15 秒載入
        （27s → 44s），首段時間沒有改善（1788ms → 1842ms，在雜訊範圍內）。

        暖機本身是有效的——只是該由引擎做，不是這一層。保留這個選項是
        為了應付「引擎的內建暖機被關掉」或換用其他後端的情況。
        """
        path = self._next_output_path(
            Sentence(text=self.warmup_text, turn_id="warmup", index=0)
        )
        try:
            self._backend.generate(
                self.warmup_text,
                path,
                on_segment_audio=None,
                language=self.language,
                emotion_overrides=self.emotion_overrides,
            )
        except Exception:  # noqa: BLE001 - 暖機失敗不該擋住正常流程
            return
        finally:
            with contextlib.suppress(OSError):
                Path(path).unlink(missing_ok=True)

    def _build_backend(self) -> Any:
        """建立真的 IndexTTS2Backend。**只有這裡 import echo_tts。**"""
        repo = self._tts_repo or config.subsystem_path("TTS")
        config.ensure_importable(repo)

        from echo_tts.runtime.indextts_backend import IndexTTS2Backend
        from echo_tts.runtime.indextts_generate import default_generation_paths

        paths = default_generation_paths(repo)
        return IndexTTS2Backend(
            paths=paths,
            device=self.device,
            use_fp16=self.use_fp16,
            emotion_vector=self.emotion_vector,
        )

    async def aclose(self) -> None:
        backend = self._backend
        if backend is not None:
            unload = getattr(backend, "unload", None)
            if callable(unload):
                with contextlib.suppress(Exception):
                    unload()
        if self._tempdir is not None:
            with contextlib.suppress(Exception):
                self._tempdir.cleanup()
            self._tempdir = None
        self._prepared = False

    # --- 串流 ---

    async def stream(
        self, sentences: AsyncIterator[Sentence], token: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        await self.prepare()
        loop = asyncio.get_running_loop()

        # 句子緩衝：讓 SentenceSplitter 在 TTS 合成期間能繼續跑，
        # 否則兩層就變成序列相加而不是重疊
        sentence_ch: StreamChannel[Sentence] = StreamChannel(
            maxsize=self.sentence_buffer, token=token
        )
        audio_ch: StreamChannel[AudioChunk] = StreamChannel(
            maxsize=self.queue_size, token=token
        )

        feeder = asyncio.ensure_future(self._feed(sentences, sentence_ch, token))
        producer = asyncio.ensure_future(self._produce(sentence_ch, audio_ch, loop, token))

        try:
            async for chunk in audio_ch:
                yield chunk
        finally:
            for task in (feeder, producer):
                if not task.done():
                    task.cancel()
            # 收掉背景 task 的例外，避免 "Task exception was never retrieved"
            for task in (feeder, producer):
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task

    async def _feed(
        self,
        sentences: AsyncIterator[Sentence],
        channel: StreamChannel[Sentence],
        token: CancellationToken,
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
    ) -> None:
        try:
            async for sentence in sentence_ch:
                token.raise_if_cancelled()
                await loop.run_in_executor(
                    None, self._synthesize_one, sentence, audio_ch, loop, token
                )
        except CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            audio_ch.fail(exc)
            return
        audio_ch.close()

    def _synthesize_one(
        self,
        sentence: Sentence,
        audio_ch: StreamChannel[AudioChunk],
        loop: asyncio.AbstractEventLoop,
        token: CancellationToken,
    ) -> None:
        """在 executor 執行緒裡跑同步推理。**這個函式不在 event loop 上。**"""
        chunk_index = 0
        output_path = self._next_output_path(sentence)

        def on_segment(tensor: Any, seg_idx: int, total: int) -> None:
            nonlocal chunk_index
            if token.is_cancelled:
                raise _Aborted
            chunk = AudioChunk(
                pcm=tensor_to_pcm16(tensor),
                sample_rate=self.sample_rate,
                turn_id=sentence.turn_id,
                sentence_index=sentence.index,
                chunk_index=chunk_index,
                is_last=sentence.is_last and seg_idx >= total - 1,
            )
            chunk_index += 1
            # 滿了會阻塞這個執行緒 = 背壓傳到 GPU；
            # 取消時會拋 CancelledError，往上中止 generate()
            audio_ch.put_threadsafe(chunk, loop)

        try:
            self._backend.generate(
                sentence.text,
                output_path,
                on_segment_audio=on_segment,
                language=self.language,
                emotion_overrides=self.emotion_overrides,
            )
        except (_Aborted, CancelledError):
            return
        finally:
            if not self.keep_wav:
                with contextlib.suppress(OSError):
                    Path(output_path).unlink(missing_ok=True)

    def _next_output_path(self, sentence: Sentence) -> Path:
        """`generate()` 一定要寫檔，即使我們只要 callback 的音訊。

        預設丟暫存目錄並在合成後刪掉；``keep_wav=True`` 時保留，
        調參比對音質時會用到。
        """
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            base = self._output_dir
        else:
            if self._tempdir is None:
                self._tempdir = tempfile.TemporaryDirectory(prefix="echo_stream_tts_")
            base = Path(self._tempdir.name)
        return base / f"{sentence.turn_id}_{sentence.index:03d}.wav"


__all__ = ["IndexTTS2SpeakStage", "tensor_to_pcm16", "DEFAULT_SAMPLE_RATE"]
