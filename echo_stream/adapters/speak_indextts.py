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
from ..contracts.cancellation import CancellationToken, CancelledError
from ..contracts.style import EMOTION_DIMENSIONS, SpeechStyle
from ..contracts.types import TTS_SAMPLE_RATE, AudioChunk, Sentence
from ..core.channel import StreamChannel

DEFAULT_SAMPLE_RATE = TTS_SAMPLE_RATE
"""22050Hz，對齊 IndexTTS2 config 的 ``s2mel.preprocess_params.sr``。"""


class _Aborted(Exception):
    """內部訊號：從 callback 拋出以中止 ``generate()``。"""


CONFIG_STRETCH_RATIO = 1.72
"""引擎 config 的 ``s2mel_stretch_ratio`` 預設值。

引擎裡的關係是 ``stretch_ratio = config_stretch_ratio - speed × 0.5``，
stretch 越大唸得越慢。
"""


def scale_emotion(vector: list[float], intensity: float) -> list[float]:
    """八維向量 × 相對強度 → 送進引擎的向量（總和落在 echo_tts 定案區間）。

    區間常數與縮放函式從 echo_tts import——TTS 那邊調區間，這裡自動跟，
    不在整合層再抄一份數字。import 不到（舊分支）退回「向量 × 強度」。
    """
    try:
        from echo_tts.config.emotion_presets import scale_to_user_strength
    except ImportError:
        alpha = max(0.0, min(1.0, intensity))
        return [v * alpha for v in vector]
    return scale_to_user_strength(list(vector), intensity)


def load_emotion_presets(
    path: str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """讀 echo_tts 的 ``emotion_presets.yaml``，轉成 core 的 preset 表形狀。

    回傳 ``(presets, aliases)``：presets 是 key → 八維分佈（只取 vector，
    ``overrides`` 那些 dynamics / text_markers 是 echo_tts session 層的功能，
    這條管線繞過它，不接）；aliases 目前空——中文別名在 core 的內建表。
    """
    from echo_tts.config.emotion_presets import get_preset_vector, load_presets

    raw = load_presets(path) if path else load_presets()
    presets: dict[str, dict[str, float]] = {}
    for key, entry in raw.items():
        vec = list(get_preset_vector(entry))
        presets[str(key)] = {
            dim: float(v)
            for dim, v in zip(EMOTION_DIMENSIONS, vec, strict=False)
            if float(v) > 0.0
        }
    return presets, {}


def speed_multiplier_to_offset(multiplier: float) -> float:
    """把 :attr:`SpeechStyle.speed` 的**倍率**換成 IndexTTS 的 speed 偏移。

    契約層用倍率（1.0 = 正常、1.1 = 快 10%），因為那是 backend 無關的說法；
    IndexTTS 用 -1~1 的偏移。換算：

        stretch = CONFIG / r        （要快 r 倍，就把拉伸縮小 r 倍）
        offset  = (CONFIG - stretch) / 0.5 = 2 × CONFIG × (1 - 1/r)

    夾在 [-1, 1]，超出範圍的倍率會被截斷而不是拋錯——
    語速設過頭應該是「頂到底」，不該讓整個 turn 掛掉。
    """
    if multiplier <= 0:
        return 0.0
    offset = 2.0 * CONFIG_STRETCH_RATIO * (1.0 - 1.0 / multiplier)
    return max(-1.0, min(1.0, offset))


_TTS_STRIP_CHARS = "「」『』【】《》〈〉“”‘’\"'`*_#~<>|（）()［］[]"
"""合成前剔除的符號。引號會被 IndexTTS 唸出怪聲（實測），
括號、markdown 記號同理——它們是視覺標記，不是語音內容。
逗號句號等韻律標點**保留**，那些影響停頓，是語音的一部分。"""


def sanitize_for_tts(text: str) -> str:
    """清掉不該被唸出來的符號，並收攏多餘空白。"""
    import re

    # 句中的列表破折號（「小說。 - 《死亡擱淺》」）換成頓點——那是列舉的
    # 停頓，不是連字號。兩側都有空白才算，"well-known" 這種不受影響。
    cleaned = re.sub(r"\s+[-–—•·・]+\s+", "，", text)
    cleaned = cleaned.translate({ord(c): None for c in _TTS_STRIP_CHARS})
    return " ".join(cleaned.split())


def smooth_segment_edges(
    pcm: bytes,
    sample_rate: int,
    fade_ms: float = 4.0,
    tail_silence_ms: float = 15.0,
) -> bytes:
    """段緣平滑：半 Hann 淡入淡出 + 段尾短靜音墊。

    段與段是獨立合成的，邊界樣本值不連續，接起來就是爆音（click）。
    調查結論（docs/design/tts-improvement-reference.md）：Hann window
    淡化 + 段間短靜音是純訊號處理的消爆音手段，UTMOS 損失僅 -0.08~-0.13。

    淡化只動段緣幾毫秒（預設 4ms ≈ 88 樣本 @22050Hz），聽不出音量變化；
    靜音墊直接附在段尾，前端排程不需要知道這件事。
    """
    import numpy as np

    arr = np.frombuffer(pcm, dtype="<i2").astype("float32")
    n_fade = min(int(sample_rate * fade_ms / 1000), len(arr) // 2)
    if n_fade > 0:
        # 半 Hann：0→1 的升沿。比線性淡化少一點高頻假影。
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
        arr[:n_fade] *= ramp
        arr[-n_fade:] *= ramp[::-1]
    out = arr.astype("<i2").tobytes()
    n_pad = int(sample_rate * tail_silence_ms / 1000)
    if n_pad > 0:
        out += b"\x00\x00" * n_pad
    return out


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
        edge_fade_ms: float = 4.0,
        segment_silence_ms: float = 15.0,
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
        self.edge_fade_ms = edge_fade_ms
        """段緣淡入淡出的長度（毫秒）。0 = 停用，消段間爆音用。"""

        self.segment_silence_ms = segment_silence_ms
        """段尾靜音墊（毫秒）。0 = 停用。與淡化搭配消爆音，見
        :func:`smooth_segment_edges`。"""

        self._default_emotion: list[float] | None = None
        """backend 開機時的情緒向量（角色檔預設）。style 退場時要**還原**——
        不還原的話，一句帶情緒的話會讓之後所有中性句都殘留那個情緒。"""

        self._current_speed_offset: float = 0.0
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
        self._current_speed_offset = self.speed or 0.0

        # 記住角色檔的預設情緒——per-sentence style 退場時要還原到這裡
        default_emotion = getattr(self._backend, "emotion_vector", None)
        self._default_emotion = list(default_emotion) if default_emotion else None

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
        self._apply_style(sentence.style)
        speak_text = sanitize_for_tts(sentence.text)
        if not speak_text:
            return  # 整句都是符號（例如純引號）——沒東西可唸

        def on_segment(tensor: Any, seg_idx: int, total: int) -> None:
            nonlocal chunk_index
            if token.is_cancelled:
                raise _Aborted
            pcm = tensor_to_pcm16(tensor)
            if self.edge_fade_ms > 0 or self.segment_silence_ms > 0:
                pcm = smooth_segment_edges(
                    pcm,
                    self.sample_rate,
                    fade_ms=self.edge_fade_ms,
                    tail_silence_ms=self.segment_silence_ms,
                )
            chunk = AudioChunk(
                pcm=pcm,
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
                speak_text,
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

    def _apply_style(self, style: SpeechStyle | None) -> None:
        """把句子的 :class:`SpeechStyle` 翻譯成 IndexTTS 的原生控制。

        在 executor 執行緒裡呼叫，且句子是**序列**合成的（一次一句），
        所以直接改 backend 狀態是安全的——不會有兩句同時在改。

        ``style`` 為 None 或中性時完全不動 backend：預設音色是既有資產，
        adapter 不該無故覆寫它。

        ``expressiveness`` 與 ``volume`` 目前沒有對應的 IndexTTS 參數，
        會被忽略——這是刻意的，SpeechStyle 是**各後端取所需**的共同描述，
        不是每個欄位都保證被實作。

        ``intensity`` 的翻譯走 echo_tts 的強度區間（2026-08-23 改）：
        0~1 刻度 → ``user_to_strength`` → 總和 0.57~0.73（0.5 = 0.65 甜蜜點），
        再 ``scale_to_strength`` 等比縮放——preset 向量只剩比例有意義。
        舊做法「向量 × intensity」送進去的總和多半 < 0.5，落在 TTS 那邊定案的
        「情緒表現不足」區，LLM 標記聽起來跟沒標一樣。

        style 為 None 或中性時**還原**角色檔預設，而不是不動——
        上一句的情緒殘留在 backend 狀態裡，是「一句開心之後全部都開心」
        這種 bug 的來源。
        """
        backend = self._backend

        # 情緒向量：有 style 就換算，沒有就還原預設
        if style is not None and not style.is_neutral:
            backend.emotion_vector = scale_emotion(style.emotion_vector(), style.intensity)
        else:
            backend.emotion_vector = (
                list(self._default_emotion) if self._default_emotion else None
            )

        # 語速：style 指定倍率就換算成偏移，否則回到啟動時的基準值。
        # 只在值真的變了才呼叫 set_speed——不必每句都戳引擎。
        target = (
            speed_multiplier_to_offset(style.speed)
            if style is not None and style.speed != 1.0
            else (self.speed or 0.0)
        )
        if target != self._current_speed_offset:
            set_speed = getattr(backend, "set_speed", None)
            if callable(set_speed):
                set_speed(target)
                self._current_speed_offset = target

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


__all__ = [
    "IndexTTS2SpeakStage",
    "sanitize_for_tts",
    "smooth_segment_edges",
    "tensor_to_pcm16",
    "speed_multiplier_to_offset",
    "scale_emotion",
    "load_emotion_presets",
    "DEFAULT_SAMPLE_RATE",
    "CONFIG_STRETCH_RATIO",
]
