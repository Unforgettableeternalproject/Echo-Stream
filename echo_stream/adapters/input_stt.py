"""InputStage adapter：真 STT（Phase 3）。

## 這一層在解什麼問題

與 TTS adapter 是鏡像問題：後端（faster-whisper 系）是**同步阻塞**推理，
契約是 **async pull**（``AsyncIterator[Utterance]``）。另外多兩件事：

1. **turn 邊界是這一層決定的，不是 STT 子系統。**
   echo_stt 自己有 VAD + segment 切分，但那是「一段可辨識的音訊」，
   不是「使用者講完一輪了」——後者是管線的職責（§7.5），而且
   之後要換 Smart Turn v3，控制權必須在這裡。
   所以走 ``transcribe_segment``（直接辨識、跳過子系統 VAD），
   VAD 與 turn 判定用 Echo Stream 自己的
   :class:`~echo_stream.core.vad.EnergyVad` + ``TurnDetector``。

2. **音訊消費不能停。** ``stream()`` 是 async generator，只有 runner
   在等它時才會推進；但 barge-in 要求機器人講話期間麥克風照聽。
   所以音訊處理放背景 task，經 :class:`StreamChannel` 出貨——
   generator 沒被拉動時，背景 task 照樣跑 VAD、照樣觸發
   ``on_voice_event``（前端據此接 InterruptionDetector → runner.interrupt）。

## 模型無關的邊界（2026-08-20 調查後的設計要求）

調查結論：Whisper 系暫時不換，但 FunASR Paraformer-streaming
（中文 CER ~10%、真串流、首字 ~600ms）是明確的 pilot 候選。
所以 transcriber 邊界收斂成 :class:`TranscriberBackend`——
**bytes 進、`TranscriptionResult` 出**，不滲漏任何子系統型別。
換 FunASR = 寫一個新 backend，這個 adapter 與管線契約都不動。

## 取樣率

依 §7.6，這裡假設 source 已是 16kHz mono int16（``STT_SAMPLE_RATE``）。
重採樣是 Frontend Adapter 的責任，不在這層做。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .. import config
from ..contracts.cancellation import CancellationToken, CancelledError
from ..contracts.stages import AudioSource
from ..contracts.turn import TurnDetector, TurnPolicy, VoiceEvent, VoiceState
from ..contracts.types import STT_SAMPLE_RATE, Utterance, new_turn_id
from ..core.channel import StreamChannel
from ..core.turn_detector import SilenceTurnDetector
from ..core.vad import EnergyVad


@dataclass(slots=True)
class TranscriptionResult:
    """一次轉錄的產物。backend 無關的最小集合。"""

    text: str
    language: str | None = None
    confidence: float = 1.0


@runtime_checkable
class TranscriberBackend(Protocol):
    """STT 後端的最小形狀。

    **同步阻塞**呼叫——adapter 負責丟進 executor。輸入是原始 PCM bytes
    而不是 numpy array，numpy 只活在各 backend 的內部。
    回傳 None 表示這段音訊沒有可用的轉錄（幻覺被過濾、純噪音）。
    """

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult | None: ...


class _Segment:
    """一段待轉錄的 turn 音訊與它的時間標記。"""

    __slots__ = ("pcm", "started_at", "ended_at")

    def __init__(self, pcm: bytes, started_at: float, ended_at: float) -> None:
        self.pcm = pcm
        self.started_at = started_at
        self.ended_at = ended_at


class SttInputStage:
    """把「PCM 流 → VAD → turn 判定 → 轉錄」包成 InputStage。

    ``backend`` 可注入——理由同 TTS adapter：橋接邏輯要能在沒有 GPU、
    沒有 echo_stt 的環境下測試，Phase 3 的多數 bug 會出在
    turn 切分與時間標記，不在模型。
    """

    name = "stt_input"

    def __init__(
        self,
        *,
        backend: TranscriberBackend | None = None,
        stt_repo: str | None = None,
        turn_detector: TurnDetector | None = None,
        policy: TurnPolicy | None = None,
        vad: EnergyVad | None = None,
        language: str | None = None,
        min_confidence: float | None = None,
        preroll_s: float = 0.2,
        max_pending_segments: int = 4,
        queue_size: int = 2,
        on_voice_event: Callable[[VoiceEvent], None] | None = None,
    ) -> None:
        self._backend = backend
        self._stt_repo = stt_repo
        self.policy = policy or TurnPolicy()
        self.turn_detector = turn_detector or SilenceTurnDetector(self.policy)
        self.vad = vad or EnergyVad()
        self.language = language or config.get("ECHO_STREAM_STT_LANGUAGE")
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else config.get_float("ECHO_STREAM_STT_MIN_CONFIDENCE", 0.35)
        )
        """低於此信心值的轉錄直接丟棄，不進 LLM。
        雜訊誤轉錄（實測 "Hello!" conf=0.2 之類）會觸發一整輪
        LLM + TTS，浪費之外還會污染對話歷史。實測正常語音 conf 0.6+。"""
        self.preroll_s = preroll_s
        """語音起點前保留的音訊量。VAD 判定總比實際起音晚一點，
        不補前導的話每句的第一個字會被削掉。"""

        self.max_pending_segments = max_pending_segments
        self.queue_size = queue_size
        self.on_voice_event = on_voice_event
        """每個 VAD 事件的同步 callback。前端在這裡接 InterruptionDetector
        （機器人講話中偵測插話 → ``runner.interrupt()``）。
        必須快——它在音訊路徑上。"""

        self._prepared = False

    # --- 生命週期 ---

    async def prepare(self) -> None:
        """載入模型。很慢（Whisper 載入數秒到數十秒），只做一次。"""
        if self._prepared:
            return
        loop = asyncio.get_running_loop()
        if self._backend is None:
            self._backend = await loop.run_in_executor(None, self._build_backend)
        self._prepared = True

    def _build_backend(self) -> TranscriberBackend:
        """建立真的 echo_stt backend。**只有這裡 import echo_stt。**"""
        repo = self._stt_repo or config.subsystem_path("STT")
        config.ensure_importable(repo)
        return _EchoSttBackend(language=self.language)

    async def aclose(self) -> None:
        backend = self._backend
        if backend is not None:
            close = getattr(backend, "close", None)
            if callable(close):
                loop = asyncio.get_running_loop()
                with contextlib.suppress(Exception):
                    await loop.run_in_executor(None, close)
        self._prepared = False

    # --- 串流 ---

    async def stream(
        self, source: AudioSource, token: CancellationToken
    ) -> AsyncIterator[Utterance]:
        await self.prepare()
        if source.sample_rate != self.vad.sample_rate:
            raise ValueError(
                f"source 取樣率 {source.sample_rate} 與 VAD 的 "
                f"{self.vad.sample_rate} 不符。重採樣屬於 Frontend Adapter"
                "（§7.6），不在這層做。"
            )
        loop = asyncio.get_running_loop()

        # 待轉錄段用有界 queue：轉錄追不上講話速度時，背壓回讀取端。
        # 順序靠單一 worker 保證——轉錄不能並行，否則 turn 會亂序進管線。
        segments: asyncio.Queue[_Segment | None] = asyncio.Queue(
            maxsize=self.max_pending_segments
        )
        out: StreamChannel[Utterance] = StreamChannel(maxsize=self.queue_size, token=token)

        reader = asyncio.ensure_future(self._read(source, token, segments))
        worker = asyncio.ensure_future(self._transcribe_worker(segments, out, loop, token))

        try:
            async for utterance in out:
                yield utterance
        except CancelledError:
            # 與 FakeInputStage 行為一致：session 取消 → 安靜結束，
            # 善後由 finally 做。取消不是 InputStage 的錯誤。
            return
        finally:
            for task in (reader, worker):
                if not task.done():
                    task.cancel()
            for task in (reader, worker):
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task

    async def _read(
        self,
        source: AudioSource,
        token: CancellationToken,
        segments: asyncio.Queue[_Segment | None],
    ) -> None:
        """音訊讀取迴圈：VAD → turn 判定 → 切出待轉錄段。

        這個 task 的壽命 = 整個 session。它**不做**任何耗時工作，
        轉錄在另一個 worker——否則辨識的一秒鐘裡 VAD 是聾的，
        插話偵測會有空窗。
        """
        preroll: deque[bytes] = deque()
        preroll_len_s = 0.0
        buffer = bytearray()
        speech_start_clock: float | None = None
        """語音起點的**音訊時鐘**。時間標記一律用音訊時鐘回推、共用同一個
        牆鐘錨點——混用兩種時鐘在比即時快的來源（測試、檔案）下會讓
        started_at / ended_at 順序反轉。"""
        speech_in_turn_s = 0.0
        last_speech_clock: float | None = None

        try:
            async for chunk in source.stream(token):
                if token.is_cancelled:
                    return
                if not chunk:
                    continue

                event = self.vad.process(chunk)
                if self.on_voice_event is not None:
                    try:
                        self.on_voice_event(event)
                    except Exception:  # noqa: BLE001 - 訂閱者壞掉不能斷麥克風
                        pass

                chunk_s = (len(chunk) // 2) / self.vad.sample_rate

                if event.state is VoiceState.SPEECH:
                    if speech_start_clock is None:
                        speech_start_clock = self.vad.clock_s - event.duration_s
                        # 補前導：語音其實在 VAD 反應之前就開始了
                        buffer.extend(b"".join(preroll))
                        preroll.clear()
                        preroll_len_s = 0.0
                    buffer.extend(chunk)
                    speech_in_turn_s += chunk_s
                    last_speech_clock = self.vad.clock_s
                else:
                    preroll.append(chunk)
                    preroll_len_s += chunk_s
                    while preroll_len_s > self.preroll_s and len(preroll) > 1:
                        dropped = preroll.popleft()
                        preroll_len_s -= (len(dropped) // 2) / self.vad.sample_rate

                decision = self.turn_detector.evaluate(event)
                if decision.is_end_of_turn and buffer:
                    # 兩個標記共用同一個牆鐘錨點，再各自用音訊時鐘回推。
                    # ended_at 回推到語音實際停止的時刻——靜音等待是
                    # turn 判定的成本，不是使用者的說話時間。
                    anchor = time.perf_counter()
                    clock = self.vad.clock_s
                    end_clock = last_speech_clock if last_speech_clock is not None else clock
                    start_clock = speech_start_clock if speech_start_clock is not None else end_clock
                    segment = _Segment(
                        pcm=bytes(buffer),
                        started_at=anchor - max(0.0, clock - start_clock),
                        ended_at=anchor - max(0.0, clock - end_clock),
                    )
                    buffer.clear()
                    speech_start_clock = None
                    speech_in_turn_s = 0.0
                    last_speech_clock = None
                    self.turn_detector.reset()
                    await segments.put(segment)
                elif (
                    buffer
                    and event.state is VoiceState.SILENCE
                    and event.duration_s >= self.policy.silence_threshold_s
                    and speech_in_turn_s < self.policy.min_utterance_s
                ):
                    # 太短不成 turn（咳嗽、關門聲）且靜音已足：丟掉。
                    # 不丟的話這段噪音會黏在下一個 turn 的開頭一起送去轉錄。
                    buffer.clear()
                    speech_start_clock = None
                    speech_in_turn_s = 0.0
                    last_speech_clock = None

            # source 正常結束（檔案播完 / 麥克風關閉）：flush 殘餘
            if buffer and speech_in_turn_s >= self.policy.min_utterance_s:
                anchor = time.perf_counter()
                clock = self.vad.clock_s
                end_clock = last_speech_clock if last_speech_clock is not None else clock
                start_clock = speech_start_clock if speech_start_clock is not None else end_clock
                await segments.put(
                    _Segment(
                        bytes(buffer),
                        anchor - max(0.0, clock - start_clock),
                        anchor - max(0.0, clock - end_clock),
                    )
                )
        except CancelledError:
            return
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                segments.put_nowait(None)

    async def _transcribe_worker(
        self,
        segments: asyncio.Queue[_Segment | None],
        out: StreamChannel[Utterance],
        loop: asyncio.AbstractEventLoop,
        token: CancellationToken,
    ) -> None:
        """依序轉錄。單一 worker——turn 的順序不能亂。"""
        try:
            while True:
                segment = await segments.get()
                if segment is None:
                    break
                token.raise_if_cancelled()
                result = await loop.run_in_executor(
                    None, self._backend.transcribe, segment.pcm, self.vad.sample_rate
                )
                if result is None or not result.text.strip():
                    continue
                if result.confidence < self.min_confidence:
                    continue  # 雜訊誤轉錄：擋在 LLM 之前（見 min_confidence）
                await out.put(
                    Utterance(
                        text=result.text.strip(),
                        turn_id=new_turn_id(),
                        is_final=True,
                        confidence=result.confidence,
                        language=result.language,
                        started_at=segment.started_at,
                        ended_at=segment.ended_at,
                    )
                )
        except CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - 傳給消費端而不是消失在 task 裡
            out.fail(exc)
            return
        out.close()


class _EchoSttBackend:
    """echo_stt（faster-whisper 系）的 TranscriberBackend 實作。

    走 ``MultiChannelSTTEngine.transcribe_segment``——同步直接辨識，
    跳過子系統自己的 VAD / channel 管線（turn 切分已由 adapter 做掉），
    但保留它的後處理（hallucination 過濾 + 繁簡轉換），那是實測踩過坑
    才加的東西，不該因為換了入口就丟掉。
    """

    def __init__(self, language: str | None = None) -> None:
        from echo_stt.core.config import load_config
        from echo_stt.core.multi_channel_engine import MultiChannelSTTEngine

        cfg = load_config()
        # transcribe_segment 需要引擎初始化，而初始化被 enabled 旗標擋著。
        # 這裡是程式化使用，不受子系統 yaml 的預設值管——強制打開。
        cfg.setdefault("multi_channel", {})["enabled"] = True
        self._language = language
        self._engine = MultiChannelSTTEngine(config=cfg)
        if not self._engine.initialize():
            raise RuntimeError(
                "MultiChannelSTTEngine 初始化失敗——檢查 echo_stt 的模型與設定"
            )

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult | None:
        import numpy as np

        if sample_rate != STT_SAMPLE_RATE:
            raise ValueError(f"echo_stt 只吃 {STT_SAMPLE_RATE}Hz，收到 {sample_rate}")
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        entry = self._engine.transcribe_segment(
            audio, channel_id="echo_stream", language=self._language
        )
        if entry is None:
            return None
        return TranscriptionResult(
            text=entry.text,
            language=entry.language,
            confidence=entry.confidence,
        )

    def close(self) -> None:
        self._engine.shutdown()


__all__ = [
    "SttInputStage",
    "TranscriberBackend",
    "TranscriptionResult",
]
