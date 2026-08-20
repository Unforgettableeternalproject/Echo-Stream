"""Web Frontend Adapter——用瀏覽器 host 整條串流管線。

## 為什麼是 Web 而不是 Discord

Discord bot 有太多與管線無關的複雜度（DAVE 加密、presence、cogs 權限）。
在那裡除錯管線，永遠分不清是管線問題還是 Discord 問題。
Web 前端只做一件事：**把管線的串流行為呈現出來，讓人能聽也能看數字。**

## 執行緒模型（這裡最容易寫錯）

HTTP server 是多執行緒的（``ThreadingHTTPServer``），管線是 asyncio 的。
橋接方式：

* 一個**常駐的 event loop** 跑在背景執行緒
* HTTP handler 用 ``asyncio.run_coroutine_threadsafe`` 提交工作並等結果
* 插話用 ``loop.call_soon_threadsafe``——``asyncio.Event.set()``
  從別的執行緒直接呼叫**不是 thread-safe** 的，這是 barge-in 會不會
  隨機失效的關鍵

音訊寫入 socket 只發生在 loop 執行緒（HTTP 執行緒此時阻塞在等結果），
所以同一時間只有一個 writer，不需要額外加鎖。

## Frame 協定

沿用 TTS 專案 ``webapp/tts_server.py`` 的格式，前端程式碼可以互相參考：

    1 byte type + 4 byte big-endian length + payload

每段音訊是**獨立完整的 WAV**（含 header），前端可以直接餵給
``decodeAudioData``，不必自己拼 PCM。
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import queue
import struct
import threading
import time
import uuid
import wave
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .. import config
from ..contracts.cancellation import CancellationToken, CancelReason
from ..contracts.types import STT_SAMPLE_RATE, AudioChunk, TurnPhase, Utterance
from ..core.pipeline import PipelineRunner
from ..core.splitter import SplitPolicy
from ..core.tracer import LatencyTracer

FRAME_AUDIO = 0x01
FRAME_META = 0x02
FRAME_ERROR = 0x03
FRAME_TEXT = 0x04
"""句子文字。讓前端能顯示字幕與逐句延遲——沒有這個就只能聽，
調參時看不出是哪一句慢。"""

FRAME_UTTER = 0x05
"""語音輸入的轉錄結果（使用者說了什麼 + STT 延遲）。"""

FRAME_EVENT = 0x06
"""狀態事件：VAD speech/silence、barge_in。前端據此顯示聆聽狀態、停播。"""

STATIC_DIR = Path(__file__).parent / "static"


def pcm_to_wav(chunk: AudioChunk) -> bytes:
    """把 AudioChunk 包成獨立可播的 WAV。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(chunk.channels)
        w.setsampwidth(2)
        w.setframerate(chunk.sample_rate)
        w.writeframes(chunk.pcm)
    return buf.getvalue()


def frame(kind: int, payload: bytes) -> bytes:
    return bytes([kind]) + struct.pack(">I", len(payload)) + payload


def json_frame(kind: int, obj: Any) -> bytes:
    return frame(kind, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


class _PushSource:
    """由 HTTP 執行緒餵資料的 AudioSource。

    瀏覽器的音訊 chunk 經 POST 進來（HTTP 執行緒），管線在 loop 執行緒
    消費——跨執行緒一律走 ``call_soon_threadsafe``。滿了丟最舊：
    對話場景裡遲到的音訊沒有價值，堵住上傳只會讓延遲雪球。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sample_rate: int,
        max_chunks: int = 256,
    ) -> None:
        self._loop = loop
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max_chunks)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def push(self, pcm: bytes) -> None:
        """從 HTTP 執行緒餵入一個 int16 PCM chunk。"""
        self._loop.call_soon_threadsafe(self._enqueue, pcm)

    def close(self) -> None:
        """從 HTTP 執行緒宣告結束。stream 消化完殘餘後正常結束（會 flush）。"""
        self._loop.call_soon_threadsafe(self._enqueue, None)

    def _enqueue(self, item: bytes | None) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(item)

    async def stream(self, token):  # noqa: ANN001 - AudioSource 契約
        while not token.is_cancelled:
            get_task = asyncio.ensure_future(self._queue.get())
            cancel_task = asyncio.ensure_future(token.wait())
            try:
                done, _ = await asyncio.wait(
                    {get_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                for task in (get_task, cancel_task):
                    if not task.done():
                        task.cancel()
            if get_task not in done:
                return
            item = get_task.result()
            if item is None:
                return
            yield item


class _FakeWebTranscriber:
    """fake 模式的 transcriber——不辨識，只回報收到了多少語音。

    存在的意義：讓「瀏覽器擷取 → 上傳 → VAD → turn 切分」這整條
    串流接收路徑能在沒有 Whisper、沒有 GPU 的機器上驗證。
    """

    def transcribe(self, pcm: bytes, sample_rate: int):
        from ..adapters.input_stt import TranscriptionResult

        seconds = len(pcm) / 2 / sample_rate
        return TranscriptionResult(text=f"（fake 轉錄：收到 {seconds:.1f}s 語音）")


class StreamService:
    """持有管線與常駐 event loop。"""

    def __init__(
        self,
        *,
        use_real_tts: bool = False,
        use_real_llm: bool = False,
        use_real_stt: bool = False,
        split_policy: SplitPolicy | None = None,
        system_prompt: str = "",
        trace_path: str | None = None,
    ) -> None:
        self.use_real_tts = use_real_tts
        self.use_real_llm = use_real_llm
        self.use_real_stt = use_real_stt
        self.split_policy = split_policy or SplitPolicy()
        self.system_prompt = system_prompt

        # 每次啟動自動開一個 session log（JSONL，逐 turn 一筆）——
        # 測試結果要能事後分析，不能只活在瀏覽器畫面上。
        session_tag = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parents[2] / "outputs" / "web_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path: Path = log_dir / f"session_{session_tag}.jsonl"
        if trace_path is None:
            trace_path = str(log_dir / f"session_{session_tag}_trace.jsonl")
        self.tracer = LatencyTracer(output_path=trace_path)
        self.trace_path = trace_path

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="echo-stream-loop"
        )
        self._thread.start()

        self._runner: PipelineRunner | None = None
        self._think = None
        self._speak = None
        self._stt = None
        self._lock = threading.Lock()
        """序列化 turn。GPU 推理本來就無法並行，排隊比搶資源好。"""

        self._voice: dict[str, Any] | None = None
        """當前的語音 session（單使用者，一次一個）。"""

        self._voice_lock = threading.Lock()

        self.ready = False
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    # --- 生命週期 ---

    def prepare(self) -> None:
        """建立並暖機管線。載入模型很慢，所以與 HTTP 啟動分開。"""
        t0 = time.perf_counter()
        try:
            fut = asyncio.run_coroutine_threadsafe(self._build(), self._loop)
            fut.result()
            self.ready = True
            self.load_seconds = time.perf_counter() - t0
            self._log(
                {
                    "type": "session_start",
                    "real_stt": self.use_real_stt,
                    "real_llm": self.use_real_llm,
                    "real_tts": self.use_real_tts,
                    "system_prompt_chars": len(self.system_prompt),
                    "load_seconds": round(self.load_seconds, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"

    async def _build(self) -> None:
        from ..adapters.input_stt import SttInputStage
        from ..fakes import FakeSpeakStage, FakeThinkStage

        # STT stage 建一次、跨 voice session 重用——真 Whisper 載入要數十秒，
        # 不能每次按下麥克風都重載。fake 模式注入假 transcriber，
        # 讓「瀏覽器擷取 → 上傳 → VAD → turn 切分」在無 GPU 環境也能驗證。
        if self.use_real_stt:
            self._stt = SttInputStage()
        else:
            self._stt = SttInputStage(backend=_FakeWebTranscriber())
        await self._stt.prepare()

        if self.use_real_tts:
            from ..adapters.speak_indextts import IndexTTS2SpeakStage

            self._speak = IndexTTS2SpeakStage()
        else:
            self._speak = FakeSpeakStage()

        if self.use_real_llm:
            from ..adapters.think_llm import LLMThinkStage

            self._think = LLMThinkStage(
                system_prompt=self.system_prompt,
                split_policy=self.split_policy,
                tracer=self.tracer,
            )
        else:
            self._think = FakeThinkStage(
                split_policy=self.split_policy, tracer=self.tracer
            )

        self._runner = PipelineRunner(
            think_stage=self._think,
            speak_stage=self._speak,
            tracer=self.tracer,
        )
        await self._runner.prepare()

    def shutdown(self) -> None:
        if self._runner is not None:
            with_timeout = asyncio.run_coroutine_threadsafe(
                self._runner.aclose(), self._loop
            )
            try:
                with_timeout.result(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # --- 插話 ---

    def interrupt(self) -> bool:
        """從 HTTP 執行緒中止當前 turn。

        **必須走 call_soon_threadsafe**——CancellationToken 內部是
        ``asyncio.Event``，從別的執行緒直接 ``set()`` 不是 thread-safe 的。
        這是 barge-in 會不會隨機失效的關鍵。
        """
        runner = self._runner
        if runner is None:
            return False
        done = threading.Event()
        result: list[bool] = []

        def _do():
            result.append(runner.interrupt(CancelReason.BARGE_IN))
            done.set()

        self._loop.call_soon_threadsafe(_do)
        done.wait(timeout=2.0)
        return bool(result and result[0])

    # --- 記錄 ---

    def _log(self, record: dict[str, Any]) -> None:
        """寫一筆 JSONL。失敗不擋主流程——log 是旁路。"""
        record["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # --- 語音 session ---

    def voice_start(self) -> dict[str, Any]:
        """建立語音 session。一次一個——重複 start 會頂掉舊的。"""
        if not self.ready:
            return {"error": "管線尚未就緒"}
        with self._voice_lock:
            self._voice_close_locked()
            sid = uuid.uuid4().hex[:12]
            session: dict[str, Any] = {
                "sid": sid,
                "source": _PushSource(self._loop, STT_SAMPLE_RATE),
                "out": queue.Queue(maxsize=256),
                "token": CancellationToken(),
            }
            self._voice = session
            asyncio.run_coroutine_threadsafe(self._voice_loop(session), self._loop)
            return {"sid": sid, "sample_rate": STT_SAMPLE_RATE}

    def voice_push(self, sid: str, pcm: bytes) -> bool:
        session = self._voice
        if session is None or session["sid"] != sid:
            return False
        session["source"].push(pcm)
        return True

    def voice_stop(self, sid: str) -> bool:
        """結束擷取。source 收尾後 STT 會 flush 殘餘語音再結束。"""
        session = self._voice
        if session is None or session["sid"] != sid:
            return False
        session["source"].close()
        return True

    def _voice_close_locked(self) -> None:
        session = self._voice
        if session is None:
            return
        session["source"].close()
        self._loop.call_soon_threadsafe(
            session["token"].cancel, CancelReason.SUPERSEDED
        )
        self._voice = None

    async def _voice_loop(self, session: dict[str, Any]) -> None:
        """語音 session 的主迴圈：STT 切出 turn → 跑管線 → frame 推回瀏覽器。

        整段跑在 loop 執行緒。turn 之間天然序列化（async for 一次一個），
        所以**不拿** ``self._lock``——那是 threading.Lock，在 loop 執行緒上
        等它會卡死整個 event loop。語音與文字輸入同時打的邊界情況
        由 GPU 排隊自然處理，單人測試不構成問題。
        """
        from ..core.turn_detector import DurationInterruptionDetector

        stage = self._stt
        out: queue.Queue = session["out"]
        token: CancellationToken = session["token"]

        def emit(payload: bytes) -> None:
            with contextlib.suppress(queue.Full):
                out.put_nowait(payload)

        # 自動 barge-in：機器人講話中偵測到持續語音 → 中止。
        # 判定器與 turn 判定分離（§7.5），門檻在 TurnPolicy。
        detector = DurationInterruptionDetector(
            self._runner.policy if self._runner else None
        )
        last_state: list[str | None] = [None]

        def on_event(event) -> None:  # noqa: ANN001 - VoiceEvent
            state = event.state.value
            if state != last_state[0]:
                last_state[0] = state
                emit(json_frame(FRAME_EVENT, {"event": state}))
            runner = self._runner
            if runner is not None and runner.is_speaking:
                if detector.evaluate(event).should_interrupt:
                    if runner.interrupt(CancelReason.BARGE_IN):
                        emit(json_frame(FRAME_EVENT, {"event": "barge_in"}))
                        self._log({"type": "barge_in"})

        stage.on_voice_event = on_event
        try:
            async for utterance in stage.stream(session["source"], token):
                stt_ms = (time.perf_counter() - utterance.ended_at) * 1000
                utter_info = {
                    "text": utterance.text,
                    "stt_ms": round(stt_ms),
                    "confidence": round(utterance.confidence, 2),
                    "language": utterance.language,
                }
                emit(json_frame(FRAME_UTTER, utter_info))
                self._log({"type": "utterance", **utter_info})
                # 傳真的 Utterance 而不是重建——ended_at 是真實的說完時刻，
                # tracer 的 TTFA 才會把 STT 的耗時算進去
                await self._run(utterance, emit)

                # turn 結束補一個 padding frame：tunnel 代理（cloudflared）會
                # 緩衝小尾巴，最後一段音訊可能卡在代理層直到下一輪資料把它
                # 擠出去——「最後一段等下次輸入才播」就是這個。前端會忽略它。
                emit(json_frame(FRAME_EVENT, {"event": "flush", "pad": "." * 16384}))
        except Exception as exc:  # noqa: BLE001
            emit(json_frame(FRAME_ERROR, {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            stage.on_voice_event = None
            with contextlib.suppress(queue.Full):
                out.put_nowait(None)
            if self._voice is session:
                self._voice = None

    def voice_frames(self, sid: str):
        """給 GET handler 迭代的 frame 流。在 HTTP 執行緒上跑。"""
        session = self._voice
        if session is None or session["sid"] != sid:
            return
        out: queue.Queue = session["out"]
        while True:
            try:
                item = out.get(timeout=60.0)
            except queue.Empty:
                continue  # 沒動靜就繼續等——長靜音是正常狀態
            if item is None:
                return
            yield item

    # --- 執行 ---

    def run_turn(self, text: str, emit: Callable[[bytes], None]) -> None:
        """跑一輪對話，逐段把音訊與字幕送給 ``emit``。

        在 HTTP 執行緒呼叫，會阻塞到這一輪結束。
        """
        if self._runner is None:
            emit(json_frame(FRAME_ERROR, {"error": "管線尚未就緒"}))
            return

        with self._lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._run(Utterance(text=text), emit), self._loop
            )
            fut.result()

    async def _run(self, utterance: Utterance, emit: Callable[[bytes], None]) -> None:
        runner = self._runner
        assert runner is not None

        turn_id = utterance.turn_id
        t0 = time.perf_counter()

        # 掛在 runner 的 sink 上會讓 PipelineRunner 負責 drain 與播放進度，
        # 但這裡的「播放」在瀏覽器端，伺服器無從得知進度。
        # 所以自己收 chunk，並把 spoken 的判定留給前端回報（Phase 5 再處理）。
        sink = _EmitSink(emit, t0)
        runner.sink = sink
        runner.think_stage = _EmitThink(self._think, emit, t0)

        result = await runner.run_turn(utterance)

        trace = self.tracer.get(turn_id)
        segments = trace.segments_ms() if trace else {}
        meta = {
            "turn_id": turn_id,
            "phase": result.phase.value,
            "cancel_reason": result.cancel_reason,
            "error": result.error,
            "generated": result.generated_text,
            "spoken": result.spoken_text,
            "segments_ms": {
                k: (round(v, 1) if v is not None else None)
                for k, v in segments.items()
            },
            "sentences": trace.sentence_count if trace else 0,
            "chunks": trace.chunk_count if trace else 0,
            "split_reasons": dict(trace.split_reasons) if trace else {},
            "audio_seconds": round(sink.written_duration_s, 2),
            "report": self.tracer.report(turn_id),
        }
        emit(json_frame(FRAME_META, meta))
        self._log({"type": "turn", "input": utterance.text, **meta})


class _EmitThink:
    """在句子流上分岔一份給前端顯示，不改變內容。

    句子不會經過 sink（那只收音訊），所以要在這裡攔。
    純轉發——包一層比在 PipelineRunner 裡加 callback 乾淨，
    因為那樣會讓核心多一個只有 Web 前端在用的參數。
    """

    def __init__(self, inner: Any, emit: Callable[[bytes], None], t0: float) -> None:
        self._inner = inner
        self._emit = emit
        self._t0 = t0

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", "think")

    async def prepare(self) -> None:
        await self._inner.prepare()

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def stream(self, utterance, token):
        async for sentence in self._inner.stream(utterance, token):
            if sentence.text:
                self._emit(
                    json_frame(
                        FRAME_TEXT,
                        {
                            "index": sentence.index,
                            "text": sentence.text,
                            "is_first": sentence.is_first,
                            "split_reason": sentence.split_reason,
                            "ms": (time.perf_counter() - self._t0) * 1000,
                        },
                    )
                )
            yield sentence

    async def commit(self, turn_id: str, spoken_text: str, generated_text: str) -> None:
        await self._inner.commit(turn_id, spoken_text, generated_text)


class _EmitSink:
    """把音訊直接送進 HTTP response 的 sink。

    ``played_duration_s`` 回傳「已送出」而非「已播出」——真正的播放進度在
    瀏覽器端，伺服器不知道。這代表 **barge-in 的 spoken_text 判定在
    Web 前端會偏樂觀**（算成整段都聽到了）。要精確得讓前端回報播放位置，
    留給 Phase 5 一起處理。
    """

    def __init__(self, emit: Callable[[bytes], None], t0: float) -> None:
        self._emit = emit
        self._t0 = t0
        self.written_duration_s = 0.0
        self._stopped = False
        self._seen_sentences: set[int] = set()

    async def write(self, chunk: AudioChunk) -> None:
        if self._stopped:
            return
        self.written_duration_s += chunk.duration_s
        self._emit(frame(FRAME_AUDIO, pcm_to_wav(chunk)))

    def stop(self) -> None:
        self._stopped = True

    async def drain(self) -> None:
        return

    @property
    def played_duration_s(self) -> float:
        return self.written_duration_s


class _Handler(BaseHTTPRequestHandler):
    service: StreamService

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A002 - 壓掉預設的逐請求 log
        return

    # --- 路由 ---

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json(
                200,
                {
                    "ready": self.service.ready,
                    "error": self.service.load_error,
                    "load_seconds": self.service.load_seconds,
                    "real_tts": self.service.use_real_tts,
                    "real_llm": self.service.use_real_llm,
                    "real_stt": self.service.use_real_stt,
                },
            )
        elif self.path.startswith("/api/voice/stream"):
            sid = self._query_param("sid")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for payload in self.service.voice_frames(sid):
                    self.wfile.write(f"{len(payload):X}\r\n".encode())
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except Exception:  # noqa: BLE001 - 瀏覽器斷線是正常結束
                return
            with contextlib.suppress(Exception):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/interrupt":
            self._json(200, {"interrupted": self.service.interrupt()})
            return
        if self.path == "/api/voice/start":
            self._json(200, self.service.voice_start())
            return
        if self.path.startswith("/api/voice/audio"):
            sid = self._query_param("sid")
            length = int(self.headers.get("Content-Length", "0"))
            pcm = self.rfile.read(length) if length else b""
            ok = bool(pcm) and self.service.voice_push(sid, pcm)
            self._json(200 if ok else 404, {"ok": ok})
            return
        if self.path.startswith("/api/voice/stop"):
            sid = self._query_param("sid")
            self._json(200, {"ok": self.service.voice_stop(sid)})
            return
        if self.path != "/api/say":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        text = (body.get("text") or "").strip()
        if not text:
            self._json(400, {"error": "text 不可為空"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(payload: bytes) -> None:
            # chunked encoding：每個 frame 一個 chunk，瀏覽器才能即時收到
            self.wfile.write(f"{len(payload):X}\r\n".encode())
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        try:
            self.service.run_turn(text, emit)
        except Exception as exc:  # noqa: BLE001
            try:
                emit(json_frame(FRAME_ERROR, {"error": f"{type(exc).__name__}: {exc}"}))
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:  # noqa: BLE001
                pass

    # --- 工具 ---

    def _query_param(self, key: str) -> str:
        from urllib.parse import parse_qs, urlparse

        values = parse_qs(urlparse(self.path).query).get(key)
        return values[0] if values else ""

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._json(404, {"error": f"missing {path.name}"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _StrictServer(ThreadingHTTPServer):
    """不設 SO_REUSEADDR 的 HTTP server。

    Python 預設 ``allow_reuse_address=1``，在 Windows 上這代表**兩個行程
    可以同時 bind 同一個 port**——舊 server 沒關乾淨時，新 server 啟動
    「成功」但流量進舊行程，人設、程式碼更新全都看似失效
    （2026-08-21 踩過，查了很久）。關掉它讓重複啟動直接炸 bind error。
    """

    allow_reuse_address = False


def serve(
    host: str | None = None,
    port: int | None = None,
    *,
    real_tts: bool = False,
    real_llm: bool = False,
    real_stt: bool = False,
    split_policy: SplitPolicy | None = None,
    system_prompt: str = "",
    trace_path: str | None = None,
) -> None:
    host = host or config.get("ECHO_STREAM_WEB_HOST", "127.0.0.1")
    port = port or int(config.get("ECHO_STREAM_WEB_PORT", "8770"))

    service = StreamService(
        use_real_tts=real_tts,
        use_real_llm=real_llm,
        use_real_stt=real_stt,
        split_policy=split_policy,
        system_prompt=system_prompt,
        trace_path=trace_path,
    )

    print("═" * 56)
    print("  Echo Stream — Web 前端")
    print("═" * 56)
    print(f"  TTS: {'真 IndexTTS2' if real_tts else 'fake'}")
    print(f"  LLM: {'真 OpenAI' if real_llm else 'fake'}")
    print(f"  STT: {'真 Whisper' if real_stt else 'fake（只驗串流接收）'}")
    print(f"  Log: {service.log_path}")
    if system_prompt:
        print(f"  System prompt: {len(system_prompt)} 字")
    print("\n  載入管線中…", flush=True)

    service.prepare()
    if service.load_error:
        print(f"  ✗ 載入失敗：{service.load_error}")
    else:
        print(f"  ✓ 就緒（{service.load_seconds:.1f}s）")

    handler = type("Handler", (_Handler,), {"service": service})
    try:
        httpd = _StrictServer((host, port), handler)
    except OSError:
        print(f"\n  ✗ {host}:{port} 已被占用——有舊的 server 沒關乾淨。")
        print("    執行 serve_web.bat stop（會按 port 掃，不管是誰起的）再重試。")
        service.shutdown()
        raise SystemExit(1)
    print(f"\n  → http://{host}:{port}\n")
    print("  Ctrl+C 結束")
    print("═" * 56)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  收工。")
    finally:
        httpd.shutdown()
        service.shutdown()


__all__ = ["serve", "StreamService", "pcm_to_wav", "frame", "json_frame"]
