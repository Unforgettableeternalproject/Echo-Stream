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
import sys
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
from ..contracts.style import EMOTION_DIMENSIONS, SpeechStyle
from ..contracts.types import STT_SAMPLE_RATE, AudioChunk, Utterance
from ..core.dream_scheduler import DreamPolicy, DreamScheduler
from ..core.emotion import EMOTION_PRESETS
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
        use_real_memory: bool = False,
        split_policy: SplitPolicy | None = None,
        system_prompt: str = "",
        trace_path: str | None = None,
        store: Any | None = None,
    ) -> None:
        self.use_real_tts = use_real_tts
        self.use_real_llm = use_real_llm
        self.use_real_stt = use_real_stt
        self.use_real_memory = use_real_memory

        self._memory = None
        """EchoMemoryAdapter。旁路——建不起來只停用記憶，不擋管線。"""
        self._memory_enabled = True
        """面板開關。關掉 = 不 retrieve、不注入、不 store；
        adapter 與 engine 留著（重載要數十秒），開回來立即生效。"""
        self._dream_running = False
        self.dream_report: dict[str, Any] | None = None
        self._dream_scheduler: DreamScheduler | None = None
        """自動 dream 的觸發策略（累積 + 閒置 / daydream）。記憶建好才啟動。"""
        self.memory_error: str | None = None
        self.memory_warm_seconds: float | None = None
        self.split_policy = split_policy or SplitPolicy()
        self.system_prompt = system_prompt

        self._store = store
        """SessionControl 的 SessionStore。``None`` 時在 prepare 建立；
        建不起來（repo 沒設定等）只停用會話功能，不擋管線。可注入供測試。"""

        self._store_error: str | None = None
        self._session_id = uuid.uuid4().hex[:8]
        self._session_created = time.time()

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

        self._discard_turn = False
        """捨棄請求打在進行中的 turn 上：voice loop 在 turn 收尾後
        把這一輪從歷史裡拿掉（「這次輸入不算在其中」）。"""

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
                    "real_memory": self.use_real_memory,
                    "memory_error": self.memory_error,
                    "memory_warm_seconds": (
                        round(self.memory_warm_seconds, 1)
                        if self.memory_warm_seconds is not None
                        else None
                    ),
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
                emotion_markers=True,
            )
        else:
            self._think = FakeThinkStage(
                split_policy=self.split_policy, tracer=self.tracer
            )

        if self.use_real_memory:
            await self._build_memory()

        self._runner = PipelineRunner(
            think_stage=self._think,
            speak_stage=self._speak,
            tracer=self.tracer,
        )
        await self._runner.prepare()

    async def _build_memory(self) -> None:
        """建立並暖機記憶層。旁路——失敗記在 memory_error，不擋管線。

        暖機的 dummy retrieve 是關鍵：embedding 模型（bge-small-zh）的
        首載實測 ~50s（含下載）/ 數秒（已快取），必須在啟動階段吸收，
        不能落在使用者的第一輪對話上。
        """
        if not self.use_real_llm:
            # 注入走 LLMThinkStage 的 _pending_memory，fake think 沒有這條路
            self.memory_error = "記憶需要 --real-llm（注入走 LLMThinkStage）"
            return
        try:
            from ..adapters.memory_echo import EchoMemoryAdapter

            memory = EchoMemoryAdapter()
            await memory.prepare()
            t0 = time.perf_counter()
            await memory.retrieve("暖機")
            self.memory_warm_seconds = time.perf_counter() - t0
            memory.set_session(self._session_id)
            # 蒸餾 LLM 沿用主模型（艾斯維爾定調）：SessionControl backend 的
            # 同步 query()，不另建第二套 LLM 存取。只在 dream（背景）用到，
            # 不在對話熱路徑。
            backend = getattr(self._think, "backend", None)
            query = getattr(backend, "query", None)
            if callable(query):
                memory.set_llm_fn(
                    lambda prompt: (query(prompt) or {}).get("text", ""),
                    # 每輪 salience 評估：五維打分不需要推理，壓到最低、限制輸出長度。
                    # 跑在背景 executor，與下一輪串流並行打同一顆 backend（query 無狀態）。
                    assess_llm_fn=lambda prompt: (
                        query(prompt, reasoning_effort="low", max_tokens=200) or {}
                    ).get("text", ""),
                )
            self._think.memory = memory
            self._memory = memory
            await self._refresh_profile()
            self._start_dream_scheduler(memory)
        except Exception as exc:  # noqa: BLE001
            self.memory_error = f"{type(exc).__name__}: {exc}"

    async def _refresh_profile(self) -> None:
        """把使用者 profile 掛到 think stage 的 system prompt。暖機與 dream 後各一次。"""
        if self._memory is None or not hasattr(self._memory, "profile"):
            return
        try:
            text = await self._memory.profile()
        except Exception as exc:  # noqa: BLE001 - 旁路
            logger_warn = f"{type(exc).__name__}: {exc}"
            self._log({"type": "profile", "error": logger_warn})
            return
        if text != getattr(self._think, "profile", ""):
            self._think.profile = text
            self._log({"type": "profile", "chars": len(text), "text": text})

    def _start_dream_scheduler(self, memory: Any) -> None:
        """門檻走 .env：ECHO_STREAM_DREAM_{IDLE_MINUTES,MIN_PENDING,DAYDREAM_PENDING}。"""
        policy = DreamPolicy(
            idle_minutes=float(config.get("ECHO_STREAM_DREAM_IDLE_MINUTES",
                                          str(DreamPolicy.idle_minutes))),
            min_pending=int(config.get("ECHO_STREAM_DREAM_MIN_PENDING",
                                       str(DreamPolicy.min_pending))),
            daydream_pending=int(config.get("ECHO_STREAM_DREAM_DAYDREAM_PENDING",
                                            str(DreamPolicy.daydream_pending))),
        )
        self._dream_scheduler = DreamScheduler(
            pending_count=memory.pending_dream_count,
            run_dream=self._run_dream,
            policy=policy,
        )
        self._dream_scheduler.start()

    def shutdown(self) -> None:
        if self._dream_scheduler is not None:
            self._loop.call_soon_threadsafe(self._dream_scheduler.stop)
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

    # --- 執行中調參 ---

    def get_config(self) -> dict[str, Any]:
        """回報當前可調參數的現值（給設定面板初始化）。"""
        style: SpeechStyle | None = getattr(self._think, "default_style", None)
        stt = self._stt
        stt_policy = getattr(stt, "policy", None)
        runner = self._runner
        backend = getattr(self._think, "_backend", None)
        return {
            "tts": {
                "speed": style.speed if style else 1.0,
                "intensity": style.intensity if style else 0.6,
                "emotion": style.normalized_emotion() if style else
                           {dim: 0.0 for dim in EMOTION_DIMENSIONS},
                "override": style is not None,
            },
            "presets": EMOTION_PRESETS,
            "split": {
                "first_min": self.split_policy.first_min_weight,
                "first_max": self.split_policy.first_max_weight,
                "min": self.split_policy.min_weight,
                "max": self.split_policy.max_weight,
            },
            "turn": {
                "silence_threshold_s": (
                    stt_policy.silence_threshold_s if stt_policy else None
                ),
                "interruption_min_speech_s": (
                    runner.policy.interruption_min_speech_s if runner else None
                ),
                "min_confidence": getattr(stt, "min_confidence", None),
            },
            "llm": {
                "reasoning_effort": getattr(backend, "reasoning_effort", None),
                "emotion_markers": getattr(self._think, "emotion_markers", False),
            },
            "memory": {
                "available": self._memory is not None,
                "enabled": self._memory is not None and self._memory_enabled,
                "error": self.memory_error,
                "top_k": getattr(self._memory, "top_k", None),
                "max_chars": getattr(self._memory, "max_chars", None),
                "min_match": getattr(self._memory, "min_match", None),
            },
        }

    def apply_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """從 HTTP 執行緒套用設定。

        **必須在 loop 執行緒上改**——asyncio 側的物件（policy dataclass、
        stage 屬性）從 HTTP 執行緒直接動，會與正在跑的 turn 產生資料競爭。
        """
        if not self.ready:
            return {"error": "管線尚未就緒"}
        fut = asyncio.run_coroutine_threadsafe(self._apply_config(body), self._loop)
        return fut.result(timeout=10.0)

    async def _apply_config(self, body: dict[str, Any]) -> dict[str, Any]:
        applied: dict[str, Any] = {}

        tts = body.get("tts")
        if isinstance(tts, dict):
            # 面板的 TTS 設定做成「覆寫 default_style」：走與情緒標記相同的
            # per-sentence _apply_style 路徑（executor 執行緒、句子序列合成，
            # 改 backend 狀態安全），而不是另闢一條直接戳 backend 的路。
            emotion = {
                dim: float(v)
                for dim, v in (tts.get("emotion") or {}).items()
                if dim in EMOTION_DIMENSIONS and float(v) > 0.0
            }
            style = SpeechStyle(
                emotion=emotion,
                intensity=float(tts.get("intensity", 0.6)),
                speed=float(tts.get("speed", 1.0)),
            )
            if style.is_neutral and style.speed == 1.0:
                style = None  # 全中性 = 撤銷覆寫，回到角色檔預設
            if hasattr(self._think, "default_style"):
                self._think.default_style = style
                applied["tts"] = {
                    "override": style is not None,
                    "speed": style.speed if style else 1.0,
                    "intensity": style.intensity if style else None,
                    "emotion": dict(style.emotion) if style else None,
                }

        split = body.get("split")
        if isinstance(split, dict):
            mapping = {
                "first_min": "first_min_weight",
                "first_max": "first_max_weight",
                "min": "min_weight",
                "max": "max_weight",
            }
            changed = {}
            for key, attr in mapping.items():
                if key in split:
                    value = float(split[key])
                    setattr(self.split_policy, attr, value)
                    changed[key] = value
            if changed:
                applied["split"] = changed

        turn = body.get("turn")
        if isinstance(turn, dict):
            changed = {}
            # 兩個 TurnPolicy 都要改：turn 切分看 STT stage 的、
            # barge-in 判定器看 runner 的（stage 持有引用，直接 mutate 生效）
            policies = [
                p
                for p in (
                    getattr(self._stt, "policy", None),
                    self._runner.policy if self._runner else None,
                )
                if p is not None
            ]
            for key in ("silence_threshold_s", "interruption_min_speech_s"):
                if key in turn:
                    value = float(turn[key])
                    for policy in policies:
                        setattr(policy, key, value)
                    changed[key] = value
            if "min_confidence" in turn and hasattr(self._stt, "min_confidence"):
                self._stt.min_confidence = float(turn["min_confidence"])
                changed["min_confidence"] = float(turn["min_confidence"])
            if changed:
                applied["turn"] = changed

        memory = body.get("memory")
        if isinstance(memory, dict) and self._memory is not None:
            changed = {}
            if "enabled" in memory:
                # 掛/卸 think stage 的引用（擋 retrieve/注入）+ service 旗標
                # （擋 store）。adapter 與 engine 留著，開回來立即生效。
                enabled = bool(memory["enabled"])
                self._memory_enabled = enabled
                if hasattr(self._think, "memory"):
                    self._think.memory = self._memory if enabled else None
                changed["enabled"] = enabled
            if "top_k" in memory:
                self._memory.top_k = max(1, int(memory["top_k"]))
                changed["top_k"] = self._memory.top_k
            if "max_chars" in memory:
                self._memory.max_chars = max(50, int(memory["max_chars"]))
                changed["max_chars"] = self._memory.max_chars
            if "min_match" in memory:
                self._memory.min_match = min(1.0, max(0.0, float(memory["min_match"])))
                changed["min_match"] = self._memory.min_match
            if changed:
                applied["memory"] = changed

        llm = body.get("llm")
        if isinstance(llm, dict):
            changed = {}
            if "emotion_markers" in llm and hasattr(self._think, "emotion_markers"):
                self._think.emotion_markers = bool(llm["emotion_markers"])
                changed["emotion_markers"] = bool(llm["emotion_markers"])
            if "reasoning_effort" in llm:
                backend = getattr(self._think, "_backend", None)
                if backend is not None and hasattr(backend, "reasoning_effort"):
                    backend.reasoning_effort = str(llm["reasoning_effort"])
                    changed["reasoning_effort"] = str(llm["reasoning_effort"])
            if changed:
                applied["llm"] = changed

        # 寫進 session log——事後分析時才知道「當時的參數是什麼」
        self._log({"type": "config", **applied})
        return {"ok": True, "applied": applied}

    # --- 會話管理 ---
    #
    # 歷史列表活在 asyncio 側（think stage 持有、turn 執行中會寫入），
    # 所以所有會話操作一律經 run_coroutine_threadsafe 到 loop 執行緒，
    # 與 apply_config 同一條規矩。store 的檔案讀寫也統一在 loop 執行緒，
    # SessionStore 不是 thread-safe 的，兩條執行緒同時碰 index 會互咬。

    def _history(self) -> list[dict[str, str]] | None:
        return getattr(self._think, "history", None)

    def _sessions_call(self, coro) -> dict[str, Any]:  # noqa: ANN001
        if not self.ready:
            coro.close()
            return {"error": "管線尚未就緒"}
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=15.0)

    def sessions_list(self) -> dict[str, Any]:
        return self._sessions_call(self._sessions_list())

    def session_new(self) -> dict[str, Any]:
        return self._sessions_call(self._session_new())

    def session_switch(self, session_id: str) -> dict[str, Any]:
        return self._sessions_call(self._session_switch(session_id))

    def session_delete(self, session_id: str) -> dict[str, Any]:
        return self._sessions_call(self._session_delete(session_id))

    async def _sessions_list(self) -> dict[str, Any]:
        if self._store is None:
            return {
                "error": self._store_error or "session store 未就緒",
                "sessions": [],
                "active": self._session_id,
            }
        sessions = []
        for entry in self._store.list_sessions():
            sid = entry["session_id"]
            title = ""
            try:
                # title 存在 session 檔的 metadata，index 沒有這個欄位。
                # 檔案很小、session 數是十位數等級，逐檔讀可接受。
                _, meta = self._store.load(sid)
                title = meta.get("title", "")
            except Exception:  # noqa: BLE001 - 壞檔不擋列表
                pass
            sessions.append(
                {
                    "id": sid,
                    "title": title or sid,
                    "message_count": entry.get("message_count", 0),
                    "saved_at": entry.get("saved_at"),
                }
            )
        sessions.sort(key=lambda s: s.get("saved_at") or 0, reverse=True)
        history = self._history() or []
        return {
            "sessions": sessions,
            "active": self._session_id,
            "active_message_count": len(history),
        }

    async def _session_new(self) -> dict[str, Any]:
        history = self._history()
        if history is None:
            return {"error": "此模式不支援會話歷史"}
        if self._runner is not None and self._runner.is_speaking:
            return {"error": "有進行中的 turn——先等它結束或插話"}
        self._save_active_session()
        history.clear()
        self._session_id = uuid.uuid4().hex[:8]
        self._session_created = time.time()
        self._sync_memory_session()
        self._log({"type": "session", "action": "new", "session": self._session_id})
        return {"ok": True, "active": self._session_id, "history": []}

    def _sync_memory_session(self) -> None:
        """會話邊界同步給記憶層——exclude_session_id（防自我回聲）與
        episode 標記都要跟著當前會話走。記憶本身跨會話，namespace 不變。"""
        if self._memory is not None:
            self._memory.set_session(self._session_id)

    async def _session_switch(self, session_id: str) -> dict[str, Any]:
        history = self._history()
        if history is None:
            return {"error": "此模式不支援會話歷史"}
        if self._store is None:
            return {"error": self._store_error or "session store 未就緒"}
        if self._runner is not None and self._runner.is_speaking:
            return {"error": "有進行中的 turn——先等它結束或插話"}
        if session_id == self._session_id:
            return {"ok": True, "active": session_id, "history": list(history)}

        self._save_active_session()
        try:
            messages, meta = self._store.load(session_id)
        except FileNotFoundError:
            return {"error": f"找不到會話 {session_id}"}

        # 只還原對話角色——之後壓縮進來可能混入 summary 等角色，
        # 那些是 ContextManager 的內部狀態，不該直接餵回歷史
        history[:] = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        self._session_id = session_id
        self._session_created = meta.get("created_at", time.time())
        self._sync_memory_session()
        self._log(
            {
                "type": "session",
                "action": "switch",
                "session": session_id,
                "messages": len(history),
            }
        )
        return {"ok": True, "active": session_id, "history": list(history)}

    async def _session_delete(self, session_id: str) -> dict[str, Any]:
        if self._store is None:
            return {"error": self._store_error or "session store 未就緒"}
        if session_id == self._session_id:
            history = self._history()
            if history is not None:
                if self._runner is not None and self._runner.is_speaking:
                    return {"error": "有進行中的 turn——先等它結束或插話"}
                history.clear()
            self._session_id = uuid.uuid4().hex[:8]
            self._session_created = time.time()
            self._sync_memory_session()
        deleted = bool(self._store.delete(session_id))
        self._log({"type": "session", "action": "delete", "session": session_id})
        return {"ok": deleted, "active": self._session_id}

    def _save_active_session(self) -> None:
        """把當前歷史寫進 store。**只在 loop 執行緒呼叫。**

        存檔失敗不擋對話——會話持久化是旁路，管線才是主體。
        """
        history = self._history()
        if not history or self._store is None:
            return
        title = next(
            (m["content"] for m in history if m.get("role") == "user"), ""
        )[:24]
        try:
            self._store.save(
                self._session_id,
                [dict(m) for m in history],
                {
                    "title": title,
                    "created_at": self._session_created,
                    "backend": "echo_stream_web",
                },
            )
        except Exception:  # noqa: BLE001
            pass

    # --- Dream（離線鞏固）---

    def memory_dream(self) -> dict[str, Any]:
        """手動觸發 Dream Engine。背景執行，不擋對話——立即回傳。

        排程不做（本輪範圍）。競態鐵律在 echo_memory 本體：
        保留原始 → 背景算 → 原子替換，絕不「先清原始再等 LLM」。
        """
        if self._memory is None:
            return {"error": self.memory_error or "記憶未啟用"}
        if self._dream_running:
            return {"error": "dream 進行中"}
        asyncio.run_coroutine_threadsafe(self._run_dream("manual"), self._loop)
        return {"ok": True, "started": True}

    async def _run_dream(self, triggered_by: str) -> None:
        """手動與排程共用的執行路徑。同時只跑一個；重入直接略過。

        中斷語意是「重做」不是「復原」——echo_memory 只在整輪結束才標
        is_dreamed，中途掛掉下次同一批從頭跑，不需要 checkpoint。
        """
        if self._memory is None or self._dream_running:
            return
        self._dream_running = True
        try:
            raw = await self._memory.dream(triggered_by)
            # DreamReport 可能含非 JSON 型別——先過一次 default=str
            report = json.loads(json.dumps(raw, ensure_ascii=False, default=str))
        except Exception as exc:  # noqa: BLE001 - 旁路
            report = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            # 旗標先放、報告後公布——讀到報告的人不會再看到 running=True
            self._dream_running = False
        self.dream_report = report
        self._log({"type": "dream", "triggered_by": triggered_by, "report": report})
        # 蒸餾可能長出新的 subject=user concept——刷新常駐 profile
        await self._refresh_profile()

    def memory_dream_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "available": self._memory is not None,
            "running": self._dream_running,
            "last_report": self.dream_report,
        }
        if self._dream_scheduler is not None:
            status["scheduler"] = self._dream_scheduler.status()
        status["profile"] = getattr(self._think, "profile", "")
        return status

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

    def _discard_input_on_loop(self) -> None:
        """丟棄進行中的輸入與 turn。**只在 loop 執行緒呼叫。**"""
        stage = self._stt
        if stage is not None and hasattr(stage, "discard_current"):
            stage.discard_current()
        runner = self._runner
        if runner is not None and runner.is_speaking:
            # turn 已經在跑（LLM / TTS）：中止它，並讓 voice loop
            # 把這一輪從歷史裡拿掉——捨棄不是插話，內容不該留下
            self._discard_turn = True
            runner.interrupt(CancelReason.BARGE_IN)

    def voice_discard(self, sid: str) -> bool:
        """丟棄這次輸入：別人講話、誤觸發——這一輪不算在對話裡。"""
        session = self._voice
        if session is None or session["sid"] != sid:
            return False
        done = threading.Event()
        self._loop.call_soon_threadsafe(
            lambda: (self._discard_input_on_loop(), done.set())
        )
        done.wait(timeout=2.0)
        with contextlib.suppress(queue.Full):
            session["out"].put_nowait(
                json_frame(FRAME_EVENT, {"event": "input_discarded"})
            )
        self._log({"type": "discard"})
        return True

    def voice_stop(self, sid: str) -> bool:
        """結束擷取。**停止 = 乾淨收場**：丟棄未完成的輸入、中止進行中的
        turn——不能有殘留的 STT flush 在停止之後又觸發一輪 LLM+TTS
        （2026-08-21 艾斯維爾實測回饋）。"""
        session = self._voice
        if session is None or session["sid"] != sid:
            return False
        done = threading.Event()
        self._loop.call_soon_threadsafe(
            lambda: (self._discard_input_on_loop(), done.set())
        )
        done.wait(timeout=2.0)
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

        # 增量轉錄：講話停頓處就把已轉出的段推給前端逐句條列，
        # 不必等整個 turn 講完才看到自己說了什麼
        partial_texts: list[str] = []

        def on_partial(text: str, index: int) -> None:
            partial_texts.append(text)
            emit(json_frame(FRAME_UTTER, {"partial": True, "index": index, "text": text}))

        stage.on_partial = on_partial
        try:
            async for utterance in stage.stream(session["source"], token):
                stt_ms = (time.perf_counter() - utterance.ended_at) * 1000
                utter_info = {
                    "text": utterance.text,
                    "parts": list(partial_texts) or [utterance.text],
                    "stt_ms": round(stt_ms),
                    "confidence": round(utterance.confidence, 2),
                    "language": utterance.language,
                }
                partial_texts.clear()
                emit(json_frame(FRAME_UTTER, utter_info))
                self._log({"type": "utterance", **utter_info})
                # 傳真的 Utterance 而不是重建——ended_at 是真實的說完時刻，
                # tracer 的 TTFA 才會把 STT 的耗時算進去
                history = self._history()
                pre_len = len(history) if history is not None else 0
                self._discard_turn = False
                await self._run(utterance, emit, origin="voice")

                if self._discard_turn:
                    # 捨棄打在這一輪上：把 commit 進歷史的內容拿掉——
                    # 「這次輸入不算在其中」，不是插話那種「講到一半算數」
                    self._discard_turn = False
                    if history is not None and len(history) > pre_len:
                        del history[pre_len:]
                    self._save_active_session()
                    self._log({"type": "turn_discarded"})

                # turn 結束補一個 padding frame：tunnel 代理（cloudflared）會
                # 緩衝小尾巴，最後一段音訊可能卡在代理層直到下一輪資料把它
                # 擠出去——「最後一段等下次輸入才播」就是這個。前端會忽略它。
                emit(json_frame(FRAME_EVENT, {"event": "flush", "pad": "." * 16384}))
        except Exception as exc:  # noqa: BLE001
            emit(json_frame(FRAME_ERROR, {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            stage.on_voice_event = None
            stage.on_partial = None
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

    async def _run(
        self,
        utterance: Utterance,
        emit: Callable[[bytes], None],
        origin: str = "text",
    ) -> None:
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
            "memory_injected": getattr(self._think, "last_injected_memory", None),
            "report": self.tracer.report(turn_id),
        }
        emit(json_frame(FRAME_META, meta))
        self._log({"type": "turn", "input": utterance.text, **meta})
        # 每輪落地一次——server 掛掉最多丟正在跑的那一輪，不會丟整個會話
        self._save_active_session()

        if self._dream_scheduler is not None:
            self._dream_scheduler.notify_interaction()

        # 記憶寫入：背景旁路，不擋下一輪。
        # **被捨棄的 turn 不寫**——voice_discard 在中止前就把 _discard_turn
        # 立起來了，這裡看得到；歷史剔除了、記憶卻留著 = 髒資料。
        # 沒播出任何內容的 turn 也不寫（錯誤或開口前就被砍，不成一輪對話）。
        # **寫 spoken 不寫 generated**——與歷史同一條語意。
        if (
            self._memory is not None
            and self._memory_enabled
            and not self._discard_turn
            and result.spoken_text
        ):
            asyncio.ensure_future(
                self._store_memory(
                    utterance.text,
                    result.spoken_text,
                    origin=origin,
                    truncated=bool(result.cancel_reason),
                )
            )

    async def _store_memory(self, user_text: str, spoken_text: str, **meta: Any) -> None:
        """背景寫入 + 把 salience 結果記進 session log（驗收看這個）。"""
        stored = await self._memory.store(user_text, spoken_text, **meta)
        if stored:
            self._log({"type": "memory_store", "input": user_text[:80], **stored})


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
        elif self.path == "/api/config":
            self._json(200, self.service.get_config())
        elif self.path == "/api/memory/dream":
            self._json(200, self.service.memory_dream_status())
        elif self.path == "/api/sessions":
            self._json(200, self.service.sessions_list())
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
        if self.path == "/api/memory/dream":
            result = self.service.memory_dream()
            self._json(200 if "error" not in result else 409, result)
            return
        if self.path == "/api/config":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return
            try:
                self._json(200, self.service.apply_config(body))
            except Exception as exc:  # noqa: BLE001 - 面板設錯值不該讓 server 掛掉
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
            return
        if self.path.startswith("/api/sessions/"):
            action = self.path.removeprefix("/api/sessions/").split("?")[0]
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return
            sid = str(body.get("id") or "")
            try:
                if action == "new":
                    result = self.service.session_new()
                elif action == "switch":
                    result = self.service.session_switch(sid)
                elif action == "delete":
                    result = self.service.session_delete(sid)
                else:
                    self._json(404, {"error": "not found"})
                    return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200 if "error" not in result else 409, result)
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
        if self.path.startswith("/api/voice/discard"):
            sid = self._query_param("sid")
            ok = self.service.voice_discard(sid)
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
    """不設 SO_REUSEADDR、且不把客戶端斷線當錯誤的 HTTP server。

    Python 預設 ``allow_reuse_address=1``，在 Windows 上這代表**兩個行程
    可以同時 bind 同一個 port**——舊 server 沒關乾淨時，新 server 啟動
    「成功」但流量進舊行程，人設、程式碼更新全都看似失效
    （2026-08-21 踩過，查了很久）。關掉它讓重複啟動直接炸 bind error。
    """

    allow_reuse_address = False

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        """客戶端斷線（WinError 10054 等）不印 traceback。

        語音串流的長連線被前端 ``abort()``、使用者關頁面、tunnel 探測——
        這些都以「強制關閉連線」收場，是串流服務的日常而非錯誤。
        預設實作會把完整 traceback 刷到主控台，嚇人且淹掉真正的錯誤。
        其他例外照常報。
        """
        exc = sys.exc_info()[1]
        if isinstance(
            exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)
        ):
            return
        super().handle_error(request, client_address)


def serve(
    host: str | None = None,
    port: int | None = None,
    *,
    real_tts: bool = False,
    real_llm: bool = False,
    real_stt: bool = False,
    real_memory: bool = False,
    split_policy: SplitPolicy | None = None,
    system_prompt: str = "",
    trace_path: str | None = None,
) -> None:
    host = host or config.get("ECHO_STREAM_WEB_HOST", "127.0.0.1")
    port = port or int(config.get("ECHO_STREAM_WEB_PORT", "8770"))

    # store 在這裡建而不是 StreamService.prepare——單元測試直接建 service
    # 時不該在 repo 根目錄長出 data/sessions/。建不起來只停用會話功能。
    store = None
    store_error: str | None = None
    try:
        from ..adapters.session_store import build_store

        store = build_store()
    except Exception as exc:  # noqa: BLE001
        store_error = f"{type(exc).__name__}: {exc}"

    service = StreamService(
        use_real_tts=real_tts,
        use_real_llm=real_llm,
        use_real_stt=real_stt,
        use_real_memory=real_memory,
        split_policy=split_policy,
        system_prompt=system_prompt,
        trace_path=trace_path,
        store=store,
    )
    if store_error:
        service._store_error = store_error

    print("═" * 56)
    print("  Echo Stream — Web 前端")
    print("═" * 56)
    print(f"  TTS: {'真 IndexTTS2' if real_tts else 'fake'}")
    print(f"  LLM: {'真 OpenAI' if real_llm else 'fake'}")
    print(f"  STT: {'真 Whisper' if real_stt else 'fake（只驗串流接收）'}")
    print(f"  Log: {service.log_path}")
    if store_error:
        print(f"  ⚠ 會話持久化停用：{store_error}")
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
        raise SystemExit(1) from None
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
