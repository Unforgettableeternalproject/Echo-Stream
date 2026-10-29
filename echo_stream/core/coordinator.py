"""TurnCoordinator——多輪輸出端的排隊、取消與結算（T-07b PR ②）。

把 bot 的「多輪排隊、讓位、regen、PASS、barge-in」執行機制收進 ES。
**機制在這裡，決策留宿主**：讓位、passive、PASS、regen 的判定都是策略插槽
（見 :mod:`echo_stream.contracts.output`），ES 只提供執行與同步呼叫點。

## 一輪的生命週期

``submit()`` 立即返回 :class:`TurnHandle`，背景 task 依序做：

1. 等 think 鎖（FIFO；等鎖期間被作廢 → ``VOIDED``，不產生任何生成／播放）
2. 生成：think → speak，音訊塊在「前一輪還沒播完」時先緩衝（合成不等前一輪），
   輪到自己才寫 sink；前一輪已結束就直接寫（**熱路徑不經緩衝、不多一次 await**）
3. 生成完成：EAGER 在釋放鎖**之前** commit；SETTLED 鎖一路持有到結算（串行語意）
4. drain（可被取消、有逾時）→ 結算 spoken → commit／settle → 恰好一個 outcome

## 取消（見草案 §7）

任何取消都先「認領」（per turn 只成功一次），再同步 ``sink.stop()``，最後取消 token：

* barge-in：``BARGE_IN``；排在後面還沒開播的輪連帶取消（detail ``cascade``）
* 讓位 ``void()``：``SUPERSEDED`` + detail ``yielded``；已開播的輪不處理（回 False）
* 同 owner 新回覆：``SUPERSEDED`` + detail ``superseded_by_owner``
* ``cancel_all()``：``SHUTDOWN``，不 commit、不 settle

## 執行緒

``on_voice_event`` / ``void`` / ``cancel_all`` 可從任何執行緒呼叫：在呼叫執行緒同步
``sink.stop()``，token 取消經 ``call_soon_threadsafe`` 回 loop。``submit`` 與策略
``refresh`` / ``SpeechTextFilter`` 只在 loop 內。

## 不做的事

不接 GapFiller（墊片音屬 ``PipelineRunner`` 單輪路徑）、不碰 ``run_turn``；
記憶寫入不在這裡（宿主依 :class:`~echo_stream.contracts.output.TurnOutcome` 自行決定）。
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence

from ..contracts.cancellation import CancellationToken, CancelledError, CancelReason
from ..contracts.output import (
    AdmissionPolicy,
    BargeInPolicy,
    CommitMode,
    ModeFlags,
    OutcomeStatus,
    PlayedEvidence,
    PolicySnapshot,
    PolicySource,
    SkipReason,
    SpokenRule,
    TurnOutcome,
    TurnRequest,
    TurnView,
    YieldPolicy,
    resolve_spoken_rule,
    sink_first_frame_at,
    sink_is_active,
    sink_played_evidence,
)
from ..contracts.stages import AudioSink, SpeakStage, ThinkStage
from ..contracts.turn import VoiceEvent
from ..contracts.types import AudioChunk, Sentence, TurnPhase
from . import tracer as marks
from .policies import (
    AcceptAllAdmission,
    AccumulatingBargeIn,
    NeverYield,
    build_sentence_spans,
    partial_spoken,
    whole_sentence_spoken,
)
from .standalone import StaticPolicySource
from .tracer import LatencyTracer

_log = logging.getLogger(__name__)

PLAY_TIMEOUT_S = 180.0
"""等一輪播完的上限，避免 TTS／播放卡死時鎖永遠不放（對應 bot 的 VOICE_WAIT_TIMEOUT_S）。"""

IDLE_TIMEOUT_S = 0.5
"""``sink.wait_idle()`` 的逾時：逾時放行並記 log（對應 bot 的 wait_vc_idle，Q7）。"""

_END = object()
_CANCELLED = object()
"""句子佇列的內部標記：結束、取消。"""

REASON_CASCADE = "cascade"
REASON_OWNER = "superseded_by_owner"
REASON_YIELDED = "yielded"


def _stop_sink(sink: AudioSink | None) -> None:
    if sink is None:
        return
    with contextlib.suppress(Exception):
        sink.stop()


class _Attempt:
    """一趟生成（regen 時同一輪會有多趟）。每趟有自己的 sink、token 與記帳。"""

    __slots__ = (
        "utterance", "token", "sink", "attempt_id", "direct",
        "generated", "voiced", "sent_audio", "timeline", "emitted",
        "buffer", "first_write_at", "wrote_any", "idle_done",
        "passed", "regen_utt", "flusher", "text_q", "text_worker",
        "pump", "pump_error",
    )

    def __init__(self, utterance, token, sink, attempt_id: int, direct: bool) -> None:
        self.utterance = utterance
        self.token: CancellationToken = token
        self.sink: AudioSink | None = sink
        self.attempt_id = attempt_id
        self.direct = direct
        """True = 前一輪已結束，音訊塊直接寫 sink（不經緩衝）。"""

        self.generated: dict[int, Sentence] = {}
        self.voiced: dict[int, str] = {}
        """句序號 → 送進 TTS 的文字（過濾後）。"""

        self.sent_audio: dict[int, float] = {}
        """句序號 → 已交給 sink 的音訊秒數（寫入前累加，同 bot）。"""

        self.timeline: list[tuple[int, float]] = []
        self.emitted = 0.0
        self.buffer: deque[AudioChunk] = deque()
        self.first_write_at: float | None = None
        self.wrote_any = False
        self.idle_done = False
        self.passed = False
        self.regen_utt = None
        self.flusher: asyncio.Future | None = None
        self.text_q: asyncio.Queue | None = None
        self.text_worker: asyncio.Future | None = None
        self.pump: asyncio.Future | None = None
        self.pump_error: BaseException | None = None


class TurnHandle:
    """``submit`` 的回傳值，同時是策略看到的 :class:`~echo_stream.contracts.output.TurnView`。"""

    def __init__(
        self,
        request: TurnRequest,
        flags: ModeFlags,
        loop: asyncio.AbstractEventLoop,
        snapshot_version: int,
        stale: bool,
        coordinator: TurnCoordinator,
    ) -> None:
        ticket = request.ticket
        self.request = request
        self.ticket = ticket
        self.turn_id = ticket.turn_id
        self.owner_id = ticket.owner_id
        self.reply_mode = ticket.reply_mode
        self.meta = request.meta
        """宿主私有旗標（與 ``TurnRequest.meta`` 同一個 dict）。"""

        self.flags = flags
        self.generation_done: asyncio.Future[None] = loop.create_future()
        """文字生成結束（完成、取消或失敗都會 resolve，不拋例外）。"""

        self.settled: asyncio.Future[TurnOutcome] = loop.create_future()
        """這一輪的唯一 outcome。"""

        self.regenerated = 0
        self._coordinator = coordinator
        self._base_barge = {u for u in request.barge_users if u}
        self._snapshot_version = snapshot_version
        self._stale = stale
        self._root = CancellationToken()
        self._att: _Attempt | None = None
        self._first_sink: AudioSink | None = None
        self._skip_reason: str | None = None
        self._predecessor: TurnHandle | None = None
        self._task: asyncio.Future | None = None
        self._lock_state = "none"
        self._play_done = asyncio.Event()
        self._done = False
        self._attempt_no = 1
        self._eager_committed = False
        self._queue_wait_s = 0.0
        self._cancel_lock = threading.Lock()
        self._cancel_claimed = False
        self._cancel_reason: CancelReason | None = None
        self._cancel_detail = ""
        self._chaining = False

    # --- TurnView ---

    @property
    def audio_started(self) -> bool:
        att = self._att
        if att is None or att.sink is None:
            return False
        if hasattr(att.sink, "first_frame_at"):
            return sink_first_frame_at(att.sink) is not None
        return att.first_write_at is not None

    @property
    def behind_playing(self) -> bool:
        pred = self._predecessor
        return pred is not None and not pred._play_done.is_set() and pred.audio_started

    @property
    def sink_active(self) -> bool:
        att = self._att
        if att is None or att.sink is None:
            return False
        fallback = att.first_write_at is not None and not att.token.is_cancelled
        return sink_is_active(att.sink, fallback=fallback)

    @property
    def barge_users(self) -> set[str]:
        users = set(self._base_barge)
        if not self._stale:
            users |= {u for u in getattr(self.request.text_filter, "addressee_ids", ()) if u}
        return users

    # --- 宿主操作 ---

    @property
    def attempt_id(self) -> int:
        """目前是第幾趟生成（regen 時 +1，turn_id 不變）。"""
        return self._attempt_no

    @property
    def token(self) -> CancellationToken:
        """目前這趟生成的 token（regen 時每趟新建）。"""
        return self._att.token if self._att is not None else self._root

    def add_barge_user(self, user_id: str) -> None:
        """生成中追加可打斷這輪的說話者（對應 bot 的 addressee_ids）。"""
        if user_id:
            self._base_barge.add(user_id)

    async def wait(self) -> TurnOutcome:
        return await self.settled


class TurnCoordinator:
    """多輪輸出協調器。見模組 docstring。

    ``barge_in`` 預設 :class:`~echo_stream.core.policies.AccumulatingBargeIn`
    （門檻取自 ``policy_source`` 的快照）；``yield_policy`` 預設 ``NeverYield``。
    """

    def __init__(
        self,
        think_stage: ThinkStage,
        speak_stage: SpeakStage,
        *,
        tracer: LatencyTracer | None = None,
        policy_source: PolicySource | None = None,
        barge_in: BargeInPolicy | None = None,
        yield_policy: YieldPolicy | None = None,
        admission: AdmissionPolicy | None = None,
        on_turn_ended: Callable[[TurnOutcome], None] | None = None,
        supersede_same_owner: bool = True,
        cascade_cancel_queued: bool = True,
        play_timeout_s: float = PLAY_TIMEOUT_S,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.think = think_stage
        self.speak = speak_stage
        self.tracer = tracer or LatencyTracer()
        self.policy_source: PolicySource = policy_source or StaticPolicySource()
        self._barge: BargeInPolicy = barge_in or AccumulatingBargeIn(self.policy_source)
        self._yield: YieldPolicy = yield_policy or NeverYield()
        self._admission: AdmissionPolicy = admission or AcceptAllAdmission()
        self._on_turn_ended = on_turn_ended
        self._supersede_same_owner = supersede_same_owner
        self._cascade = cascade_cancel_queued
        self._play_timeout_s = play_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._clock = clock

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._live: list[TurnHandle] = []
        self._think_lock = asyncio.Lock()
        self._lock_users = 0
        """持有或等待 think 鎖的輪數（admission 的 busy／waiting 依此）。"""

    # ───────────────────────── 對外介面 ─────────────────────────

    def submit(self, req: TurnRequest) -> TurnHandle:
        """送出一輪。必須在 loop 內呼叫；立即返回，不等生成。

        ticket 與 utterance 不對應、或 turn_id 與還在處理的輪重複，是呼叫端的程式錯誤，
        拋 ``ValueError``（不算 submit，也沒有 outcome）。其餘情況每個 submit 恰好一個 outcome。
        """
        if not req.ticket.matches(req.utterance):
            raise ValueError("ticket 與 utterance 不對應（source_turn_id != utterance.turn_id）")
        if any(t.turn_id == req.ticket.turn_id and not t._done for t in self._live):
            raise ValueError(f"turn_id 重複: {req.ticket.turn_id}")
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._loop_thread = threading.get_ident()

        snap = self.policy_source.current()
        stale = self._is_stale(snap)
        flags = self._flags_for(snap, req.ticket.reply_mode)
        t = TurnHandle(req, flags, loop, snap.version, stale, self)

        busy = self._lock_users > 0
        waiting = max(0, self._lock_users - 1)
        try:
            admitted = self._admission.admit(req, busy, waiting)
        except Exception:  # noqa: BLE001 - 閘門壞掉不該擋住回覆
            _log.warning("admission 失敗，視為接受", exc_info=True)
            admitted = True
        if not admitted:
            self._deliver_dropped(t)
            return t

        tid = t.turn_id
        self.tracer.start(tid)
        self.tracer.mark(
            tid, marks.MARK_SPEECH_END,
            at=req.speech_end_at if req.speech_end_at is not None else req.utterance.ended_at,
        )
        self.tracer.mark(tid, marks.MARK_INPUT_FINAL, at=req.stt_done_at)
        for name, at in req.marks.items():
            self.tracer.mark(tid, name, at=at)

        self._supersede_owner(t)
        queue = [
            o for o in self._live
            if not o._play_done.is_set() and o._cancel_reason is not CancelReason.SUPERSEDED
        ]
        t._predecessor = queue[-1] if queue else None

        if not flags.speak:
            t._skip_reason = SkipReason.MODE_SILENT
        else:
            t._first_sink, t._skip_reason = self._make_sink(req, t)

        t._lock_state = "waiting"
        self._lock_users += 1
        self._live.append(t)
        t._task = asyncio.ensure_future(self._run(t))
        return t

    def on_voice_event(self, ev: VoiceEvent) -> None:
        """使用者聲音事件（任何執行緒）。依序餵 barge-in 與讓位策略。

        命中 barge-in 時，**呼叫執行緒**內同步 ``sink.stop()``，token 取消經
        ``call_soon_threadsafe`` 回 loop，且每輪只取消一次。
        """
        loop = self._loop
        if loop is None:
            return
        live = [t for t in list(self._live) if not t._done and not t._play_done.is_set()]
        if not live:
            return
        for t in live:
            if t._cancel_claimed or not t.flags.barge_in:
                continue
            try:
                decision = self._barge.on_voiced(t, ev)
            except Exception:  # noqa: BLE001 - 策略壞掉不能斷掉音訊執行緒
                _log.debug("barge-in 策略失敗", exc_info=True)
                continue
            if decision.should_interrupt:
                self._barge_in(t)

        if self._is_stale(self.policy_source.current()):
            return  # 快照過期：讓位停用（退保守），barge-in 停播是安全方向所以照常
        try:
            actions = self._yield.on_voiced(ev, live)
        except Exception:  # noqa: BLE001
            _log.debug("讓位策略失敗", exc_info=True)
            return
        for action in actions:
            self.void(action.turn_id, action.reason)

    def void(self, turn_id: str, reason: str = REASON_YIELDED) -> bool:
        """作廢一輪（讓位）：SUPERSEDED，detail 為 ``reason``。任何執行緒皆可。

        已開播的輪不處理（回 False，交給 barge-in）；已被認領取消的輪也回 False。
        """
        for t in list(self._live):
            if t.turn_id != turn_id:
                continue
            if t._done or t._play_done.is_set() or t.audio_started:
                return False
            if not self._claim(t, CancelReason.SUPERSEDED, reason):
                return False
            self._dispatch(t)
            return True
        return False

    def cancel_all(self, reason: str = "shutdown") -> None:
        """取消所有進行中與排隊中的輪（SHUTDOWN）：停播、不 commit、不 settle。"""
        for t in list(self._live):
            if t._done or t._play_done.is_set():
                continue
            if self._claim(t, CancelReason.SHUTDOWN, reason):
                self._dispatch(t)

    def live_turns(self) -> Sequence[TurnView]:
        """還沒結束的輪（依排隊順序）。"""
        return [t for t in list(self._live) if not t._done]

    async def join(self) -> None:
        """等目前所有輪都處理完（測試與關閉用）。"""
        while True:
            tasks = [t._task for t in list(self._live) if t._task is not None]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        self.cancel_all("shutdown")
        await self.join()

    # ───────────────────────── 取消機制 ─────────────────────────

    def _on_loop_thread(self) -> bool:
        return threading.get_ident() == self._loop_thread

    def _claim(self, t: TurnHandle, reason: CancelReason, detail: str) -> bool:
        """認領取消權：每輪只會成功一次（跨執行緒安全）。"""
        with t._cancel_lock:
            if t._cancel_claimed:
                return False
            t._cancel_claimed = True
            t._cancel_reason = reason
            t._cancel_detail = detail
        return True

    def _dispatch(self, t: TurnHandle, *, cascade: bool = False) -> None:
        """已認領的取消：同步停 sink，token 取消在 loop 內做。"""
        if self._on_loop_thread():
            self._cancel_in_loop(t, cascade)
            return
        att = t._att
        _stop_sink(att.sink if att is not None else t._first_sink)
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._cancel_in_loop, t, cascade)

    def _cancel_in_loop(self, t: TurnHandle, cascade: bool) -> None:
        att = t._att
        _stop_sink(att.sink if att is not None else t._first_sink)
        t._root.cancel(t._cancel_reason or CancelReason.SHUTDOWN, t._cancel_detail)
        if cascade and self._cascade:
            self._cancel_queued_after(t)

    def _barge_in(self, t: TurnHandle) -> None:
        if not self._claim(t, CancelReason.BARGE_IN, "voice"):
            return
        self._dispatch(t, cascade=True)

    def _cancel_queued_after(self, interrupted: TurnHandle) -> None:
        """正在播的輪被插話：排在後面、還沒開播的輪一併取消（在 loop 內執行）。"""
        if interrupted not in self._live:
            return
        later = self._live[self._live.index(interrupted) + 1:]
        for o in later:
            if o._done or o._play_done.is_set() or o.audio_started:
                continue
            if self._claim(o, CancelReason.BARGE_IN, REASON_CASCADE):
                self._cancel_in_loop(o, False)

    def _supersede_owner(self, new: TurnHandle) -> None:
        if not (self._supersede_same_owner and new.flags.supersede_same_owner):
            return
        if new.owner_id is None:
            return
        for old in list(self._live):
            if (
                old._done or old._play_done.is_set()
                or old.owner_id != new.owner_id
                or old.reply_mode != new.reply_mode
                or not old.flags.supersede_same_owner
            ):
                continue
            if self._claim(old, CancelReason.SUPERSEDED, REASON_OWNER):
                self._dispatch(old)

    # ───────────────────────── 快照／旗標 ─────────────────────────

    @staticmethod
    def _is_stale(snap: PolicySnapshot) -> bool:
        return snap.is_stale()

    @staticmethod
    def _flags_for(snap: PolicySnapshot, reply_mode: str | None) -> ModeFlags:
        try:
            return snap.flags_for(reply_mode)
        except Exception:  # noqa: BLE001
            _log.warning("ModeRules 失敗，退回預設旗標", exc_info=True)
            return ModeFlags()

    def _make_sink(self, req: TurnRequest, t: TurnHandle) -> tuple[AudioSink | None, str | None]:
        try:
            result = req.sink_factory(t)
        except Exception as exc:  # noqa: BLE001
            _log.warning("建立 sink 失敗: %s", exc)
            return None, f"sink_error: {exc}"
        if isinstance(result, str):
            return None, result
        return result, None

    # ───────────────────────── 每輪主流程 ─────────────────────────

    async def _run(self, t: TurnHandle) -> None:
        try:
            if not await self._acquire_think(t):
                self._conclude_voided(t)
                self._mark_play_done(t)
            else:
                await self._produce(t)
        except asyncio.CancelledError:
            self._fallback_outcome(t, "task_cancelled", cancelled=True)
            raise
        except Exception as exc:  # noqa: BLE001 - 單輪失敗不該拖垮其他輪
            _log.exception("TurnCoordinator 內部錯誤")
            self._fallback_outcome(t, f"{type(exc).__name__}: {exc}")
        finally:
            self._lock_done(t)
            self._resolve_gen_done(t)
            self._mark_play_done(t)
            try:
                await t._play_done.wait()
            finally:
                if t in self._live:
                    self._live.remove(t)
                forget = getattr(self._barge, "forget", None)
                if forget is not None:
                    with contextlib.suppress(Exception):
                        forget(t.turn_id)

    async def _acquire_think(self, t: TurnHandle) -> bool:
        """取得 think 鎖。等待期間被取消回 False（未開始生成）。"""
        lock = self._think_lock
        if self._lock_users == 1 and not t._root.is_cancelled:
            await lock.acquire()  # 只有自己時不會讓出 event loop
            t._lock_state = "held"
            return True
        acq = asyncio.ensure_future(lock.acquire())
        cancel = asyncio.ensure_future(t._root.wait())
        try:
            await asyncio.wait({acq, cancel}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancel.cancel()
            if not acq.done():
                acq.cancel()
            await asyncio.gather(acq, cancel, return_exceptions=True)
        got = acq.done() and not acq.cancelled() and acq.exception() is None
        if got:
            t._lock_state = "held"
            if t._root.is_cancelled:
                self._lock_done(t)
                return False
            return True
        return False

    def _lock_done(self, t: TurnHandle) -> None:
        """離開 think 鎖系統（釋放或放棄等待），冪等。"""
        state = t._lock_state
        if state == "held":
            self._think_lock.release()
        if state in ("held", "waiting"):
            self._lock_users -= 1
        t._lock_state = "released"

    def _new_attempt(self, t: TurnHandle, utterance) -> _Attempt:
        if t._att is None:
            sink, skip = t._first_sink, t._skip_reason
        else:
            # regen：舊 sink 已 stop（stop 是永久的），每趟重新建
            sink, skip = (
                self._make_sink(t.request, t) if t.flags.speak else (None, SkipReason.MODE_SILENT)
            )
            t._skip_reason = skip
        pred = t._predecessor
        direct = pred is None or pred._play_done.is_set()
        return _Attempt(utterance, t._root.child(), sink, t._attempt_no, direct)

    async def _produce(self, t: TurnHandle) -> None:
        req = t.request
        tid = t.turn_id
        self.tracer.mark(tid, marks.MARK_QUEUE_ACQUIRED)
        trace = self.tracer.get(tid)
        if trace is not None:
            trace.phase = TurnPhase.THINKING

        phase = TurnPhase.DONE
        error: str | None = None
        utt = req.utterance
        att: _Attempt | None = None
        try:
            while True:
                att = self._new_attempt(t, utt)
                t._att = att
                if await self._run_attempt(t, att) != "regen":
                    break
                _stop_sink(att.sink)
                t.regenerated += 1
                t._attempt_no += 1
                utt = att.regen_utt
        except CancelledError as exc:
            phase = TurnPhase.CANCELLED
            if self._claim(t, exc.reason, exc.detail):
                self._cancel_in_loop(t, False)
        except Exception as exc:  # noqa: BLE001 - 單一 turn 失敗不該拖垮 session
            phase = TurnPhase.FAILED
            error = f"{type(exc).__name__}: {exc}"
            if self._claim(t, CancelReason.UPSTREAM_ERROR, error):
                self._cancel_in_loop(t, False)

        assert att is not None
        if (
            phase is TurnPhase.CANCELLED
            and req.commit_mode is CommitMode.EAGER
            and not t._eager_committed
            and t._cancel_reason is CancelReason.BARGE_IN
        ):
            # 生成途中被插話：文字只生成到一半，已送出的部分照 EAGER 語意先寫進歷史
            generated = self._generated_text(att)
            t._eager_committed = True
            with contextlib.suppress(Exception):
                await self.think.commit(tid, generated, generated)

        if req.commit_mode is CommitMode.EAGER:
            self._lock_done(t)  # EAGER：commit 之後下一輪就能開始生成
        self._resolve_gen_done(t)
        self._mark_play_done(t)
        await self._conclude(t, att, phase, error)

    async def _run_attempt(self, t: TurnHandle, att: _Attempt) -> str:
        """跑一趟生成與播放。回傳 ``"done"`` 或 ``"regen"``。

        think 在獨立的 pump task 裡跑到底（輸出進無界佇列），speak 從佇列拉：
        文字生成完成與 TTS 消化速度脫鉤，鎖才能在「文字完成」時釋放，
        下一輪的 LLM 不必等這一輪的合成與播放。
        """
        tid = t.turn_id
        token = att.token
        sink = att.sink
        if sink is not None:
            token.on_cancel(lambda _r, s=sink: _stop_sink(s))
        feed = None
        chunk_stream = None
        try:
            try:
                if sink is None:
                    await self._generate(t, att, None)
                else:
                    queue: asyncio.Queue = asyncio.Queue()
                    # 取消時喚醒可能卡在 queue.get() 的 speak（token.cancel 只在 loop 內呼叫）
                    token.on_cancel(lambda _r: queue.put_nowait(_CANCELLED))
                    att.pump = asyncio.ensure_future(self._pump(t, att, queue))
                    feed = self._feed(att, queue)
                    if not att.direct:
                        att.flusher = asyncio.ensure_future(self._flush(t, att))
                    chunk_stream = self.speak.stream(feed, token)
                    async for chunk in chunk_stream:
                        token.raise_if_cancelled()
                        self.tracer.mark(tid, marks.MARK_SPEAK_FIRST_CHUNK)
                        if att.direct:
                            await self._write(t, att, chunk)
                        else:
                            att.buffer.append(chunk)
                    self.tracer.mark(tid, marks.MARK_SPEAK_LAST_CHUNK)
                    if att.pump is not None:
                        await att.pump

                if att.regen_utt is not None and not t._root.is_cancelled:
                    return "regen"
                if att.flusher is not None:
                    await att.flusher
                token.raise_if_cancelled()
                if att.wrote_any:
                    await self._drain(t, att)
                return "done"
            except CancelledError:
                if att.regen_utt is not None and not t._root.is_cancelled:
                    return "regen"
                if att.pump_error is not None:
                    raise att.pump_error from None
                raise
        finally:
            await self._close_attempt(att, feed, chunk_stream)

    async def _close_attempt(self, att: _Attempt, feed, chunk_stream) -> None:
        for stream in (chunk_stream, feed):
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        for task in (att.pump, att.flusher):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        worker = att.text_worker
        if worker is not None:
            if att.regen_utt is not None:
                worker.cancel()  # 被丟棄的趟：還沒送出的文字不送
            elif att.text_q is not None:
                att.text_q.put_nowait(None)
            await asyncio.gather(worker, return_exceptions=True)

    # --- 生成端 ---

    @staticmethod
    async def _feed(att: _Attempt, queue: asyncio.Queue):
        """speak 的輸入：從佇列拉過濾後的句子，收到結束／取消／錯誤標記就收尾。"""
        while True:
            item = await queue.get()
            if item is _END:
                return
            if item is _CANCELLED:
                raise CancelledError(
                    att.token.reason or CancelReason.SHUTDOWN, att.token.detail
                )
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _pump(self, t: TurnHandle, att: _Attempt, queue: asyncio.Queue) -> None:
        try:
            await self._generate(t, att, queue.put_nowait)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 含管線自己的 CancelledError
            if not isinstance(exc, CancelledError):
                att.pump_error = exc
                att.token.cancel(CancelReason.UPSTREAM_ERROR, f"{type(exc).__name__}: {exc}")
            queue.put_nowait(exc)
            return
        queue.put_nowait(_END)

    async def _generate(self, t: TurnHandle, att: _Attempt, emit) -> None:
        """跑 think 到底：打點、記帳、過濾（PASS）、文字回呼；``emit`` 收過濾後要唸的句子。"""
        req = t.request
        tid = t.turn_id
        filt = req.text_filter
        speaking = emit is not None
        stream = self.think.stream(att.utterance, att.token)
        try:
            async for sentence in stream:
                self.tracer.mark(tid, marks.MARK_THINK_FIRST_SENTENCE)
                att.generated[sentence.index] = sentence
                self.tracer.count_sentence(tid, sentence.split_reason)

                voiced: Sentence | None = sentence
                if filt is not None and speaking:
                    try:
                        text = filt.feed(sentence.text)
                        passed = bool(filt.passed)
                    except Exception:  # noqa: BLE001 - 語音支線不能影響文字路徑
                        _log.warning("語音句子過濾失敗", exc_info=True)
                        text, passed = None, False
                    if passed:
                        att.passed = True
                        break
                    if not text:
                        voiced = None
                    elif text != sentence.text:
                        voiced = dataclasses.replace(sentence, text=text)

                if voiced is not None and voiced.text:
                    att.voiced[sentence.index] = voiced.text
                # 語音先、文字後：句子先進語音佇列，worker 之後才送文字（不延遲 TTS）
                if speaking and voiced is not None and voiced.text:
                    emit(voiced)
                self._push_text(t, att, sentence)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

        self.tracer.mark(tid, marks.MARK_THINK_DONE, overwrite=True)
        if (
            not att.passed
            and req.regen is not None
            and t.regenerated < req.max_regenerate
            and not t.audio_started
            and not att.token.is_cancelled
        ):
            try:
                fresh = req.regen.refresh(t)
            except Exception:  # noqa: BLE001
                _log.warning("regen 判定失敗", exc_info=True)
                fresh = None
            if fresh is not None:
                # 沿用同一 turn_id；舊趟語音由 token 取消停掉，不 commit
                att.regen_utt = dataclasses.replace(fresh, turn_id=req.utterance.turn_id)
                att.token.cancel(CancelReason.SUPERSEDED, "regenerated")
                return
        await self._finish_generation(t, att)

    async def _finish_generation(self, t: TurnHandle, att: _Attempt) -> None:
        req = t.request
        if req.commit_mode is CommitMode.EAGER and not t._eager_committed:
            generated = self._generated_text(att)
            t._eager_committed = True
            with contextlib.suppress(Exception):
                await self.think.commit(t.turn_id, generated, generated)
        if req.commit_mode is CommitMode.EAGER:
            self._lock_done(t)  # commit 之後才放鎖：下一輪的 prompt 才看得到這一輪
        self._resolve_gen_done(t)

    def _push_text(self, t: TurnHandle, att: _Attempt, sentence: Sentence) -> None:
        """句子交給宿主的文字回呼：丟進佇列由 worker 依序處理，不阻塞 TTS。"""
        callback = t.request.on_sentence
        if callback is None:
            return
        if att.text_q is None:
            att.text_q = asyncio.Queue()
            att.text_worker = asyncio.ensure_future(self._text_worker(t, att, callback))
        att.text_q.put_nowait(sentence)

    @staticmethod
    async def _text_worker(t: TurnHandle, att: _Attempt, callback) -> None:
        assert att.text_q is not None
        while True:
            sentence = await att.text_q.get()
            if sentence is None:
                return
            try:
                await callback(t, sentence)
            except Exception:  # noqa: BLE001 - 文字送出失敗不影響語音
                _log.warning("on_sentence 失敗", exc_info=True)

    # --- 播放端 ---

    async def _flush(self, t: TurnHandle, att: _Attempt) -> None:
        """等前一輪播完，把緩衝的音訊塊依序寫出，然後切成直寫模式。"""
        token = att.token
        pred = t._predecessor
        started = self._clock()
        if pred is not None and not pred._play_done.is_set():
            await self._wait_event(pred._play_done, token)
        t._queue_wait_s = self._clock() - started
        while not token.is_cancelled:
            if not att.buffer:
                att.direct = True  # 檢查與切換之間沒有 await，不會漏掉新塊
                return
            await self._write(t, att, att.buffer.popleft())

    @staticmethod
    async def _wait_event(event: asyncio.Event, token: CancellationToken) -> None:
        if event.is_set():
            return
        waiter = asyncio.ensure_future(event.wait())
        canceller = asyncio.ensure_future(token.wait())
        try:
            await asyncio.wait({waiter, canceller}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (waiter, canceller):
                task.cancel()
            await asyncio.gather(waiter, canceller, return_exceptions=True)

    async def _write(self, t: TurnHandle, att: _Attempt, chunk: AudioChunk) -> None:
        sink = att.sink
        assert sink is not None
        if not att.idle_done:
            att.idle_done = True
            await self._wait_idle(sink)
        trace = self.tracer.get(t.turn_id)
        if trace is not None:
            trace.phase = TurnPhase.SPEAKING
        duration = chunk.duration_s
        idx = chunk.sentence_index
        att.sent_audio[idx] = att.sent_audio.get(idx, 0.0) + duration  # 寫入前累加（同 bot）
        await sink.write(chunk)
        att.emitted += duration
        att.timeline.append((idx, att.emitted))
        self.tracer.count_chunk(t.turn_id, duration)
        if att.first_write_at is None:
            att.first_write_at = self._clock()
            att.wrote_any = True
            frame_at = sink_first_frame_at(sink)
            self.tracer.mark(
                t.turn_id, marks.MARK_FIRST_FRAME,
                at=frame_at if frame_at is not None else att.first_write_at,
            )

    async def _wait_idle(self, sink: AudioSink) -> None:
        """sink 選用能力：等底層播放器真的釋放。逾時放行並記 log。"""
        wait_idle = getattr(sink, "wait_idle", None)
        if wait_idle is None:
            return
        try:
            await asyncio.wait_for(wait_idle(), self._idle_timeout_s)
        except asyncio.TimeoutError:
            _log.warning("sink.wait_idle 逾時（%.2fs），放行", self._idle_timeout_s)
        except Exception:  # noqa: BLE001
            _log.warning("sink.wait_idle 失敗，放行", exc_info=True)

    async def _drain(self, t: TurnHandle, att: _Attempt) -> None:
        """等 sink 播完，但保持可被取消；逾時視為 TIMEOUT 取消。"""
        sink = att.sink
        assert sink is not None
        token = att.token
        drain_task = asyncio.ensure_future(sink.drain())
        cancel_task = asyncio.ensure_future(token.wait())
        try:
            done, _ = await asyncio.wait(
                {drain_task, cancel_task},
                timeout=self._play_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                _log.warning("播放逾時（%.0fs），強制停止", self._play_timeout_s)
                if self._claim(t, CancelReason.TIMEOUT, "play_timeout"):
                    self._cancel_in_loop(t, False)
            # 播放完成與取消可能同時到達，取消優先；播放失敗也要往上拋成 FAILED
            token.raise_if_cancelled()
            drain_task.result()
        finally:
            for task in (drain_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(drain_task, cancel_task, return_exceptions=True)

    # --- 結算 ---

    @staticmethod
    def _generated_text(att: _Attempt) -> str:
        return "".join(
            att.generated[i].text for i in sorted(att.generated) if att.generated[i].text
        )

    @staticmethod
    async def _confirm_stopped(sink: AudioSink | None) -> None:
        confirm = getattr(sink, "confirm_stopped", None)
        if confirm is None:
            return
        with contextlib.suppress(Exception):
            await confirm()

    def _resolve_spoken(
        self, t: TurnHandle, att: _Attempt, cut_short: bool
    ) -> tuple[str, float, bool, list]:
        """依 SpokenRule 推算實際聽到的內容。回傳 ``(spoken, 秒數, 是否確知, spans)``。"""
        sink = att.sink
        evidence = sink_played_evidence(sink) if sink is not None else PlayedEvidence.TRANSPORT_SENT
        rule = resolve_spoken_rule(t.request.spoken_rule, evidence)
        played: float | None = None
        if sink is not None:
            with contextlib.suppress(Exception):
                played = float(sink.played_duration_s)
        spans = build_sentence_spans(att.timeline, att.voiced)
        if rule is SpokenRule.PARTIAL:
            spoken, duration = partial_spoken(att.voiced, att.timeline, att.emitted, played)
            return spoken, duration, True, spans
        text = whole_sentence_spoken(att.voiced, att.sent_audio, played, cut_short)
        duration = 0.0 if played is None else max(0.0, min(att.emitted, played))
        return (text or ""), duration, text is not None, spans

    async def _conclude(
        self, t: TurnHandle, att: _Attempt, phase: TurnPhase, error: str | None
    ) -> None:
        req = t.request
        tid = t.turn_id
        sink = att.sink
        cancelled = phase is TurnPhase.CANCELLED
        reason = t._cancel_reason if cancelled else None
        detail = t._cancel_detail if cancelled else ""
        barge_in = reason is CancelReason.BARGE_IN
        superseded = cancelled and detail == REASON_OWNER
        aborted = cancelled and not (barge_in and detail != REASON_CASCADE)
        cut_short = barge_in or superseded or aborted or bool(error)

        # 被停掉的 sink 可能還有「最後實播位置」在回報路上：先等它確認再推算 spoken
        if phase is not TurnPhase.DONE and sink is not None:
            await self._confirm_stopped(sink)

        spoken, spoken_s, known, spans = self._resolve_spoken(t, att, cut_short)
        generated = self._generated_text(att)

        # --- commit／settle：嚴格排在確認實播量之後（Pipecat #4111）---
        no_history = reason in (CancelReason.SHUTDOWN, CancelReason.SUPERSEDED)
        settle = getattr(self.think, "settle", None)
        if no_history:
            if (
                reason is CancelReason.SUPERSEDED
                and req.retract_on_yield
                and t._eager_committed
                and settle is not None
            ):
                with contextlib.suppress(Exception):
                    await settle(tid, "", generated)
        elif req.commit_mode is CommitMode.SETTLED:
            with contextlib.suppress(Exception):
                await self.think.commit(tid, spoken, generated)
        elif req.history_spoken_only and t._eager_committed and settle is not None:
            with contextlib.suppress(Exception):
                await settle(tid, spoken, generated)

        # --- 狀態與跳過原因 ---
        skipped: str | None = None
        if phase is TurnPhase.DONE:
            if t._skip_reason:
                skipped = t._skip_reason
            elif att.passed:
                skipped = SkipReason.PASS
            elif sink is not None:
                play_error = getattr(sink, "play_error", None)
                if not att.voiced:
                    skipped = SkipReason.NO_SPOKEN_TEXT
                elif not att.wrote_any or (play_error and not t.audio_started):
                    skipped = f"play_failed: {play_error}" if play_error else SkipReason.NO_AUDIO
        if phase is TurnPhase.FAILED:
            status = OutcomeStatus.FAILED
        elif cancelled:
            status = OutcomeStatus.CANCELLED
        elif skipped is not None:
            status = OutcomeStatus.SKIPPED
        else:
            status = OutcomeStatus.COMPLETED

        frame_at = sink_first_frame_at(sink) if sink is not None else None
        if frame_at is not None:
            self.tracer.mark(tid, marks.MARK_FIRST_FRAME, at=frame_at, overwrite=True)
        trace = self.tracer.get(tid)
        if trace is not None:
            trace.spoken_duration_s = spoken_s
        cancel_value = reason.value if reason is not None else None
        self.tracer.finish(tid, phase, cancel_value, error)

        outcome = TurnOutcome(
            turn_id=tid,
            phase=phase,
            utterance_text=req.utterance.text,
            generated_text=generated,
            spoken_text=spoken,
            spoken_duration_s=spoken_s,
            cancel_reason=cancel_value,
            error=error,
            metadata=self._outcome_meta(t, reason, detail),
            skipped_reason=skipped,
            cut_short=cut_short,
            audio_started=t.audio_started,
            first_audio_at=frame_at if frame_at is not None else att.first_write_at,
            sentence_spans=spans,
            status=status,
            spoken_known=known,
            barge_in=barge_in,
            superseded=superseded,
            aborted=aborted,
            queue_wait_s=t._queue_wait_s,
            regenerated=t.regenerated,
            attempt_id=t._attempt_no,
            snapshot_version=t._snapshot_version,
        )
        self._lock_done(t)  # SETTLED：commit 完才放鎖
        self._deliver(t, outcome)

    # --- outcome 出口 ---

    @staticmethod
    def _outcome_meta(t: TurnHandle, reason: CancelReason | None, detail: str) -> dict:
        meta: dict = {}
        if reason is not None:
            if reason is CancelReason.SUPERSEDED:
                code = REASON_YIELDED if detail == REASON_YIELDED else "superseded"
            else:
                code = reason.value
            meta["reason_code"] = code
            meta["cancel_detail"] = detail
            if detail == REASON_CASCADE:
                meta["cascade"] = True
        if t._stale:
            meta["stale_snapshot"] = True
        return meta

    def _deliver(self, t: TurnHandle, outcome: TurnOutcome) -> None:
        """送出這一輪唯一的 outcome（冪等：第二次呼叫無效）。"""
        if t._done:
            return
        t._done = True
        if not t.settled.done():
            t.settled.set_result(outcome)
        callback = self._on_turn_ended
        if callback is not None:
            try:
                callback(outcome)
            except Exception:  # noqa: BLE001
                _log.warning("on_turn_ended 失敗", exc_info=True)

    def _deliver_dropped(self, t: TurnHandle) -> None:
        self._resolve_gen_done(t)
        t._play_done.set()
        self._deliver(t, TurnOutcome(
            turn_id=t.turn_id,
            phase=TurnPhase.CANCELLED,
            utterance_text=t.request.utterance.text,
            skipped_reason=SkipReason.DROPPED_BUSY,
            status=OutcomeStatus.DROPPED_BUSY,
            snapshot_version=t._snapshot_version,
        ))

    def _conclude_voided(self, t: TurnHandle) -> None:
        """等鎖期間被作廢：沒有生成、沒有播放、不 commit。"""
        reason = t._cancel_reason or CancelReason.SHUTDOWN
        detail = t._cancel_detail
        barge_in = reason is CancelReason.BARGE_IN
        _stop_sink(t._first_sink)
        self.tracer.finish(t.turn_id, TurnPhase.CANCELLED, reason.value, None)
        self._deliver(t, TurnOutcome(
            turn_id=t.turn_id,
            phase=TurnPhase.CANCELLED,
            utterance_text=t.request.utterance.text,
            cancel_reason=reason.value,
            metadata=self._outcome_meta(t, reason, detail),
            skipped_reason=SkipReason.VOIDED,
            cut_short=True,
            status=OutcomeStatus.VOIDED,
            barge_in=barge_in,
            superseded=detail == REASON_OWNER,
            aborted=not (barge_in and detail != REASON_CASCADE),
            snapshot_version=t._snapshot_version,
        ))

    def _fallback_outcome(self, t: TurnHandle, error: str, *, cancelled: bool = False) -> None:
        """內部錯誤或 task 被外部取消時的保底：確保每個 submit 都有 outcome。"""
        if t._done:
            return
        self.tracer.finish(
            t.turn_id, TurnPhase.CANCELLED if cancelled else TurnPhase.FAILED, None, error
        )
        self._deliver(t, TurnOutcome(
            turn_id=t.turn_id,
            phase=TurnPhase.CANCELLED if cancelled else TurnPhase.FAILED,
            utterance_text=t.request.utterance.text,
            error=error,
            cut_short=True,
            status=OutcomeStatus.CANCELLED if cancelled else OutcomeStatus.FAILED,
            aborted=cancelled,
            snapshot_version=t._snapshot_version,
        ))

    def _resolve_gen_done(self, t: TurnHandle) -> None:
        if not t.generation_done.done():
            t.generation_done.set_result(None)

    def _mark_play_done(self, t: TurnHandle) -> None:
        """這一輪的播放位置讓出來。必須等前一輪也結束：被取消的排隊輪
        若搶先結束，會讓更後面的輪在前一輪還在播時開播而疊音。"""
        if t._play_done.is_set():
            return
        pred = t._predecessor
        if pred is None or pred._play_done.is_set():
            t._play_done.set()
        elif not t._chaining:
            t._chaining = True
            asyncio.ensure_future(self._chain(t, pred))

    @staticmethod
    async def _chain(t: TurnHandle, pred: TurnHandle) -> None:
        await pred._play_done.wait()
        t._play_done.set()


__all__ = [
    "IDLE_TIMEOUT_S",
    "PLAY_TIMEOUT_S",
    "TurnCoordinator",
    "TurnHandle",
]
