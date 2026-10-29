"""TurnCoordinator（T-07b PR ②）——排隊、取消語意、regen、PASS、commit、跨執行緒。

沿用 ``scenario_support`` 的手動推進零件：沒有任何用來賭時序的 sleep，
狀態推進一律靠 ``ManualSink.play/finish``、``GateSpeak.open``、``until``。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import pytest

from echo_stream.adapters.think_llm import LLMThinkStage
from echo_stream.contracts.cancellation import CancelReason
from echo_stream.contracts.output import (
    CommitMode,
    ModeFlags,
    OutcomeStatus,
    PolicySnapshot,
    SkipReason,
    TurnOutcome,
    TurnRequest,
    YieldAction,
)
from echo_stream.contracts.turn import VoiceEvent, VoiceState
from echo_stream.contracts.types import TTS_SAMPLE_RATE, AudioChunk, Sentence, Utterance
from echo_stream.core import tracer as marks
from echo_stream.core.coordinator import TurnCoordinator
from echo_stream.core.standalone import StandaloneTicketIssuer, StaticPolicySource
from echo_stream.core.tracer import LatencyTracer

from .scenario_support import (
    RESPONSE,
    SENTENCES,
    GateSpeak,
    ManualSink,
    ScriptBackend,
    build_think,
    until,
)

S0 = SENTENCES[0]
SAFETY_S = 5.0


def voiced(t: float, speaker: str = "u1", chunk: float = 0.5) -> VoiceEvent:
    return VoiceEvent(
        state=VoiceState.SPEECH, timestamp=t, duration_s=0.0, speaker_id=speaker,
        is_voiced=True, chunk_s=chunk,
    )


@dataclass
class Env:
    co: TurnCoordinator
    think: LLMThinkStage
    backend: ScriptBackend
    speak: GateSpeak
    tracer: LatencyTracer
    log: list[str] = field(default_factory=list)
    commits: list[tuple[str, str, str]] = field(default_factory=list)
    outcomes: list[TurnOutcome] = field(default_factory=list)
    sinks: list[ManualSink] = field(default_factory=list)

    def req(
        self, text: str = "你好", *, owner: str = "u1", sink: ManualSink | None = None,
        barge: tuple[str, ...] = ("u1",), commit: CommitMode = CommitMode.EAGER, **kw,
    ) -> tuple[TurnRequest, ManualSink]:
        utt = Utterance(text=text, speaker_id=owner)
        ticket = StandaloneTicketIssuer().issue(utt)
        s = sink or ManualSink(self.log)
        self.sinks.append(s)
        return TurnRequest(
            ticket=ticket, utterance=utt, sink_factory=lambda _v, s=s: s,
            barge_users=barge, commit_mode=commit, **kw,
        ), s


def build(
    responses=(RESPONSE,), *, gate: asyncio.Event | None = None, open_all: bool = True,
    platform: str = "discord", **co_kw,
) -> Env:
    log: list[str] = []
    backend = ScriptBackend(responses, gate=gate)
    think = build_think(backend, log)
    original = think.commit
    commits: list[tuple[str, str, str]] = []

    async def spy(turn_id: str, spoken: str, generated: str) -> None:
        commits.append((turn_id, spoken, generated))
        await original(turn_id, spoken, generated)

    think.commit = spy  # type: ignore[method-assign]
    speak = GateSpeak(open_all=open_all)
    tracer = LatencyTracer()
    outcomes: list[TurnOutcome] = []
    co = TurnCoordinator(
        think, speak, tracer=tracer,
        policy_source=StaticPolicySource(PolicySnapshot.for_platform(platform)),
        on_turn_ended=outcomes.append, **co_kw,
    )
    _COORDINATORS.append(co)
    return Env(co, think, backend, speak, tracer, log, commits, outcomes)


_COORDINATORS: list[TurnCoordinator] = []


@pytest.fixture(autouse=True)
async def _drain_coordinators():
    """每個測試結束都收掉還沒結束的輪，避免 task 外洩到下一個測試。"""
    yield
    while _COORDINATORS:
        co = _COORDINATORS.pop()
        co.cancel_all("test_end")
        await asyncio.wait_for(co.join(), SAFETY_S)


async def settled(handle) -> TurnOutcome:
    return await asyncio.wait_for(handle.settled, SAFETY_S)


async def playing(sink: ManualSink, n: int = 4) -> None:
    await until(lambda: len(sink.chunks) == n and sink.draining.is_set())


# ───────────────────────── 排隊 ─────────────────────────


async def test_queued_turn_writes_only_after_predecessor_ends() -> None:
    env = build()
    ra, sink_a = env.req("A", owner="u1")
    ha = env.co.submit(ra)
    await playing(sink_a)

    rb, sink_b = env.req("B", owner="u2", barge=("u2",))
    hb = env.co.submit(rb)
    await asyncio.wait_for(hb.generation_done, SAFETY_S)  # B 的文字已生成完（不等 A 播完）
    await until(lambda: hb._att is not None and len(hb._att.buffer) == 4)
    assert sink_b.chunks == []  # 合成完了，但 A 還在播，不寫 sink
    assert not hb.audio_started

    sink_a.finish()
    out_a = await settled(ha)
    await until(lambda: len(sink_b.chunks) == 4)  # A 結束後才寫
    sink_b.finish()
    out_b = await settled(hb)
    assert out_a.status is OutcomeStatus.COMPLETED and out_b.status is OutcomeStatus.COMPLETED
    assert out_b.queue_wait_s >= 0.0
    assert env.backend.calls and len(env.backend.calls) == 2


async def test_barge_in_cascades_to_queued_unstarted_turns() -> None:
    env = build()
    ra, sink_a = env.req("A", owner="u1")
    ha = env.co.submit(ra)
    await playing(sink_a)
    sink_a.play(1.0)  # 使用者聽到第一句
    rb, sink_b = env.req("B", owner="u2", barge=("u2",))
    hb = env.co.submit(rb)
    await asyncio.wait_for(hb.generation_done, SAFETY_S)

    # barge-in 只累積 A 的指定使用者；1.5s 不停、2.0s 停
    for k in range(3):
        env.co.on_voice_event(voiced(1.0 + 0.5 * k))
    assert not ha.token.is_cancelled
    env.co.on_voice_event(voiced(2.5))
    assert ha.token.is_cancelled

    out_a, out_b = await settled(ha), await settled(hb)
    assert out_a.status is OutcomeStatus.CANCELLED and out_a.barge_in and out_a.cut_short
    assert out_a.cancel_reason == CancelReason.BARGE_IN.value
    assert out_a.spoken_text == S0 and out_a.spoken_known  # 被打斷時後面有別句音訊，第一句算確知
    assert out_b.barge_in and out_b.metadata.get("cascade") is True
    assert out_b.status is OutcomeStatus.CANCELLED
    assert sink_b.chunks == [] and sink_b.stop_calls >= 1


# ───────────────────────── 取消語意表（§7）─────────────────────────


async def test_void_before_generation_is_voided_and_never_generates() -> None:
    gate = asyncio.Event()
    env = build(gate=gate)
    ra, _ = env.req("A", owner="u1")
    ha = env.co.submit(ra)
    await until(lambda: len(env.backend.calls) == 1)
    rb, sink_b = env.req("B", owner="u2")
    hb = env.co.submit(rb)
    assert env.co.void(hb.turn_id, "yielded") is True
    out_b = await settled(hb)
    assert out_b.status is OutcomeStatus.VOIDED and out_b.skipped_reason == SkipReason.VOIDED
    assert out_b.cancel_reason == CancelReason.SUPERSEDED.value
    assert out_b.metadata["reason_code"] == "yielded"
    assert len(env.backend.calls) == 1 and sink_b.chunks == []
    assert env.co.void(hb.turn_id) is False  # 已結束
    gate.set()
    await playing(env.sinks[0])
    env.sinks[0].finish()
    await settled(ha)
    assert [o.turn_id for o in env.outcomes].count(hb.turn_id) == 1


async def test_void_mid_generation_cancels_without_commit() -> None:
    gate = asyncio.Event()
    env = build(gate=gate)
    ra, sink_a = env.req("A")
    ha = env.co.submit(ra)
    await until(lambda: len(env.backend.calls) == 1)
    assert env.co.void(ha.turn_id, "yielded") is True
    out = await settled(ha)
    assert out.status is OutcomeStatus.CANCELLED and out.aborted and not out.barge_in
    assert out.cancel_reason == CancelReason.SUPERSEDED.value
    assert env.commits == [] and sink_a.chunks == []


async def test_void_after_audio_started_is_refused() -> None:
    env = build()
    ra, sink_a = env.req("A")
    ha = env.co.submit(ra)
    await playing(sink_a)
    assert ha.audio_started and ha.sink_active  # ManualSink 沒有選用成員，走推定
    assert env.co.void(ha.turn_id) is False
    sink_a.finish()
    await settled(ha)


async def test_same_owner_new_turn_supersedes_old_one() -> None:
    env = build()
    r1, sink_1 = env.req("一", owner="u1")
    h1 = env.co.submit(r1)
    await playing(sink_1)
    r2, sink_2 = env.req("二", owner="u1")
    h2 = env.co.submit(r2)
    out_1 = await settled(h1)
    assert out_1.superseded and out_1.status is OutcomeStatus.CANCELLED
    assert out_1.cancel_reason == CancelReason.SUPERSEDED.value and sink_1.stopped
    await playing(sink_2)  # 舊輪被取代不佔位，新輪不必等它
    sink_2.finish()
    assert (await settled(h2)).status is OutcomeStatus.COMPLETED


async def test_cancel_all_is_shutdown_without_commit_in_settled_mode() -> None:
    env = build()
    ra, sink_a = env.req("A", commit=CommitMode.SETTLED)
    ha = env.co.submit(ra)
    await playing(sink_a)
    env.co.cancel_all("left_channel")
    out = await settled(ha)
    assert out.cancel_reason == CancelReason.SHUTDOWN.value and out.aborted
    assert out.metadata["cancel_detail"] == "left_channel"
    assert env.commits == []  # SHUTDOWN 不 commit、不 settle


async def test_retract_on_yield_settles_eager_history() -> None:
    env = build((RESPONSE, "戊戊戊。"))
    ra, sink_a = env.req("A", owner="u1", retract_on_yield=True)
    ha = env.co.submit(ra)
    await playing(sink_a)
    assert any(h["content"] == RESPONSE for h in env.think.history)  # EAGER 已 commit
    r2, _ = env.req("二", owner="u1")  # 同 owner 取代走 SUPERSEDED
    env.co.submit(r2)
    await settled(ha)
    assert all(h["content"] != RESPONSE for h in env.think.history)  # 撤回舊輪 assistant 項


# ───────────────────────── 跨執行緒 ─────────────────────────


class ThreadSpySink(ManualSink):
    def __init__(self, log=None):
        super().__init__(log)
        self.stop_threads: list[int] = []

    def stop(self) -> None:
        self.stop_threads.append(threading.get_ident())
        super().stop()


async def test_on_voice_event_from_other_threads_stops_sync_and_cancels_once() -> None:
    env = build()
    sink = ThreadSpySink(env.log)
    ra, _ = env.req("A", sink=sink)
    ha = env.co.submit(ra)
    await playing(sink)

    cancels: list[int] = []
    ha.token.on_cancel(lambda _r: cancels.append(threading.get_ident()))
    loop_thread = threading.get_ident()
    caller_threads: list[int] = []

    def send(offset: float) -> None:
        caller_threads.append(threading.get_ident())
        for k in range(8):
            env.co.on_voice_event(voiced(offset + 0.5 * k))

    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, send, 1.0), loop.run_in_executor(None, send, 1.0)
    )
    # stop 在「第一個命中的呼叫執行緒」同步發生（早於 token 取消回到 loop）
    assert sink.stop_threads[0] in caller_threads and sink.stop_threads[0] != loop_thread
    await until(lambda: ha.token.is_cancelled)
    assert cancels == [loop_thread]  # token 只取消一次，且在 loop 內
    out = await settled(ha)
    assert out.barge_in


# ───────────────────────── regen ─────────────────────────


class OnceRegen:
    def __init__(self, text: str, times: int = 99) -> None:
        self.text, self.times, self.calls = text, times, 0

    def refresh(self, turn):
        self.calls += 1
        if self.calls > self.times:
            return None
        return Utterance(text=self.text, turn_id="ignored")


class MemorySpy:
    def __init__(self) -> None:
        self.retrieves = 0
        self.stores = 0

    async def retrieve(self, query: str, **_kw) -> str:
        self.retrieves += 1
        return ""

    async def store(self, *_a, **_kw) -> None:
        self.stores += 1


async def test_regen_reuses_turn_id_stops_old_sink_and_commits_once() -> None:
    env = build(("甲甲甲。", "乙乙乙。"), open_all=False)
    memory = MemorySpy()
    env.think.memory = memory
    regen = OnceRegen("你好", times=1)
    r, _ = env.req("你好", regen=regen, max_regenerate=1)
    made: list[ManualSink] = []

    def factory(_v) -> ManualSink:
        s = ManualSink(env.log)
        made.append(s)
        return s

    r.sink_factory = factory
    h = env.co.submit(r)
    await until(lambda: len(made) == 2)  # 舊趟 abort，新趟重新建 sink（stop 是永久的）
    assert made[0].stopped and made[0].chunks == []
    env.speak.open(0)
    await until(lambda: len(made[1].chunks) == 1 and made[1].draining.is_set())
    made[1].finish()
    out = await settled(h)
    assert out.regenerated == 1 and out.attempt_id == 2
    assert out.turn_id == r.ticket.turn_id
    assert out.generated_text == "乙乙乙。"
    assert len(env.commits) == 1
    assert memory.retrieves == 1  # 同 turn_id 同 query 不重 retrieve
    assert regen.calls == 1
    trace = env.tracer.get(r.ticket.turn_id)
    assert trace is not None and marks.MARK_THINK_DONE in trace.marks


async def test_regen_respects_limit() -> None:
    env = build(("甲甲甲。", "乙乙乙。", "丙丙丙。"), open_all=False)
    regen = OnceRegen("你好")  # 每次都想重算，但上限 1
    r, _ = env.req("你好", regen=regen, max_regenerate=1)
    made: list[ManualSink] = []

    def factory(_v) -> ManualSink:
        made.append(ManualSink(env.log))
        return made[-1]

    r.sink_factory = factory
    h = env.co.submit(r)
    await until(lambda: len(made) == 2 and h.generation_done.done())
    assert regen.calls == 1 and len(env.backend.calls) == 2
    env.co.cancel_all()
    out = await settled(h)
    assert out.regenerated == 1


async def test_no_regen_once_audio_started() -> None:
    class StartedSink(ManualSink):
        first_frame_at = 1.0

    env = build()
    regen = OnceRegen("你好")
    r, _ = env.req("你好", sink=StartedSink(env.log), regen=regen, max_regenerate=1)
    h = env.co.submit(r)
    await until(lambda: h.generation_done.done())
    assert regen.calls == 0
    env.co.cancel_all()
    await settled(h)


# ───────────────────────── PASS ─────────────────────────


class PassOnFirst:
    def __init__(self) -> None:
        self.passed = False
        self.addressee_ids: set[str] = set()

    def feed(self, text: str):
        if text.startswith("甲"):
            self.passed = True
            return None
        return text


@pytest.mark.parametrize("mode", [CommitMode.EAGER, CommitMode.SETTLED])
async def test_pass_writes_nothing_and_commits_per_mode(mode: CommitMode) -> None:
    env = build()
    r, sink = env.req("A", commit=mode, text_filter=PassOnFirst())
    h = env.co.submit(r)
    out = await settled(h)
    assert sink.chunks == [] and sink.stop_calls == 0
    assert out.status is OutcomeStatus.SKIPPED and out.skipped_reason == SkipReason.PASS
    assert len(env.commits) == 1
    _tid, spoken, generated = env.commits[0]
    if mode is CommitMode.EAGER:
        assert spoken == generated == S0  # bot 現況：PASS 照樣 commit 已生成的文字
    else:
        assert spoken == "" and generated == S0


# ───────────────────────── commit／settle ─────────────────────────


async def test_eager_commits_before_drain_and_settled_after() -> None:
    env = build()
    ra, sink_a = env.req("A", commit=CommitMode.EAGER)
    ha = env.co.submit(ra)
    await playing(sink_a)
    await until(lambda: "commit" in env.log)  # 文字完成就 commit，播放還在 drain
    assert not ha.settled.done()
    sink_a.finish()
    await settled(ha)

    env2 = build()
    rb, sink_b = env2.req("B", commit=CommitMode.SETTLED)
    hb = env2.co.submit(rb)
    await playing(sink_b)
    assert "commit" not in env2.log  # SETTLED：播完才 commit
    # 且鎖一路持有到結算：下一輪的 think 不會搶先開始
    rc, _ = env2.req("C", owner="u2")
    hc = env2.co.submit(rc)
    assert len(env2.backend.calls) == 1
    sink_b.finish()
    await settled(hb)
    assert env2.commits[0][1] == RESPONSE
    env2.co.cancel_all()
    await settled(hc)


class NoSettleThink:
    """只有 stream/commit、沒有 settle 的 ThinkStage。"""

    name = "no_settle"

    def __init__(self, inner: LLMThinkStage) -> None:
        self.inner = inner

    def stream(self, utterance, token):
        return self.inner.stream(utterance, token)

    async def commit(self, turn_id, spoken, generated):
        await self.inner.commit(turn_id, spoken, generated)


async def test_missing_settle_does_not_error() -> None:
    env = build()
    env.co.think = NoSettleThink(env.think)  # type: ignore[assignment]
    r, sink = env.req("A", history_spoken_only=True)
    h = env.co.submit(r)
    await playing(sink)
    sink.finish()
    assert (await settled(h)).status is OutcomeStatus.COMPLETED


async def test_history_spoken_only_settles_to_spoken_after_barge_in() -> None:
    env = build()
    r, sink = env.req("A", history_spoken_only=True)
    h = env.co.submit(r)
    await playing(sink)
    sink.play(1.0)
    for k in range(4):
        env.co.on_voice_event(voiced(1.0 + 0.5 * k))
    out = await settled(h)
    assert out.spoken_text == S0
    assert env.think.history[-1] == {"role": "assistant", "content": S0}


# ───────────────────────── sink 能力與降級 ─────────────────────────


async def test_sink_factory_string_skips_voice_but_keeps_text() -> None:
    env = build()
    sent: list[str] = []

    async def on_sentence(_view, sentence: Sentence) -> None:
        sent.append(sentence.text)

    r, _ = env.req("A", on_sentence=on_sentence)
    r.sink_factory = lambda _v: "tts_not_ready"
    h = env.co.submit(r)
    out = await settled(h)
    assert out.status is OutcomeStatus.SKIPPED and out.skipped_reason == "tts_not_ready"
    assert out.generated_text == RESPONSE and "".join(sent) == RESPONSE
    assert not out.spoken_known  # 沒有語音資料 = 未知，不是「什麼都沒播」


async def test_mode_silent_flag_means_text_only() -> None:
    class Rules:
        def flags_for(self, reply_mode):
            return ModeFlags(speak=False)

    env = build()
    env.co.policy_source = StaticPolicySource(PolicySnapshot(mode_rules=Rules()))
    r, sink = env.req("A")
    out = await settled(env.co.submit(r))
    assert out.skipped_reason == SkipReason.MODE_SILENT and sink.chunks == []


async def test_wait_idle_timeout_releases_and_logs(caplog) -> None:
    class StuckIdleSink(ManualSink):
        async def wait_idle(self) -> None:
            await asyncio.Event().wait()

    env = build(idle_timeout_s=0.01)
    sink = StuckIdleSink(env.log)
    r, _ = env.req("A", sink=sink)
    with caplog.at_level("WARNING"):
        h = env.co.submit(r)
        await playing(sink)
    assert any("wait_idle" in rec.message for rec in caplog.records)
    sink.finish()
    await settled(h)


# ───────────────────────── admission／每個 submit 恰一個 outcome ─────────────────────────


class MaxPending:
    def __init__(self, n: int) -> None:
        self.n = n

    def admit(self, request, busy: bool, waiting: int) -> bool:
        return not (busy and waiting >= self.n)


async def test_every_submit_gets_exactly_one_outcome() -> None:
    gate = asyncio.Event()
    env = build(gate=gate, admission=MaxPending(1))
    handles = []
    for name in ("A", "B", "C"):
        r, _ = env.req(name, owner=name)
        handles.append(env.co.submit(r))
    dropped = await settled(handles[2])  # C：busy 且已有人在等 → 丟棄，不觸發任何生成
    assert dropped.status is OutcomeStatus.DROPPED_BUSY
    assert dropped.skipped_reason == SkipReason.DROPPED_BUSY
    env.co.void(handles[1].turn_id)  # B：等鎖期間作廢
    gate.set()
    await until(lambda: env.sinks[0].draining.is_set())
    env.sinks[0].finish()
    outs = [await settled(h) for h in handles]
    assert [o.status for o in outs] == [
        OutcomeStatus.COMPLETED, OutcomeStatus.VOIDED, OutcomeStatus.DROPPED_BUSY
    ]
    ids = [o.turn_id for o in env.outcomes]
    assert sorted(ids) == sorted(h.turn_id for h in handles)  # 每輪恰好一次
    assert len(env.backend.calls) == 1


def test_submit_rejects_ticket_mismatch() -> None:
    async def go() -> None:
        env = build()
        r, _ = env.req("A")
        r.ticket = StandaloneTicketIssuer().issue(Utterance(text="別人"))
        with pytest.raises(ValueError):
            env.co.submit(r)

    asyncio.run(go())


# ───────────────────────── 打點／記憶／TTFA ─────────────────────────


async def test_marks_queue_first_frame_and_separate_stt_done() -> None:
    env = build()
    r, sink = env.req("A", speech_end_at=10.0, stt_done_at=10.8, marks={"decision_done": 11.0})
    h = env.co.submit(r)
    await playing(sink)
    sink.finish()
    await settled(h)
    m = env.tracer.get(r.ticket.turn_id).marks  # type: ignore[union-attr]
    assert m[marks.MARK_SPEECH_END] == 10.0 and m[marks.MARK_INPUT_FINAL] == 10.8
    assert m["decision_done"] == 11.0
    for name in (marks.MARK_QUEUE_ACQUIRED, marks.MARK_FIRST_FRAME, marks.MARK_SPEAK_FIRST_CHUNK):
        assert name in m


async def test_coordinator_never_writes_memory() -> None:
    env = build()
    memory = MemorySpy()
    env.think.memory = memory
    r, sink = env.req("A")
    h = env.co.submit(r)
    await playing(sink)
    sink.finish()
    await settled(h)
    assert memory.stores == 0  # 記憶寫入屬宿主（bot 的 VoiceMemoryHub），ES 不雙寫
    import echo_stream.core.coordinator as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert ".store(" not in src and "memory" not in src.lower().replace("memory_", "")


async def test_first_chunk_to_sink_write_has_no_extra_loop_iteration() -> None:
    ticks = {"n": 0}

    async def ticker() -> None:
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0)

    class Speak:
        name = "one"
        at_yield: list[int] = []

        async def stream(self, sentences, token):
            async for s in sentences:
                self.at_yield.append(ticks["n"])
                yield AudioChunk(
                    pcm=bytes(2000), sample_rate=TTS_SAMPLE_RATE, turn_id=s.turn_id,
                    sentence_index=s.index,
                )

    class Sink(ManualSink):
        at_write: list[int] = []

        async def write(self, chunk):
            self.at_write.append(ticks["n"])
            await super().write(chunk)

    env = build()
    speak = Speak()
    env.co.speak = speak  # type: ignore[assignment]
    sink = Sink(env.log)
    r, _ = env.req("A", sink=sink)
    task = asyncio.ensure_future(ticker())
    try:
        h = env.co.submit(r)
        await until(lambda: len(sink.at_write) >= 1)
        assert sink.at_write[0] == speak.at_yield[0]  # 首 chunk 產出到 sink.write 之間沒讓出 loop
        sink.finish()
        await settled(h)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert env.co.live_turns() == []


async def test_yield_policy_is_called_and_stale_snapshot_disables_it() -> None:
    class Y:
        def __init__(self) -> None:
            self.calls = 0

        def on_voiced(self, ev, live):
            self.calls += 1
            return [YieldAction(live[0].turn_id)] if live and not live[0].audio_started else []

    gate = asyncio.Event()
    y = Y()
    env = build(gate=gate, yield_policy=y)
    r, _ = env.req("A")
    h = env.co.submit(r)
    await until(lambda: len(env.backend.calls) == 1)
    snap = PolicySnapshot(ttl_s=0.0, issued_at=-100.0)  # 已過期
    env.co.policy_source = StaticPolicySource(snap)
    env.co.on_voice_event(voiced(1.0))
    assert y.calls == 0 and not h.token.is_cancelled  # 過期：讓位停用
    env.co.policy_source = StaticPolicySource(PolicySnapshot())
    env.co.on_voice_event(voiced(2.0))
    assert y.calls == 1
    out = await settled(h)
    assert out.metadata["reason_code"] == "yielded"
    gate.set()
