"""輸出端（多輪播放協調）的契約層。

T-07b PR ①：只定義型別、Protocol 與常數，**不含執行機制**——
``TurnCoordinator``（排隊、取消、結算）在後續 PR 才實作，``run_turn`` 行為不變。

## 設計要點

* **執行權來自 TurnTicket。** turn_id 由 Core 簽發寫在 :class:`TurnTicket`；
  ``Utterance.turn_id`` 是輸入端 id，ticket 以 ``source_turn_id`` 對應。
  Web／CLI 沒有 Core，用 :class:`~echo_stream.core.standalone.StandaloneTicketIssuer`。
* **熱路徑只讀 PolicySnapshot。** 策略插槽全部同步、純記憶體、不得 I/O
  （barge-in 每個 VAD chunk 約 20-100ms 就會呼叫一次，任何等待都會變成「叫不停」）。
  門檻是平台 preset，放在快照裡，不寫死在 ES。
* **spoken 規則可切換。** 預設 :attr:`SpokenRule.AUTO`：sink 以
  ``played_evidence`` 宣告證據等級，有播放確認走「部分播出算已播」，
  否則退回整句制。
* **sink 的新成員全是選用能力**（見 :class:`OptionalSinkCapabilities`）。
  不加進 :class:`~echo_stream.contracts.stages.AudioSink`——那是
  ``runtime_checkable``，加成員會讓既有 sink 全部驗不過 isinstance。
  一律用本檔的 ``sink_*`` helper 以 ``getattr`` 探測。

* **每個 submit 必定剛好一個 TurnOutcome。** 不論這一輪是正常播完、被插話、
  生成失敗、被 admission 丟棄（busy）、還是在等鎖期間被作廢，coordinator
  都必須恰好送出一次 :class:`TurnOutcome`，不多不少。結束路徑由
  :attr:`TurnOutcome.status` 與 ``skipped_reason`` 表達（見 :class:`OutcomeStatus`）。
  被丟棄的輪次不觸發任何生成／播放回呼，但仍有 outcome。

本檔**不 import 任何子系統**，也不引入第三方依賴。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .stages import AudioSink
from .turn import InterruptionDecision, VoiceEvent
from .types import Sentence, SentenceSpan, TurnResult, Utterance

# ─────────────────────────── spoken／commit 語意 ───────────────────────────


class SpokenRule(str, Enum):
    """「使用者實際聽到了什麼」的推算規則（P3 D74）。"""

    PARTIAL = "partial"
    """部分播出算已播：句子的 chunk 起點 < 實播時長即算。寧多算不少算，
    避免下一輪重複已講過的內容（與 ``PipelineRunner._resolve_spoken`` 同義）。"""

    WHOLE_SENTENCE = "whole_sentence"
    """整句音訊都已播出才算。保守；證據只有「送出去了」時使用。"""

    AUTO = "auto"
    """依 sink 的 :class:`PlayedEvidence` 決定：播放確認走 PARTIAL，其餘走 WHOLE_SENTENCE。"""


class PlayedEvidence(str, Enum):
    """``played_duration_s`` 的證據等級，由 sink 宣告。"""

    PLAYBACK_CONFIRMED = "playback_confirmed"
    """播放端回報的實播量（瀏覽器 ACK、本機喇叭的播放游標）。"""

    TRANSPORT_SENT = "transport_sent"
    """只知道交給傳輸層／player 的量（例如 Discord），不代表使用者聽到。"""


class OutcomeStatus(str, Enum):
    """一輪的結束路徑。每個 submit 恰好對應一個。"""

    COMPLETED = "completed"
    """正常播完（含無 sink 的純量測）。"""

    CANCELLED = "cancelled"
    """播放／生成中被取消（插話、shutdown 等；原因看 ``cancel_reason``）。"""

    FAILED = "failed"
    """生成或播放拋錯（看 ``error``）。"""

    SKIPPED = "skipped"
    """進入處理但不出聲（PASS、無可唸文字、sink 不可用；原因看 ``skipped_reason``）。"""

    DROPPED_BUSY = "dropped_busy"
    """被 :class:`AdmissionPolicy` 拒絕（busy 且佇列已滿）：不觸發任何回呼。"""

    VOIDED = "voided"
    """等鎖期間被作廢（讓位、同 owner 取代、離開頻道）：未開始生成。"""


class SkipReason:
    """``skipped_reason`` 的標準值（字串常數；宿主可另加自己的）。"""

    PASS = "pass"
    NO_SPOKEN_TEXT = "no_spoken_text"
    TTS_NOT_READY = "tts_not_ready"
    DROPPED_BUSY = "dropped_busy"
    VOIDED = "voided"


class CommitMode(str, Enum):
    """歷史寫回時機。"""

    SETTLED = "settled"
    """播完／確認停止後才 ``commit(spoken, generated)``（現況，Web 用）。"""

    EAGER = "eager"
    """文字生成完立刻 ``commit(generated, generated)``，播放結算後再 ``settle``
    （多輪重疊的宿主用，下一輪 prompt 才看得到上一輪）。"""


def resolve_spoken_rule(rule: SpokenRule, evidence: PlayedEvidence) -> SpokenRule:
    """把 AUTO 依證據等級解成具體規則；明確指定的規則原樣回傳（宿主覆寫優先）。"""
    if rule is not SpokenRule.AUTO:
        return rule
    if evidence is PlayedEvidence.PLAYBACK_CONFIRMED:
        return SpokenRule.PARTIAL
    return SpokenRule.WHOLE_SENTENCE


# ─────────────────────────── AudioSink 選用能力 ───────────────────────────


class OptionalSinkCapabilities(Protocol):
    """AudioSink 的選用成員（文件用，不做 isinstance 檢查）。

    缺少時的預設行為寫在對應的 ``sink_*`` helper。
    """

    is_active: bool
    """已開始出聲且未結束（barge-in 武裝條件）。"""

    first_frame_at: float | None
    """第一個真音訊 frame 交給 player 的 perf_counter 時刻。"""

    played_evidence: PlayedEvidence
    """``played_duration_s`` 的證據等級。"""

    async def confirm_stopped(self) -> None:
        """停止後等最終實播量，時限由 sink 負責。"""

    def reset(self) -> None:
        """清掉停止狀態與緩衝，供長壽命 sink 重用。不存在 = 每輪必須新建。"""


def sink_is_active(sink: AudioSink, *, fallback: bool = False) -> bool:
    """sink 是否正在出聲；沒有 ``is_active`` 時回傳 ``fallback``
    （coordinator 以「已 write 過首個 chunk 且未 stop」推定後傳入）。"""
    value = getattr(sink, "is_active", None)
    return fallback if value is None else bool(value)


def sink_first_frame_at(sink: AudioSink) -> float | None:
    """首個真音訊 frame 的時刻；sink 不提供則 None（呼叫端退回首次 write 完成時刻）。"""
    return getattr(sink, "first_frame_at", None)


def sink_played_evidence(sink: AudioSink) -> PlayedEvidence:
    """sink 宣告的證據等級；未宣告視為 ``TRANSPORT_SENT``（保守）。"""
    value = getattr(sink, "played_evidence", None)
    if value is None:
        return PlayedEvidence.TRANSPORT_SENT
    try:
        return PlayedEvidence(value)
    except ValueError:
        return PlayedEvidence.TRANSPORT_SENT


def sink_can_reset(sink: AudioSink) -> bool:
    """sink 是否支援重用（有 ``reset``）。沒有的 sink 停止後必須新建。"""
    return callable(getattr(sink, "reset", None))


# ─────────────────────────── 簽發與快照 ───────────────────────────


@dataclass(frozen=True, slots=True)
class TurnTicket:
    """一輪輸出的執行權憑證。ES 不自行開 turn，turn_id 取自這裡。"""

    turn_id: str
    """Core 簽發的 turn 識別碼（ES 輸出端、tracer 的帳號）。"""

    source_turn_id: str
    """對應的輸入端 id（``Utterance.turn_id``）。"""

    owner_id: str | None = None
    """這一輪回應的對象（通常是 ``Utterance.speaker_id``）。"""

    reply_mode: str | None = None
    """不透明字串，ES 不定義語意，只交給 :class:`ModeRules` 翻成機制旗標。"""

    snapshot_version: int = 0
    """簽發當下的 :class:`PolicySnapshot` 版本。"""

    issued_at: float = field(default_factory=time.perf_counter)
    meta: dict = field(default_factory=dict)

    def matches(self, utterance: Utterance) -> bool:
        """ticket 是否對應這個輸入（``source_turn_id == utterance.turn_id``）。"""
        return self.source_turn_id == utterance.turn_id


class TicketIssuer(Protocol):
    """簽發 :class:`TurnTicket`。Core 宿主由 Core 實作；Web／CLI 用 standalone 版。"""

    def issue(self, utterance: Utterance, *, reply_mode: str | None = None) -> TurnTicket: ...


@dataclass(frozen=True, slots=True)
class ModeFlags:
    """``reply_mode`` 翻譯後的機制旗標。"""

    speak: bool = True
    """False = 這一輪不出聲（例如 passive）。"""

    barge_in: bool = True
    """False = 這一輪不可被插話打斷。"""


class ModeRules(Protocol):
    """``reply_mode``（不透明字串）→ 機制旗標。同步、純計算，建立 turn 時評估一次。"""

    def flags_for(self, reply_mode: str | None) -> ModeFlags: ...


BARGE_IN_PRESETS: dict[str, float] = {
    "web": 0.4,
    "discord": 2.0,
}
"""平台 preset：barge-in 的有聲秒數門檻（Q4）。

兩者語意不同、不統一：Web 看單次事件的 ``duration_s``，Discord 是跨事件累積的有聲秒數。
"""


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """熱路徑唯一讀取的策略快照（同步、純記憶體）。"""

    version: int = 0
    platform: str = "web"
    barge_in_min_speech_s: float = BARGE_IN_PRESETS["web"]
    """連續（或累積）有聲達此秒數才算插話。"""

    barge_in_gap_reset_s: float = 0.5
    """有聲事件間隔超過此值，累積歸零。"""

    mode_rules: ModeRules | None = None
    """None = 一律 ``ModeFlags()``（出聲、可插話）。"""

    issued_at: float = field(default_factory=time.perf_counter)
    ttl_s: float | None = None
    """None = 永不過期。過期的快照由 coordinator 退回最保守行為。"""

    def is_stale(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return False
        return (time.perf_counter() if now is None else now) - self.issued_at > self.ttl_s

    def flags_for(self, reply_mode: str | None) -> ModeFlags:
        return ModeFlags() if self.mode_rules is None else self.mode_rules.flags_for(reply_mode)

    @classmethod
    def for_platform(cls, platform: str, **overrides: Any) -> PolicySnapshot:
        """依平台 preset 建立快照；未知平台拋 ``ValueError``。"""
        try:
            threshold = BARGE_IN_PRESETS[platform]
        except KeyError:
            raise ValueError(f"未知的平台 preset: {platform!r}") from None
        overrides.setdefault("barge_in_min_speech_s", threshold)
        return cls(platform=platform, **overrides)


class PolicySource(Protocol):
    """快照來源。``current()`` 必須同步且不阻塞。"""

    def current(self) -> PolicySnapshot: ...


# ─────────────────────────── 策略插槽 ───────────────────────────


class TurnView(Protocol):
    """策略看到的一輪狀態（唯讀視圖）。"""

    turn_id: str
    owner_id: str | None
    reply_mode: str | None
    audio_started: bool
    """``sink.first_frame_at is not None``。"""

    behind_playing: bool
    """前一輪正在出聲。"""

    barge_users: set[str]
    """可打斷這一輪的說話者；宿主可在生成中動態加入。"""

    meta: dict
    """宿主私有旗標。"""


class AdmissionPolicy(Protocol):
    """收件閘門：``submit`` 時決定這一輪要不要進入處理。

    政策留宿主（例：busy 且 ``waiting >= max_pending`` 且 trigger 不在 never-drop
    集合 → 拒絕）。拒絕的輪次不觸發任何生成／播放回呼，只送出
    ``status=DROPPED_BUSY`` 的 :class:`TurnOutcome`。同步、不得 I/O。
    """

    def admit(self, request: TurnRequest, busy: bool, waiting: int) -> bool: ...


class BargeInPolicy(Protocol):
    """每個有聲事件評估一次，呼叫於 STT／接收執行緒。同步、不得 I/O。"""

    def on_voiced(self, turn: TurnView, ev: VoiceEvent) -> InterruptionDecision: ...


@dataclass(frozen=True, slots=True)
class YieldAction:
    """讓位指令：對 ``turn_id`` 以 SUPERSEDED（detail ``reason``）作廢。"""

    turn_id: str
    reason: str = "yielded"


class YieldPolicy(Protocol):
    """讓位判定。同步、不得 I/O。預設實作永不讓位。"""

    def on_voiced(self, ev: VoiceEvent, live: Sequence[TurnView]) -> list[YieldAction]: ...


class SpeechTextFilter(Protocol):
    """句級文字過濾（標記剝除、PASS 偵測）。協定內容屬宿主。"""

    passed: bool
    """已偵測到 PASS：ES 關閉該輪語音輸入、不寫 sink。"""

    def feed(self, sentence_text: str) -> str | None:
        """回傳要唸的文字；None = 這句不唸。"""
        ...


class RegenPolicy(Protocol):
    """生成完成時檢查是否要丟棄重算。ES 強制沿用同一 turn_id、
    並負責次數上限與「已出聲不重算」。"""

    def refresh(self, turn: TurnView) -> Utterance | None: ...


SinkFactory = Callable[[TurnView], "AudioSink | str"]
"""每輪建立 sink；回傳字串 = 不出聲的原因（對應 ``skipped_reason``）。"""

OnSentence = Callable[[TurnView, Sentence], Awaitable[None]]
"""句子送進語音佇列**之後**才呼叫，宿主的文字送出不會延遲 TTS。"""


# ─────────────────────────── 請求與結果 ───────────────────────────


@dataclass(slots=True)
class TurnRequest:
    """宿主交給 coordinator 的一輪輸出請求。"""

    ticket: TurnTicket
    utterance: Utterance
    sink_factory: SinkFactory
    spoken_rule: SpokenRule = SpokenRule.AUTO
    commit_mode: CommitMode = CommitMode.SETTLED
    text_filter: SpeechTextFilter | None = None
    regen: RegenPolicy | None = None
    max_regenerate: int = 0
    on_sentence: OnSentence | None = None
    speech_end_at: float | None = None
    """使用者說完的 perf_counter 時刻（蓋 ``MARK_SPEECH_END``）。None = 取 utterance.ended_at。"""

    stt_done_at: float | None = None
    """STT 轉錄完成的 perf_counter 時刻。與 speech_end 是**兩個不同的點**，不可互為別名。"""

    marks: dict[str, float] = field(default_factory=dict)
    """宿主自有打點（名稱由宿主定），ES 在 start 時蓋章，不新增 MARK_* 常數。"""

    history_spoken_only: bool = False
    """Q2：EAGER 下是否於結算時 ``settle`` 成只留 spoken。預設 False = 維持現況。"""

    retract_on_yield: bool = False
    """Q3：已 eager commit 的輪次被讓位時是否撤回整輪歷史。預設 False = 維持現況。"""

    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class TurnOutcome(TurnResult):
    """coordinator 一輪的結果。全部欄位有預設值，是 :class:`TurnResult` 的加法延伸。"""

    status: OutcomeStatus = OutcomeStatus.COMPLETED
    """結束路徑；每個 submit 恰好一個 outcome。"""

    barge_in: bool = False
    superseded: bool = False
    aborted: bool = False
    queue_wait_s: float = 0.0
    regenerated: int = 0
    attempt_id: int = 1
    """第幾次生成（ES 遞增，regen 時 turn_id 不變、attempt_id +1）。"""

    snapshot_version: int = 0
    """這輪實際使用的 :class:`PolicySnapshot` 版本。"""


__all__ = [
    "BARGE_IN_PRESETS",
    "AdmissionPolicy",
    "BargeInPolicy",
    "CommitMode",
    "ModeFlags",
    "ModeRules",
    "OnSentence",
    "OutcomeStatus",
    "OptionalSinkCapabilities",
    "PlayedEvidence",
    "PolicySnapshot",
    "PolicySource",
    "RegenPolicy",
    "SentenceSpan",
    "SinkFactory",
    "SkipReason",
    "SpeechTextFilter",
    "SpokenRule",
    "TicketIssuer",
    "TurnOutcome",
    "TurnRequest",
    "TurnTicket",
    "TurnView",
    "YieldAction",
    "YieldPolicy",
    "resolve_spoken_rule",
    "sink_can_reset",
    "sink_first_frame_at",
    "sink_is_active",
    "sink_played_evidence",
]
