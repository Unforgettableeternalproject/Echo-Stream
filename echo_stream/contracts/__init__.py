"""Pipeline 契約層。

這個套件**不 import 任何子系統**（echo_stt / echo_tts / echo_memory /
session control），也不含任何推理邏輯。所有能力都由 adapter 在邊界包裝。
這是「不變成第六個孤島」的具體防線——契約層一旦開始 import 子系統，
這個專案就從整合層退化成又一個耦合點。
"""

from .cancellation import CancellationToken, CancelledError, CancelReason
from .stages import AudioSink, AudioSource, InputStage, SpeakStage, Stage, ThinkStage
from .style import EMOTION_DIMENSIONS, NEUTRAL, SpeechStyle
from .turn import (
    AddressingClassifier,
    InterruptionDecision,
    InterruptionDetector,
    TurnDecision,
    TurnDetector,
    TurnPolicy,
    VoiceEvent,
    VoiceState,
)
from .types import (
    DISCORD_SAMPLE_RATE,
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    AudioChunk,
    Sentence,
    TurnPhase,
    TurnResult,
    Utterance,
    new_turn_id,
)

__all__ = [
    # cancellation
    "CancellationToken",
    "CancelledError",
    "CancelReason",
    # stages
    "AudioSink",
    "AudioSource",
    "InputStage",
    "SpeakStage",
    "Stage",
    "ThinkStage",
    # style
    "EMOTION_DIMENSIONS",
    "NEUTRAL",
    "SpeechStyle",
    # turn
    "AddressingClassifier",
    "InterruptionDecision",
    "InterruptionDetector",
    "TurnDecision",
    "TurnDetector",
    "TurnPolicy",
    "VoiceEvent",
    "VoiceState",
    # types
    "AudioChunk",
    "Sentence",
    "TurnPhase",
    "TurnResult",
    "Utterance",
    "new_turn_id",
    "STT_SAMPLE_RATE",
    "TTS_SAMPLE_RATE",
    "DISCORD_SAMPLE_RATE",
]
