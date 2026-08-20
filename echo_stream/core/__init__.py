"""Pipeline Core——串接、背壓、取消、量測。

與 :mod:`echo_stream.contracts` 一樣，**不 import 任何子系統**。
"""

from .channel import ChannelClosed, StreamChannel
from .pipeline import PipelineRunner
from .splitter import SentenceSplitter, SplitPolicy, char_weight, text_weight
from .tracer import BUDGET_MS, LatencyTracer, TurnTrace
from .turn_detector import (
    AlwaysAddressed,
    DurationInterruptionDetector,
    SilenceTurnDetector,
)

__all__ = [
    "ChannelClosed",
    "StreamChannel",
    "PipelineRunner",
    "SentenceSplitter",
    "SplitPolicy",
    "char_weight",
    "text_weight",
    "BUDGET_MS",
    "LatencyTracer",
    "TurnTrace",
    "AlwaysAddressed",
    "DurationInterruptionDetector",
    "SilenceTurnDetector",
]
