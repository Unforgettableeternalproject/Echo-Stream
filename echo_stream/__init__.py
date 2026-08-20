"""Echo Stream——串流語音對話管線整合層。

## 這個專案做什麼

把五個獨立的子系統（echo_stt / echo_tts / echo_memory / session control / LLM）
串成一條低延遲的對話管線。目標是 **TTFA < 2.5s**。

## 這個專案不做什麼

* ❌ 不實作 STT / LLM / TTS / Memory 的任何推理邏輯
* ❌ 不做 Discord 的功能（權限、指令、presence）
* ❌ 不做人設、prompt 工程（那是 Session Control 的事）
* ❌ 不做 UI
* ✅ 只做：契約、串接、背壓、取消、量測

範圍限制是刻意的。這個專案最大的風險是變成第六個孤島——一旦開始實作推理
邏輯，它就從中立的整合層退化成又一個耦合點。所有實際能力都來自 adapter
包裝的既有 repo。

## 核心洞察

上次 Discord 慢（2026-06-22 實測 30-40s）的根因**不是「沒有串流」，
是每一層都等前一層 100% 完成**，四段耗時直接相加。

解法不是「每層都做成串流」，而是**在句子邊界讓各層重疊**：LLM 吐出第一句
就立刻送 TTS，LLM 生成第二句的時間被 TTS 合成第一句吃掉。
"""

from .contracts import (
    AudioChunk,
    CancellationToken,
    CancelledError,
    CancelReason,
    Sentence,
    TurnPhase,
    TurnPolicy,
    TurnResult,
    Utterance,
)
from .core import LatencyTracer, PipelineRunner, SentenceSplitter, SplitPolicy

__version__ = "0.1.0"

__all__ = [
    "AudioChunk",
    "CancellationToken",
    "CancelledError",
    "CancelReason",
    "Sentence",
    "TurnPhase",
    "TurnPolicy",
    "TurnResult",
    "Utterance",
    "LatencyTracer",
    "PipelineRunner",
    "SentenceSplitter",
    "SplitPolicy",
    "__version__",
]
