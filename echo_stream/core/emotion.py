"""逐句情緒控制——LLM 情緒標記 → SpeechStyle（Speech Planner 起步版）。

## 這一層在解什麼問題

現況是每句都用角色檔的預設情緒向量，聽起來像機器人。基礎設施早就在：
:attr:`~echo_stream.contracts.types.Sentence.style` → adapter 的
``_apply_style`` 翻譯成後端原生控制。缺的是**產生** style 的那一端。

起步做法：讓 LLM 在回應裡夾帶輕量標記（``[開心]``），ThinkStage 逐句
解析成 :class:`~echo_stream.contracts.style.SpeechStyle`，標記本身從文字
剔除。完整的 Speech Planner 設計見 docs/design/tts-improvement-reference.md。

## 為什麼是「剝掉所有短標記」而不是只剝認得的

``sanitize_for_tts`` 會剔除方括號**符號**但保留內容——LLM 若寫了不在
表裡的 ``[大笑]``，不剝掉就會被唸成「大笑」。所以策略是：

* 認得的標記 → 轉成 style，剝掉
* 不認得的短標記 → 靜默剝掉，退回預設 style（**不炸管線**）

正常語音內容不該出現方括號（sanitize 本來就會清掉符號），
誤殺合法內容的風險可忽略。
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..contracts.style import SpeechStyle

EMOTION_PRESETS: dict[str, dict[str, float]] = {
    "開心": {"happy": 0.8},
    "興奮": {"happy": 0.7, "surprised": 0.4},
    "難過": {"sad": 0.8},
    "生氣": {"angry": 0.8},
    "害怕": {"afraid": 0.7},
    "厭惡": {"disgusted": 0.7},
    "憂鬱": {"melancholic": 0.7},
    "驚訝": {"surprised": 0.8},
    "平靜": {"calm": 0.8},
    "溫柔": {"calm": 0.5, "happy": 0.3},
}
"""標記名 → 八維情緒分佈。

值刻意不頂滿 1.0——引擎的 ``normalize_emotion_vector`` 會把總和壓到
``max_strength``，但在那之前保留一點餘裕讓 intensity 還有調整空間。
Web 設定面板的 preset 下拉也吃這張表，兩邊共用一份定義。
"""

MARKER_PROMPT = (
    "【語音情緒標記】\n"
    "你的回覆會被轉成語音。當某句話的情緒有明顯轉折時，在該句開頭加上"
    "情緒標記，格式是方括號包住標記名，例如「[開心]太好了！」。\n"
    f"可用標記：{'、'.join(EMOTION_PRESETS)}。\n"
    "不需要每句都加——只在情緒改變時標，平鋪直敘不用。"
    "除了情緒標記之外，回覆中不要使用方括號。\n"
    "回覆是口語對話：不要使用 markdown 格式（列表符號、標題、粗體），"
    "列舉事情時用自然的語句串接。"
)
"""接在人設之後的 system prompt 補充。人設是基底，這是本功能的附加指示。"""

_MARKER_RE = re.compile(r"\[([^\[\]\s]{1,6})\]")
"""短標記：方括號包 1-6 個無空白字元。放寬到任意長度會開始誤殺正常內容。"""


def extract_style(
    text: str, base: SpeechStyle | None = None
) -> tuple[str, SpeechStyle | None]:
    """剝掉文字裡的情緒標記，回傳（乾淨文字, style 或 None）。

    ``base`` 是 turn 級的預設 style（例如設定面板的覆寫）——標記只換掉
    **情緒分佈**，語速、強度沿用 base，兩層控制才不會互相打架。

    一句裡有多個標記時取第一個認得的；解析不出任何認得的標記就回 None，
    由呼叫端退回預設。**這個函式不拋例外**——標記格式錯誤不能炸管線。
    """
    style: SpeechStyle | None = None

    def _consume(match: re.Match[str]) -> str:
        nonlocal style
        preset = EMOTION_PRESETS.get(match.group(1))
        if preset is not None and style is None:
            style = (
                replace(base, emotion=dict(preset))
                if base is not None
                else SpeechStyle(emotion=dict(preset))
            )
        return ""  # 認不認得都剝掉——見模組 docstring

    cleaned = _MARKER_RE.sub(_consume, text).strip()
    return cleaned, style


__all__ = ["EMOTION_PRESETS", "MARKER_PROMPT", "extract_style"]
