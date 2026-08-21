"""情緒標記解析（core/emotion.py）的測試。

重點是**壞輸入不炸管線**：LLM 的標記格式不受我們控制，
解析失敗的正確行為是靜默剝除、退回預設。
"""

from __future__ import annotations

from echo_stream.contracts.style import SpeechStyle
from echo_stream.core.emotion import EMOTION_PRESETS, MARKER_PROMPT, extract_style

# --- 基本解析 ---


def test_認得的標記轉成_style_並剝除():
    text, style = extract_style("[開心]太好了，成功了！")
    assert text == "太好了，成功了！"
    assert style is not None
    assert style.emotion == EMOTION_PRESETS["開心"]


def test_無標記時回傳原文與_None():
    text, style = extract_style("今天天氣不錯。")
    assert text == "今天天氣不錯。"
    assert style is None


def test_不認得的標記剝除但不產生_style():
    # sanitize_for_tts 只剝括號符號、保留內容——這裡不剝掉的話
    # 「大笑」兩個字會被唸出來
    text, style = extract_style("[大笑]真的假的？")
    assert text == "真的假的？"
    assert style is None


def test_多個標記取第一個認得的():
    text, style = extract_style("[大笑][難過]怎麼會這樣…")
    assert text == "怎麼會這樣…"
    assert style is not None
    assert style.emotion == EMOTION_PRESETS["難過"]


def test_句中標記也會被剝除():
    text, style = extract_style("我本來以為沒事，[驚訝]結果嚇死。")
    assert text == "我本來以為沒事，結果嚇死。"
    assert style.emotion == EMOTION_PRESETS["驚訝"]


# --- 與 base style 的關係 ---


def test_base_的語速與強度被保留():
    base = SpeechStyle(speed=1.2, intensity=0.9)
    _, style = extract_style("[生氣]夠了！", base)
    assert style.speed == 1.2
    assert style.intensity == 0.9
    assert style.emotion == EMOTION_PRESETS["生氣"]


def test_base_的情緒被標記覆蓋而非合併():
    base = SpeechStyle(emotion={"calm": 0.8})
    _, style = extract_style("[開心]好耶！", base)
    assert style.emotion == EMOTION_PRESETS["開心"]


# --- 邊界情況（不能炸） ---


def test_過長的括號內容不視為標記():
    text, style = extract_style("[這不是一個情緒標記]內容照舊。")
    assert text == "[這不是一個情緒標記]內容照舊。"
    assert style is None


def test_含空白的括號內容不視為標記():
    text, style = extract_style("[not marker]內容照舊。")
    assert text == "[not marker]內容照舊。"
    assert style is None


def test_空字串():
    text, style = extract_style("")
    assert text == ""
    assert style is None


def test_只有標記的句子剝完是空字串():
    text, style = extract_style("[開心]")
    assert text == ""
    assert style is not None


# --- prompt ---


def test_prompt_列出所有_preset():
    for name in EMOTION_PRESETS:
        assert name in MARKER_PROMPT
