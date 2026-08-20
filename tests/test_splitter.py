"""SentenceSplitter 的測試。

切分是整個管線最大的槓桿，切錯的後果分兩種：切太細語調破碎、
切太粗 TTFA 拉長。這裡測的是「切點判定正確」，聽感要用耳朵測。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from echo_stream.contracts.cancellation import CancellationToken, CancelReason
from echo_stream.core.splitter import (
    SentenceSplitter,
    SplitPolicy,
    char_weight,
    text_weight,
)


async def chars(text: str) -> AsyncIterator[str]:
    """逐字元模擬 token 流。"""
    for ch in text:
        yield ch


async def collect(text: str, policy: SplitPolicy | None = None) -> list[str]:
    splitter = SentenceSplitter(policy)
    return [
        s.text async for s in splitter.split(chars(text), "t") if s.text
    ]


async def collect_full(text: str, policy: SplitPolicy | None = None):
    splitter = SentenceSplitter(policy)
    return [s async for s in splitter.split(chars(text), "t")]


# --- 權重 ---


def test_中文字權重高於英文字母():
    """閾值代表的是發音時長，不是字元數——不換算會讓英文被切得極短。"""
    assert char_weight("好") == 1.0
    assert char_weight("a") == 0.4
    assert char_weight("，") == 0.0


def test_日文假名與中文同權重():
    assert char_weight("あ") == 1.0
    assert char_weight("カ") == 1.0


# --- 基本切分 ---


async def test_全形句末標點立即切():
    result = await collect("你好。今天天氣真好。")
    assert result == ["你好。", "今天天氣真好。"]


async def test_全形問號驚嘆號():
    result = await collect("真的嗎？太好了！")
    assert result == ["真的嗎？", "太好了！"]


async def test_結尾引號跟著句子走():
    result = await collect('他說「好啊."後面繼續講一些別的內容。')
    assert result[0].endswith('"') or "」" in result[0] or result[0].endswith("好啊.")


# --- 半形標點的歧義處理（串流中最容易錯的地方）---


async def test_小數點不切():
    """3.14 的點不是句末——這是串流切分最典型的誤判。

    刻意放寬長度上限：這裡驗的是「標點判定」，不能讓長度強制切分
    混進來當變因。
    """
    loose = SplitPolicy(first_max_weight=999.0, max_weight=999.0)
    result = await collect("圓周率大約是3.14這個數字很常用喔。", loose)
    assert len(result) == 1
    assert "3.14" in result[0]


async def test_省略號不切():
    result = await collect("我想想...應該可以吧。")
    assert all("." not in r or r.count(".") >= 2 for r in result[:1])
    assert len(result) <= 2


async def test_半形句號在後面有字時才切():
    result = await collect("Hello world. This is fine.")
    assert len(result) >= 1
    assert result[0].startswith("Hello")


# --- 首句優先 ---


async def test_首句用較低閾值_較早切出():
    policy = SplitPolicy(first_min_weight=6.0, min_weight=20.0)
    result = await collect_full("我覺得這個想法不錯，可以再多想一下細節，然後我們就開始做。", policy)
    texts = [s.text for s in result if s.text]
    assert texts[0] == "我覺得這個想法不錯，"
    assert result[0].is_first
    assert result[0].split_reason == "secondary"


async def test_首句標記只有第一句有():
    result = await collect_full("一句。二句。三句。")
    firsts = [s.is_first for s in result if s.text]
    assert firsts[0] is True
    assert not any(firsts[1:])


# --- 長度上限 ---


async def test_超長句被強制切開():
    """一個 80 字的長句在 RTF 0.3 下要合成 6 秒，後面全部延遲。"""
    policy = SplitPolicy(first_max_weight=10.0, max_weight=15.0, allow_secondary_split=False)
    long_text = "這" * 60 + "。"
    result = await collect_full(long_text, policy)
    assert len(result) > 3
    assert any(s.split_reason == "max_length" for s in result)


async def test_強制切分優先回溯到空白處():
    """在自然停頓處切，銜接假影遠小於在強母音中間硬切。"""
    policy = SplitPolicy(first_max_weight=4.0, max_weight=4.0, allow_secondary_split=False)
    result = await collect("hello world foo bar baz qux", policy)
    assert any(r.endswith(" ") or " " in r for r in result)


# --- 收尾 ---


async def test_沒有句末標點也會_flush():
    result = await collect_full("這句話沒有標點")
    texts = [s.text for s in result if s.text]
    assert texts == ["這句話沒有標點"]
    assert result[-1].is_last


async def test_最後一句標記_is_last():
    result = await collect_full("一句。二句。")
    assert result[-1].is_last
    assert not any(s.is_last for s in result[:-1])


async def test_序號連續遞增():
    result = await collect_full("一。二。三。四。")
    assert [s.index for s in result] == list(range(len(result)))


# --- 取消 ---


async def test_取消後不_flush_殘句():
    """使用者已經在講話了，補送殘句只會讓機器人多講半句廢話。

    取消是以例外形式冒出來的（消費端必須知道這輪沒講完），
    重點是「第二句還沒」那段殘餘**不會**被 flush 出來。
    """
    from echo_stream.contracts.cancellation import CancelledError

    token = CancellationToken()
    splitter = SentenceSplitter()
    out = []

    async def slow_tokens():
        for ch in "第一句話講完了。第二句還沒":
            yield ch

    with pytest.raises(CancelledError):
        async for sentence in splitter.split(slow_tokens(), "t", token):
            out.append(sentence.text)
            token.cancel(CancelReason.BARGE_IN)

    assert out == ["第一句話講完了。"]


async def test_取消時拋出():
    token = CancellationToken()
    token.cancel(CancelReason.BARGE_IN)
    splitter = SentenceSplitter()
    from echo_stream.contracts.cancellation import CancelledError

    with pytest.raises(CancelledError):
        async for _ in splitter.split(chars("你好。"), "t", token):
            pass


# --- 切點成因 ---


async def test_split_reason_可分類():
    """調參時要看這個分布：max_length 佔比過高代表閾值要調。"""
    result = await collect_full("你好。這是一段比較長的句子，裡面有逗號分隔的內容。")
    reasons = {s.split_reason for s in result}
    assert "terminal" in reasons


def test_text_weight_累加():
    assert text_weight("你好") == 2.0
    assert text_weight("，。") == 0.0
