"""SentenceSplitter——把 LLM 的 token 流切成可合成的句子。

## 這是整個管線最大的槓桿

不是「每一層都做成串流」，而是**在句子邊界讓各層重疊**。LLM 吐出第一句就
立刻送 TTS，LLM 生成第二句的時間被 TTS 合成第一句吃掉——兩層重疊而非相加。

## 為什麼是「切句子」而不是更細的粒度

調查過 CosyVoice 2/3、Orpheus、FlashTTS 這些宣稱 150-325ms 首包的方案後，
結論很明確：**它們能「不等句子邊界」是模型架構層級的能力，不是推論期的
wrapper。**

CosyVoice 2 的 chunk-aware causal flow matching 靠的是 text token 與
speech token 在訓練時就按約 5:15 比例交錯排列（text-speech interleaved LM），
所以文字還在進來時就能吐音訊。Orpheus 是 SNAC-frame-level 逐 token 解碼。
FlashTTS 是 lagged multi-track + MTP。三者都要求模型本身就是流式訓練的。

IndexTTS 2.5 是「先生成完整 semantic token 序列 → DiT + vocoder」的架構，
沒有交錯生成的基礎。換模型的代價遠高於收益（會丟掉情感向量系統這個既有資產）。

而且連專門研究更細切法的 Prosodic Boundary-Aware Streaming 論文，結論也是
「智能切分」而非「完全不切」。F5-TTS、XTTS 這些架構相近的模型，官方推薦
做法同樣是句子級 chunk streaming。

**所以句子級切分就是串接式管線的正解。** 可優化的是切點怎麼選，不是要不要切。

## 核心張力

**切太細，語調破碎；切太粗，TTFA 拉長。**

解法是首句與後續用不同閾值：首句只求快（越短越好，反正通常是承接語），
後續放寬求品質。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..contracts.cancellation import CancellationToken
from ..contracts.types import Sentence

# --- 標點分類 ---

TERMINAL_PUNCT_FULLWIDTH = frozenset("。！？；…")
"""全形句末標點。全形標點不會出現在數字或縮寫裡，可以**立即切**，不需要 lookahead。"""

TERMINAL_PUNCT_ASCII = frozenset(".!?")
"""半形句末標點。有歧義（3.14、Mr.、U.S.A.），需要看下一個字元才能決定。"""

SECONDARY_PUNCT = frozenset("，、,:：")
"""次級標點。累積長度夠時可以當切點——這是「韻律邊界」的簡化版：
自然停頓處切分比在強母音區硬切，銜接假影小得多。"""

CLOSING_PUNCT = frozenset("」』）\")'】》>")
"""結尾引號括號。句末標點後面若接這些，要一起帶走再切。"""

_NO_SPLIT_AFTER = frozenset("0123456789")
"""句末標點後面接數字 → 是小數點，不切。"""


def char_weight(ch: str) -> float:
    """字元的「發音時長權重」。

    切分閾值真正代表的是**合成後的音訊長度**，不是字元數。中文一個字約一個
    音節，英文要好幾個字母才一個音節。用同一組閾值套中英混合文字，
    不做換算就會讓英文句子被切得極短。

    權重取 CJK=1.0、拉丁字母=0.4（約 2.5 個字母 ≈ 一個中文字的時長），
    標點與空白 0。這是估算值，調參時要用聽感校正。
    """
    code = ord(ch)
    # CJK 統一表意文字 + 日文假名 + 韓文
    if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
        return 1.0
    if ch.isalnum():
        return 0.4
    return 0.0


def text_weight(text: str) -> float:
    """整段文字的發音時長權重。"""
    return sum(char_weight(c) for c in text)


@dataclass(slots=True)
class SplitPolicy:
    """切分參數。

    預設值是**起點不是定論**——實際值要用聽感測，就像調情緒強度那樣。
    調參時看 :attr:`Sentence.split_reason` 的分布：``max_length`` 佔比過高
    代表 LLM 在生成長句，閾值需要調整。
    """

    first_min_weight: float = 8.0
    """首句的最小長度。低到 8 是因為首句通常是承接語（「嗯，我想想」），
    切短對語調傷害小，但對 TTFA 幫助極大。"""

    first_max_weight: float = 20.0
    """首句的強制切分長度。首句拖長 = TTFA 直接受害，所以上限壓得比後續緊。"""

    min_weight: float = 12.0
    """後續句子在次級標點處切分的最小長度。"""

    max_weight: float = 40.0
    """後續句子的強制切分長度。超過就算在句子中間也要切——
    一個 80 字的長句會讓 TTS 卡住好幾秒，後面全部延遲。"""

    allow_secondary_split: bool = True
    """是否允許在逗號等次級標點切分。關掉可用來比較「只在句末切」的音質差異。"""

    metadata: dict = field(default_factory=dict)


class SentenceSplitter:
    """把 token 流切成 :class:`Sentence` 流。

    用法::

        splitter = SentenceSplitter(policy)
        async for sentence in splitter.split(token_stream, turn_id, token):
            ...

    實例是**有狀態**的（buffer、句序號），一個 turn 用一個實例。
    """

    def __init__(self, policy: SplitPolicy | None = None) -> None:
        self.policy = policy or SplitPolicy()

    async def split(
        self,
        tokens: AsyncIterator[str],
        turn_id: str,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[Sentence]:
        """消費 token 流，逐句產出。

        取消時直接停止產出——**不 flush 剩餘 buffer**。被 barge-in 中止時，
        使用者已經在講話了，把殘句補送去合成只會讓機器人多講半句廢話。
        """
        buffer = ""
        index = 0
        # 半形句末標點的 lookahead 狀態：記住「上次看到 ASCII 標點的位置」，
        # 等下一個字元進來才能判斷那是句末還是小數點
        pending_ascii_at = -1

        async for tok in tokens:
            if token is not None:
                token.raise_if_cancelled()
            if not tok:
                continue

            for ch in tok:
                buffer += ch

                # 1. 先處理待決的半形標點（已經看到下一個字元了）
                if pending_ascii_at >= 0:
                    # 這個 ch 就是標點後的下一個字元
                    if ch in _NO_SPLIT_AFTER or ch in TERMINAL_PUNCT_ASCII:
                        # 小數點或省略號 → 不是句末
                        pending_ascii_at = -1
                    else:
                        cut = pending_ascii_at + 1
                        # 把結尾引號一起帶走
                        while cut < len(buffer) and buffer[cut] in CLOSING_PUNCT:
                            cut += 1
                        head, buffer = buffer[:cut], buffer[cut:]
                        pending_ascii_at = -1
                        if head.strip():
                            yield self._make(head, turn_id, index, "terminal")
                            index += 1
                        continue

                weight = text_weight(buffer)
                is_first = index == 0
                max_w = (
                    self.policy.first_max_weight if is_first else self.policy.max_weight
                )
                min_w = (
                    self.policy.first_min_weight if is_first else self.policy.min_weight
                )

                # 2. 全形句末標點：可以立即切（不會有小數點歧義）
                if ch in TERMINAL_PUNCT_FULLWIDTH:
                    head, buffer = buffer, ""
                    if head.strip():
                        yield self._make(head, turn_id, index, "terminal")
                        index += 1
                    continue

                # 3. 半形句末標點：進入待決狀態，等下一個字元
                if ch in TERMINAL_PUNCT_ASCII:
                    pending_ascii_at = len(buffer) - 1
                    continue

                # 4. 次級標點 + 長度達標
                if (
                    self.policy.allow_secondary_split
                    and ch in SECONDARY_PUNCT
                    and weight >= min_w
                ):
                    head, buffer = buffer, ""
                    if head.strip():
                        yield self._make(head, turn_id, index, "secondary")
                        index += 1
                    continue

                # 5. 長度上限強制切
                if weight >= max_w:
                    head, buffer = self._force_cut(buffer)
                    if head.strip():
                        yield self._make(head, turn_id, index, "max_length")
                        index += 1

        # token 流正常結束 → flush 殘餘
        if token is not None and token.is_cancelled:
            return
        if buffer.strip():
            yield self._make(buffer, turn_id, index, "flush", is_last=True)
        elif index > 0:
            # 沒有殘餘，但要讓下游知道結束了。用零長度的收尾句標記。
            yield self._make("", turn_id, index, "flush", is_last=True)

    def _force_cut(self, buffer: str) -> tuple[str, str]:
        """長度超限時的切點選擇。

        優先回溯到最近的次級標點或空白處——**在自然停頓處切分，銜接假影
        遠小於在強母音中間硬切**。回溯範圍限制在尾端 40%，避免切出過短的頭段。
        """
        limit = int(len(buffer) * 0.6)
        for i in range(len(buffer) - 1, limit, -1):
            if buffer[i] in SECONDARY_PUNCT or buffer[i].isspace():
                return buffer[: i + 1], buffer[i + 1 :]
        return buffer, ""

    @staticmethod
    def _make(
        text: str,
        turn_id: str,
        index: int,
        reason: str,
        is_last: bool = False,
    ) -> Sentence:
        return Sentence(
            text=text.strip(),
            turn_id=turn_id,
            index=index,
            is_first=index == 0,
            is_last=is_last,
            split_reason=reason,
        )
