"""三層 Stage 的契約。

整個設計的重點：**任何一層都不回傳「完整結果」，只回傳流。**
三個 stream() 全部是 AsyncIterator，這樣下游能在上游還沒講完時就開工，
四段耗時從「相加」變成「重疊」。

契約設計上的兩個非顯而易見決策：

1. **token 是顯式參數，不是建構子注入。**
   一個 stage 實例會服務多個 turn（模型載入很貴，不可能每 turn 重建），
   但取消是 per-turn 的。所以 token 跟著 turn 走，不跟著 stage 走。

2. **SpeakStage 吃 AsyncIterator 而不是單句。**
   TTS 需要看到「還有沒有下一句」才能決定要不要保留尾音、以及做批次。
   逐句呼叫會失去這個資訊。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .cancellation import CancellationToken
from .types import AudioChunk, Sentence, Utterance


@runtime_checkable
class AudioSource(Protocol):
    """音訊來源（麥克風 / 檔案 / Discord voice receive）。

    產出 int16 PCM bytes。取樣率由實作宣告，InputStage 負責確認相容。
    """

    @property
    def sample_rate(self) -> int: ...

    def stream(self, token: CancellationToken) -> AsyncIterator[bytes]: ...


@runtime_checkable
class AudioSink(Protocol):
    """音訊輸出（喇叭 / WAV 檔 / Discord voice send）。

    ``stop()`` 必須是**同步且立即**的——barge-in 時等不了 event loop 排程。
    這也是 :meth:`CancellationToken.on_cancel` 存在的主因。
    """

    async def write(self, chunk: AudioChunk) -> None: ...

    def stop(self) -> None:
        """立刻停止播放並丟棄已排入的緩衝。同步、冪等。"""
        ...

    async def drain(self) -> None:
        """等到緩衝內的音訊全部播完。

        **正常結束時必須 await 這個**，否則 ``played_duration_s`` 會嚴重低估：
        TTS 合成通常比即時播放快（RTF < 1），管線邏輯跑完時音訊還在排隊播，
        此時去算「使用者聽到了什麼」只會拿到最前面那一兩句。

        被 barge-in 取消時**不要** drain——那時緩衝已經被 ``stop()`` 丟棄，
        剩下的本來就不該播出去。
        """
        ...

    @property
    def played_duration_s(self) -> float:
        """至今實際播出的音訊時長。

        barge-in 善後要用：據此推算「講到第幾句」，
        決定哪些文字要寫回對話歷史。
        """
        ...


@runtime_checkable
class Stage(Protocol):
    """所有 Stage 的共同生命週期。

    ``prepare()`` 與 ``aclose()`` 是 Phase 1+ 才有實質內容的
    （載入模型、啟動 worker process），Phase 0 的 fake 實作為 no-op。
    先放進契約是為了避免屆時改介面。
    """

    name: str

    async def prepare(self) -> None:
        """暖機。載入模型、啟動 worker、跑一次 dummy 推理。"""
        ...

    async def aclose(self) -> None:
        """釋放資源。必須冪等。"""
        ...


@runtime_checkable
class InputStage(Stage, Protocol):
    """音訊或文字 → 使用者的一個 turn。

    產出多個 Utterance（一個 session 有多輪對話），
    邊界由內部的 TurnDetector 決定。
    """

    def stream(
        self, source: AudioSource, token: CancellationToken
    ) -> AsyncIterator[Utterance]: ...


@runtime_checkable
class ThinkStage(Protocol):
    """turn → 逐句吐出回應。

    內部組成（§7.3 定案）::

        Utterance → SessionControl(ContextManager 壓縮) → LLM 串流 → SentenceSplitter
                          ▲                                    │
                          └────── EchoMemory(retrieve/store) ──┘

    §7.4 定案：Memory retrieve 與 LLM prefill **平行**，首句不等它。

    ## 首句一致性問題的正解（2026-08-20 調查後修訂）

    平行跑會有個副作用：首句可能與後續記憶內容矛盾（首句說「我不記得了」，
    第二句記憶回來了）。原本設想用「late-context 注入」補救——**這條路不通。**

    Gemini / Ollama / vLLM 全都是「一次 prefill + 自迴歸 decode」，KV cache
    一旦建立就無法替換已編碼的前綴。這不是產品限制，是 Transformer 自迴歸
    推論的結構性限制，沒有任何一家支援「串流生成到一半改 context」。

    真正的解法是**從源頭排除**，而非事後補救：

    1. **首句改用固定 filler**（「嗯，讓我想想」「這個問題」），刻意
       **不含任何可被推翻的記憶斷言**。注意「我不記得了」這種話是斷言，
       不能用；「讓我想一下」才是安全的。
    2. retrieve 完成後，把已說出的 filler 當作 assistant 前綴放進 prompt，
       **開新請求**續寫真正有記憶依據的內容。

    這樣就沒有「第一句錯了要被第二句推翻」的問題——第一句本來就不含
    可能被推翻的東西。實作成本低（只改 prompt 策略，不動 Memory /
    SessionControl 架構），且直接切斷問題根源，比生成後再做一致性檢查
    （那要多一輪 LLM 判斷，反而增加延遲）便宜得多。

    中期優化方向是 speculative retrieval：使用者還在講話時就用 partial
    transcript 先 retrieve，爭取時間差讓記憶趕上首句。

    ## 壓縮的競態陷阱

    SessionControl 的上下文壓縮**必須背景跑、下一輪才生效**，絕對不能
    採用「先清空原始內容再等 LLM 回摘要」的模式。Claude Code（#40352）、
    Codex（#13946）都踩過這個坑：API 失敗或中途有新訊息進來時，
    原始對話被永久吞掉。正確順序是「保留原始 → 背景算 → 算完才原子性替換
    → 中途變髒就作廢重算」。
    """

    name: str

    async def prepare(self) -> None: ...

    async def aclose(self) -> None: ...

    def stream(
        self, utterance: Utterance, token: CancellationToken
    ) -> AsyncIterator[Sentence]: ...

    async def commit(self, turn_id: str, spoken_text: str, generated_text: str) -> None:
        """turn 結束後回寫對話歷史與長期記憶。

        **``spoken_text`` 才是寫進對話歷史的東西**，不是 generated_text。
        被 barge-in 中止時兩者不同，寫錯會讓 LLM 以為自己講完了整段。
        """
        ...


@runtime_checkable
class SpeakStage(Protocol):
    """句子流 → 音訊塊流。

    實作要點（Phase 1 接 IndexTTS2 時會撞到）：真 TTS 後端是
    **同步 push callback**（``on_segment_audio(tensor, idx, total)``），
    這裡的契約是 **async pull**。adapter 必須做 push→pull 橋接——
    在 executor 執行緒跑同步推理，callback 用
    ``loop.call_soon_threadsafe`` 把 chunk 塞進有界 queue，
    這裡再從 queue 拉。有界是重點：無界 queue 會讓 TTS 一路合成到底，
    barge-in 就白做了（已經算完的東西丟掉，GPU 時間全浪費）。
    """

    name: str

    async def prepare(self) -> None: ...

    async def aclose(self) -> None: ...

    def stream(
        self, sentences: AsyncIterator[Sentence], token: CancellationToken
    ) -> AsyncIterator[AudioChunk]: ...
