"""Memory adapter：Echo Memory（Phase 4a）。

**這是唯一 import echo_memory 的地方**，其餘程式碼只認
:class:`EchoMemoryAdapter` 的三個方法（retrieve / store / set_session）。

## 為什麼是 adapter 而不是直接用 MemoryEngine

echo_memory 是獨立子系統（EcphoryRAG + Dream Engine + 神經化學狀態），
API 是同步的、回傳自己的 dataclass。Echo Stream 的管線是 asyncio，
think stage 只需要「一段可以塞進 prompt 的文字」。adapter 負責：

* 同步 → 非同步（``run_in_executor``，不擋 event loop）
* ``MemoryContext`` → 截斷過的純文字（token 預算是管線的責任）
* session 邊界（``exclude_session_id`` 防自我回聲）

## ⚠️ 分支依賴

需要 echo_memory repo 在 ``feature/agent-memory-spike`` 分支——
``namespace_id`` / ``exclude_session_id`` / ``llm_fn`` 這些 API
**master 上沒有**（2026-08-22 確認）。build_engine 會驗 API 形狀，
缺了就給出可行動的錯誤而不是深處的 TypeError。

## 設計決策（見 docs/hidden/phase4-memory-integration-plan.md）

* **namespace 固定、跨會話**——記憶跨會話是它存在的意義。
  session_id 只用於 episode 標記與 exclude。
* **store 是旁路**：失敗 log 一筆，不炸管線。
* **retrieve 也是旁路**：與 LLM prefill 平行跑（§7.4），結果下一輪生效。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import config

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "echo-stream"
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHARS = 500
"""注入 prompt 的記憶文字上限。記憶放在 user message 內（保 prompt cache），
太長會吃掉 token 預算、也會影響切分行為——先保守，profiling 後再調。"""
DEFAULT_MIN_MATCH = 0.45
ASSESS_MODES = ("llm", "heuristic", "off")
"""salience 評估模式（ECHO_STREAM_MEMORY_ASSESS）：
llm = SubconsciousAssessor 打一次 LLM（背景，不在熱路徑）；
heuristic = echo_memory 的文字啟發式，零 token；off = 全部 0.5（Phase 4a 行為）。"""
"""檢索相似度下限。echo_memory 預設 0.3，實測 35-41% 的匹配是純雜訊
（「攀岩」撈出「深藍色」），注入只會讓 LLM 亂講。"""


def build_engine(
    *,
    data_path: str | Path | None = None,
    namespace_id: str | None = None,
    llm_fn: Callable[[str], str] | None = None,
    memory_repo: str | None = None,
) -> Any:
    """從 echo_memory repo 建一個 MemoryEngine。

    embedding 走本地 bge-small-zh、CPU——不跟 GPU 上的 STT/TTS 搶資源
    （艾斯維爾 2026-08-21 定調）。
    """
    repo = memory_repo or config.get("ECHO_STREAM_MEMORY_REPO")
    config.ensure_importable(
        Path(repo).expanduser().resolve() if repo else config.subsystem_path("MEMORY")
    )

    # echo_memory 自帶的 APScheduler 是 lazy + 無差別的（見 core/dream_scheduler.py），
    # 觸發時機改由 Echo Stream 的 DreamScheduler 決定。其 Config 在 import 時讀 env，
    # 所以要在 import 前設；已 import 過則直接改 config 物件。setdefault 保留人工覆寫。
    import os

    os.environ.setdefault("DREAM_ENABLE_IDLE", "false")
    os.environ.setdefault("DREAM_ENABLE_CRON", "false")

    from echo_memory import MemoryEngine

    try:
        from echo_memory.config import get_config

        cfg = get_config()
        cfg.dream_enable_idle = os.environ["DREAM_ENABLE_IDLE"].lower() == "true"
        cfg.dream_enable_cron = os.environ["DREAM_ENABLE_CRON"].lower() == "true"
    except Exception:  # noqa: BLE001 - 舊版沒有 get_config 就算了
        pass

    params = inspect.signature(MemoryEngine.__init__).parameters
    if "namespace_id" not in params:
        raise RuntimeError(
            "echo_memory 的 MemoryEngine 沒有 namespace_id 參數——"
            "repo 不在 feature/agent-memory-spike 分支上。\n"
            "到 TestSeperateMemorySystem 執行："
            "git checkout feature/agent-memory-spike"
        )

    data = Path(
        data_path
        or config.get("ECHO_STREAM_MEMORY_DATA")
        or Path(__file__).resolve().parent.parent.parent / "data" / "memory"
    )
    return MemoryEngine(
        storage_path=data,
        device="cpu",
        llm_fn=llm_fn,
        namespace_id=namespace_id
        or config.get("ECHO_STREAM_MEMORY_NAMESPACE", DEFAULT_NAMESPACE),
    )


class EchoMemoryAdapter:
    """把 MemoryEngine 包成管線期望的形狀。

    stage 實例服務多個 turn；session 邊界用 :meth:`set_session` 更新，
    與 SttInputStage / SessionStore 的會話生命週期同步。
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        top_k: int | None = None,
        max_chars: int | None = None,
        llm_fn: Callable[[str], str] | None = None,
        assess_llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._engine = engine
        self._llm_fn = llm_fn
        self._assess_llm_fn = assess_llm_fn
        """評估用 LLM。五維打分不需要 reasoning，server 會給一個壓低 effort 的版本；
        沒給就退回 dream 用的 llm_fn。"""
        self._assessor: Any = None
        self.assess_mode = config.get("ECHO_STREAM_MEMORY_ASSESS", "llm") or "llm"
        if self.assess_mode not in ASSESS_MODES:
            logger.warning("ECHO_STREAM_MEMORY_ASSESS=%r 不認得，改用 llm", self.assess_mode)
            self.assess_mode = "llm"
        self._contexts: dict[str, Any] = {}
        """retrieve 的 MemoryContext 暫存（key = user text），給同一輪的 store
        評估 novelty / dream_resonance 用。retrieve 與 store 之間隔一整輪對話，
        所以不能只留「最後一個」。"""
        self.top_k = top_k or int(
            config.get("ECHO_STREAM_MEMORY_TOP_K", str(DEFAULT_TOP_K))
        )
        self.max_chars = max_chars or int(
            config.get("ECHO_STREAM_MEMORY_MAX_CHARS", str(DEFAULT_MAX_CHARS))
        )
        self.min_match = float(
            config.get("ECHO_STREAM_MEMORY_MIN_MATCH", str(DEFAULT_MIN_MATCH))
        )
        self._session_id: str | None = None
        self.load_seconds: float | None = None

    @property
    def engine(self) -> Any:
        if self._engine is None:
            import time

            t0 = time.perf_counter()
            self._engine = build_engine(llm_fn=self._llm_fn)
            self.load_seconds = time.perf_counter() - t0
        return self._engine

    async def prepare(self) -> None:
        """提早建立 engine——設定錯誤在暖機階段就爆，而不是第一次 retrieve。

        embedding 模型（sentence-transformers）的首載也發生在這裡之後的
        第一次呼叫；真正的冷啟成本用 memory-check 量。
        """
        await asyncio.get_running_loop().run_in_executor(None, lambda: self.engine)

    def set_session(self, session_id: str | None) -> None:
        """換會話。之後的 retrieve 會排除這個會話的 episodes（防自我回聲），
        store 會把 episodes 標上這個 session_id。"""
        self._session_id = session_id

    # --- 管線介面 ---

    async def retrieve(self, text: str) -> str | None:
        """檢索相關記憶，回傳可塞進 prompt 的文字；沒撈到就 None。

        同步引擎丟 executor——EcphoryRAG 的向量搜尋 + 圖擴散是 CPU 工作，
        不能讓它擋住 event loop 上正在流的音訊。
        """
        if not text or not text.strip():
            return None

        def _work() -> str | None:
            context = self.engine.retrieve(
                text,
                top_k=self.top_k,
                min_similarity=self.min_match,
                exclude_session_id=self._session_id,
            )
            # 第二道門檻：EcphoryRAG 圖擴散出來的 episode 在 similarity_to_query
            # 回 None 時不會被 min_similarity 擋（2026-08-22 實測注入了 30-36%
            # 的東西）。顯示給 LLM 的匹配度必須是真的過了門檻的。
            sims = getattr(context, "similarities", {}) or {}
            kept = [
                ep for ep in context.episodes
                if sims.get(ep.id, 0.0) >= self.min_match
            ]
            if len(kept) != len(context.episodes):
                logger.info("memory retrieve：圖擴散漏過門檻，丟棄 %d 筆",
                            len(context.episodes) - len(kept))
                context.episodes = kept
            self._remember_context(text, context)
            if context.is_empty():
                return None
            rendered = context.to_text()
            if len(rendered) > self.max_chars:
                rendered = rendered[: self.max_chars] + "…"
            return rendered

        return await asyncio.get_running_loop().run_in_executor(None, _work)

    _CONTEXT_KEEP = 8

    def _remember_context(self, text: str, context: Any) -> None:
        self._contexts[text] = context
        while len(self._contexts) > self._CONTEXT_KEEP:
            self._contexts.pop(next(iter(self._contexts)))

    # --- salience（Phase 4b）---

    @property
    def assessor(self) -> Any:
        """SubconsciousAssessor，lazy。heuristic 模式不給 LLM，它自己會降級。"""
        if self._assessor is None:
            from echo_memory.affect.subconscious import SubconsciousAssessor

            fn = None
            if self.assess_mode == "llm":
                fn = self._assess_llm_fn or self._llm_fn
                if fn is None:
                    logger.warning("assess=llm 但沒有 llm_fn，退回 heuristic")
            self._assessor = SubconsciousAssessor(llm_fn=fn)
        return self._assessor

    def _evaluate_salience(
        self, user_text: str, spoken_text: str, context: Any
    ) -> dict[str, Any]:
        """Entry.py Step 3 的精簡版：assess → signals → 三值 → 更新 neurochem。

        StorageJudge（第二次 LLM 微調）刻意不接——先看三值本身夠不夠用。
        推理全在 echo_memory；這裡只是把它們串起來。
        """
        from echo_memory.affect.salience import SalienceEvaluator
        from echo_memory.affect.signals import SalienceSignals

        assessment = self.assessor.assess(user_text, context)
        signals = SalienceSignals.from_subconscious(assessment)
        evaluator = SalienceEvaluator(self.engine.neurochem)
        salience, da, ht = evaluator.evaluate(
            signals, user_input=user_text, agent_response=spoken_text
        )
        evaluator.update_neurochem(signals)
        return {
            "salience": salience,
            "da_weight": da,
            "ht_weight": ht,
            "assessment": assessment.to_dict(),
        }

    async def store(
        self,
        user_text: str,
        spoken_text: str,
        session_id: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any] | None:
        """把一輪對話寫進記憶。**寫 spoken 不寫 generated**——與歷史同一條語意。

        回傳存檔摘要（episode_id / salience 三值 / 評估維度）給 server 記 log；
        旁路：任何失敗只 log、回 None，不往上拋。metadata 建議帶 origin
        （voice/text、是否被 barge-in 截斷）——agent 產物混進使用者指示
        會污染記憶（spike 實測）。

        salience 評估失敗不擋寫入——退回 0.5 照存，寧可有一筆平淡的記憶
        也不要丟掉一輪對話。
        """
        if not user_text or not user_text.strip():
            return None
        sid = session_id or self._session_id or "unknown"
        context = self._contexts.pop(user_text, None)

        def _work() -> dict[str, Any]:
            scores: dict[str, Any] = {}
            if self.assess_mode != "off":
                try:
                    scores = self._evaluate_salience(user_text, spoken_text, context)
                except Exception as exc:  # noqa: BLE001 - 評估是加分項
                    logger.warning("salience 評估失敗，退回 0.5：%s: %s",
                                   type(exc).__name__, exc)
                    scores = {}
            assessment = scores.pop("assessment", None)
            extra = dict(metadata)
            if assessment is not None:
                extra["assessment"] = assessment
            episode_id = self.engine.store_episode(
                user_input=user_text,
                agent_response=spoken_text,
                session_id=sid,
                **scores,
                **extra,
            )
            return {"episode_id": episode_id, **scores, "assessment": assessment}

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _work)
        except Exception as exc:  # noqa: BLE001 - 旁路，失敗不該炸管線
            logger.warning("memory store 失敗（不影響對話）：%s: %s",
                           type(exc).__name__, exc)
            return None

    async def dream(self, triggered_by: str = "manual") -> dict[str, Any]:
        """離線鞏固（蒸餾/重播/修剪）。背景執行，不擋對話。

        蒸餾走 ``llm_fn``（gpt-5.6-luna，同一顆 backend 的同步 query）；
        沒有 llm_fn 時 Dream Engine 會優雅跳過蒸餾步驟。
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.engine.dream(triggered_by)
        )

    DISTILL_MIN_SALIENCE = 0.5
    """與 echo_memory KnowledgeDistillation.MIN_SALIENCE_THRESHOLD 對齊——
    低於這個值的 episode 蒸餾根本不會看，算進 pending 只會讓排程空跑。"""

    async def pending_dream_count(self) -> int:
        """有多少 episode 等著被蒸餾（未 dreamed 且顯著性過門檻）。

        DreamScheduler 用這個數字決定「跑了會不會有產出」。
        list_episodes 單頁上限 200，所以分頁掃。
        """

        def _work() -> int:
            episodic = self.engine.episodic
            total = episodic.count()
            pending = 0
            offset = 0
            while offset < total:
                page = episodic.list_episodes(limit=200, offset=offset)
                if not page:
                    break
                pending += sum(
                    1
                    for ep in page
                    if not ep.is_dreamed
                    and ep.salience_score >= self.DISTILL_MIN_SALIENCE
                )
                offset += len(page)
            return pending

        return await asyncio.get_running_loop().run_in_executor(None, _work)

    def set_llm_fn(
        self,
        llm_fn: Callable[[str], str] | None,
        assess_llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        """事後補上 LLM——backend 建立時機晚於 adapter 時用。

        ``llm_fn`` 給 dream 蒸餾；``assess_llm_fn`` 給每輪的 salience 評估
        （可以是壓低 reasoning 的便宜版本）。
        """
        self._llm_fn = llm_fn
        self._assess_llm_fn = assess_llm_fn
        self._assessor = None  # 下次用到時以新的 fn 重建
        if self._engine is not None and hasattr(self._engine, "set_llm_fn"):
            self._engine.set_llm_fn(llm_fn)


__all__ = ["EchoMemoryAdapter", "build_engine", "DEFAULT_NAMESPACE"]
