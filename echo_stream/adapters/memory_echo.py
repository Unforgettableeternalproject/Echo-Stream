"""Memory adapter：Echo Memory（Phase 4a）。

**這是唯一 import echo_memory 的地方**，其餘程式碼只認
:class:`EchoMemoryAdapter` 的幾個方法（retrieve / store / set_session /
tool_definitions / execute_tool）。

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
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
"""檢索相似度下限。echo_memory 預設 0.3，實測 35-41% 的匹配是純雜訊
（「攀岩」撈出「深藍色」），注入只會讓 LLM 亂講。"""
DEFAULT_TOOLS = "deep_recall,force_remember"
"""預設開放給 LLM 的記憶工具（ECHO_STREAM_MEMORY_TOOLS，逗號分隔，``off`` 關閉）。
echo_memory 有四個：deep_recall（純讀）、force_remember（寫一筆高顯著性 episode）、
adjust_importance（改既有 episode 的 salience）、trigger_dream（觸發鞏固）。
後兩個讓模型自主決定風險高，預設不給——要開自己寫進 .env。"""
DEFAULT_TOOL_MAX_CHARS = 1500
"""工具結果回給 LLM 的字數上限。deep_recall 的 top_k 可到 20、門檻比自動注入鬆
（0.2 vs 0.45），不截會把第二段續寫的 prompt 塞爆。"""
ASSESS_MODES = ("llm", "heuristic", "off")
"""salience 評估模式（ECHO_STREAM_MEMORY_ASSESS）：
llm = SubconsciousAssessor 打一次 LLM（背景，不在熱路徑）；
heuristic = echo_memory 的文字啟發式，零 token；off = 全部 0.5（Phase 4a 行為）。"""


def _coerce_evidence(item: Any) -> Any:
    """dict → ``EvidenceInput``；已是 ``EvidenceInput`` 原樣回傳。"""
    from echo_memory.schemas.evidence import EvidenceInput, SourceRef, TextSpan

    if isinstance(item, EvidenceInput):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"evidence 項目必須是 EvidenceInput 或 dict，收到 {type(item).__name__}")
    data = dict(item)
    ref = data.get("source_ref")
    if isinstance(ref, dict):
        ref = dict(ref)
        ref["ref_ids"] = tuple(ref.get("ref_ids") or ())
        data["source_ref"] = SourceRef(**ref)
    span = data.get("text_span")
    if isinstance(span, dict):
        data["text_span"] = TextSpan(**span)
    for key in ("origin_evidence_ids", "derived_from"):
        if key in data:
            data[key] = tuple(data[key] or ())
    return EvidenceInput(**data)


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
        executor: ThreadPoolExecutor | None = None,
        engine_lock: threading.Lock | None = None,
    ) -> None:
        self._engine = engine
        self._owns_executor = executor is None
        """外部注入的 executor 由注入方管理生命週期，``aclose`` 不關它。"""
        self._engine_lock = engine_lock
        """可選的 engine 互斥鎖（``threading.Lock``）。同一個 engine 若另有執行緒
        會碰（例如 bot 文字頻道路徑），雙方持同一把鎖才不會撞出 WinError 5。
        不給 = 只靠單工作緒 executor 序列化（原行為）。"""
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="echo-memory"
        )
        """engine 專用的單工作緒 executor——**所有** engine 呼叫都走這裡序列化。

        echo_memory 不是 thread-safe 的：EntityStore._save 是「固定 tmp 檔名 +
        整檔 replace」且無鎖，retrieve（會更新 mention）/ store / dream / 工具
        呼叫若各自丟共用執行緒池，兩條執行緒同時進 engine 就會撞出
        WinError 5（replace 時檔案被另一條執行緒開著，2026-08-24 實錄）。
        代價：dream 蒸餾期間 retrieve 會排隊——但併發進去換到的不只是
        WinError 5，還有整份 JSON 互相覆寫的資料損毀，序列化是底線。"""
        self._llm_fn = llm_fn
        self._assess_llm_fn = assess_llm_fn
        """評估用 LLM。五維打分不需要 reasoning，server 會給一個壓低 effort 的版本；
        沒給就退回 dream 用的 llm_fn。"""
        self._assessor: Any = None
        self._judge: Any = None
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
        self.last_retrieve_degraded = False
        """最近一次 retrieve 是否被引擎標為降級（``MemoryContext.degraded``）。
        retrieve 對外只回文字，降級旗標存這裡給宿主 log 讀。"""
        self.last_degraded_reason = ""
        self._evidence_supported: bool | None = None
        self._idempotency_supported: bool | None = None
        self._tool_executor: Any = None
        self.tool_max_chars = int(
            config.get("ECHO_STREAM_MEMORY_TOOL_MAX_CHARS", str(DEFAULT_TOOL_MAX_CHARS))
        )
        raw_tools = config.get("ECHO_STREAM_MEMORY_TOOLS", DEFAULT_TOOLS) or DEFAULT_TOOLS
        self.enabled_tools: list[str] = (
            [] if raw_tools.strip().lower() in ("off", "none", "false", "0")
            else [t.strip() for t in raw_tools.split(",") if t.strip()]
        )

    @property
    def engine(self) -> Any:
        if self._engine is None:
            import time

            t0 = time.perf_counter()
            self._engine = build_engine(llm_fn=self._llm_fn)
            self.load_seconds = time.perf_counter() - t0
        return self._engine

    def _locked(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """有 ``engine_lock`` 就包一層持鎖；沒有就原樣回傳。"""
        lock = self._engine_lock
        if lock is None:
            return fn

        def run() -> Any:
            with lock:
                return fn()

        return run

    async def _run(self, fn: Callable[[], Any]) -> Any:
        """把 engine 工作丟進專用 executor（持鎖）。"""
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._locked(fn))

    async def prepare(self) -> None:
        """提早建立 engine——設定錯誤在暖機階段就爆，而不是第一次 retrieve。

        embedding 模型（sentence-transformers）的首載也發生在這裡之後的
        第一次呼叫；真正的冷啟成本用 memory-check 量。
        """
        await self._run(lambda: self.engine)

    async def aclose(self) -> None:
        """關閉 engine 專用 executor。

        不取消排隊中的工作——store / dream 是寫入，丟掉就是記憶遺失；
        等它們跑完再收，且不擋 event loop。重複呼叫是安全的。
        外部注入的 executor 不在這裡關（可能還有別的 adapter 在用）。
        """
        if not self._owns_executor:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._executor.shutdown(wait=True)
        )

    def set_session(self, session_id: str | None) -> None:
        """換會話。之後的 retrieve 會排除這個會話的 episodes（防自我回聲），
        store 會把 episodes 標上這個 session_id。"""
        self._session_id = session_id
        self._tool_executor = None  # echo_memory 的 ToolExecutor 綁 session_id，換會話重建

    # --- 工具呼叫（C 段）---

    def tool_definitions(self) -> list[dict[str, Any]]:
        """開放給 LLM 的工具，OpenAI function schema。空列表 = 不給工具。

        schema 來自 echo_memory 的 ``TOOL_DECLARATIONS``（它是 Gemini 格式的 dict，
        欄位剛好與 OpenAI 的 function 一致，只差外面要包一層 ``type: function``）。
        """
        if not self.enabled_tools:
            return []
        try:
            from llm_integration.tools.definitions import TOOL_DECLARATIONS
        except ImportError as exc:
            logger.warning("echo_memory 沒有 llm_integration.tools——工具停用：%s", exc)
            return []
        known = {d["name"]: d for d in TOOL_DECLARATIONS}
        out = []
        for name in self.enabled_tools:
            decl = known.get(name)
            if decl is None:
                logger.warning(
                    "ECHO_STREAM_MEMORY_TOOLS 裡的 %r 不是 echo_memory 的工具，略過", name
                )
                continue
            out.append({"type": "function", "function": dict(decl)})
        return out

    @property
    def tool_executor(self) -> Any:
        """echo_memory 的 ToolExecutor，lazy、綁當前 session。"""
        if self._tool_executor is None:
            from llm_integration.tools.executor import ToolExecutor

            self._tool_executor = ToolExecutor(
                self.engine, session_id=self._session_id or "unknown"
            )
        return self._tool_executor

    async def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """給 LLMThinkStage 的 ``tool_executor``。

        白名單再擋一次——schema 只給了兩個，模型照理叫不出別的，
        但「照理」不是防線。結果截到 ``tool_max_chars``。
        """
        if name not in self.enabled_tools:
            return {"success": False, "result": f"Tool {name!r} is not available."}

        def _work() -> dict[str, Any]:
            outcome = self.tool_executor.execute(name, dict(args))
            text = str(outcome.get("result", ""))
            if len(text) > self.tool_max_chars:
                outcome = {**outcome, "result": text[: self.tool_max_chars] + "…"}
            return outcome

        return await self._run(_work)

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
            # 降級旗標：引擎在 embedding / 後端不可用時回 degraded=True（空結果也要知道原因）。
            # 用 ``is True`` 判斷，避免替身物件的任意屬性被當成真值
            degraded = getattr(context, "degraded", False) is True
            reason = getattr(context, "degraded_reason", "")
            self.last_retrieve_degraded = degraded
            self.last_degraded_reason = reason if degraded and isinstance(reason, str) else ""
            if degraded:
                logger.warning(
                    "memory retrieve 降級：%s", self.last_degraded_reason or "未註明原因"
                )
            self._remember_context(text, context)
            if context.is_empty():
                return None
            rendered = context.to_text()
            if len(rendered) > self.max_chars:
                rendered = rendered[: self.max_chars] + "…"
            return rendered

        return await self._run(_work)

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

    async def _evaluate_salience(
        self, user_text: str, spoken_text: str, context: Any
    ) -> dict[str, Any]:
        """Entry.py Step 3：assess → signals → 三值 → neurochem → StorageJudge 微調。

        推理全在 echo_memory；這裡只是把它們串起來。

        **鎖範圍**：assess 與 judge 是 LLM 呼叫（數秒），丟預設執行緒池、不持
        engine 鎖也不佔 engine 專用 executor；只有碰 ``engine.neurochem`` 的
        評估段（純 CPU、極短）走 :meth:`_run`（executor + 鎖）。

        StorageJudge 是第二段：OpenSoul 公式偏重 complexity/novelty，
        「我妹妹要去筑波大學」「別再糾正我的英文」這種簡單但該記住的話
        baseline 只有 0.35-0.45，過不了蒸餾門檻（2026-08-22 實測）。
        Judge 的 JudgeGate 只在信號進灰區才打 LLM，用 salience_delta /
        force_remember / metadata_tags（correction、preference…）補這一段。
        """
        from echo_memory.affect.salience import SalienceEvaluator
        from echo_memory.affect.signals import SalienceSignals

        loop = asyncio.get_running_loop()
        assessor = self.assessor
        assessment = await loop.run_in_executor(
            None, lambda: assessor.assess(user_text, context)
        )
        signals = SalienceSignals.from_subconscious(assessment)

        def _score() -> tuple[float, float, float]:
            evaluator = SalienceEvaluator(self.engine.neurochem)
            scored = evaluator.evaluate(
                signals, user_input=user_text, agent_response=spoken_text
            )
            evaluator.update_neurochem(signals)
            return scored

        baseline, da, ht = await self._run(_score)
        result: dict[str, Any] = {
            "salience": baseline,
            "da_weight": da,
            "ht_weight": ht,
            "assessment": assessment.to_dict(),
        }
        if self.judge_enabled:
            try:
                judge = self.judge
                decision = await loop.run_in_executor(
                    None,
                    lambda: judge.storage_decision(
                        user_input=user_text,
                        agent_response=spoken_text,
                        baseline_salience=baseline,
                        assessment=assessment,
                        context=context if context is not None and not context.is_empty()
                        else None,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - judge 是加分項，炸了用 baseline
                logger.warning("StorageJudge 失敗，用 baseline：%s: %s",
                               type(exc).__name__, exc)
            else:
                result["salience"] = max(0.0, min(1.0, baseline + decision.salience_delta))
                judge_info: dict[str, Any] = {"source": decision.source}
                if decision.salience_delta:
                    judge_info["delta"] = round(decision.salience_delta, 3)
                if decision.metadata_tags:
                    judge_info["tags"] = list(decision.metadata_tags)
                if decision.force_remember:
                    judge_info["force_remember"] = True
                    result["force_remember"] = True
                if decision.entity_triggers:
                    result["entity_triggers"] = True
                if decision.reasoning:
                    judge_info["reasoning"] = decision.reasoning[:200]
                result["judge"] = judge_info
        return result

    @property
    def judge_enabled(self) -> bool:
        return self.assess_mode == "llm" and (
            config.get("ECHO_STREAM_MEMORY_JUDGE", "on") or "on"
        ).lower() not in ("off", "false", "0")

    @property
    def judge(self) -> Any:
        """StorageJudge，lazy。與 assessor 共用評估用的 LLM。"""
        if self._judge is None:
            from echo_memory.affect.judge import StorageJudge

            self._judge = StorageJudge(llm_fn=self._assess_llm_fn or self._llm_fn)
        return self._judge

    async def store(
        self,
        user_text: str,
        spoken_text: str,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any] | None:
        """把一輪對話寫進記憶。**寫 spoken 不寫 generated**——與歷史同一條語意。

        回傳存檔摘要（episode_id / salience 三值 / 評估維度）給 server 記 log；
        旁路：任何失敗只 log、回 None，不往上拋。metadata 建議帶 origin
        （voice/text、是否被 barge-in 截斷）——agent 產物混進使用者指示
        會污染記憶（spike 實測）。

        ``idempotency_key``：同 key 重送由引擎回原 episode_id、不重寫；引擎簽名沒有
        此參數（舊版）或為 None 時不傳。

        salience 評估失敗不擋寫入——退回 0.5 照存，寧可有一筆平淡的記憶
        也不要丟掉一輪對話。
        """
        if not user_text or not user_text.strip():
            return None
        sid = session_id or self._session_id or "unknown"
        context = self._contexts.pop(user_text, None)

        try:
            scores: dict[str, Any] = {}
            if self.assess_mode != "off":
                try:
                    scores = await self._evaluate_salience(user_text, spoken_text, context)
                except Exception as exc:  # noqa: BLE001 - 評估是加分項
                    logger.warning("salience 評估失敗，退回 0.5：%s: %s",
                                   type(exc).__name__, exc)
                    scores = {}
            assessment = scores.pop("assessment", None)
            extra = dict(metadata)
            if assessment is not None:
                extra["assessment"] = assessment
            evidence = self._prepare_evidence(extra.pop("evidence", None))
            if evidence:
                extra["evidence"] = evidence
            if idempotency_key is not None and self._supports_idempotency():
                extra["idempotency_key"] = idempotency_key

            def _write() -> str:
                return self.engine.store_episode(
                    user_input=user_text,
                    agent_response=spoken_text,
                    session_id=sid,
                    **scores,
                    **extra,
                )

            episode_id = await self._run(_write)
            return {"episode_id": episode_id, **scores, "assessment": assessment}
        except Exception as exc:  # noqa: BLE001 - 旁路，失敗不該炸管線
            logger.warning("memory store 失敗（不影響對話）：%s: %s",
                           type(exc).__name__, exc)
            return None

    def _supports_evidence(self) -> bool:
        """引擎的 ``store_episode`` 是否有 ``evidence`` 參數（舊版沒有，不能傳）。"""
        if self._evidence_supported is None:
            try:
                params = inspect.signature(self.engine.store_episode).parameters
                self._evidence_supported = "evidence" in params
            except (TypeError, ValueError):
                self._evidence_supported = False
        return self._evidence_supported

    def _supports_idempotency(self) -> bool:
        """引擎的 ``store_episode`` 是否有 ``idempotency_key`` 參數（舊版沒有，不能傳）。"""
        if self._idempotency_supported is None:
            try:
                params = inspect.signature(self.engine.store_episode).parameters
                self._idempotency_supported = "idempotency_key" in params
            except (TypeError, ValueError):
                self._idempotency_supported = False
        return self._idempotency_supported

    def _prepare_evidence(self, evidence: Any) -> list[Any] | None:
        """把呼叫端的 evidence 轉成引擎吃的 ``EvidenceInput`` 清單。

        接受 ``EvidenceInput`` 或同欄位的 dict（``source_ref`` / ``text_span`` 也可為 dict），
        讓宿主不必 import echo_memory。引擎不支援 evidence（舊版）或宣告格式無法解析時
        退回不帶 evidence——寧可以 legacy_unknown 存一筆，也不要丟掉一輪對話。
        內容與 user_input / agent_response 不一致由引擎丟 ValueError，不在這裡吞掉。
        """
        if not evidence:
            return None
        if not self._supports_evidence():
            logger.debug("引擎不支援 evidence 參數，略過來源宣告")
            return None
        try:
            return [_coerce_evidence(item) for item in evidence]
        except (TypeError, ValueError, KeyError, ImportError) as exc:
            logger.warning("evidence 宣告無法解析，退回不帶來源：%s: %s",
                           type(exc).__name__, exc)
            return None

    async def profile(self) -> str:
        """使用者 profile 文字（subject=user 的 concept 全列）。沒有就空字串。

        常駐脈絡，不是檢索結果——server 在暖機與每次 dream 後刷新一次，
        不在每輪熱路徑呼叫。
        """
        max_chars = int(config.get("ECHO_STREAM_MEMORY_PROFILE_MAX_CHARS", "800"))
        get_profile = getattr(self.engine, "get_profile", None)
        if get_profile is None:
            logger.warning("echo_memory 沒有 get_profile——分支太舊，profile 停用")
            return ""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._locked(lambda: get_profile(max_chars=max_chars) or "")
        )

    async def activity_profile(self, top_n: int = 3) -> list[dict[str, Any]]:
        """遊玩紀錄摘要（依累計時長排序）：[{name, play_count, total_minutes, last_played}]。

        引擎沒有此方法（舊版）或查詢失敗一律回 []——不讓 prompt 組裝因此失敗。
        """
        fn = getattr(self.engine, "activity_profile", None)
        if fn is None:
            return []
        try:
            return list(await self._run(lambda: fn(top_n=top_n)) or [])
        except Exception as exc:  # noqa: BLE001 - 加分項，失敗只降級
            logger.warning("activity_profile 失敗，退回空清單：%s: %s",
                           type(exc).__name__, exc)
            return []

    async def dream(self, triggered_by: str = "manual") -> dict[str, Any]:
        """離線鞏固（蒸餾/重播/修剪）。背景執行，不擋對話。

        蒸餾走 ``llm_fn``（gpt-5.6-luna，同一顆 backend 的同步 query）；
        沒有 llm_fn 時 Dream Engine 會優雅跳過蒸餾步驟。
        """
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._locked(lambda: self.engine.dream(triggered_by))
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

        return await self._run(_work)

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
        self._judge = None
        if self._engine is not None and hasattr(self._engine, "set_llm_fn"):
            self._engine.set_llm_fn(llm_fn)


__all__ = ["EchoMemoryAdapter", "build_engine", "DEFAULT_NAMESPACE"]
