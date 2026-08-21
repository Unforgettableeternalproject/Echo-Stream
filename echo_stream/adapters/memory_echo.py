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

    from echo_memory import MemoryEngine

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
    ) -> None:
        self._engine = engine
        self._llm_fn = llm_fn
        self.top_k = top_k or int(
            config.get("ECHO_STREAM_MEMORY_TOP_K", str(DEFAULT_TOP_K))
        )
        self.max_chars = max_chars or int(
            config.get("ECHO_STREAM_MEMORY_MAX_CHARS", str(DEFAULT_MAX_CHARS))
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
                exclude_session_id=self._session_id,
            )
            if context.is_empty():
                return None
            rendered = context.to_text()
            if len(rendered) > self.max_chars:
                rendered = rendered[: self.max_chars] + "…"
            return rendered

        return await asyncio.get_running_loop().run_in_executor(None, _work)

    async def store(
        self,
        user_text: str,
        spoken_text: str,
        session_id: str | None = None,
        **metadata: Any,
    ) -> None:
        """把一輪對話寫進記憶。**寫 spoken 不寫 generated**——與歷史同一條語意。

        旁路：任何失敗只 log，不往上拋。metadata 建議帶 origin
        （voice/text、是否被 barge-in 截斷）——agent 產物混進使用者指示
        會污染記憶（spike 實測）。
        """
        if not user_text or not user_text.strip():
            return
        sid = session_id or self._session_id or "unknown"

        def _work() -> None:
            self.engine.store_episode(
                user_input=user_text,
                agent_response=spoken_text,
                session_id=sid,
                **metadata,
            )

        try:
            await asyncio.get_running_loop().run_in_executor(None, _work)
        except Exception as exc:  # noqa: BLE001 - 旁路，失敗不該炸管線
            logger.warning("memory store 失敗（不影響對話）：%s: %s",
                           type(exc).__name__, exc)

    async def dream(self, triggered_by: str = "manual") -> dict[str, Any]:
        """離線鞏固（蒸餾/重播/修剪）。背景執行，不擋對話。

        蒸餾走 ``llm_fn``（gpt-5.6-luna，同一顆 backend 的同步 query）；
        沒有 llm_fn 時 Dream Engine 會優雅跳過蒸餾步驟。
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.engine.dream(triggered_by)
        )

    def set_llm_fn(self, llm_fn: Callable[[str], str] | None) -> None:
        """事後補上蒸餾用的 LLM——backend 建立時機晚於 adapter 時用。"""
        self._llm_fn = llm_fn
        if self._engine is not None and hasattr(self._engine, "set_llm_fn"):
            self._engine.set_llm_fn(llm_fn)


__all__ = ["EchoMemoryAdapter", "build_engine", "DEFAULT_NAMESPACE"]
