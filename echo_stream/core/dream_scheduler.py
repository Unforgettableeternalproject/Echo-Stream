"""DreamScheduler：決定「什麼時候該做夢」的純策略層。

**不 import 任何子系統**——只拿兩個 callable：
``pending_count()``（有多少 episode 等著被蒸餾）與 ``run_dream(trigger)``。
真正的蒸餾邏輯在 echo_memory，這裡只管觸發時機。

## 為什麼不用 echo_memory 自帶的 APScheduler

它是 lazy 的（第一次手動 dream 才啟動）、無差別的（閒置到點就跑，
不管有沒有東西可蒸餾），而且跟我們的 event loop 是兩套時鐘。
艾斯維爾 2026-08-22 定調：**dream 不是無預警與無差別做的，每次都要
期待能萃取出 concept 或 procedure**——所以觸發條件是「累積」先於「閒置」。

## 兩種觸發

* ``idle``：閒置 ≥ ``idle_minutes`` **且** pending ≥ ``min_pending``。
  閒置但沒料 → 不跑（省 token，也不會產出空報告）。
* ``daydream``：pending ≥ ``daydream_pending``，**不等閒置**、背景跑。
  堆太多再一次蒸餾，concept 品質會變差（batch 只看前 10 個高顯著性），
  所以寧可在對話中就先消化一批。

## 中斷語意：重做，不復原

echo_memory 的 ``dream_now`` 只在全部階段結束後才統一標 ``is_dreamed``，
concept 走 upsert。中途掛掉 → 下次同一批從頭再跑，結果一致。
**刻意不做 checkpoint**——做了反而會引入「標了 dreamed 但 concept 沒落地」
的不一致狀態。失敗後退避一個 ``idle_minutes`` 再試，不每分鐘重撞。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_IDLE_MINUTES = 45.0
DEFAULT_MIN_PENDING = 5
DEFAULT_DAYDREAM_PENDING = 20
DEFAULT_TICK_SECONDS = 60.0


@dataclass
class DreamPolicy:
    """觸發門檻。全部可由 .env 覆寫（見 web/server.py）。"""

    idle_minutes: float = DEFAULT_IDLE_MINUTES
    min_pending: int = DEFAULT_MIN_PENDING
    daydream_pending: int = DEFAULT_DAYDREAM_PENDING
    tick_seconds: float = DEFAULT_TICK_SECONDS

    def decide(self, *, pending: int, idle_seconds: float) -> str | None:
        """回傳觸發原因（``daydream`` / ``idle``）或 None。純函數，好測。"""
        if pending >= self.daydream_pending > 0:
            return "daydream"
        if pending >= self.min_pending and idle_seconds >= self.idle_minutes * 60:
            return "idle"
        return None


class DreamScheduler:
    """背景 tick，按 :class:`DreamPolicy` 呼叫 ``run_dream``。

    ``run_dream`` 由呼叫端負責「同時只有一個 dream」（server 有
    ``_dream_running`` 旗標）——這裡只是不在自己跑的時候重複觸發。
    """

    def __init__(
        self,
        *,
        pending_count: Callable[[], Awaitable[int]],
        run_dream: Callable[[str], Awaitable[Any]],
        policy: DreamPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pending_count = pending_count
        self._run_dream = run_dream
        self.policy = policy or DreamPolicy()
        self._clock = clock
        self._last_interaction = clock()
        self._last_attempt: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self.last_pending: int | None = None
        self.last_trigger: str | None = None
        self.last_error: str | None = None

    # --- 外部事件 ---

    def notify_interaction(self) -> None:
        """有一輪對話發生——重置閒置計時。server 在每輪 turn 收尾呼叫。"""
        self._last_interaction = self._clock()

    @property
    def idle_seconds(self) -> float:
        return self._clock() - self._last_interaction

    # --- 生命週期 ---

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.policy.tick_seconds)
                await self.tick()
        except asyncio.CancelledError:
            pass

    # --- 單次判斷（拆出來給測試與手動檢查用）---

    async def tick(self) -> str | None:
        """做一次判斷；觸發了就回傳原因。"""
        if self._running:
            return None
        # 失敗退避：上次嘗試後至少隔一個 idle 週期
        if (
            self.last_error is not None
            and self._last_attempt is not None
            and self._clock() - self._last_attempt < self.policy.idle_minutes * 60
        ):
            return None
        try:
            pending = await self._pending_count()
        except Exception as exc:  # noqa: BLE001 - 旁路，數不到就當 0
            logger.warning("dream pending 計數失敗：%s: %s", type(exc).__name__, exc)
            return None
        self.last_pending = pending
        reason = self.policy.decide(pending=pending, idle_seconds=self.idle_seconds)
        if reason is None:
            return None
        await self._fire(reason)
        return reason

    async def _fire(self, reason: str) -> None:
        self._running = True
        self._last_attempt = self._clock()
        self.last_trigger = reason
        logger.info("dream 觸發 [%s]（pending=%s, idle=%.0fs）",
                    reason, self.last_pending, self.idle_seconds)
        try:
            await self._run_dream(reason)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - 旁路
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("dream [%s] 失敗，退避 %.0f 分鐘：%s",
                           reason, self.policy.idle_minutes, self.last_error)
        finally:
            self._running = False

    def status(self) -> dict[str, Any]:
        return {
            "idle_seconds": round(self.idle_seconds),
            "pending": self.last_pending,
            "last_trigger": self.last_trigger,
            "last_error": self.last_error,
            "policy": {
                "idle_minutes": self.policy.idle_minutes,
                "min_pending": self.policy.min_pending,
                "daydream_pending": self.policy.daydream_pending,
            },
        }


__all__ = ["DreamPolicy", "DreamScheduler"]
