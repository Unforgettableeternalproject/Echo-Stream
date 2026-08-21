"""DreamScheduler：觸發時機是純策略，這裡用假時鐘逐條驗。"""

from __future__ import annotations

import pytest

from echo_stream.core.dream_scheduler import DreamPolicy, DreamScheduler


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _make(pending: int, *, policy: DreamPolicy | None = None, fail: bool = False):
    clock = FakeClock()
    fired: list[str] = []
    state = {"pending": pending}

    async def pending_count() -> int:
        return state["pending"]

    async def run_dream(reason: str) -> None:
        fired.append(reason)
        if fail:
            raise RuntimeError("boom")
        state["pending"] = 0  # 做完就消化掉

    sched = DreamScheduler(
        pending_count=pending_count,
        run_dream=run_dream,
        policy=policy or DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=20),
        clock=clock,
    )
    return sched, clock, fired, state


# --- 純策略 ---


def test_policy_閒置但沒料不跑():
    p = DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=20)
    assert p.decide(pending=0, idle_seconds=10_000) is None
    assert p.decide(pending=4, idle_seconds=10_000) is None


def test_policy_有料但沒閒置不跑():
    p = DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=20)
    assert p.decide(pending=10, idle_seconds=44 * 60) is None


def test_policy_閒置且有料是idle():
    p = DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=20)
    assert p.decide(pending=5, idle_seconds=45 * 60) == "idle"


def test_policy_堆太多不等閒置是daydream():
    p = DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=20)
    assert p.decide(pending=20, idle_seconds=0) == "daydream"


def test_policy_daydream門檻為零代表停用():
    p = DreamPolicy(idle_minutes=45, min_pending=5, daydream_pending=0)
    assert p.decide(pending=999, idle_seconds=0) is None


# --- 排程器行為 ---


async def test_對話會重置閒置計時():
    sched, clock, fired, _ = _make(pending=10)
    clock.advance(40 * 60)
    sched.notify_interaction()
    clock.advance(10 * 60)  # 距上次對話只有 10 分鐘
    assert await sched.tick() is None
    clock.advance(35 * 60)
    assert await sched.tick() == "idle"
    assert fired == ["idle"]


async def test_做完一次不會每tick重跑():
    sched, clock, fired, _ = _make(pending=10)
    clock.advance(46 * 60)
    assert await sched.tick() == "idle"
    # pending 已歸零——再閒置多久都不該跑
    clock.advance(120 * 60)
    assert await sched.tick() is None
    assert fired == ["idle"]


async def test_失敗後退避一個閒置週期():
    sched, clock, fired, _ = _make(pending=30, fail=True)
    assert await sched.tick() == "daydream"
    assert sched.last_error is not None
    clock.advance(60)
    assert await sched.tick() is None, "失敗後一分鐘不該重撞"
    clock.advance(45 * 60)
    assert await sched.tick() == "daydream"
    assert fired == ["daydream", "daydream"]


async def test_pending計數失敗旁路():
    async def bad_count() -> int:
        raise RuntimeError("storage gone")

    fired: list[str] = []

    async def run_dream(reason: str) -> None:
        fired.append(reason)

    sched = DreamScheduler(pending_count=bad_count, run_dream=run_dream)
    assert await sched.tick() is None
    assert fired == []


async def test_status_形狀():
    sched, clock, _, _ = _make(pending=3)
    await sched.tick()
    s = sched.status()
    assert s["pending"] == 3
    assert s["last_trigger"] is None
    assert s["policy"]["idle_minutes"] == 45


@pytest.mark.parametrize("pending,expected", [(19, None), (20, "daydream")])
async def test_daydream_邊界(pending, expected):
    sched, _, _, _ = _make(pending=pending)
    assert await sched.tick() == expected
