"""EchoMemoryAdapter 的共用 executor / engine_lock（voice-memory 批 A3）。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from echo_stream.adapters.memory_echo import EchoMemoryAdapter

from .test_memory_echo import _Ctx, _Engine, _StoreEngine


async def test_注入executor_aclose不關它():
    shared = ThreadPoolExecutor(max_workers=1)
    try:
        adapter = EchoMemoryAdapter(engine=_Engine(_Ctx()), executor=shared)
        await adapter.retrieve("問題")
        await adapter.aclose()
        assert shared.submit(lambda: 42).result() == 42  # 仍可用
    finally:
        shared.shutdown(wait=True)


async def test_自建executor_aclose會關():
    adapter = EchoMemoryAdapter(engine=_Engine(_Ctx()))
    await adapter.aclose()
    with pytest.raises(RuntimeError):
        adapter._executor.submit(lambda: 1)


async def test_engine_lock_在engine呼叫期間持有():
    lock = threading.Lock()
    held: list[bool] = []

    class _LockedEngine(_StoreEngine):
        def retrieve(self, text, **kw):
            held.append(lock.locked())
            return super().retrieve(text, **kw)

        def store_episode(self, **kw):
            held.append(lock.locked())
            return super().store_episode(**kw)

    adapter = EchoMemoryAdapter(engine=_LockedEngine(), engine_lock=lock)
    adapter.assess_mode = "off"
    await adapter.retrieve("問題")
    await adapter.store("問題", "回答")
    assert held == [True, True]
    assert not lock.locked(), "工作結束要釋放"


async def test_外部持鎖時_adapter工作會等():
    lock = threading.Lock()
    adapter = EchoMemoryAdapter(engine=_Engine(_Ctx()), engine_lock=lock)
    lock.acquire()
    task = asyncio.ensure_future(adapter.retrieve("問題"))
    await asyncio.sleep(0.05)
    assert not task.done(), "鎖被別條路徑持有時要排隊"
    lock.release()
    await asyncio.wait_for(task, 2)


async def test_沒給lock時行為與原本相同():
    engine = _Engine(_Ctx())
    adapter = EchoMemoryAdapter(engine=engine)
    assert adapter._engine_lock is None
    await adapter.retrieve("問題")
    assert len(engine.calls) == 1
