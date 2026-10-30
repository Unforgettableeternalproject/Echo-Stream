"""member_recall 指名查詢工具：假引擎驗 adapter 的解析、透傳、掛名與消費記錄。"""

from __future__ import annotations

import pytest

from echo_stream.adapters.memory_echo import EchoMemoryAdapter

from .test_memory_echo import _Ctx, _Engine
from .test_memory_echo_sharing import _Item, _Shared


class _OwnerEngine(_Engine):
    """帶 owner_uid 的 retrieve_shared 替身。"""

    def __init__(self):
        super().__init__(_Ctx())
        self.calls2: list[tuple] = []
        self.shared = _Shared([_Item("p1"), _Item("p2")])

    def retrieve_shared(self, query, *, scope_id, viewer_uid, present_uids=(), owner_uid=None):
        self.calls2.append((query, scope_id, viewer_uid, tuple(present_uids), owner_uid))
        return self.shared


class _OldEngine(_Engine):
    """舊版引擎：retrieve_shared 不認得 owner_uid。"""

    def __init__(self):
        super().__init__(_Ctx())
        self.called = 0

    def retrieve_shared(self, query, *, scope_id, viewer_uid, present_uids=()):
        self.called += 1
        return _Shared([_Item("x")])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ECHO_STREAM_MEMORY_TOOLS", raising=False)


def _adapter(engine, resolver=None, tools=("member_recall",)):
    adapter = EchoMemoryAdapter(engine=engine)
    adapter.enabled_tools = list(tools)
    adapter.set_member_resolver(resolver)
    adapter.set_member_recall_context(
        scope_id="scope-g1", viewer_uid="v1", present_uids=["v1", "a"],
        display_name=lambda uid: "雅烈",
    )
    return adapter


async def test_解析成功_帶owner_uid_掛名_ids進入消費記錄():
    eng = _OwnerEngine()
    adapter = _adapter(eng, lambda name: "u_yalie" if name == "雅烈" else None)
    out = await adapter.execute_tool("member_recall", {"member": "雅烈", "query": "遊戲"})
    assert out["success"] is True
    assert eng.calls2 == [("遊戲", "scope-g1", "v1", ("v1", "a"), "u_yalie")]
    assert "雅烈" in out["result"] and "p1,p2" in out["result"]
    assert adapter.take_member_recall_ids() == ["p1", "p2"]
    assert adapter.take_member_recall_ids() == [], "取走即清空"


async def test_query空白可用():
    eng = _OwnerEngine()
    adapter = _adapter(eng, lambda name: "u1")
    out = await adapter.execute_tool("member_recall", {"member": "雅烈"})
    assert out["success"] is True
    assert eng.calls2[0][0] == "" and eng.calls2[0][4] == "u1"


@pytest.mark.parametrize("resolver", [None, lambda name: None])
async def test_解析失敗或未注入_不呼叫引擎(resolver):
    eng = _OwnerEngine()
    adapter = _adapter(eng, resolver)
    out = await adapter.execute_tool("member_recall", {"member": "路人"})
    assert "找不到" in out["result"]
    assert eng.calls2 == []
    assert adapter.take_member_recall_ids() == []


async def test_resolver拋例外_視同找不到():
    def boom(name):
        raise RuntimeError("x")

    eng = _OwnerEngine()
    adapter = _adapter(eng, boom)
    out = await adapter.execute_tool("member_recall", {"member": "雅烈"})
    assert "找不到" in out["result"] and eng.calls2 == []


async def test_未設場域脈絡_不呼叫引擎():
    eng = _OwnerEngine()
    adapter = EchoMemoryAdapter(engine=eng)
    adapter.enabled_tools = ["member_recall"]
    adapter.set_member_resolver(lambda name: "u1")
    out = await adapter.execute_tool("member_recall", {"member": "雅烈"})
    assert "找不到" in out["result"] and eng.calls2 == []


async def test_舊引擎不支援owner_uid_回空不退化成全體查詢():
    eng = _OldEngine()
    adapter = EchoMemoryAdapter(engine=eng)
    out = await adapter.retrieve_shared("", scope_id="s", viewer_uid="v", owner_uid="u1")
    assert out is not None and out.text == "" and out.ids == []
    assert eng.called == 0

    adapter2 = _adapter(eng, lambda name: "u1")
    res = await adapter2.execute_tool("member_recall", {"member": "雅烈"})
    assert eng.called == 0 and adapter2.take_member_recall_ids() == []
    assert res["success"] is True


async def test_retrieve_shared_有owner_uid時允許空text_沒有時仍回None():
    eng = _OwnerEngine()
    adapter = EchoMemoryAdapter(engine=eng)
    out = await adapter.retrieve_shared("", scope_id="s", viewer_uid="v", owner_uid="u1")
    assert out is not None and out.ids == ["p1", "p2"]
    assert eng.calls2[-1][4] == "u1"
    assert await adapter.retrieve_shared("  ", scope_id="s", viewer_uid="v") is None


def test_工具未啟用時不在schema():
    names = [d["function"]["name"] for d in EchoMemoryAdapter(_OwnerEngine()).tool_definitions()]
    assert "member_recall" not in names


def test_工具啟用時出現在schema(monkeypatch):
    monkeypatch.setenv("ECHO_STREAM_MEMORY_TOOLS", "member_recall")
    defs = EchoMemoryAdapter(_OwnerEngine()).tool_definitions()
    assert [d["function"]["name"] for d in defs] == ["member_recall"]
    params = defs[0]["function"]["parameters"]
    assert params["required"] == ["member"] and "query" in params["properties"]


async def test_未啟用時execute_tool擋掉():
    eng = _OwnerEngine()
    adapter = _adapter(eng, lambda name: "u1", tools=("deep_recall",))
    out = await adapter.execute_tool("member_recall", {"member": "雅烈"})
    assert out["success"] is False and eng.calls2 == []
