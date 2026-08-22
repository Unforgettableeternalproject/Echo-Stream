"""EchoMemoryAdapter：不碰真 echo_memory，用假 engine 驗 adapter 自己的責任。"""

from __future__ import annotations

from dataclasses import dataclass, field

from echo_stream.adapters.memory_echo import EchoMemoryAdapter


@dataclass
class _Ep:
    id: str
    user_input: str
    is_dreamed: bool = False
    salience_score: float = 0.5


@dataclass
class _Ctx:
    episodes: list = field(default_factory=list)
    similarities: dict = field(default_factory=dict)
    concepts: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    procedures: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.episodes or self.concepts or self.entities or self.procedures)

    def to_text(self) -> str:
        return "|".join(f"{e.user_input}:{self.similarities[e.id]:.2f}" for e in self.episodes)


class _Episodic:
    def __init__(self, eps):
        self._eps = eps

    def count(self):
        return len(self._eps)

    def list_episodes(self, limit=50, offset=0, **_):
        return self._eps[offset : offset + min(limit, 200)]


class _Engine:
    def __init__(self, ctx, eps=()):
        self._ctx = ctx
        self.calls: list[dict] = []
        self.episodic = _Episodic(list(eps))

    def retrieve(self, text, **kw):
        self.calls.append(kw)
        return self._ctx


async def test_retrieve_丟棄圖擴散漏過門檻的episode():
    ctx = _Ctx(
        episodes=[_Ep("a", "攀岩"), _Ep("b", "深藍色"), _Ep("c", "無分數")],
        similarities={"a": 0.62, "b": 0.31},  # c 沒有分數 = 圖擴散漏出來的
    )
    adapter = EchoMemoryAdapter(engine=_Engine(ctx), top_k=3)
    adapter.min_match = 0.45
    assert await adapter.retrieve("問題") == "攀岩:0.62"


async def test_retrieve_全部低於門檻回None():
    ctx = _Ctx(episodes=[_Ep("b", "深藍色")], similarities={"b": 0.31})
    adapter = EchoMemoryAdapter(engine=_Engine(ctx))
    adapter.min_match = 0.45
    assert await adapter.retrieve("問題") is None


async def test_retrieve_把門檻傳給引擎():
    engine = _Engine(_Ctx())
    adapter = EchoMemoryAdapter(engine=engine)
    adapter.min_match = 0.5
    adapter.set_session("s1")
    await adapter.retrieve("問題")
    assert engine.calls[0]["min_similarity"] == 0.5
    assert engine.calls[0]["exclude_session_id"] == "s1"


async def test_pending_dream_count_只算未dreamed且過顯著性():
    eps = [
        _Ep("1", "x", is_dreamed=False, salience_score=0.5),
        _Ep("2", "x", is_dreamed=True, salience_score=0.9),
        _Ep("3", "x", is_dreamed=False, salience_score=0.3),
        _Ep("4", "x", is_dreamed=False, salience_score=0.7),
    ]
    adapter = EchoMemoryAdapter(engine=_Engine(_Ctx(), eps))
    assert await adapter.pending_dream_count() == 2


async def test_pending_dream_count_跨頁():
    eps = [_Ep(str(i), "x") for i in range(450)]
    adapter = EchoMemoryAdapter(engine=_Engine(_Ctx(), eps))
    assert await adapter.pending_dream_count() == 450
