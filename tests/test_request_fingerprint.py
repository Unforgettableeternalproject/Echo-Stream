"""請求指紋（last_request_fingerprint）與預熱 effort 一致性的測試（假 backend，不打真 API）。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from echo_stream.adapters.think_llm import LLMThinkStage
from echo_stream.contracts.cancellation import CancellationToken
from echo_stream.contracts.types import Utterance

_TOOLS = [{"type": "function", "function": {"name": "deep_recall"}}]


class _Backend:
    """只吐文字、絕不回工具呼叫（避免無限往返）。"""

    def __init__(self, api: str = "responses", effort: str | None = "high") -> None:
        self.api = api
        self.reasoning_effort = effort
        self.requests: list[dict[str, Any]] = []

    async def stream_query(
        self, messages: Sequence[Any], system_instruction: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        self.requests.append({"system_instruction": system_instruction, **kwargs})
        yield "好"

    def get_last_stream_usage(self) -> dict[str, int]:
        return {"input_tokens": 100, "cached_tokens": 64, "output_tokens": 1}


async def _run_turn(stage: LLMThinkStage) -> None:
    async for _ in stage.stream(Utterance(text="嗨"), CancellationToken()):
        pass


async def _executor(name, args):  # pragma: no cover - 假 backend 不會呼叫
    return ""


def _stage(b: _Backend, **kw: Any) -> LLMThinkStage:
    return LLMThinkStage(b, system_prompt="人設", prompt_cache_key="k", **kw)


async def test_指紋欄位存在且預熱與同前綴正常輪_hash_相同():
    stage = _stage(_Backend())
    assert stage.last_request_fingerprint is None
    await stage.prewarm()
    warm = stage.last_request_fingerprint
    assert warm is not None and warm["kind"] == "prewarm"
    assert warm["effort"] == "high"
    assert warm["prompt_cache_key"] == "k"
    assert warm["input_tokens"] == 100 and warm["cached_tokens"] == 64
    assert len(warm["instructions_hash"]) == 8 and warm["instructions_len"] > 0

    await _run_turn(stage)
    real = stage.last_request_fingerprint
    assert real["kind"] == "turn"
    for key in ("effort", "tools", "instructions_hash", "instructions_len", "prompt_cache_key"):
        assert real[key] == warm[key], key


async def test_instructions_變動時_hash_改變():
    stage = _stage(_Backend())
    await stage.prewarm()
    before = stage.last_request_fingerprint["instructions_hash"]
    stage.profile = "使用者喜歡咖啡"
    await _run_turn(stage)
    assert stage.last_request_fingerprint["instructions_hash"] != before


async def test_effort_與_backend_一致_chat_帶_tools_降_none():
    # responses：帶 tools 也維持 backend 設定
    r = _stage(_Backend("responses", "high"), tools=_TOOLS, tool_executor=_executor)
    await r.prewarm()
    assert r.last_request_fingerprint["effort"] == "high"
    assert r.last_request_fingerprint["tools"] == 1

    # chat：帶 tools 時 SessionControl 會降成 none；不帶 tools 維持
    c = _stage(_Backend("chat", "high"), tools=_TOOLS, tool_executor=_executor)
    await c.prewarm()
    assert c.last_request_fingerprint["effort"] == "none"
    await _run_turn(c)
    assert c.last_request_fingerprint["effort"] == "none"

    n = _stage(_Backend("chat", "high"))
    await n.prewarm()
    assert n.last_request_fingerprint["effort"] == "high"
    assert n.last_request_fingerprint["tools"] == 0


async def test_預熱與正常輪都不自帶_reasoning_參數():
    b = _Backend()
    stage = _stage(b)
    await stage.prewarm()
    await _run_turn(stage)
    for req in b.requests:
        assert "reasoning" not in req and "reasoning_effort" not in req
