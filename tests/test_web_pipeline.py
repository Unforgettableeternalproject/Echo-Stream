"""直接驗證 Web 與管線的播放契約，不建立 HTTP listener 或服務執行緒。"""

from __future__ import annotations

import asyncio
import json

import pytest

from echo_stream.contracts.cancellation import CancelReason
from echo_stream.contracts.types import Utterance
from echo_stream.core.gap_filler import GapFiller
from echo_stream.core.pipeline import PipelineRunner
from echo_stream.core.tracer import LatencyTracer
from echo_stream.fakes import FakeSpeakStage, FakeThinkStage
from echo_stream.web.server import FRAME_AUDIO_END, FRAME_META, StreamService


def build_service():
    # 跳過建構子的執行緒與日誌設定，只注入 _run 使用的 fake 依賴。
    service = object.__new__(StreamService)
    service._loop = asyncio.get_running_loop()
    service._playbacks = {}
    service.tracer = LatencyTracer()
    service._think = FakeThinkStage(
        responses=["這是一句測試。"],
        first_token_delay_s=0,
        token_interval_s=0,
    )
    service._runner = PipelineRunner(
        think_stage=service._think,
        speak_stage=FakeSpeakStage(rtf=0, first_chunk_extra_s=0),
        tracer=service.tracer,
    )
    service.gap_filler = GapFiller()
    service._tool_counts = {}
    service._log = lambda event: None
    service._save_active_session = lambda: None
    service._dream_scheduler = None
    service._memory = None
    return service


@pytest.mark.parametrize("action", ["complete", "interrupt", "aborted_ack"])
async def test_Web生成結束後仍等播放確認並保留插話能力(action):
    service = build_service()
    frames = []
    ended = asyncio.Event()

    async def emit(payload):
        frames.append(payload)
        if payload[0] == FRAME_AUDIO_END:
            ended.set()

    task = asyncio.create_task(service._run(Utterance(text="你好"), emit))
    try:
        await asyncio.wait_for(ended.wait(), timeout=2)
        assert not task.done()
        assert service._runner.is_speaking
        assert not service._think.committed
        assert all(payload[0] != FRAME_META for payload in frames)
        end = json.loads(next(p[5:] for p in frames if p[0] == FRAME_AUDIO_END))
        state = service._playbacks[end["generation"]]
        if action == "interrupt":
            assert service._runner.interrupt(CancelReason.BARGE_IN)
        elif action == "aborted_ack":
            assert await service._apply_playback_report_on_loop(
                state,
                {
                    "generation": end["generation"],
                    "turn_id": end["turn_id"],
                    "played_s": 0,
                    "status": "aborted",
                },
            )
        else:
            assert await state.apply_report(
                {
                    "generation": end["generation"],
                    "turn_id": end["turn_id"],
                    "played_s": end["duration_s"],
                    "status": "complete",
                }
            )
        await asyncio.wait_for(task, timeout=2)
        meta = json.loads(next(p[5:] for p in frames if p[0] == FRAME_META))
        interrupted = action != "complete"
        assert meta["phase"] == ("cancelled" if interrupted else "done")
        assert len(service._think.committed) == 1
        assert meta["spoken"] == ("" if interrupted else meta["generated"])
        assert not service._runner.is_speaking
        assert not service._playbacks
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
