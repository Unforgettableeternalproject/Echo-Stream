"""直接驗證 Web 與管線的播放契約，不建立 HTTP listener 或服務執行緒。"""

from __future__ import annotations

import asyncio
import collections
import json

import pytest

from echo_stream.contracts.cancellation import CancellationToken, CancelReason
from echo_stream.contracts.types import Utterance
from echo_stream.core.gap_filler import GapFiller
from echo_stream.core.pipeline import PipelineRunner
from echo_stream.core.tracer import LatencyTracer
from echo_stream.fakes import FakeSpeakStage, FakeThinkStage
from echo_stream.web.server import (
    FRAME_AUDIO_ABORT,
    FRAME_AUDIO_END,
    FRAME_META,
    StreamService,
)


def build_service():
    # 跳過建構子的執行緒與日誌設定，只注入 _run 使用的 fake 依賴。
    service = object.__new__(StreamService)
    service._loop = asyncio.get_running_loop()
    service._playbacks = {}
    service._finished_playbacks = collections.OrderedDict()
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


def _end_info(frames):
    return json.loads(next(p[5:] for p in frames if p[0] == FRAME_AUDIO_END))


async def _wait_for_end(frames, ended):
    await asyncio.wait_for(ended.wait(), timeout=2)
    return _end_info(frames)


async def test_後端發起停止會等前端回報最終播放量再結算():
    service = build_service()
    service.playback_stop_confirm_s = 1.0
    frames = []
    ended = asyncio.Event()
    replies = []

    async def emit(payload):
        frames.append(payload)
        if payload[0] == FRAME_AUDIO_END:
            ended.set()
        elif payload[0] == FRAME_AUDIO_ABORT:
            # 模擬前端：收到後端停止後回報停在哪裡（這裡是整段都已播完）
            info = json.loads(payload[5:])
            state = service._playbacks[info["generation"]]
            replies.append(asyncio.ensure_future(service._apply_playback_report_on_loop(
                state,
                {
                    "generation": info["generation"],
                    "turn_id": info["turn_id"],
                    "played_s": state._written,
                    "status": "aborted",
                },
            )))

    task = asyncio.create_task(service._run(Utterance(text="你好"), emit))
    try:
        end = await _wait_for_end(frames, ended)
        # 後端自動插話（例如 voice loop 的 barge-in 偵測），前端尚未回報任何進度
        assert service._runner.interrupt(CancelReason.BARGE_IN)
        await asyncio.wait_for(task, timeout=2)
        assert replies and all(await asyncio.gather(*replies))
        meta = json.loads(next(p[5:] for p in frames if p[0] == FRAME_META))
        assert meta["phase"] == "cancelled"
        assert meta["played_seconds"] == pytest.approx(end["duration_s"], abs=1e-3)
        assert meta["spoken"] == meta["generated"]
        assert service._think.committed[0][1] == meta["generated"]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_後端停止確認逾時沿用最後已知播放量():
    service = build_service()
    service.playback_stop_confirm_s = 0.05
    frames = []
    ended = asyncio.Event()

    async def emit(payload):
        frames.append(payload)
        if payload[0] == FRAME_AUDIO_END:
            ended.set()

    task = asyncio.create_task(service._run(Utterance(text="你好"), emit))
    try:
        end = await _wait_for_end(frames, ended)
        state = service._playbacks[end["generation"]]
        assert await state.apply_report({
            "generation": end["generation"], "turn_id": end["turn_id"],
            "played_s": end["duration_s"], "status": "progress",
        })
        assert service._runner.interrupt(CancelReason.BARGE_IN)
        await asyncio.wait_for(task, timeout=1)
        meta = json.loads(next(p[5:] for p in frames if p[0] == FRAME_META))
        assert meta["spoken"] == meta["generated"]
        # 逾時後才到的回報不得再改寫已結算的播放量
        assert not await state.apply_report({
            "generation": end["generation"], "turn_id": end["turn_id"],
            "played_s": 0.0, "status": "aborted",
        })
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_重複的完成ACK冪等_回合結束後重送仍成功():
    service = build_service()
    frames = []
    ended = asyncio.Event()

    async def emit(payload):
        frames.append(payload)
        if payload[0] == FRAME_AUDIO_END:
            ended.set()

    task = asyncio.create_task(service._run(Utterance(text="你好"), emit))
    try:
        end = await _wait_for_end(frames, ended)
        report = {
            "generation": end["generation"], "turn_id": end["turn_id"],
            "played_s": end["duration_s"], "status": "complete",
        }
        state = service._playbacks[end["generation"]]
        assert await state.apply_report(report)
        assert await state.apply_report(report)  # 回合進行中的重送
        await asyncio.wait_for(task, timeout=2)
        assert not service._playbacks
        # 回合已結算：同一份 ACK 重送（前一次回應遺失）必須仍回成功
        assert await asyncio.to_thread(service.report_playback, dict(report))
        # 衝突的最終狀態仍拒絕
        assert not await asyncio.to_thread(
            service.report_playback, {**report, "status": "aborted"}
        )
        assert len(service._think.committed) == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _OneUtteranceStage:
    """吐一句話後保持 session 開著的 fake STT。"""

    def __init__(self, text):
        self.text = text
        self.on_voice_event = None
        self.on_partial = None

    async def stream(self, source, token):
        yield Utterance(text=self.text)
        await token.wait()


async def test_語音佇列塞滿時插話仍能解除等待且中止訊號優先送出():
    service = build_service()
    service.playback_stop_confirm_s = 0.05
    service._stt = _OneUtteranceStage("你好")
    service._voice = None
    service._discard_turn = False
    session = {
        "sid": "s1",
        "source": None,
        "out": asyncio.Queue(maxsize=1),
        "token": CancellationToken(),
        "control_tasks": set(),
        "disconnected_event": asyncio.Event(),
        "urgent": collections.deque(),
        "urgent_event": asyncio.Event(),
    }
    await session["out"].put(b"stale")  # 前端讀得慢，佇列已滿
    service._voice = session
    loop_task = asyncio.create_task(service._voice_loop(session))
    try:
        for _ in range(200):
            if service._runner.is_speaking:
                break
            await asyncio.sleep(0.005)
        assert service._runner.is_speaking
        await asyncio.sleep(0.05)  # 讓管線卡在佇列 put 上
        assert service._runner.interrupt(CancelReason.BARGE_IN)
        for _ in range(200):
            if service._think.committed:
                break
            await asyncio.sleep(0.005)
        assert service._think.committed, "插話後回合仍卡在塞滿的語音佇列上"
        first = await asyncio.wait_for(service._voice_next(session), timeout=1)
        assert first[0] == FRAME_AUDIO_ABORT, "中止訊號被排在塞滿的佇列後面"
    finally:
        # 模擬前端斷線收尾，避免 voice loop 的結束哨兵卡在塞滿的佇列上
        session["disconnected"] = True
        session["disconnected_event"].set()
        session["token"].cancel(CancelReason.SUPERSEDED)
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
