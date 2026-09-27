"""不啟動 HTTP listener 的 Web 播放協定測試。"""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from echo_stream.contracts.cancellation import CancelledError, CancelReason
from echo_stream.contracts.types import AudioChunk
from echo_stream.web.server import (
    FRAME_AUDIO,
    FRAME_AUDIO_ABORT,
    FRAME_AUDIO_END,
    FRAME_EVENT,
    _EmitSink,
    _PlaybackState,
)


def test_playback_generation_拒絕舊回報並只累進實播秒數():
    async def scenario():
        state = _PlaybackState("turn-a", "a" * 32, asyncio.get_running_loop())
        state.add_chunk(2.0)
        assert not await state.apply_report({
            "generation": "b" * 32, "turn_id": "turn-a", "played_s": 1.0,
        })
        assert await state.apply_report({
            "generation": "a" * 32, "turn_id": "turn-a", "played_s": 1.25,
        })
        assert await state.apply_report({
            "generation": "a" * 32, "turn_id": "turn-a", "played_s": 0.5,
        })
        assert state.played_duration_s == 1.25

    asyncio.run(scenario())


def test_sink_先送_generation_end_再等待完整播放回報():
    async def scenario():
        state = _PlaybackState("turn-b", "c" * 32, asyncio.get_running_loop())
        frames: list[bytes] = []
        sink = _EmitSink(
            frames.append, 0.0, state, drain_timeout_s=0.2, flush_after_end=True
        )
        await sink.write(AudioChunk(
            pcm=b"\x00\x00" * 100,
            sample_rate=1_000,
            turn_id="turn-b",
            sentence_index=0,
        ))

        drain = asyncio.create_task(sink.drain())
        await asyncio.sleep(0)
        assert frames[-2][0] == FRAME_AUDIO_END
        assert frames[-1][0] == FRAME_EVENT
        assert json.loads(frames[-1][5:].decode("utf-8"))["event"] == "flush"
        end = json.loads(frames[-2][5:].decode("utf-8"))
        assert end["generation"] == "c" * 32
        assert not drain.done()

        assert await state.apply_report({
            "generation": "c" * 32,
            "turn_id": "turn-b",
            "played_s": end["duration_s"],
            "status": "complete",
        })
        await drain
        assert state.status == "complete"
        assert sink.played_duration_s == pytest.approx(0.1)
        audio = next(payload for payload in frames if payload[0] == FRAME_AUDIO)
        assert struct.unpack(">I", audio[1:5])[0] >= 28
        assert audio[5 + 28 : 5 + 32] == b"RIFF"

    asyncio.run(scenario())


def test_sink_播放回報逾時明確失敗且不虛報已播():
    async def scenario():
        state = _PlaybackState("turn-c", "d" * 32, asyncio.get_running_loop())
        frames: list[bytes] = []
        sink = _EmitSink(frames.append, 0.0, state, drain_timeout_s=0.01)
        await sink.write(AudioChunk(
            pcm=b"\x00\x00" * 100,
            sample_rate=1_000,
            turn_id="turn-c",
            sentence_index=0,
        ))
        with pytest.raises(TimeoutError, match="逾時"):
            await sink.drain()
        assert state.status == "timeout"
        assert sink.played_duration_s == 0.0
        sink.stop()
        assert any(payload[0] == FRAME_AUDIO_ABORT for payload in frames)

    asyncio.run(scenario())


def test_先到的_aborted_ACK_保留_barge_in_取消語意():
    async def scenario():
        state = _PlaybackState("turn-stop", "f" * 32, asyncio.get_running_loop())
        state.add_chunk(1.0)
        assert await state.apply_report({
            "generation": "f" * 32, "turn_id": "turn-stop",
            "played_s": 0.3, "status": "aborted",
        })
        assert not await state.apply_report({
            "generation": "f" * 32, "turn_id": "turn-stop",
            "played_s": 1.0, "status": "complete",
        })
        with pytest.raises(CancelledError) as exc:
            await state.drain(lambda _payload: None, timeout_s=0.1)
        assert exc.value.reason is CancelReason.BARGE_IN
        assert state.played_duration_s == pytest.approx(0.3)

    asyncio.run(scenario())


def test_bounded_voice_queue_滿時等待而不丟音訊_frame():
    async def scenario():
        out: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        await out.put(b"first")
        token = asyncio.Event()

        async def emit(payload: bytes) -> None:
            put_task = asyncio.create_task(out.put(payload))
            cancel_task = asyncio.create_task(token.wait())
            try:
                done, _ = await asyncio.wait(
                    {put_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_task in done and put_task not in done:
                    put_task.cancel()
                    return
                put_task.result()
            finally:
                for task in (put_task, cancel_task):
                    if not task.done():
                        task.cancel()

        sink = _EmitSink(emit, 0.0, _PlaybackState("turn-q", "e" * 32, asyncio.get_running_loop()))
        chunk = AudioChunk(
            pcm=b"\x00\x00" * 100,
            sample_rate=1_000,
            turn_id="turn-q",
            sentence_index=0,
        )
        producer = asyncio.create_task(sink.write(chunk))
        await asyncio.sleep(0)
        assert not producer.done()
        assert await out.get() == b"first"
        # sink 先送 begin，再送真正的 AUDIO frame；兩段都必須保留。
        begin = await out.get()
        await asyncio.wait_for(producer, timeout=0.1)
        audio = await out.get()
        assert begin[0] != FRAME_AUDIO
        assert audio[0] == FRAME_AUDIO

        # 佇列塞滿時取消會解除 sink.write，不留背景 put waiter。
        await out.put(b"occupy")
        blocked = asyncio.create_task(sink.write(chunk))
        await asyncio.sleep(0)
        token.set()
        await asyncio.wait_for(blocked, timeout=0.1)
        assert await out.get() == b"occupy"

    asyncio.run(scenario())


# --- 播放逾時診斷（純觀察，不改判定）---


def _diag_report(generation, turn_id, played, status="progress", **diag):
    return {
        "generation": generation, "turn_id": turn_id,
        "played_s": played, "status": status, "diag": diag,
    }


def test_播放診斷只保存不改變回報判定():
    async def scenario():
        gen = "d" * 32
        state = _PlaybackState("turn-d", gen, asyncio.get_running_loop())
        for _ in range(3):
            state.add_chunk(1.0)
        # 超量與超長輸入會被截斷；diag 本身不影響回報被接受
        assert await state.apply_report(_diag_report(
            gen, "turn-d", 2.0,
            ended_chunks=2, expected_chunks=None, received_chunks=3,
            missing_count=1, issue_count=15,
            missing=[{"index": 2, "stage": "decoding"}] + [{"index": i} for i in range(20)],
            issues=["x" * 500] * 15,
        ))
        diag = state._client_diag
        assert diag["missing"][0] == {"index": 2, "stage": "decoding"}
        assert len(diag["missing"]) == 10
        assert len(diag["issues"]) == 10
        assert all(len(x) == 200 for x in diag["issues"])
        assert state.status == "pending"

        # 格式錯的 diag 當作沒送：回報照常接受、先前保存的診斷保留
        assert await state.apply_report({
            "generation": gen, "turn_id": "turn-d", "played_s": 2.5,
            "status": "progress", "diag": "garbage",
        })
        assert state._client_diag is diag
        assert state.played_duration_s == 2.5

        # complete 判定仍只看 played 與 ended，diag 說缺席也不擋
        drain = asyncio.create_task(state.drain(lambda _p: None, timeout_s=1.0))
        await asyncio.sleep(0)
        assert await state.apply_report(_diag_report(
            gen, "turn-d", 3.0, status="complete",
            ended_chunks=2, expected_chunks=3, missing=[{"index": 2, "stage": "scheduled"}],
        ))
        await asyncio.wait_for(drain, timeout=0.5)
        assert state.status == "complete"
        assert state.timeout_diag is None

    asyncio.run(scenario())


def test_播放逾時時快照缺席_chunk_與前端_issues():
    async def scenario():
        gen = "f" * 32
        state = _PlaybackState("turn-f", gen, asyncio.get_running_loop())
        for duration in (2.0, 1.5, 4.25):
            state.add_chunk(duration)
        assert await state.apply_report(_diag_report(
            gen, "turn-f", 3.5,
            ended_chunks=2, expected_chunks=3, received_chunks=3,
            missing_count=1, issue_count=1,
            missing=[{"index": 2, "stage": "failed"}],
            issues=["failed_at_decoding #2 4.25s: EncodingError"],
        ))
        with pytest.raises(TimeoutError, match="逾時"):
            await state.drain(lambda _p: None, timeout_s=0.05)

        diag = state.timeout_diag
        assert diag["timeout_s"] == 0.05
        assert diag["written_s"] == 7.75
        assert diag["played_s"] == 3.5
        assert diag["chunks"] == 3
        assert diag["reports"] == 1
        assert diag["last_report_age_s"] >= 0.04
        assert diag["client"] == {
            "ended_chunks": 2, "expected_chunks": 3, "received_chunks": 3,
            "missing_count": 1, "issue_count": 1,
        }
        assert diag["missing"] == [{"index": 2, "stage": "failed", "duration_s": 4.25}]
        assert diag["issues"] == ["failed_at_decoding #2 4.25s: EncodingError"]

    asyncio.run(scenario())


def test_舊前端沒有_diag_逾時快照仍輸出後端側數字():
    async def scenario():
        gen = "0" * 32
        state = _PlaybackState("turn-0", gen, asyncio.get_running_loop())
        state.add_chunk(1.0)
        with pytest.raises(TimeoutError):
            await state.drain(lambda _p: None, timeout_s=0.01)
        diag = state.timeout_diag
        assert diag["client"] is None
        assert diag["missing"] == [] and diag["issues"] == []
        assert diag["reports"] == 0 and diag["last_report_age_s"] is None
        assert diag["written_s"] == 1.0 and diag["played_s"] == 0.0

    asyncio.run(scenario())
