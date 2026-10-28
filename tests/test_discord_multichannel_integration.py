"""DiscordAudioSource → MultiChannelSttInputStage 端到端（fake backend、無 GPU）。

驗 bot 實際會走的那條路：48k stereo 封包 → 每成員重取樣 → 多聲道 stage
→ 帶 speaker_id 的 Utterance，以及 Discord 不說話就不送封包時，turn 仍會由背景 tick 補靜音結束。
"""

from __future__ import annotations

import asyncio
import struct
import time

from echo_stream.adapters.stt_discord import DiscordAudioSource

from .test_input_multichannel import EchoBackend, make_stage, open_session

DISCORD_SR = 48_000
FRAME_BYTES = 3840  # Discord 20ms：960 frames × 2 聲道 × int16


def stereo_dc(value: int, seconds: float) -> bytes:
    n = int(DISCORD_SR * seconds)
    return struct.pack(f"<{n * 2}h", *([value] * (n * 2)))


def frames(pcm: bytes) -> list[bytes]:
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


async def test_兩位成員經_Discord_入口各自出一則_Utterance() -> None:
    stage = make_stage(EchoBackend(), tick_interval_s=0.02)
    source = DiscordAudioSource(stage)
    session = await open_session(stage)
    try:
        a, b = frames(stereo_dc(3000, 0.5)), frames(stereo_dc(6000, 0.4))
        for i in range(max(len(a), len(b))):
            if i < len(a):
                source.feed("111", a[i])
            if i < len(b):
                source.feed("222", b[i])
        # 之後完全不送封包（Discord 靜音行為）
        results = await session.wait_for(2, timeout=4)
        by_speaker = {u.speaker_id: u for u in results}
        assert set(by_speaker) == {"111", "222"}
        # 重取樣 DC 增益為 1；backend 取最大值，階躍起點的 Gibbs 過衝約 8%，容差 12%
        assert abs(int(by_speaker["111"].text) - 3000) <= 3000 * 0.12
        assert abs(int(by_speaker["222"].text) - 6000) <= 6000 * 0.12
        # 未注入 clock 時由 stage 蓋時間戳：與 perf_counter 同一時域，可直接相減
        md = by_speaker["111"].metadata
        assert abs(time.perf_counter() - md["stt_done_ts"]) < 5
        assert md["speech_end_ts"] <= md["turn_end_ts"] <= md["stt_done_ts"]
    finally:
        await session.close()


async def test_斷線重連_leave_全部再_join_後照常運作() -> None:
    stage = make_stage(EchoBackend(), tick_interval_s=0.02)
    source = DiscordAudioSource(stage)
    session = await open_session(stage)
    try:
        for f in frames(stereo_dc(3000, 0.2)):
            source.feed("111", f)
        source.leave("111", flush=False)
        await asyncio.sleep(0.05)
        source.join("111")
        for f in frames(stereo_dc(4000, 0.5)):
            source.feed("111", f)
        (utt,) = await session.wait_for(1, timeout=4)
        assert utt.speaker_id == "111"
        assert abs(int(utt.text) - 4000) <= 4000 * 0.12
    finally:
        await session.close()
