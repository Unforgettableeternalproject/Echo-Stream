"""命令列入口。

Phase 0 驗收條件::

    python -m echo_stream demo --fake

要能完整跑一輪對話並輸出延遲報告。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from .contracts.cancellation import CancelReason
from .contracts.turn import TurnPolicy
from .core.pipeline import PipelineRunner
from .core.splitter import SentenceSplitter, SplitPolicy
from .core.tracer import LatencyTracer
from .fakes import (
    FakeAudioSink,
    FakeAudioSource,
    FakeInputStage,
    FakeSpeakStage,
    FakeThinkStage,
)

DEFAULT_PROMPTS = [
    "你好，今天過得怎麼樣？",
    "那你覺得這個做法可行嗎？",
]

DEFAULT_RESPONSES = [
    "嗯，我想想。今天大致上還算順利，把幾個卡了很久的問題處理掉了，"
    "感覺輕鬆不少。不過還有一些收尾的工作要做，晚點再繼續吧。",
    "可行，但有前提。如果資料量維持在現在的規模，這個做法完全沒問題；"
    "一旦成長到十倍，瓶頸就會出現在檢索那一段，到時候要換索引結構。",
]


async def _run_demo(args: argparse.Namespace) -> int:
    tracer = LatencyTracer(output_path=args.trace)
    split_policy = SplitPolicy(
        first_min_weight=args.first_min,
        first_max_weight=args.first_max,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    turns = args.turns
    prompts = (DEFAULT_PROMPTS * ((turns // len(DEFAULT_PROMPTS)) + 1))[:turns]

    input_stage = FakeInputStage(texts=prompts, stt_delay_s=args.stt_delay)
    think_stage = FakeThinkStage(
        responses=DEFAULT_RESPONSES,
        first_token_delay_s=args.llm_delay,
        memory_delay_s=args.memory_delay,
        split_policy=split_policy,
        tracer=tracer,
    )
    speak_stage = FakeSpeakStage(first_chunk_extra_s=args.tts_extra, rtf=args.rtf)
    sink = FakeAudioSink(realtime=not args.no_realtime)

    runner = PipelineRunner(
        input_stage=input_stage,
        think_stage=think_stage,
        speak_stage=speak_stage,
        sink=sink,
        tracer=tracer,
        policy=TurnPolicy(),
    )
    await runner.prepare()

    barge_task = None
    if args.barge_in is not None:
        barge_task = asyncio.ensure_future(_barge_in_after(runner, args.barge_in))

    print("═" * 56)
    print("  Echo Stream — Phase 0 骨架驗證（全 fake，不需要 GPU）")
    print("═" * 56)

    try:
        async for result in runner.run_session(FakeAudioSource()):
            print()
            print(f"▸ 使用者：{result.utterance_text}")
            print(f"▸ 生成　：{result.generated_text}")
            if result.spoken_text != result.generated_text:
                print(f"▸ 實際播出：{result.spoken_text or '（什麼都沒播出）'}")
                if result.cancel_reason:
                    print(
                        f"  （被 {result.cancel_reason} 中止——"
                        f"寫回對話歷史的是這一段，不是完整生成內容）"
                    )
            print()
            print(tracer.report(result.turn_id))
    finally:
        if barge_task is not None and not barge_task.done():
            barge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await barge_task
        await runner.aclose()

    print()
    print("═" * 56)
    print("  " + tracer.summary())
    if args.trace:
        print(f"  逐 turn 打點已寫入 {args.trace}")
    print("═" * 56)
    return 0


async def _barge_in_after(runner: PipelineRunner, delay_s: float) -> None:
    """模擬使用者在機器人講話途中插話。"""
    await asyncio.sleep(delay_s)
    if runner.interrupt(CancelReason.BARGE_IN):
        print(f"\n  ⚡ [{delay_s:.1f}s] 使用者插話 → 中止播放並往上游傳播")


def _run_split(args: argparse.Namespace) -> int:
    """離線檢查切分結果。調參時用——不必跑整條管線就能看切點。"""

    async def _go() -> None:
        async def tokens():
            for ch in args.text:
                yield ch

        policy = SplitPolicy(
            first_min_weight=args.first_min,
            first_max_weight=args.first_max,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        )
        splitter = SentenceSplitter(policy)
        async for sentence in splitter.split(tokens(), "cli"):
            if not sentence.text:
                continue
            flag = "首句" if sentence.is_first else "　　"
            print(f"  [{sentence.index:>2}] {flag} ({sentence.split_reason:<10}) {sentence.text}")

    asyncio.run(_go())
    return 0


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--first-min", type=float, default=8.0, help="首句最小長度權重")
    parser.add_argument("--first-max", type=float, default=20.0, help="首句強制切分權重")
    parser.add_argument("--min-weight", type=float, default=12.0, help="後續句最小長度權重")
    parser.add_argument("--max-weight", type=float, default=40.0, help="後續句強制切分權重")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echo_stream",
        description="Echo Stream — 串流語音對話管線整合層",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="跑一輪 fake 管線並輸出延遲報告")
    demo.add_argument("--fake", action="store_true", help="使用 fake 模組（Phase 0 唯一選項）")
    demo.add_argument("--turns", type=int, default=1, help="要跑幾輪對話")
    demo.add_argument("--barge-in", type=float, default=None, metavar="秒",
                      help="在第 N 秒模擬使用者插話")
    demo.add_argument("--trace", type=str, default=None, help="打點輸出的 JSONL 路徑")
    demo.add_argument("--no-realtime", action="store_true",
                      help="sink 不模擬即時播放（測試用，會讓 barge-in 善後失真）")
    demo.add_argument("--stt-delay", type=float, default=1.2, help="模擬 STT 轉錄延遲（秒）")
    demo.add_argument("--llm-delay", type=float, default=0.4, help="模擬 LLM 首 token 延遲（秒）")
    demo.add_argument("--memory-delay", type=float, default=1.5,
                      help="模擬 Memory retrieve 耗時（秒，與 LLM 平行）")
    demo.add_argument("--tts-extra", type=float, default=0.6, help="模擬 TTS 首段額外開銷（秒）")
    demo.add_argument("--rtf", type=float, default=0.30, help="模擬 TTS 的 Real-Time Factor")
    _add_split_args(demo)
    demo.set_defaults(func=lambda a: asyncio.run(_run_demo(a)))

    split = sub.add_parser("split", help="檢查切分結果（調參用）")
    split.add_argument("text", help="要切分的文字")
    _add_split_args(split)
    split.set_defaults(func=_run_split)

    return parser


def _force_utf8_stdout() -> None:
    """Windows 主控台預設 cp950，會讓報告裡的方框繪製字元與全形標點炸掉。

    這不是美觀問題——UnicodeEncodeError 會直接中斷 demo，讓 Phase 0 的
    驗收看起來像管線壞了，實際上只是終端機編碼。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n（中斷）", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
