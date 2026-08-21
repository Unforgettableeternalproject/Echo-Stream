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
from .core.tracer import BUDGET_MS, MARK_THINK_FIRST_TOKEN, LatencyTracer
from .fakes import (
    FakeAudioSink,
    FakeAudioSource,
    FakeInputStage,
    FakeSpeakStage,
    FakeThinkStage,
)
from .sinks import WavFileSink

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


def _build_speak_stage(args: argparse.Namespace):
    """依旗標選 fake 或真 TTS。真 TTS 需要 GPU 與 echo_tts repo。"""
    if not getattr(args, "real_tts", False):
        return FakeSpeakStage(first_chunk_extra_s=args.tts_extra, rtf=args.rtf), None

    from .adapters.speak_indextts import IndexTTS2SpeakStage

    stage = IndexTTS2SpeakStage(
        device=args.device,
        language=args.language,
        speed=getattr(args, "speed", None),
        keep_wav=False,
    )
    return stage, stage.sample_rate


def _combined_system_prompt(explicit: str | None) -> str:
    """把 .env 的 ``SYSTEM_PROMPT``（U.E.P 人設）接在最前面。

    順序是刻意的：人設是基底，`--system` 是這次測試的補充指示，
    後者不該蓋掉前者。
    """
    from . import config

    parts = [config.get("SYSTEM_PROMPT") or "", explicit or ""]
    return "\n\n".join(p for p in parts if p)


def _run_serve(args: argparse.Namespace) -> int:
    from .web.server import serve

    serve(
        host=args.host,
        port=args.port,
        real_tts=args.real_tts or args.real,
        real_llm=args.real_llm or args.real,
        real_stt=args.real_stt or args.real,
        split_policy=SplitPolicy(
            first_min_weight=args.first_min,
            first_max_weight=args.first_max,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        ),
        system_prompt=_combined_system_prompt(args.system),
        trace_path=args.trace,
    )
    return 0


async def _run_llm_check(args: argparse.Namespace) -> int:
    """Phase 2 驗收：只跑 ThinkStage，量首 token 與首句延遲。

    不經過 PipelineRunner，也不接 TTS——這裡要看的是 LLM 段本身。
    特別是 reasoning effort 對首字延遲的影響：強度越高，吐出第一個 token
    前思考得越久，那段時間全部計入 TTFA。
    """
    import time

    from .adapters.think_llm import LLMThinkStage, build_backend
    from .contracts.cancellation import CancellationToken
    from .contracts.types import Utterance

    print("═" * 56)
    print("  Echo Stream — Phase 2 LLM 驗收（真 API，會消耗 token）")
    print("═" * 56)

    overrides = {}
    if args.effort:
        overrides["reasoning_effort"] = args.effort
    if args.model:
        overrides["model"] = args.model

    try:
        backend = build_backend(**overrides)
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 無法建立 backend：{type(exc).__name__}: {exc}")
        print("\n  檢查項：")
        print("  - .env 的 OPENAI_API_KEY 是否已填")
        print("  - ECHO_STREAM_SESSION_REPO 是否指向 TestSeparateSessionControl")
        return 1

    effort = getattr(backend, "reasoning_effort", "?")
    model = getattr(backend, "model_name", "?")
    print(f"\n▸ model={model}  effort={effort}  api={getattr(backend, 'api', '?')}")
    if hasattr(backend, "get_context_window_size"):
        window = backend.get_context_window_size()
        print(f"  有效 context window：{window:,}")

    split_policy = SplitPolicy(
        first_min_weight=args.first_min,
        first_max_weight=args.first_max,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    # 用 tracer 取得「首 token」的打點——它與「首句」是兩件事：
    # 首 token 反映 API 往返 + prefill + 推理，首句還要加上累積到切點的時間。
    # 分不開就不知道該優化哪一邊。
    tracer = LatencyTracer()
    stage = LLMThinkStage(
        backend,
        system_prompt=args.system or "",
        split_policy=split_policy,
        tracer=tracer,
    )

    print(f"\n▸ 送出：{args.text}\n")
    token = CancellationToken()
    utterance = Utterance(text=args.text)
    tracer.start(utterance.turn_id)
    t0 = time.perf_counter()
    first_sentence_ms: float | None = None
    sentences = []

    try:
        async for sentence in stage.stream(utterance, token):
            now = (time.perf_counter() - t0) * 1000
            if first_sentence_ms is None:
                first_sentence_ms = now
            if sentence.text:
                sentences.append((sentence, now))
                flag = "首句" if sentence.is_first else "　　"
                print(f"  [{sentence.index:>2}] {flag} {now:>7.0f}ms  {sentence.text}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 串流失敗：{type(exc).__name__}: {exc}")
        return 1
    finally:
        await stage.aclose()

    elapsed = (time.perf_counter() - t0) * 1000
    trace = tracer.get(utterance.turn_id)
    mark = trace.marks.get(MARK_THINK_FIRST_TOKEN) if trace else None
    start = trace.marks.get("turn_start") if trace else None
    first_token_ms = (mark - start) * 1000 if (mark and start) else None

    print()
    print("─" * 56)
    budget = BUDGET_MS["llm_first_sentence"]
    verdict = "✓" if first_sentence_ms and first_sentence_ms <= budget else "✗"
    if first_token_ms is not None:
        print(f"  首 token      {first_token_ms:>8.0f}ms   （API 往返 + prefill + 推理）")
    print(f"  LLM 首句      {first_sentence_ms:>8.0f}ms   預算 {budget:.0f}ms   {verdict}")
    if first_token_ms is not None and first_sentence_ms is not None:
        accumulate = first_sentence_ms - first_token_ms
        print(f"    └ 累積成句  {accumulate:>8.0f}ms   （切分閾值決定，可調）")
    print(f"  全部生成完    {elapsed:>8.0f}ms")
    print(f"  句數          {len(sentences)}")

    if first_sentence_ms and first_sentence_ms > budget:
        print()
        if first_token_ms is not None and first_token_ms > budget * 0.7:
            print("  ⚠ 瓶頸在**首 token**，不是切分。")
            print("    切短首句救不了——那只影響上面的「累積成句」那一段。")
            print(f"    目前 effort={effort}；若調整 effort 沒有明顯差異，")
            print("    代表延遲來自 API 往返而非推理，要往 prompt caching /")
            print("    網路路徑 / 換更小模型的方向查。")
        else:
            print("  ⚠ 瓶頸在**累積成句**——首句切短就能改善。")
            print("    試試 --first-min 4 --first-max 10。")
    print("─" * 56)
    return 0


async def _run_tts_check(args: argparse.Namespace) -> int:
    """Phase 1 驗收：只跑 SpeakStage，量首段延遲。

    不經過 PipelineRunner——這裡要驗的是 adapter 與模型本身，
    混進管線只會讓「哪一層慢」變得不明確。
    """
    import time

    from .adapters.speak_indextts import IndexTTS2SpeakStage
    from .contracts.cancellation import CancellationToken
    from .contracts.types import Sentence, new_turn_id
    from .core.splitter import SentenceSplitter, SplitPolicy

    print("═" * 56)
    print("  Echo Stream — Phase 1 TTS 驗收（真 IndexTTS2，需要 GPU）")
    print("═" * 56)

    stage = IndexTTS2SpeakStage(
        device=args.device,
        language=args.language,
        speed=args.speed,
        warmup=args.warmup,
    )

    print(f"\n▸ 載入模型（device={stage.device}, speed={stage.speed:+.2f}）…")
    load_t0 = time.perf_counter()
    try:
        await stage.prepare()
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 模型載入失敗：{type(exc).__name__}: {exc}")
        print("\n  檢查項：")
        print("  - .env 的 ECHO_STREAM_TTS_REPO 是否指向 TestSeparateTTSSystem")
        print("  - 是否在有 torch + CUDA 的環境下執行")
        return 1
    load_ms = (time.perf_counter() - load_t0) * 1000
    print(f"  載入完成 {load_ms:.0f}ms")

    turn_id = new_turn_id()
    policy = SplitPolicy(
        first_min_weight=args.first_min,
        first_max_weight=args.first_max,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    async def chars():
        for ch in args.text:
            yield ch

    sentences: list[Sentence] = []
    async for s in SentenceSplitter(policy).split(chars(), turn_id):
        if s.text:
            sentences.append(s)

    print(f"\n▸ 切成 {len(sentences)} 句：")
    for s in sentences:
        flag = "首句" if s.is_first else "　　"
        print(f"    [{s.index:>2}] {flag} ({s.split_reason:<10}) {s.text}")

    async def sentence_stream():
        for s in sentences:
            yield s

    sink = WavFileSink(args.out, realtime=False)
    token = CancellationToken()

    print("\n▸ 合成中…")
    t0 = time.perf_counter()
    first_ms: float | None = None
    per_sentence: dict[int, float] = {}
    count = 0

    try:
        async for chunk in stage.stream(sentence_stream(), token):
            now = (time.perf_counter() - t0) * 1000
            if first_ms is None:
                first_ms = now
                print(f"    ⚡ 首段音訊 {now:.0f}ms")
            per_sentence.setdefault(chunk.sentence_index, now)
            await sink.write(chunk)
            count += 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 合成失敗：{type(exc).__name__}: {exc}")
        return 1
    finally:
        await stage.aclose()

    elapsed = (time.perf_counter() - t0) * 1000
    path = sink.flush()

    print()
    print("─" * 56)
    print(f"  首段音訊（TTS 段的 TTFA）  {first_ms:.0f}ms   預算 1000ms"
          f"   {'✓' if first_ms and first_ms <= 1000 else '✗'}")
    print(f"  總合成時間                {elapsed:.0f}ms")
    print(f"  音訊段數                  {count}")
    print(f"  音訊總長                  {sink.written_duration_s:.2f}s")
    if sink.written_duration_s > 0:
        print(f"  RTF                       {elapsed / 1000 / sink.written_duration_s:.3f}")
    print("\n  逐句首段時間：")
    for idx in sorted(per_sentence):
        print(f"    句 {idx}：{per_sentence[idx]:.0f}ms")
    if path:
        print(f"\n  音訊已寫入 {path}")
    print("─" * 56)
    return 0


async def _run_stt_check(args: argparse.Namespace) -> int:
    """Phase 3 驗收：只跑 SttInputStage，量 turn 切分與轉錄延遲。

    不經過 PipelineRunner——這裡要驗的是 VAD、turn 判定、轉錄的接線。
    「使用者說完 → 拿到 Utterance」這段就是 TTFA 裡 STT 要吃掉的預算。
    """
    import time

    from .adapters.input_stt import SttInputStage
    from .contracts.cancellation import CancellationToken

    print("═" * 56)
    print("  Echo Stream — Phase 3 STT 驗收（真 Whisper，載入需要時間）")
    print("═" * 56)

    if args.wav:
        from .sources import WavFileSource

        source = WavFileSource(args.wav, realtime=True)
        print(f"\n▸ 來源：{args.wav}（realtime 播放）")
    else:
        from .sources import MicrophoneSource

        source = MicrophoneSource(device=args.device_index)
        print("\n▸ 來源：麥克風（Ctrl+C 結束）")

    policy = TurnPolicy(
        silence_threshold_s=args.silence,
        min_utterance_s=args.min_utterance,
    )
    stage = SttInputStage(language=args.language, policy=policy)

    print("▸ 載入模型…")
    load_t0 = time.perf_counter()
    try:
        await stage.prepare()
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 模型載入失敗：{type(exc).__name__}: {exc}")
        print("\n  檢查項：")
        print("  - ECHO_STREAM_STT_REPO 是否指向 TestSeparateSTTSystem")
        print("  - 該 repo 的模型與設定是否可用（faster-whisper checkpoint）")
        return 1
    print(f"  載入完成 {(time.perf_counter() - load_t0) * 1000:.0f}ms")
    print(f"  靜音門檻 {policy.silence_threshold_s * 1000:.0f}ms"
          f" · 最短語音 {policy.min_utterance_s * 1000:.0f}ms\n")

    token = CancellationToken()
    count = 0
    latencies: list[float] = []
    try:
        async for utterance in stage.stream(source, token):
            now = time.perf_counter()
            stt_ms = (now - utterance.ended_at) * 1000
            speech_s = max(0.0, utterance.ended_at - utterance.started_at)
            latencies.append(stt_ms)
            count += 1
            print(f"  [{count:>2}] 語音 {speech_s:>4.1f}s → 轉錄 {stt_ms:>6.0f}ms"
                  f"  ({utterance.language or '?'}, conf={utterance.confidence:.2f})")
            print(f"       {utterance.text}")
            if args.turns and count >= args.turns:
                token.cancel(CancelReason.SHUTDOWN)
                break
    except KeyboardInterrupt:
        token.cancel(CancelReason.SHUTDOWN)
    finally:
        await stage.aclose()

    if latencies:
        print()
        print("─" * 56)
        budget = BUDGET_MS.get("stt", 700.0)
        median = sorted(latencies)[len(latencies) // 2]
        verdict = "✓" if median <= budget else "✗"
        print(f"  轉錄延遲 med  {median:>8.0f}ms   預算 {budget:.0f}ms   {verdict}")
        print(f"  轉錄延遲 max  {max(latencies):>8.0f}ms")
        print(f"  turn 數       {count}")
        print("─" * 56)
    else:
        print("\n  （沒有切出任何 turn——檢查麥克風音量或 VAD 閾值）")
    return 0


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
    speak_stage, real_sample_rate = _build_speak_stage(args)
    if args.out:
        sink = WavFileSink(
            args.out, sample_rate=real_sample_rate, realtime=not args.no_realtime
        )
    else:
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

    written = getattr(sink, "flush", None)
    if callable(written):
        path = written()
        if path:
            print(f"\n  音訊已寫入 {path}")

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
    demo.add_argument("--real-tts", action="store_true",
                      help="SpeakStage 換成真的 IndexTTS2（需要 GPU 與 echo_tts repo）")
    demo.add_argument("--device", default=None, help="TTS 裝置（cuda / cpu）")
    demo.add_argument("--language", default=None, help="TTS 語言（留空自動偵測）")
    demo.add_argument("--speed", type=float, default=None, metavar="-1~1",
                      help="TTS 語速（引擎預設 0.0，正值變快）")
    demo.add_argument("--out", default=None, metavar="檔案", help="把音訊寫成 WAV")
    _add_split_args(demo)
    demo.set_defaults(func=lambda a: asyncio.run(_run_demo(a)))

    web = sub.add_parser("serve", help="開 Web 前端 host 整條管線")
    web.add_argument("--host", default=None)
    web.add_argument("--port", type=int, default=None)
    web.add_argument("--real-tts", action="store_true", help="用真 IndexTTS2（需要 GPU）")
    web.add_argument("--real-llm", action="store_true", help="用真 OpenAI（會消耗 token）")
    web.add_argument("--real-stt", action="store_true", help="用真 Whisper（載入需要時間）")
    web.add_argument("--real", action="store_true", help="STT / LLM / TTS 全用真的")
    web.add_argument("--system", default=None, help="system prompt")
    web.add_argument("--trace", default=None, help="打點輸出的 JSONL 路徑")
    _add_split_args(web)
    web.set_defaults(func=_run_serve)

    llm = sub.add_parser("llm-check", help="Phase 2 驗收：只跑真 LLM，量首句延遲")
    llm.add_argument("text", help="要送出的訊息")
    llm.add_argument("--effort", default=None,
                     metavar="none|low|medium|high|xhigh|max",
                     help="推理強度（覆寫 .env）。直接影響首字延遲")
    llm.add_argument("--model", default=None, help="模型（覆寫 .env）")
    llm.add_argument("--system", default=None, help="system prompt")
    _add_split_args(llm)
    llm.set_defaults(func=lambda a: asyncio.run(_run_llm_check(a)))

    tts = sub.add_parser("tts-check", help="Phase 1 驗收：只跑真 TTS，量首段延遲")
    tts.add_argument("text", help="要合成的文字")
    tts.add_argument("--out", default="outputs/tts_check.wav", help="音訊輸出路徑")
    tts.add_argument("--device", default=None, help="TTS 裝置（cuda / cpu）")
    tts.add_argument("--language", default=None, help="語言（留空自動偵測）")
    tts.add_argument("--speed", type=float, default=None, metavar="-1~1",
                     help="語速（引擎預設 0.0，正值變快。stretch=1.72-speed*0.5）")
    tts.add_argument("--warmup", action="store_true",
                     help="載入後先跑一次丟棄的合成（實測無效益，預設關閉）")
    _add_split_args(tts)
    tts.set_defaults(func=lambda a: asyncio.run(_run_tts_check(a)))

    stt = sub.add_parser("stt-check", help="Phase 3 驗收：只跑真 STT，量 turn 切分與轉錄延遲")
    stt.add_argument("--wav", default=None, metavar="檔案",
                     help="用 WAV 檔當來源（16kHz mono；不給就用麥克風）")
    stt.add_argument("--device-index", type=int, default=None, help="麥克風裝置索引")
    stt.add_argument("--language", default=None, help="轉錄語言（留空自動偵測）")
    stt.add_argument("--turns", type=int, default=None, help="收滿 N 個 turn 後結束")
    stt.add_argument("--silence", type=float, default=0.7,
                     help="turn 結束的靜音門檻（秒）")
    stt.add_argument("--min-utterance", type=float, default=0.3,
                     help="短於此的語音不成 turn（秒）")
    stt.set_defaults(func=lambda a: asyncio.run(_run_stt_check(a)))

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
