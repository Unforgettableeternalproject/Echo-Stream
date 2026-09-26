"""唯讀彙整舊 trace，並以阻塞假後端驗證 TTS 工作生命週期；不載入模型。"""

# 直接執行腳本需要先加入 repo 根目錄，再匯入專案模組。
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echo_stream.adapters.speak_indextts import IndexTTS2SpeakStage
from echo_stream.contracts.cancellation import CancellationToken
from echo_stream.contracts.types import Sentence
from echo_stream.core.gap_filler import GapFiller, GapFillerPolicy


def historical_summary() -> list[dict]:
    """每份檔案分開統計，不將不同設定的舊 session 混成基準。"""
    reports = []
    for path in sorted((ROOT / "outputs/web_logs").glob("*_trace.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        completed = [row for row in rows if row.get("phase") == "done"]
        stats = {}
        for key in ("stt", "llm_first_token", "llm_first_sentence", "tts_first_chunk", "ttfa"):
            values = [
                r["segments_ms"][key]
                for r in completed
                if r.get("segments_ms", {}).get(key) is not None
            ]
            if values:
                stats[key] = {
                    "n": len(values),
                    "median_ms": round(statistics.median(values), 1),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
        reports.append(
            {"file": path.name, "turns": len(rows), "completed": len(completed), "stats": stats}
        )
    return reports


class BlockingBackend:
    """用事件固定工作交錯，避免以 GPU 或時間猜測重入。"""

    emotion_vector = None

    def __init__(self) -> None:
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0
        self.entered: list[str] = []

    def generate(self, text, output_path, **kwargs):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.entered.append(text)
        try:
            if not self.release.wait(timeout=5):
                raise TimeoutError("診斷工作未收到釋放訊號")
        finally:
            with self.lock:
                self.active -= 1


async def until(predicate) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=2)


async def lifecycle_probe() -> dict:
    backend = BlockingBackend()
    with TemporaryDirectory(prefix="echo_latency_audit_") as tmp:
        stage = IndexTTS2SpeakStage(
            backend=backend, output_dir=tmp, edge_fade_ms=0, segment_silence_ms=0
        )
        # 略過 prepare 的模型載入與舊暫存目錄清理，僅測現有橋接程式。
        stage._prepared = True
        gap = GapFiller(GapFillerPolicy(enabled=True, start_delay_s=0, phrases={"zh": ["墊片"]}))
        gap.attach(stage)

        class Sink:
            async def write(self, chunk):
                raise AssertionError("此診斷不應產生音訊")

        async def sentences():
            yield Sentence(text="正式回答", turn_id="audit", index=0)

        async def consume():
            async for _ in stage.stream(sentences(), CancellationToken()):
                pass

        main_task = None
        try:
            gap.start(Sink(), "audit", CancellationToken(), language="zh")
            await until(lambda: backend.active == 1)
            await gap.stop()
            active_after_stop = backend.active
            main_task = asyncio.create_task(consume())
            # 給第二條 stream 機會取得執行權；固定觀察窗僅用來診斷，
            # 正確性另由事件控制的單元測試保證。
            await asyncio.sleep(0.1)
            entered_before_release = list(backend.entered)
            backend.release.set()
            await asyncio.wait_for(main_task, timeout=2)
            result = {
                "active_after_gap_stop": active_after_stop,
                "maximum_simultaneous_generate": backend.maximum,
                "entered_before_release": entered_before_release,
                "entered": list(backend.entered),
            }
        finally:
            backend.release.set()
            await gap.stop()
            if main_task is not None:
                await main_task
            await until(lambda: backend.active == 0)
            await stage.aclose()
        return result


if __name__ == "__main__":
    print(
        json.dumps(
            {"historical": historical_summary(), "lifecycle_probe": asyncio.run(lifecycle_probe())},
            ensure_ascii=False,
            indent=2,
        )
    )
