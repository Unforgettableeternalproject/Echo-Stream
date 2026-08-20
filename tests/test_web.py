"""Web 前端的測試。

實際起一個 server 打幾個請求——frame 協定與執行緒橋接是這一層最容易錯的
地方，光測純函式驗不出來。全程用 fake stages，不碰 GPU 也不打 API。
"""

from __future__ import annotations

import json
import struct
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from echo_stream.contracts.types import AudioChunk
from echo_stream.web.server import (
    FRAME_AUDIO,
    FRAME_META,
    FRAME_TEXT,
    StreamService,
    _Handler,
    frame,
    json_frame,
    pcm_to_wav,
)


# --- 純函式 ---


def test_frame_格式():
    """1 byte type + 4 byte big-endian length + payload。"""
    f = frame(FRAME_AUDIO, b"abc")
    assert f[0] == FRAME_AUDIO
    assert struct.unpack(">I", f[1:5])[0] == 3
    assert f[5:] == b"abc"


def test_json_frame_保留中文():
    f = json_frame(FRAME_META, {"text": "你好"})
    payload = json.loads(f[5:].decode("utf-8"))
    assert payload["text"] == "你好"


def test_pcm_轉成可播的_wav():
    chunk = AudioChunk(
        pcm=b"\x00\x00" * 100, sample_rate=22050, turn_id="t", sentence_index=0
    )
    wav = pcm_to_wav(chunk)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"

    import io
    import wave

    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 22050
        assert w.getnchannels() == 1
        assert w.getnframes() == 100


# --- 端到端 ---


@pytest.fixture
def server():
    """起一個跑 fake 管線的 server。"""
    service = StreamService(use_real_tts=False, use_real_llm=False)
    service.prepare()

    handler = type("H", (_Handler,), {"service": service})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_port}", service

    httpd.shutdown()
    service.shutdown()


def parse_frames(data: bytes) -> list[tuple[int, bytes]]:
    out = []
    i = 0
    while i + 5 <= len(data):
        kind = data[i]
        length = struct.unpack(">I", data[i + 1 : i + 5])[0]
        out.append((kind, data[i + 5 : i + 5 + length]))
        i += 5 + length
    return out


def post(url: str, payload: dict) -> bytes:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def test_首頁能開(server):
    base, _ = server
    with urllib.request.urlopen(f"{base}/", timeout=10) as r:
        body = r.read().decode("utf-8")
    assert r.status == 200
    assert "Echo Stream" in body


def test_狀態端點(server):
    base, _ = server
    with urllib.request.urlopen(f"{base}/api/status", timeout=10) as r:
        status = json.loads(r.read())
    assert status["ready"] is True
    assert status["error"] is None
    assert status["real_tts"] is False


def test_跑一輪對話會收到音訊與字幕(server):
    base, _ = server
    frames = parse_frames(post(f"{base}/api/say", {"text": "你好"}))
    kinds = [k for k, _ in frames]

    assert FRAME_TEXT in kinds, "要有字幕，不然調參時看不出哪一句慢"
    assert FRAME_AUDIO in kinds
    assert kinds[-1] == FRAME_META, "META 必須最後送，它含完整報告"


def test_字幕含索引與時間(server):
    base, _ = server
    frames = parse_frames(post(f"{base}/api/say", {"text": "你好"}))
    texts = [json.loads(p) for k, p in frames if k == FRAME_TEXT]

    assert texts[0]["is_first"] is True
    assert texts[0]["index"] == 0
    assert texts[0]["ms"] > 0
    assert all(t["text"] for t in texts)


def test_音訊是獨立可播的_wav(server):
    """每段獨立完整，前端才能直接 decodeAudioData。"""
    base, _ = server
    frames = parse_frames(post(f"{base}/api/say", {"text": "你好"}))
    audio = [p for k, p in frames if k == FRAME_AUDIO]

    assert audio
    for wav in audio:
        assert wav[:4] == b"RIFF"


def test_meta_含延遲報告(server):
    base, _ = server
    frames = parse_frames(post(f"{base}/api/say", {"text": "你好"}))
    meta = json.loads([p for k, p in frames if k == FRAME_META][-1])

    assert meta["phase"] == "done"
    assert meta["segments_ms"]["ttfa"] is not None
    assert meta["sentences"] > 0
    assert meta["chunks"] > 0
    assert "report" in meta


def test_空文字被拒絕(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(f"{base}/api/say", {"text": "   "})
    assert exc.value.code == 400


def test_沒有進行中的_turn_時插話回傳_false(server):
    base, _ = server
    result = json.loads(post(f"{base}/api/interrupt", {}))
    assert result["interrupted"] is False


def test_插話會中止進行中的_turn(server):
    """插話要走 call_soon_threadsafe——直接從 HTTP 執行緒動
    asyncio.Event 不是 thread-safe 的，會隨機失效。"""
    base, service = server
    result: dict = {}

    def run():
        result["frames"] = parse_frames(
            post(f"{base}/api/say", {"text": "講一段很長的話"})
        )

    worker = threading.Thread(target=run)
    worker.start()

    # 等到真的開始播了才插話，否則會插在還沒開始的空檔
    for _ in range(300):
        if service._runner is not None and service._runner.is_speaking:
            break
        threading.Event().wait(0.01)
    interrupted = json.loads(post(f"{base}/api/interrupt", {}))["interrupted"]

    worker.join(timeout=60)
    meta = json.loads([p for k, p in result["frames"] if k == FRAME_META][-1])

    assert interrupted is True
    assert meta["phase"] == "cancelled"
    assert meta["cancel_reason"] == "barge_in"


def test_未知路徑回_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base}/nope", timeout=10)
    assert exc.value.code == 404
