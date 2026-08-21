"""Web 前端的測試。

實際起一個 server 打幾個請求——frame 協定與執行緒橋接是這一層最容易錯的
地方，光測純函式驗不出來。全程用 fake stages，不碰 GPU 也不打 API。
"""

from __future__ import annotations

import json
import struct
import threading
import time
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


# --- 語音 session ---


def _tone_pcm(seconds: float, amplitude: float = 0.3) -> bytes:
    import math
    import struct as _struct

    n = int(16_000 * seconds)
    return _struct.pack(
        f"<{n}h",
        *(
            int(amplitude * 32767 * math.sin(2 * math.pi * 220.0 * i / 16_000))
            for i in range(n)
        ),
    )


def _post_raw(url: str, body: bytes) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def test_語音_session_端到端(server):
    """瀏覽器擷取 → 上傳 → VAD 切 turn →（fake）轉錄 → 管線回應。

    這條路徑驗的是串流音訊接收：chunk 經 POST 進來、跨執行緒餵給
    asyncio 管線、turn 切出來後 frame 從長連線推回去。
    """
    base, service = server

    info = json.loads(post(f"{base}/api/voice/start", {}))
    sid = info["sid"]
    assert info["sample_rate"] == 16_000

    # 收 frame 的長連線（在背景執行緒讀）
    collected: dict = {"raw": b""}

    def read_stream():
        req = urllib.request.Request(f"{base}/api/voice/stream?sid={sid}")
        with urllib.request.urlopen(req, timeout=60) as r:
            collected["raw"] = r.read()

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()

    # 推 0.8s 語音 + 1.2s 靜音（切出一個 turn），再 stop（觸發收尾）
    speech = _tone_pcm(0.8)
    quiet = b"\x00\x00" * int(16_000 * 1.2)
    step = int(16_000 * 0.2) * 2  # 200ms 一個 chunk，模擬前端上傳節奏
    stream_bytes = speech + quiet
    for i in range(0, len(stream_bytes), step):
        _post_raw(f"{base}/api/voice/audio?sid={sid}", stream_bytes[i : i + step])

    # 等 turn 跑完再 stop——stop 現在是「乾淨收場」，會中斷進行中的
    # turn，太早按會把這一輪打斷，測試就變成在驗捨棄而不是正常流程
    for _ in range(300):
        time.sleep(0.1)
        if service.log_path.exists() and '"type": "turn"' in service.log_path.read_text(
            encoding="utf-8"
        ):
            break

    _post_raw(f"{base}/api/voice/stop?sid={sid}", b"x")
    reader.join(timeout=60)

    frames = parse_frames(collected["raw"])
    kinds = [k for k, _ in frames]

    from echo_stream.web.server import FRAME_EVENT, FRAME_UTTER

    assert FRAME_EVENT in kinds, "要有 VAD 狀態事件（前端聆聽指示）"
    assert FRAME_UTTER in kinds, "要有轉錄結果 frame"
    assert FRAME_AUDIO in kinds, "turn 要走完管線並回音訊"
    assert FRAME_META in kinds

    utters = [json.loads(p) for k, p in frames if k == FRAME_UTTER]
    finals = [u for u in utters if not u.get("partial")]
    assert finals, "要有定案的轉錄 frame（partial frame 之外）"
    assert "fake 轉錄" in finals[0]["text"]
    assert finals[0]["stt_ms"] >= 0
    assert finals[0]["parts"], "定案 frame 要帶逐段條列"

    meta = json.loads([p for k, p in frames if k == FRAME_META][-1])
    assert meta["phase"] == "done"


def test_語音_audio_無效_sid_回_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post_raw(f"{base}/api/voice/audio?sid=nope", b"\x00\x00" * 100)
    assert exc.value.code == 404


def test_語音_start_會頂掉舊_session(server):
    base, _ = server
    first = json.loads(post(f"{base}/api/voice/start", {}))
    second = json.loads(post(f"{base}/api/voice/start", {}))
    assert first["sid"] != second["sid"]
    # 舊 sid 已失效
    with pytest.raises(urllib.error.HTTPError):
        _post_raw(f"{base}/api/voice/audio?sid={first['sid']}", b"\x00\x00" * 100)
    # 清掉
    _post_raw(f"{base}/api/voice/stop?sid={second['sid']}", b"x")


# --- 執行中調參（/api/config） ---


def test_config_GET_回報現值(server):
    base, service = server
    with urllib.request.urlopen(f"{base}/api/config", timeout=10) as r:
        cfg = json.loads(r.read())
    assert cfg["split"]["first_min"] == service.split_policy.first_min_weight
    assert "presets" in cfg and "開心" in cfg["presets"]
    assert cfg["turn"]["silence_threshold_s"] is not None
    assert cfg["turn"]["min_confidence"] is not None


def test_config_POST_套用切分與_turn_參數(server):
    base, service = server
    res = json.loads(post(f"{base}/api/config", {
        "split": {"first_min": 4, "min": 24},
        "turn": {"silence_threshold_s": 1.1, "min_confidence": 0.5},
    }))
    assert res["ok"]
    assert service.split_policy.first_min_weight == 4
    assert service.split_policy.min_weight == 24
    assert service._stt.policy.silence_threshold_s == 1.1
    assert service._stt.min_confidence == 0.5
    # barge-in 判定器讀的是 runner 的 policy——兩份都要同步
    res = json.loads(post(f"{base}/api/config", {
        "turn": {"interruption_min_speech_s": 0.8},
    }))
    assert service._runner.policy.interruption_min_speech_s == 0.8
    assert service._stt.policy.interruption_min_speech_s == 0.8


def test_config_POST_設定寫進_session_log(server):
    base, service = server
    post(f"{base}/api/config", {"split": {"first_min": 5}})
    records = [
        json.loads(line)
        for line in service.log_path.read_text(encoding="utf-8").splitlines()
    ]
    config_records = [r for r in records if r["type"] == "config"]
    assert config_records
    assert config_records[-1]["split"] == {"first_min": 5.0}


def test_config_POST_壞_json_回_400(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post_raw(f"{base}/api/config", b"{not json")
    assert exc.value.code == 400


# --- 會話管理（/api/sessions） ---


class StubStore:
    """記憶體版 SessionStore——測會話流程不需要真的檔案系統。

    介面形狀對齊 echo_thought_core.session_store.SessionStore。
    """

    def __init__(self):
        self.data = {}
        self._clock = 0.0

    def save(self, session_id, messages, metadata=None):
        self._clock += 1
        self.data[session_id] = (list(messages), dict(metadata or {}), self._clock)

    def load(self, session_id):
        if session_id not in self.data:
            raise FileNotFoundError(session_id)
        messages, meta, _ = self.data[session_id]
        return list(messages), dict(meta)

    def list_sessions(self):
        return [
            {"session_id": sid, "message_count": len(m), "saved_at": t}
            for sid, (m, _meta, t) in self.data.items()
        ]

    def delete(self, session_id):
        return self.data.pop(session_id, None) is not None


@pytest.fixture
def session_server():
    """帶 StubStore 的 fake 管線 server。"""
    service = StreamService(use_real_tts=False, use_real_llm=False, store=StubStore())
    service.prepare()

    handler = type("H", (_Handler,), {"service": service})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_port}", service

    httpd.shutdown()
    service.shutdown()


def _get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def test_每輪對話後自動存檔(session_server):
    base, service = session_server
    post(f"{base}/api/say", {"text": "第一個會話的第一句"})

    store = service._store
    assert service._session_id in store.data
    messages, meta, _ = store.data[service._session_id]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "第一個會話的第一句"
    assert messages[1]["role"] == "assistant"
    assert meta["title"].startswith("第一個會話")


def test_新對話會清空歷史並保留舊會話(session_server):
    base, service = session_server
    post(f"{base}/api/say", {"text": "舊會話的內容"})
    old_id = service._session_id

    res = json.loads(post(f"{base}/api/sessions/new", {}))
    assert res["ok"]
    assert res["active"] != old_id
    assert service._think.history == []
    assert old_id in service._store.data  # 舊的還在 store 裡

    listing = _get_json(f"{base}/api/sessions")
    assert any(s["id"] == old_id for s in listing["sessions"])
    assert listing["active"] == res["active"]


def test_切換會話會還原歷史(session_server):
    base, service = session_server
    post(f"{base}/api/say", {"text": "會話A的訊息"})
    sid_a = service._session_id

    json.loads(post(f"{base}/api/sessions/new", {}))
    post(f"{base}/api/say", {"text": "會話B的訊息"})
    sid_b = service._session_id
    assert sid_a != sid_b

    res = json.loads(post(f"{base}/api/sessions/switch", {"id": sid_a}))
    assert res["ok"]
    assert res["history"][0]["content"] == "會話A的訊息"
    assert service._think.history[0]["content"] == "會話A的訊息"
    assert service._session_id == sid_a
    # 切走前 B 有落地
    assert sid_b in service._store.data


def test_切換到不存在的會話回錯誤(session_server):
    base, _ = session_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(f"{base}/api/sessions/switch", {"id": "nope0000"})
    assert exc.value.code == 409


def test_刪除使用中的會話會開新會話(session_server):
    base, service = session_server
    post(f"{base}/api/say", {"text": "要被刪掉的會話"})
    sid = service._session_id

    res = json.loads(post(f"{base}/api/sessions/delete", {"id": sid}))
    assert res["ok"]
    assert sid not in service._store.data
    assert service._session_id != sid
    assert service._think.history == []


def test_沒有_store_時回報錯誤但不炸(server):
    base, _ = server
    listing = _get_json(f"{base}/api/sessions")
    assert "error" in listing
    assert listing["sessions"] == []


def test_會話事件寫進_session_log(session_server):
    base, service = session_server
    post(f"{base}/api/sessions/new", {})
    records = [
        json.loads(line)
        for line in service.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(r["type"] == "session" and r["action"] == "new" for r in records)


def test_語音_discard_無效_sid_回_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post_raw(f"{base}/api/voice/discard?sid=nope", b"x")
    assert exc.value.code == 404


# --- 記憶寫入（Phase 4a）---


class StubMemory:
    """記錄 store/set_session 呼叫的假記憶 adapter。"""

    def __init__(self):
        self.stored: list[dict] = []
        self.sessions: list[str] = []

    def set_session(self, session_id):
        self.sessions.append(session_id)

    async def retrieve(self, text):
        return None

    async def store(self, user_text, spoken_text, session_id=None, **metadata):
        self.stored.append(
            {"user": user_text, "spoken": spoken_text, **metadata}
        )


def _wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_每輪對話後背景寫入記憶(server):
    base, service = server
    memory = StubMemory()
    service._memory = memory

    post(f"{base}/api/say", {"text": "我養了一隻叫毛球的貓"})

    assert _wait_for(lambda: memory.stored), "store 是背景任務，等它落地"
    record = memory.stored[0]
    assert record["user"] == "我養了一隻叫毛球的貓"
    assert record["spoken"]  # 寫的是 spoken_text
    assert record["origin"] == "text"
    assert record["truncated"] is False


def test_被捨棄的_turn_不寫入記憶(server):
    base, service = server
    memory = StubMemory()
    service._memory = memory

    # 模擬 voice_discard 在 turn 進行中把捨棄旗標立起來：
    # _run 收尾時看到旗標就不該寫記憶（歷史剔除了、記憶留著 = 髒資料）
    import asyncio as aio

    from echo_stream.contracts.types import Utterance

    service._discard_turn = True
    fut = aio.run_coroutine_threadsafe(
        service._run(Utterance(text="不該被記住的話"), lambda _: None,
                     origin="voice"),
        service._loop,
    )
    fut.result(timeout=30)
    service._discard_turn = False

    assert not _wait_for(lambda: memory.stored, timeout=1.0), "捨棄的 turn 寫進記憶了"


def test_會話切換同步給記憶層(session_server):
    base, service = session_server
    memory = StubMemory()
    service._memory = memory

    post(f"{base}/api/sessions/new", {})
    assert memory.sessions, "session new 沒有同步給記憶層"
    assert memory.sessions[-1] == service._session_id
