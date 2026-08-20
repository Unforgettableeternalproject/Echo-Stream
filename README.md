# Echo Stream

串流語音對話管線整合層。把五個獨立的子系統串成一條低延遲的對話管線。

**目標：TTFA（Time To First Audio）< 2.5s**，現況約 30s。

---

## 為什麼有這個專案

上次 Discord 實測 LLM 回覆延遲 30-40 秒。根因**不是「沒有串流」，
是每一層都等前一層 100% 完成**：

```
STT 整段轉錄 1-2s → Memory retrieve 1-3s → LLM 整段生成 20-30s → TTS 整段合成 5-15s
```

四段耗時直接相加。光把 TTS 做成串流救不了——TTS 只佔 5-15s。

解法是**在句子邊界讓各層重疊**：LLM 吐出第一句就立刻送 TTS，
LLM 生成第二句的時間被 TTS 合成第一句吃掉。

---

## 範圍限制（重要）

這個專案**不實作任何推理邏輯**，只做五件事：**契約、串接、背壓、取消、量測**。

所有實際能力都來自 adapter 包裝的既有 repo（echo_stt / echo_tts /
echo_memory / session control / LLM）。

程式碼層級的防線：`contracts/` 與 `core/` **不得 import 任何子系統**，
只有 `adapters/` 可以。這條線一旦破了，整合層就退化成第六個孤島。

---

## 快速開始

環境沿用 U.E.P Core：

```
C:\Users\Bernie\source\repos\Unforgettableeternalproject\U.E.P-s-Core\env\Scripts\python.exe
```

Phase 0 是純 asyncio + 標準庫，**零第三方依賴、不需要 GPU**。

```bash
PY="C:/Users/Bernie/source/repos/Unforgettableeternalproject/U.E.P-s-Core/env/Scripts/python.exe"

# 跑一輪 fake 管線並輸出延遲報告
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream demo --fake

# 模擬使用者在第 4 秒插話（驗證 barge-in 與上游取消傳播）
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream demo --fake --barge-in 4.0

# 檢查切分結果（調參用，不必跑整條管線）
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream split "你的文字。第二句在這裡。"

# Web 前端（fake 管線，秒開）
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream serve
# 接真模組：--real-tts（要 GPU）/ --real-llm（消耗 token）/ --real（兩者）

# 各段獨立驗收
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream tts-check "要合成的文字。"   # Phase 1，要 GPU
PYTHONIOENCODING=utf-8 "$PY" -m echo_stream llm-check "要問的問題。"     # Phase 2，消耗 token

# 測試
PYTHONIOENCODING=utf-8 "$PY" -m pytest tests/ -q
```

真模組的設定放 `.env`（複製 `.env.example`）。⚠️ `ECHO_STREAM_LLM_TEMPERATURE`
必須留空——gpt-5.6-luna 只接受預設值 1，傳自訂值會被 API 直接拒絕。

輸出範例：

```
─── 延遲報告  turn=1b9f71fceab5  phase=done ───
  階段                        實測        預算   狀態
  STT 最終轉錄              1202ms     700ms   ✗ 超支 1.7×
  Memory retrieve       1509ms     300ms   ✗ 超支 5.0×
  LLM 首 token            404ms         —
  LLM 首句                 482ms     800ms   ✓
  TTS 首段                 911ms    1000ms   ✓
  ──────────────────────────────────────────────
  TTFA                  2596ms    2500ms   ✗ 超支 1.0×
  句數 6 / 音訊段 5 / 播出 13.0s
  切點成因：flush=1  secondary=2  terminal=3
```

---

## 結構

```
echo_stream/
  contracts/          契約層——不 import 任何子系統
    types.py          Utterance / Sentence / AudioChunk / TurnResult
    stages.py         InputStage / ThinkStage / SpeakStage / AudioSource / AudioSink
    cancellation.py   CancellationToken（barge-in 的骨幹）
    turn.py           TurnDetector / InterruptionDetector / AddressingClassifier
  core/               串接、背壓、取消、量測——同樣不 import 子系統
    pipeline.py       PipelineRunner
    splitter.py       SentenceSplitter（整個管線最大的槓桿）
    channel.py        StreamChannel（有界通道 + push→pull 橋接）
    tracer.py         LatencyTracer
    turn_detector.py  Phase 0 的固定閾值實作
  fakes/              Phase 0 的 fake 三層，零依賴、不需要 GPU
  adapters/           Phase 1+ ——唯一允許 import 子系統的地方
  cli.py
docs/
  design/             設計文件
  research/           外部工程參考的調查結論
tests/
```

---

## 進度

| Phase | 內容 | 狀態 |
|-------|------|------|
| 0 | 骨架 + fake 三層、契約、背壓、取消、打點 | ✅ 完成 |
| 1 | 接真 TTS（IndexTTS2，push→pull 橋接） | ✅ 完成 |
| 2 | 接真 LLM（GPT-5.6-luna，經 SessionControl） | ✅ 完成 |
| — | Web 前端 host 整條管線 | ✅ 完成 |
| 3 | 接真 STT（MultiChannelSTTEngine）+ Smart Turn v3 | ⬜ |
| 4 | Memory / Session 接入 | ⬜ |
| 5 | Discord Frontend | ⬜ |

### 實測延遲（2026-08-20，各段獨立量測）

| 段 | 實測 | 預算 | 性質 |
|----|------|------|------|
| STT | ~700ms | 700ms | 本地（尚未接） |
| LLM 首句 | ~2600ms | 800ms | 其中 **~800ms 是純網路往返** |
| TTS 首段 | ~1800ms | 1800ms | 本地，**每次合成的固定開銷** |
| **TTFA** | **≈5100ms** | 3500ms | |

兩個大頭性質完全不同，優化方向也不同：

* **TTS 1.8s** — 本地算力問題。首句就算只有一個字也要 1.8 秒，切短救不了。
  解法是 TensorRT 加速（Faster IndexTTS-2 論文：端到端 3.46-3.60×）。
* **LLM 2.6s** — 網路距離問題。實測純網路往返（`models.list`）就要
  721/806/1215ms（min/med/max），程式碼優化不了。
  且 **reasoning effort 對它沒有可辨識的影響**（none 2389 / low 2877 /
  medium 2192 / high 2665 ms，全在雜訊範圍內）。

拆解 LLM 段可以看得更清楚——瓶頸完全在首 token，不在切分：

```
首 token      2563ms   ← 全部在這裡
累積成句       137ms   ← 切分閾值只影響這一段
```

---

## 文件

- [設計文件](docs/design/echo-stream-design.md) — 架構、延遲預算、分階段計畫、設計決策
- [調查結論](docs/research/2026-08-20-調查結論.md) — CosyVoice / turn detection /
  late-context / AEC 等外部工程參考的結論與待驗證清單
