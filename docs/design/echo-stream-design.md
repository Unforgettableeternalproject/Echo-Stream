# Echo Stream 設計文件

> 狀態：Phase 0 已完成
> 建立：2026-08-20（原稿暫放於 TestSeparateTTSSystem/docs/design/，已移轉至此）
> 最近修訂：2026-08-20（四份外部調查後修正延遲預算與 §7.4 / §7.5）

---

## 1. 問題診斷：上次 Discord 為什麼慢

2026-06-22 的實測記錄了兩個數字：**LLM 回覆延遲 30-40 秒**、
**STT 管線因 event loop 被阻塞 30 秒而停擺**。

後者已修（`_handle_decision` 改 fire-and-forget），但前者是結構性的：

```
使用者說完話
  └─ STT 等整段音訊轉錄完          ~1-2s
      └─ Memory retrieve（bge-m3）  ~1-3s
          └─ LLM 等整段生成完        ~20-30s
              └─ TTS 等整段合成完    ~5-15s
                  └─ 開始播放
```

**每一層都等前一層 100% 完成，四段耗時直接相加。**

這就是為什麼「先做 TTS 串流」不能解決問題——TTS 只是最後 5-15 秒。

### 關鍵指標搞錯了

真正決定體感的不是總處理時間，是 **TTFA（Time To First Audio）**——
使用者說完話到聽見第一個字的間隔。人類對話的自然停頓約 200-500ms，
超過 2 秒明顯尷尬，超過 5 秒對話就斷了。

總時長可以長，但 TTFA 必須短。

---

## 2. 核心設計：句子級 pipeline

不是「每一層都做成串流」，而是**在句子邊界讓各層重疊**。

```
t=0.0  使用者說完
t=0.7  STT 吐出最終轉錄
t=0.9  LLM 開始串流 token
t=1.5  ├─ 累積到第一個句子邊界「我覺得這個想法不錯。」
t=1.5  │   └─ 立刻送 TTS 開始合成 ─────┐
t=2.3  ├─ LLM 還在生成第二句           │
t=2.5  │                          第一句音訊就緒 → 播放  ← TTFA 2.5s
t=2.8  ├─ 第二句邊界 → 送 TTS         │
t=4.2  └─ LLM 結束                    │
t=5.0                            第二句接上播放，無縫
```

LLM 生成第二句的時間，被 TTS 合成第一句的時間吃掉了。**兩層重疊，不是相加。**

### 為什麼是「切句子」而不是更細的粒度（2026-08-20 調查確認）

CosyVoice 2/3、Orpheus、FlashTTS 能做到 150-325ms 首包、「不等句子邊界」，
靠的是**模型訓練架構層級的能力**（text/speech token 交錯、chunk-aware
causal flow matching），不是推論期的 wrapper。IndexTTS 2.5 沒有這個基礎。

**句子級切分就是串接式管線的正解。** 詳見
[docs/research/2026-08-20-調查結論.md](../research/2026-08-20-調查結論.md)。

---

## 3. 為什麼要獨立專案

1. **Discord bot 有太多與管線無關的複雜度** — DAVE 加密、presence、cogs 權限。
   在那裡除錯管線，永遠分不清是管線問題還是 Discord 問題。
2. **管線需要能單獨量測** — 每一段的延遲要能打點、比較、回歸測試。
3. **五個子系統各自獨立 repo，缺一個中立的整合層**。
4. **可以用 fake 模組先跑通骨架**（Phase 0）。

**風險：變成第六個孤島。** 防範方式：這個專案**不實作任何推理邏輯**，
只做契約、串接、背壓、取消、量測。所有實際能力都來自 adapter 包裝的既有 repo。

程式碼層級的防線：`contracts/` 與 `core/` **不得 import 任何子系統**，
只有 `adapters/` 可以。

---

## 4. 架構

```
┌─────────────────────────────────────────────────────────┐
│  Frontend Adapter（可插拔）                              │
│  CLI / Web(現有 tts_server) / Discord / 檔案批次          │
│  ── 重採樣在這裡做（16k in / 22.05k out ↔ 48k Discord）── │
└────────────┬────────────────────────────▲───────────────┘
             │ AudioChunk / Text          │ AudioChunk
             ▼                            │
┌─────────────────────────────────────────┴───────────────┐
│                    Pipeline Core                         │
│                                                          │
│   InputStage ──► ThinkStage ──► SpeakStage               │
│   (STT)          (LLM+Mem)      (TTS)                    │
│      │               │              │                    │
│      └───────────────┴──────────────┘                    │
│              LatencyTracer（全程打點）                     │
└──────────────────────────────────────────────────────────┘
             │              │              │
             ▼              ▼              ▼
      echo_stt        Memory/Session    echo_tts
      (worker)        (in-process)      (worker)
```

沿用 **Option D 混合模式**：GPU 推理（STT/TTS）走獨立 worker process，
CPU 模組（Memory/Session）in-process。

### 三個 Stage 的契約

三個 `stream()` 全部是 `AsyncIterator`——**任何一層都不回傳「完整結果」**。

實作細節見 `echo_stream/contracts/stages.py`。兩個非顯而易見的決策：

- **token 是顯式參數，不是建構子注入**：stage 實例服務多個 turn
  （模型載入很貴），但取消是 per-turn 的。
- **SpeakStage 吃 AsyncIterator 而不是單句**：TTS 需要看到「還有沒有下一句」
  才能決定要不要保留尾音。

### SentenceSplitter

介於 LLM 與 TTS 之間，切點優先序：

1. 全形句末標點（。！？；…）→ **立即切**（無歧義）
2. 半形句末標點（.!?）→ **等下一個字元**才判定（避免 3.14、Mr. 誤切）
3. 次級標點（，、；:）且累積長度達標
4. 長度上限強制切（優先回溯到最近的次級標點或空白處）

**張力：切太細語調破碎，切太粗 TTFA 拉長。** 首句用較低閾值（快速出聲），
後續放寬（品質優先）。實際值要用聽感測。

長度用**發音時長權重**而非字元數（CJK=1.0、拉丁字母=0.4），
否則同一組閾值套中英混合會讓英文被切得極短。

---

## 5. 延遲預算（2026-08-20 修訂）

目標 **TTFA < 2.5s**（含 STT）。

| 階段 | 預算 | 現況 | 備註 |
|------|------|------|------|
| STT 最終轉錄 | **700ms** | 1-2s | ⬆️ 原訂 400ms 不切實際，見下 |
| Memory retrieve | 300ms | 1-3s | **與 LLM 平行，不計入 TTFA** |
| LLM 首句 | 800ms | 20-30s（整段）| 串流後首 token ~200-400ms |
| TTS 首段 | 1000ms | 4-5s（整段）| 已實測段級串流首段 4.0s |
| **TTFA 合計** | **2500ms** | **~30s** | |

**STT 從 400ms 上修為 700ms**：400ms 是雲端專用串流 ASR 的量級
（AssemblyAI P50 ~150ms、Deepgram Flux <300ms）。Whisper 系是
encoder-decoder attention 架構，即使 turbo + int8 也先天不利。
改造成增量串流 + LocalAgreement 後，社群實測落在 500-800ms。

**最大的兩個槓桿**：LLM 改串流（省 ~20s）、首句切短（省 ~3s）。

`LatencyTracer` 對每個 turn 記錄這些數字並寫 JSONL。
沒有量測就沒有優化——上次只有「30-40 秒」這個總數，不知道該修哪裡。

---

## 6. 分階段計畫

### ✅ Phase 0：骨架 + Fake 模組（已完成 2026-08-20）

三個 Stage 全用 fake（stdin 文字 / 固定回應模擬 token 串流 / sine wave）。
**全程不需要 GPU，秒級迭代。**

驗收：`python -m echo_stream demo --fake` 跑完一輪並輸出延遲報告 ✅
（81 個測試通過，barge-in 與取消上游傳播已驗證）

### Phase 1：接真 TTS

只換 SpeakStage，其餘維持 fake。adapter 包 `IndexTTS2Backend`。

⚠️ **push→pull 橋接**：後端是同步 callback
`on_segment_audio(tensor, idx, total)`，契約是 async pull。
在 executor 執行緒跑推理，callback 用 `StreamChannel.put_threadsafe`
塞進**有界** queue。有界是重點——無界會讓 TTS 一路合成到底，
barge-in 就白做了。

同時導入：**Hann window cross-fade + 段間短靜音墊**（消除拼接爆音）。

驗收：TTFA < 1.5s（fake LLM 首句立即給），音訊無縫。

### Phase 2：接真 LLM + SentenceSplitter 調參

換 ThinkStage。**LLM 改用 GPT-5.6-luna**（2026-08-20 決定，原 Gemini）。
adapter 設計成 provider 無關——`SentenceSplitter` 吃的是純 token 流，
換 provider 不影響管線契約。

驗收：文字輸入 → 語音輸出，TTFA < 2s，長回應的句間銜接自然。

### Phase 3：接真 STT

換 InputStage，包 `MultiChannelSTTEngine`。同時導入
Pipecat Smart Turn v3（8MB、CPU 推論 12ms）取代固定閾值 TurnDetector。

驗收：麥克風輸入 → 語音輸出全鏈路，TTFA < 2.5s。

### Phase 4：Memory / Session 接入

ThinkStage 內部接 `MemoryEngine` + `ContextManager`。
重點是**不能讓 retrieve 阻塞首句**（見 §7.4）。

⚠️ 前置：先對 EcphoryRAG 做 profiling——1-3s 是現象不是診斷，
bge-m3 的 embedding 推論本身只要 5-10ms。

### Phase 5：Discord Frontend

管線已驗證過，Discord 只是換一個 Frontend Adapter。

✅ **AEC 風險已解除**：Discord voice receive 是逐使用者獨立 Opus 流，
bot 自己的 TTS 不會回灌到接收流，架構上沒有 loopback 路徑。
順帶：per-user stream 天然語者分離，**pyannote diarization 可能是多餘的**。

---

## 7. 設計決策

### 已定案

**7.1 專案名稱：`Echo Stream`**

**7.2 支援插話（barge-in）** — 中止訊號往上游傳播（TTS 停播 → LLM 停生成 →
丟棄剩餘 token）。每個 Stage 都要能取消，且取消後不留半寫入的狀態。

Phase 0 的骨架從一開始就有取消路徑——事後補會很痛。

⚠️ **善後順序是死的**：音訊送 sink → 確認實際播出量（`played_duration_s`）
→ 推算 `spoken_text` → 才 commit。commit 絕不能提前
（Pipecat issue #4111 就是這個 bug）。

**7.3 Memory + Session Control 都要接**

```
Utterance → SessionControl(ContextManager 壓縮) → LLM 串流
                    ▲                              │
                    └── EchoMemory(retrieve/store) ┘
```

⚠️ 壓縮**必須背景跑、下一輪才生效**，絕不能「先清空原始內容再等 LLM 回摘要」
（Claude Code #40352、Codex #13946 都踩過，API 失敗時原始對話被永久吞掉）。

**7.4 Memory retrieve 時機：與 LLM prefill 平行，首句不等它** ✅ 定案

**副作用的正解已收斂（2026-08-20 調查後修訂）**：

原本設想用「late-context 注入」補救首句與記憶的矛盾——**這條路不通**。
Gemini / Ollama / vLLM 全都是「一次 prefill + 自迴歸 decode」，
KV cache 一旦建立無法替換前綴。這是 Transformer 自迴歸推論的
**結構性限制**，沒有任何一家支援串流生成中改 context。

**正解是從源頭排除**：

1. 首句改用**固定 filler**（「嗯，讓我想想」「這個問題」），
   刻意不含任何可被推翻的記憶斷言。
   注意「我不記得了」是斷言**不能用**，「讓我想一下」才安全。
2. retrieve 完成後，把已說出的 filler 當 assistant 前綴，
   **開新請求**續寫真正有記憶依據的內容。

平行跑這個時機決策是對的，缺的是**首句生成內容的邊界控制**。

中期優化：speculative retrieval（partial transcript 先查）。

**7.5 turn 邊界：TurnDetector 與 InterruptionDetector 分離** ✅ 定案

錯誤代價不對稱——turn-end 判太慢使用者乾等（要靈敏）、barge-in 判太浮動
一聲咳嗽就打斷（要保守）。合併就只能取一組閾值。
LiveKit / Pipecat 都是這樣分的。

`TurnDecision` 回傳**信心分數 + 建議等待時間**而非 bool，
因為純閾值靜音判定有調參數解不掉的矛盾（閾值短則思考停頓被誤判、
長則反應遲鈍），業界一致解法是動態延展等待視窗。
Phase 0 用固定閾值實作，Phase 3 換語意模型時**介面不需要動**。

Phase 0 參數：靜音門檻 700ms、barge-in 連續語音門檻 400ms、
字數門檻停用（沒有即時轉錄可用）。

⚠️ 字數門檻與時長門檻定義為 **AND**——LiveKit 是覆蓋關係（issue #3515），
會讓人以為設了兩層保護但只有一層生效。

**7.6 音訊格式統一** — 重採樣放 Frontend Adapter，Pipeline 內部用各模組
原生取樣率（16kHz in / 22.05kHz out）。soxr 最快（~1-11ms），
但有 ~20ms 固有演算法延遲要計入預算。

**7.7 addressee 判定留介面、Phase 0-4 永遠回 True** — 單人測試沒有第二個人
可以講話。學術上 addressee detection 至今是開放問題，沒有現成方案，
Phase 5 用 wake-word / 名字偵測當務實起點。

---

## 8. 這個專案不做什麼

- ❌ 不實作 STT / LLM / TTS / Memory 的任何推理邏輯
- ❌ 不做 Discord 的功能（權限、指令、presence）
- ❌ 不做人設、prompt 工程（那是 Session Control 的事）
- ❌ 不做 UI
- ✅ 只做：契約、串接、背壓、取消、量測

---

## 9. 待驗證清單

見 [docs/research/2026-08-20-調查結論.md](../research/2026-08-20-調查結論.md) 末節。
其中影響 Phase 1-2 排程的兩項：

- **TensorRT-LLM 加速 IndexTTS-2**（Faster IndexTTS-2, arXiv:2607.21042）：
  GPT 元件 4.8-5.0× 加速，是同一個模型，CP 值最高的優化
- **EcphoryRAG profiling**：1-3s 的瓶頸位置未知
