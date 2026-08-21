### 1. IndexTTS 2.5：我會繼續保留它作為 U.E.P 的控制介面基準

Index 現在最漂亮的地方就是這個：

```python
emo_vector = [
    happy,
    angry,
    sad,
    afraid,
    disgusted,
    melancholic,
    surprised,
    calm
]
```

再配：

```python
emo_alpha = 0.6
duration_factor = 0.9
```

甚至可以：

```python
emo_text = "有點高興，但是刻意壓抑著自己的興奮"
```

而且 emotion reference 和 speaker reference 是分開的，所以可以：

```text
U.E.P 的聲音
     │
     ├── Speaker Reference → 身分
     │
     └── Emotion Reference → 情緒
```

這種設計對桌面角色特別實用，因為「我是誰」和「我現在什麼心情」不應該綁死。IndexTTS2.5 就是特別做了 timbre-emotion disentanglement。([GitHub][1])

所以我會把 Index 的設計保留在 **UEP TTS abstraction layer**，即使底下最後不是跑 Index。

例如：

```python
SpeechStyle(
    emotion={
        "happy": 0.2,
        "sad": 0.0,
        "angry": 0.0,
        "calm": 0.8,
    },
    intensity=0.6,
    speed=0.95,
)
```

這種 API 我覺得比直接把：

```text
"speak happily"
```

丟給模型可靠很多。

---

## 2. Fish Audio S2：這個是我現在最想叫你參考的東西

不是因為我要你直接換 Fish，而是它有一個設計**非常適合 U.E.P**：

### sub-word / phrase level emotion control

Fish Audio S2 Pro 現在可以直接：

```text
你好，[whisper]其實我剛剛想到一件事情，
[excited]我覺得這個方法真的可以！
```

它支援類似：

```text
[whisper]
[excited]
[angry]
```

這種自然語言 tag，而且控制粒度可以小到 sub-word / phrase level。([GitHub][2])

這跟 Index 最大不同就在：

Index 比較像：

```text
整句：
sad = 0.4
calm = 0.6
```

Fish S2 可以：

```text
「我本來以為沒問題，
 [hesitant]但好像有哪裡怪怪的……
 [excited]啊，我知道了！」
```

這個我覺得**非常適合 U.E.P。**

因為 U.E.P 的 LLM 本來就知道自己正在講什麼。

你甚至可以讓 LLM 內部生成：

```xml
<speech emotion="calm">
我檢查了一下，
<emotion type="hesitant">這裡好像有點奇怪。</emotion>
<emotion type="excited">啊，我知道問題在哪裡了！</emotion>
</speech>
```

然後 TTS layer 再轉成不同 backend 的控制方式。

這比「一個 response 只能有一個 emotion」自然太多。

另外 Fish S2 的官方 technical report 給出的 production streaming 數據是 RTF 0.195、TTFA <100 ms，並且官方有 fine-tuning code。([arXiv][3])

問題是它現在 S2 Pro 是 **4B**，而且使用 Fish Audio Research License，不像 Apache/MIT 那麼舒服。([GitHub][2])

所以：

> **非常值得偷它的控制方法，不一定值得拿它當 U.E.P 最終模型。**

---

## 3. Qwen3-TTS：最值得參考「語意式情緒控制」

Qwen3-TTS 的思路跟 Index 不太一樣。

它比較偏：

```text
「用輕柔、有點疲憊，但仍然保持友善的語氣說話。
語速偏慢，句尾稍微放輕。」
```

模型自己理解：

* timbre
* emotion
* prosody
* speaking rate

然後決定怎麼念。([GitHub][4])

這對 U.E.P 很有價值，因為情緒其實並不只有：

```text
happy / sad / angry
```

像：

> 無奈
> 欲言又止
> 假裝冷靜
> 興奮但克制
> 帶點懷疑
> 調侃

八維 emotion vector 很難直接表達。

Qwen3-TTS 就很適合這一層。

而且它有 0.6B / 1.7B，官方支援 streaming，最低宣稱 97 ms，而且 Base 可以 3 秒 reference audio voice cloning，也提供 fine-tuning。([GitHub][4])

但這邊有一個我覺得很重要的限制：

**Qwen 現在的 model family 把功能拆開了。**

`CustomVoice / VoiceDesign`：

```text
instruction control ✓
```

`Base / Voice Clone`：

```text
voice cloning ✓
fine tuning ✓
instruction control -
```

官方 model table 就是這樣分。([GitHub][4])

所以如果你要的是：

> U.E.P 自己的 LoRA 聲音 + 強 instruction emotion control

它不像 Index 一樣天然就是同一條 pathway。

這一點需要實測。

---

## 4. CosyVoice 3：我覺得是「工程面」最值得你研究的

CosyVoice3 有個很適合 assistant 的定位：

```text
0.5B
+ voice cloning
+ instruction
+ streaming
+ pronunciation control
```

它的 instruction 可以直接控制：

* language
* dialect
* emotion
* speed
* volume

而且支援 text-in streaming + audio-out streaming，官方標示最低約 150 ms。([GitHub][5])

也就是：

```text
LLM:
「我覺得這個方法……」

       ↓ 還沒生成完

CosyVoice:
「我覺得——」

       ↓

LLM:
「可能不太適合。」

       ↓

CosyVoice:
「這個方法可能不太適合。」
```

不用等整句完成。

對 U.E.P 這種 assistant，這其實比純 RTF 還重要。

而且 CosyVoice3 還特別訓練了 speech emotion recognition、language identification、audio event detection、speaker analysis 等任務來改善 tokenizer 的 prosody representation。([arXiv][6])

所以我會研究它的：

> **streaming architecture + instruction conditioning**

而不是單純拿它來比較 MOS。

---

## 5. VoxCPM2：最值得參考 U.E.P 的「角色聲音訓練系統」

這個則是另一個方向。

VoxCPM2：

```text
2B
30 languages
48kHz
Voice Design
Voice Clone
Style Guidance
LoRA
```

而且官方直接提供：

```bash
train_voxcpm_finetune.py
```

支援 SFT 和 LoRA，只需要約 **5–10 分鐘語音**就可以適應 speaker / language / domain。還支援 LoRA hot swapping。([GitHub][7])

這其實正中你之前在做的：

> 台灣腔 U.E.P LoRA

所以如果你的 500 句 / 30+ 分鐘語料還留著，我一定會拿 VoxCPM2 做一次實驗。

甚至可以做：

```text
Base VoxCPM2
     │
     ├── UEP_voice.lora
     ├── Taiwan_Mandarin.lora
     └── other_character.lora
```

官方甚至已有：

* GGUF
* llama.cpp-omni
* CUDA
* Vulkan
* Metal
* ONNX
* Rust

等生態。([GitHub][7])

這跟 U.E.P 的 local-first 方向很合。

它唯一讓我比較保留的是官方自己也承認：

> Voice Design / controllable cloning 不同次 inference 的控制一致性仍會變動。

也就是它比較「生成式」，不像 Index 的 emotion vector 那麼 deterministic。([GitHub][7])

---

## 6. Chatterbox Multilingual V3：有一個很有趣的控制方法

Chatterbox 不像 Index：

```text
sad = 0.6
```

它主要有：

```python
exaggeration = 0.5
cfg_weight = 0.5
```

`exaggeration` 控制：

> 「這個人講話到底有多戲劇化」

官方甚至建議 expressive speech：

```python
exaggeration ~= 0.7+
cfg_weight ~= 0.3
```

([GitHub][8])

這個概念我反而建議你**直接偷到 U.E.P API 裡**。

因為 emotion 跟 expressiveness 本來就是兩回事。

例如：

```text
emotion = sad
intensity = 0.8
expressiveness = 0.2
```

可以代表：

> 很悲傷，但幾乎沒有表現出來。

而：

```text
emotion = happy
intensity = 0.4
expressiveness = 0.9
```

則是：

> 其實只有一點高興，但表現得非常誇張。

這對角色表演其實很重要。

Chatterbox V3 本身只有 0.5B、MIT，而且現在有中文專門的 Single Language Pack。([GitHub][8])

所以它也很適合作為 lightweight benchmark。

---

## GPT-SoVITS 和 F5-TTS 呢？

我會留著，但不是 U.E.P 情緒系統的主要參考。

GPT-SoVITS 現在仍然非常強在：

```text
5 秒 zero-shot
1 分鐘 few-shot training
```

而且它整個 dataset preparation / segmentation / ASR / training WebUI 非常成熟。([GitHub][9])

所以：

> **值得參考訓練工作流。**

不是特別值得參考 emotion API。

F5-TTS 則是非常漂亮的 flow-matching architecture，而且模型只有約 0.3B，官方 PyTorch RTF 約 0.147，TensorRT server benchmark 可以做到約 0.039。([GitHub][10])

但它的核心強項是：

> simple architecture + zero-shot naturalness

不是 explicit emotional control。

而且官方 pretrained weights 是 CC-BY-NC，所以如果未來 U.E.P 有商業用途，也要注意。([GitHub][10])

---

所以如果換成我來設計現在的 **UEP TTS vNext**，我反而不會選一套模型把所有控制方式綁死。

我會把這幾年的好點子拼起來：

```text
                   UEP LLM
                      │
                Speech Planner
                      │
        ┌─────────────┼──────────────┐
        │             │              │
     Emotion        Style         Prosody
        │             │              │
   Index-style    Qwen-style      duration
     vector       description     speed/pause
        │             │              │
        └─────────────┼──────────────┘
                      │
                Speech Markup
                      │
        「我本來覺得沒問題，
         <hesitant>但是……</hesitant>
         <excited>啊，我知道了！</excited>」
                      │
              Backend Adapter
        ┌─────────────┼──────────────┐
        │             │              │
     Index2.5     VoxCPM2       Qwen3-TTS
```

而內部的標準 interface 我甚至會弄成：

```python
SpeechStyle(
    emotion={
        "happy": 0.1,
        "sad": 0.0,
        "angry": 0.0,
        "calm": 0.8,
    },

    # Index / Chatterbox 思路
    intensity=0.55,
    expressiveness=0.35,

    # prosody
    speed=0.95,
    volume=1.0,

    # Qwen / Vox / CosyVoice 思路
    style="relaxed, slightly curious, conversational",

    # Fish S2 思路
    segments=[
        Segment("這裡好像有點奇怪", style="hesitant"),
        Segment("啊，我知道了！", style="excited"),
    ],
)
```
