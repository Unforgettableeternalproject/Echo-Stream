"""Discord 平台的 STT adapter：轉錄 backend 與 bot 入口。

Echo Stream 是跨平台唯一語音主系統；Discord 特有的行為（固定語言、RMS 門檻、
歌曲 credit 過濾、簡體才轉繁體、動態熱詞）在這一層以可插拔元件表達，
bot 只負責餵「48k stereo PCM + 成員 id」。

邏輯移植自 bot repo（U.E.P's Mind Reflourished）``cogs/stt.py``，**不 import bot**：
過濾與後處理是 ``SegmentProcessor._recognize`` 的等價實作。

- ``DiscordSttProfile``：平台設定集合（日後 Host PC / Web 可並列自己的 profile）。
- ``DiscordTranscriberBackend``：實作 ``TranscriberBackend``（同步阻塞，stage 丟 executor）。
- ``DiscordAudioSource``：bot 入口，每成員一個有狀態重取樣器，轉送給多聲道 stage。

opencc / faster_whisper / echo_stt 一律延遲 import；模組 import 與建構都不載模型。
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ..contracts.types import STT_SAMPLE_RATE
from .discord_resample import StreamingResampler48to16
from .input_stt import TranscriptionResult

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 文字過濾與後處理（移植自 bot cogs/stt.py）
# ---------------------------------------------------------------------------

# 幻覺黑名單：比對前先正規化（小寫、去標點與空白）。bot :97-112
_HALLUCINATION_EXACT = {
    "thankyou", "thankyouforwatching", "thanksforwatching", "thanks", "bye", "byebye",
    "you", "謝謝", "謝謝觀看", "謝謝大家", "感謝觀看", "谢谢", "谢谢观看",
    "ご視聴ありがとうございました", "ありがとうございました",
}
_HALLUCINATION_CONTAINS = [
    "ご視聴いただきありがとうございます", "ご視聴ありがとうございました", "チャンネル登録",
    "請不吝點贊訂閱", "請不吝點贊", "點贊訂閱轉發", "訂閱我的頻道", "訂閱頻道", "订阅我的频道",
    "字幕由", "字幕提供", "詞曲李宗盛", "詞曲 李宗盛", "可能夾雜英文詞彙", "以下是繁體中文的對話",
    "字幕志愿者", "字幕志願者", "amara.org", "明鏡與點點欄目",
    "thank you for watching", "please subscribe", "subtitles by", "subscribe to my channel",
    "感謝您的觀看", "感谢您的观看", "謝謝收看", "謝謝你的觀看",
]
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)
_TRAD_SIMP_SHARED = set("才回只台吃托后里干")  # 繁簡同形，不能當簡體判斷依據
_CJK_RE = re.compile(r"[一-鿿]")
_LONE_FULLWIDTH_B_RE = re.compile(r"(?<![A-Za-z0-9Ａ-Ｚａ-ｚ])Ｂ(?![A-Za-z0-9Ａ-Ｚａ-ｚ])")

# 字幕／歌曲 credit 格式：「詞・作曲・編曲 李宗盛」「字幕由 XXX 提供」「XXX 字幕組」
_CREDIT_KW = (
    "作詞|作曲|編曲|填詞|詞曲|詞|曲|監製|製作|演唱|原唱|混音|母帶|字幕|翻譯|校對|時間軸|志願者|"
    "作词|编曲|填词|词曲|词|监制|制作|演唱|翻译|校对|时间轴|志愿者|字幕"
)
_CREDIT_SEP = r"[\s・·•/／:：,，、\-－]"
# 名字 token：<=8 字，且不含常見口語虛詞（避免「曲 這首歌很好聽」之類句子被誤傷）
_CREDIT_NAME = r"[^\s・·•/／:：,，、\-－的是了嗎呢吧啊不很我你他好難太也都就這那有會]{1,8}"
_CREDIT_RES = [
    re.compile(
        rf"^(?:{_CREDIT_KW})(?:{_CREDIT_SEP}*(?:{_CREDIT_KW}))*{_CREDIT_SEP}+{_CREDIT_NAME}$"
    ),
    re.compile(
        rf"^(?:{_CREDIT_KW})(?:{_CREDIT_SEP}*(?:{_CREDIT_KW}))+{_CREDIT_SEP}*{_CREDIT_NAME}$"
    ),
    re.compile(
        rf"^(?:字幕|翻譯|翻译)由{_CREDIT_SEP}*.{{1,16}}?{_CREDIT_SEP}*"
        r"(?:提供|製作|制作|翻譯|翻译)?$"
    ),
    re.compile(rf"^{_CREDIT_NAME}{_CREDIT_SEP}*(?:字幕組|字幕组|字幕志願者|字幕志愿者)$"),
]


def clean_whisper_garbage(text: str) -> str:
    """保底正規化 Whisper 的標點亂碼（byte-level token 被破壞）。僅限含中文的文字。

    獨立全形「Ｂ」→「，」、「﹑」→「、」、移除 U+FFFD。（bot :120）
    """
    if not _CJK_RE.search(text):
        return text
    text = _LONE_FULLWIDTH_B_RE.sub("，", text)
    return text.replace("﹑", "、").replace("�", "")


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).lower()


def credit_hallucination(text: str) -> str | None:
    """整個 segment 只是「字幕／歌曲 credit + 名字」格式時回傳原因。（bot :237）

    只比對整段格式，不看人名本身：討論李宗盛的真實句子不會命中。
    """
    t = text.strip().rstrip(".。!！?？").strip()
    if not t:
        return None
    for i, r in enumerate(_CREDIT_RES):
        if r.match(t):
            return f"credit_format:{i}"
    return None


def is_hallucination(text: str) -> str | None:
    """命中幻覺黑名單回傳原因，否則 None。（bot :509）"""
    norm = _normalize_text(text)
    if not norm:
        return "empty"
    if norm in _HALLUCINATION_EXACT:
        return f"exact:{norm}"
    for p in _HALLUCINATION_CONTAINS:
        if _normalize_text(p) in norm:
            return f"contains:{p}"
    return None


def hotword_leak_reason(seg_text: str, hotwords: Iterable[str] | None) -> str | None:
    """hotwords 外洩：segment 幾乎只由詞表內的詞組成且含 >=3 個不同詞。（bot :248）"""
    words = list(hotwords or [])
    if not words:
        return None
    seg = _normalize_text(seg_text)
    if not seg:
        return None
    hit: set[str] = set()
    covered = [False] * len(seg)
    for w in words:
        nw = _normalize_text(str(w))
        if not nw:
            continue
        start = 0
        while True:
            i = seg.find(nw, start)
            if i < 0:
                break
            hit.add(nw)
            for t in range(i, i + len(nw)):
                covered[t] = True
            start = i + len(nw)
    ratio = sum(covered) / len(seg)
    if len(hit) >= 3 and ratio >= 0.6:
        return f"hotword_list:{len(hit)}words,{ratio:.2f}"
    return None


def prompt_leak_reason(seg_text: str, prompt: str | None, seg_dur: float) -> str | None:
    """segment 是否為 initial_prompt 外洩（模型把提示詞念出來）。（bot :477）"""
    if not prompt:
        return None
    seg, pr = _normalize_text(seg_text), _normalize_text(prompt)
    if not seg or not pr:
        return None
    if 2 <= len(seg) <= 4 and seg in pr and seg_dur <= 1.0:
        return f"short_exact:{seg}"
    covered = [False] * len(seg)
    for i in range(len(seg)):
        longest = 0
        for j in range(len(pr)):
            k = 0
            while i + k < len(seg) and j + k < len(pr) and seg[i + k] == pr[j + k]:
                k += 1
            longest = max(longest, k)
        if longest >= 4:
            for t in range(i, i + longest):
                covered[t] = True
    ratio = sum(covered) / len(seg)
    if ratio >= 0.5:
        return f"coverage:{ratio:.2f}"
    return None


def _ascii_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace() and c not in ",.?!'\"-:;…"]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def english_guard_reason(
    text: str, conf: float, rms: float, max_words: int, max_conf: float, max_rms: float
) -> str | None:
    """純英文、短、信心低、音量小 → 回傳原因（多半是中文被誤聽）。（bot :466）"""
    if _CJK_RE.search(text) or _ascii_ratio(text) < 0.95:
        return None
    words = len(text.split())
    if words <= max_words and conf < max_conf and rms < max_rms:
        return f"words={words},conf={conf:.2f},rms={rms:.4f}"
    return None


def _worst(segments: list[Any], attr: str, pick: Any) -> float | None:
    """保留段落中某個解碼品質欄位的最差值；欄位拿不到就回 None。"""
    values = [v for v in (getattr(seg, attr, None) for seg in segments) if v is not None]
    return float(pick(values)) if values else None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscordSttProfile:
    """Discord 平台的 STT 設定。預設值 = bot ``DEFAULT_STT_SETTINGS``（cogs/stt.py:51）。"""

    language: str | None = "zh"  # None = 自動偵測
    min_rms: float = 0.005  # 16k float RMS，約 -46 dBFS
    convert_simplified: bool = True  # 只在偵測到簡體時才轉
    convert_scheme: str = "s2twp"
    filter_credits: bool = True
    clean_garbage: bool = True
    filter_hallucination: bool = True
    # 模型來源：None = 走 echo_stt 引擎預載模型；有路徑 = faster_whisper 載自訂模型（如 Breeze）
    model_path: str | None = None
    compute_type: str = "int8_float16"
    device: str = "auto"
    # faster-whisper 參數
    beam_size: int = 5
    condition_on_previous_text: bool = False
    repetition_penalty: float = 1.0
    no_speech_threshold: float = 0.6
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "min_silence_duration_ms": 500, "speech_pad_ms": 200, "threshold": 0.5,
        }
    )
    # 逐 segment 過濾
    max_no_speech_prob: float = 0.7
    min_avg_logprob: float = -1.2
    max_compression_ratio: float = 2.4
    min_text_chars: int = 1
    # 熱詞初始值（執行中用 set_hotwords 替換）。initial_prompt 預設空——Whisper 不做一般用途 prompt
    hotwords: tuple[str, ...] = ()
    hotwords_max: int = 30
    initial_prompt: str = ""
    # 過度英文化防線（預設關）
    english_guard: bool = False
    english_guard_max_words: int = 2
    english_guard_max_conf: float = 0.5
    english_guard_max_rms: float = 0.02


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _load_custom_whisper(profile: DiscordSttProfile) -> Any:
    """faster_whisper 直接載自訂模型。（bot ``load_custom_whisper`` :194）"""
    from faster_whisper import WhisperModel

    return WhisperModel(
        profile.model_path,
        device=profile.device or "auto",
        compute_type=profile.compute_type or "int8_float16",
        local_files_only=True,
    )


def _load_echo_stt_model() -> Any:
    """走 echo_stt 引擎載入預設模型並取出其 faster-whisper 物件。（bot ``_init_engine`` :974）"""
    import copy

    from echo_stt.core.config import load_config
    from echo_stt.core.multi_channel_engine import MultiChannelSTTEngine

    cfg = copy.deepcopy(load_config())
    cfg["multi_channel"] = {
        "enabled": True, "num_workers": 1, "max_channels": 8,
        "vad_config": {"sample_rate": STT_SAMPLE_RATE},
    }
    engine = MultiChannelSTTEngine(config=cfg)
    if not engine.initialize():
        raise RuntimeError("MultiChannelSTTEngine 初始化失敗——檢查 echo_stt 的模型與設定")
    return engine._stt_engine._fw_model  # noqa: SLF001 - 與 bot 相同的取法


class DiscordTranscriberBackend:
    """Discord 場景的 ``TranscriberBackend``。

    模型只載一次：``model`` 可由宿主預載後注入；否則第一次 ``transcribe``（或明確呼叫
    ``preload``）時才載。``transcribe`` 同步阻塞，由 stage 丟 executor。

    :param profile: 設定集合，None = 預設 profile。
    :param model: 已載入的 faster-whisper 風格模型（``transcribe(audio, **kw)`` →
        ``(segments, info)``）。
    :param overrides: 直接覆寫 profile 欄位，例如 ``min_rms=0.01``。
    """

    def __init__(
        self,
        profile: DiscordSttProfile | None = None,
        *,
        model: Any = None,
        **overrides: Any,
    ) -> None:
        base = profile or DiscordSttProfile()
        self.profile = replace(base, **overrides) if overrides else base
        self._model = model
        self._model_lock = threading.Lock()
        self._converter: Any = None
        self._s2t: Any = None
        self._converter_loaded = False
        self._hw_lock = threading.Lock()
        self._global_hotwords: tuple[str, ...] = self._clip(self.profile.hotwords)
        self._global_prompt: str = self.profile.initial_prompt or ""
        self._speaker_hotwords: dict[str, tuple[str, ...]] = {}
        self._speaker_prompt: dict[str, str] = {}

    # -- 模型 -------------------------------------------------------------

    def preload(self) -> None:
        """明確載入模型（宿主可在啟動時呼叫）。已有模型則什麼都不做。"""
        self._get_model()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                if self.profile.model_path:
                    self._model = _load_custom_whisper(self.profile)
                else:
                    self._model = _load_echo_stt_model()
            return self._model

    # -- 動態熱詞 ---------------------------------------------------------

    def _clip(self, words: Iterable[str] | None) -> tuple[str, ...]:
        """去空白、不分大小寫去重、超過 hotwords_max 截斷（bot HotwordStore.get :421-455）。"""
        merged: list[str] = []
        seen: set[str] = set()
        for w in words or ():
            w = str(w).strip()
            if w and w.casefold() not in seen:
                seen.add(w.casefold())
                merged.append(w)
        limit = int(self.profile.hotwords_max or 0)
        if limit and len(merged) > limit:
            _log.warning(
                "[DiscordSTT] 詞表共 %d 個詞，超過 hotwords_max=%d，截斷最後 %d 個",
                len(merged), limit, len(merged) - limit,
            )
            merged = merged[:limit]
        return tuple(merged)

    def set_hotwords(
        self, words: Iterable[str] | None, speaker_id: str | None = None
    ) -> None:
        """替換熱詞。``speaker_id=None`` 是全域；指定成員則只影響該聲道（優先於全域）。

        空清單 = 清除（逐聲道清除後退回全域；全域清除 = 不傳熱詞）。thread-safe。
        """
        clipped = self._clip(words)
        with self._hw_lock:
            if speaker_id is None:
                self._global_hotwords = clipped
            elif clipped:
                self._speaker_hotwords[speaker_id] = clipped
            else:
                self._speaker_hotwords.pop(speaker_id, None)

    def set_initial_prompt(self, prompt: str | None, speaker_id: str | None = None) -> None:
        """替換 initial_prompt（範圍規則同 ``set_hotwords``）。預設不使用——見 profile 說明。"""
        prompt = prompt or ""
        with self._hw_lock:
            if speaker_id is None:
                self._global_prompt = prompt
            elif prompt:
                self._speaker_prompt[speaker_id] = prompt
            else:
                self._speaker_prompt.pop(speaker_id, None)

    def clear_speaker(self, speaker_id: str) -> None:
        """成員離開時清掉逐聲道的熱詞與提示詞。"""
        with self._hw_lock:
            self._speaker_hotwords.pop(speaker_id, None)
            self._speaker_prompt.pop(speaker_id, None)

    def get_hotwords(self, speaker_id: str | None = None) -> tuple[str, ...]:
        """目前對該聲道生效的熱詞（逐聲道優先、否則全域）。"""
        return self._effective(speaker_id)[0]

    def _effective(self, speaker_id: str | None) -> tuple[tuple[str, ...], str]:
        with self._hw_lock:
            hw = (
                self._speaker_hotwords[speaker_id]
                if speaker_id is not None and speaker_id in self._speaker_hotwords
                else self._global_hotwords
            )
            prompt = (
                self._speaker_prompt[speaker_id]
                if speaker_id is not None and speaker_id in self._speaker_prompt
                else self._global_prompt
            )
        return hw, prompt

    # -- 繁體化 -----------------------------------------------------------

    def _convert_zh(self, text: str) -> str:
        """只在偵測到簡體時轉 s2twp。（bot ``_convert_zh`` :552）

        已是純繁體（Breeze 原生輸出）就不轉：s2twp 對繁體輸入會誤傷
        （演算法→演演算法、台灣→臺灣）。繁簡同形字不算簡體證據。
        """
        p = self.profile
        if not p.convert_simplified or not p.convert_scheme:
            return text
        if not self._converter_loaded:
            self._converter_loaded = True
            try:
                from opencc import OpenCC

                self._converter = OpenCC(p.convert_scheme)
                self._s2t = OpenCC("s2t")
            except Exception as e:  # noqa: BLE001 - OpenCC 缺席不能讓 STT 掛掉
                _log.warning("[DiscordSTT] OpenCC 不可用，略過繁體化: %s", e)
        if not self._converter:
            return text
        if not any(
            a != b and a not in _TRAD_SIMP_SHARED
            for a, b in zip(text, self._s2t.convert(text), strict=False)
        ):
            return text
        return self._converter.convert(text)

    # -- 轉錄 -------------------------------------------------------------

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult | None:
        """``TranscriberBackend``：全域熱詞轉錄。"""
        return self.transcribe_for(None, pcm, sample_rate)

    def transcribe_for(
        self, speaker_id: str | None, pcm: bytes, sample_rate: int
    ) -> TranscriptionResult | None:
        """多聲道 stage 的可選入口：該成員的熱詞優先於全域。回傳 None = 丟棄。"""
        import numpy as np

        if sample_rate != STT_SAMPLE_RATE:
            raise ValueError(f"Discord backend 只吃 {STT_SAMPLE_RATE}Hz，收到 {sample_rate}")
        p = self.profile
        usable = len(pcm) - len(pcm) % 2
        audio = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
        if audio.size == 0:
            return None
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < p.min_rms:
            _log.debug("[DiscordSTT] 丟棄 low_energy rms=%.5f<%s", rms, p.min_rms)
            return None

        hotwords, prompt = self._effective(speaker_id)
        kwargs: dict[str, Any] = {
            "task": "transcribe",
            "language": p.language,
            "beam_size": p.beam_size,
            "condition_on_previous_text": p.condition_on_previous_text,
            "repetition_penalty": p.repetition_penalty,
            "no_speech_threshold": p.no_speech_threshold,
            "vad_filter": p.vad_filter,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        if hotwords:
            kwargs["hotwords"] = ", ".join(hotwords)
        if p.vad_filter and p.vad_parameters:
            kwargs["vad_parameters"] = dict(p.vad_parameters)

        segments_iter, info = self._get_model().transcribe(audio, **kwargs)

        kept = []
        for seg in segments_iter:
            text = seg.text.strip()
            reasons = []
            if seg.no_speech_prob > p.max_no_speech_prob:
                reasons.append("no_speech")
            if seg.avg_logprob < p.min_avg_logprob:
                reasons.append("avg_logprob")
            if seg.compression_ratio > p.max_compression_ratio:
                reasons.append("compression_ratio")
            if p.filter_credits and credit_hallucination(text):
                reasons.append("credit")
            if prompt and prompt_leak_reason(text, prompt, seg.end - seg.start):
                reasons.append("prompt_leak")
            if hotword_leak_reason(text, hotwords):
                reasons.append("hotword_leak")
            if text and not reasons:
                kept.append(seg)
            elif reasons:
                _log.debug("[DiscordSTT] segment 被過濾 %s: %r", reasons, text)
        if not kept:
            return None

        text = " ".join(seg.text.strip() for seg in kept)
        if p.clean_garbage:
            text = clean_whisper_garbage(text)
        text = self._convert_zh(text).strip()
        if len(_normalize_text(text)) < p.min_text_chars:
            return None
        if p.filter_hallucination and is_hallucination(text):
            return None

        # 信心度：以時長加權的 avg_logprob 取 exp，落在 0-1
        weights = [max(seg.end - seg.start, 1e-3) for seg in kept]
        mean_lp = sum(w * s.avg_logprob for w, s in zip(weights, kept, strict=True)) / sum(weights)
        confidence = float(min(1.0, max(0.0, math.exp(mean_lp))))
        if p.english_guard and english_guard_reason(
            text, confidence, rms, p.english_guard_max_words,
            p.english_guard_max_conf, p.english_guard_max_rms,
        ):
            return None
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=confidence,
            temperature=_worst(kept, "temperature", max),
            avg_logprob=_worst(kept, "avg_logprob", min),
            compression_ratio=_worst(kept, "compression_ratio", max),
        )


# ---------------------------------------------------------------------------
# bot 入口
# ---------------------------------------------------------------------------


class MultiChannelStageLike(Protocol):
    """多聲道 InputStage 的最小形狀（duck typing，不 import 實作類別）。"""

    def feed(self, speaker_id: str, pcm: bytes, ts: float | None = None) -> None: ...
    def join(self, speaker_id: str) -> None: ...
    def leave(self, speaker_id: str, flush: bool = True) -> None: ...
    def discard(self, speaker_id: str | None = None) -> None: ...


class _Member:
    __slots__ = ("lock", "resampler")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.resampler = StreamingResampler48to16()


class DiscordAudioSource:
    """bot 的入口：``feed(member_id, 48k stereo PCM)`` → 每成員獨立重取樣 → 轉給多聲道 stage。

    ``feed`` 可從 Discord sink 執行緒呼叫；各成員的重取樣器互不共享狀態，
    同一成員的 feed 以成員鎖保序。重連時 bot 先 ``leave`` 全部再 ``join``。
    """

    def __init__(
        self,
        stage: MultiChannelStageLike,
        *,
        clock: Callable[[], float] | None = None,
        on_leave: Callable[[str], None] | None = None,
    ) -> None:
        self._stage = stage
        self._clock = clock
        self._on_leave = on_leave
        self._members: dict[str, _Member] = {}
        self._members_lock = threading.Lock()

    def _stamp(self, ts: float | None) -> float | None:
        if ts is not None or self._clock is None:
            return ts
        return self._clock()

    def _member(self, member_id: str) -> _Member:
        with self._members_lock:
            m = self._members.get(member_id)
            if m is None:
                m = self._members[member_id] = _Member()
            return m

    def feed(self, member_id: str, pcm48k_stereo: bytes, ts: float | None = None) -> None:
        """餵一個 chunk（任意大小）。

        ``ts`` 為 None 且未注入 ``clock`` 時原樣傳 None，由 stage 用自己的時鐘蓋時間戳——
        ``ts`` 必須與 stage 的時鐘同一時域，否則 speech_*_ts 與 TTFA 無法相減。
        """
        m = self._member(member_id)
        with m.lock:
            out = m.resampler.process(pcm48k_stereo)
            if out:
                self._stage.feed(member_id, out, self._stamp(ts))

    def join(self, member_id: str) -> None:
        """成員加入：重取樣器歸零（重連後不沿用舊濾波歷史）。"""
        m = self._member(member_id)
        with m.lock:
            m.resampler.reset()
        self._stage.join(member_id)

    def leave(self, member_id: str, flush: bool = True) -> None:
        """成員離開：``flush`` 時先把重取樣器尾端送進 stage，再通知 stage，最後移除狀態。"""
        with self._members_lock:
            m = self._members.pop(member_id, None)
        if m is not None:
            with m.lock:
                tail = m.resampler.flush() if flush else b""
                m.resampler.reset()
                if tail:
                    self._stage.feed(member_id, tail, self._stamp(None))
        self._stage.leave(member_id, flush=flush)
        if self._on_leave is not None:
            self._on_leave(member_id)

    def discard(self, member_id: str | None = None) -> None:
        """丟棄重取樣狀態（None = 全部成員）並轉給 stage。"""
        with self._members_lock:
            if member_id is None:
                targets = list(self._members.values())
            else:
                found = self._members.get(member_id)
                targets = [found] if found is not None else []
        for m in targets:
            with m.lock:
                m.resampler.reset()
        self._stage.discard(member_id)


__all__ = [
    "DiscordAudioSource",
    "DiscordSttProfile",
    "DiscordTranscriberBackend",
    "MultiChannelStageLike",
    "clean_whisper_garbage",
    "credit_hallucination",
    "english_guard_reason",
    "hotword_leak_reason",
    "is_hallucination",
    "prompt_leak_reason",
]
