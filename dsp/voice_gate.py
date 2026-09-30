"""Voiceprint gate (module G1): "is the speaker me right now".

Contract: VoiceGate(model_dir, threshold=0.5) (models auto-found under
models/); enroll(wav16k) accumulates a voiceprint and returns a report dict;
decide(wav16k) returns a smoothed 0-1 similarity, None when undecidable;
vad_prob() returns voice probability; save_profile()/load_profile() persist
the voiceprint; suggest_threshold() proposes a threshold without mutating.

Behavior: pure CPU (CAM++ embedding plus Silero VAD); decide never
thresholds — the caller compares; no cross-call audio state, only the
enrolled vector and one smoothed score. Measured: 1 s windows separate
same-speaker 0.60-0.77 from different-speaker 0.00-0.38 at threshold 0.5.
"""
from __future__ import annotations

import glob
import json
import math
import os
import threading
import time

import numpy as np

#: Voiceprint/VAD working rate (the pipeline feeds the trailing 1 s at 16k).
SAMPLE_RATE = 16000
#: Fixed Silero analysis window at 16k (32 ms).
VAD_WINDOW = 512
#: VAD state shapes: h/c build (2, batch, 64), v5 build (2, batch, 128).
_VAD_HC_STATE_SHAPE = (2, 1, 64)
_VAD_V5_STATE_SHAPE = (2, 1, 128)
#: Shorter audio is undecidable (decide returns None); CAM++ needs enough frames.
MIN_SPEECH_SECONDS = 0.4
#: VAD probability readout: mean of the top 25% window scores.
_VAD_PEAK_RATIO = 0.25
#: Pre-VAD normalization target (-20 dBFS) and silence floor.
_VAD_NORM_RMS = 0.1
_VAD_SILENCE_RMS = 1e-5
#: Enrollment segmentation: max silence gap and edge margins (seconds).
_SPEECH_GAP_SECONDS = 0.35
_SPEECH_MARGIN_SECONDS = 0.15
#: Cap on segments per enroll call (bounds long recordings).
_MAX_ENROLL_PARTS = 50
#: Default voiceprint match threshold (cosine).
DEFAULT_THRESHOLD = 0.5
#: Below this VAD score decide reports no speech (None) without embedding.
DEFAULT_VAD_THRESHOLD = 0.5
#: Cross-call smoothing factor: higher follows faster.
DEFAULT_SMOOTH_ALPHA = 0.4
#: Outlier rejection: keep line is max(mean - margin, floor).
_ENROLL_OUTLIER_MARGIN = 0.2
_ENROLL_MIN_KEEP_SIM = 0.6
#: Profile format version.
PROFILE_VERSION = 1

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Model dir, shared with app.gui.
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
#: Default voiceprint profile path, shared with app.gui.
DEFAULT_PROFILE_PATH = os.path.join(DEFAULT_MODEL_DIR, "voice_profile.npz")

#: Model filename keywords (lowercased substring match).
_SPEAKER_KEYWORDS = ("campplus", "cam++")
_VAD_KEYWORDS = ("silero", "vad")


def _find_onnx(model_dir, keywords, exclude=("pyannote", "segmentation")):
    """Pick a matching .onnx under model_dir; int8 builds preferred."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("模型目录不存在：%s" % model_dir)
    hits = []
    for path in glob.glob(os.path.join(model_dir, "*.onnx")):
        name = os.path.basename(path).lower()
        if not any(key in name for key in keywords):
            continue
        if any(key in name for key in exclude):
            continue
        hits.append(path)
    if not hits:
        return None
    hits.sort(
        key=lambda p: (0 if "int8" in os.path.basename(p).lower() else 1, os.path.basename(p).lower())
    )
    return hits[0]


def _as_mono_float32(wav) -> np.ndarray:
    """Coerce torch tensor / list / ndarray to (n,) float32 mono."""
    if wav is None:
        raise ValueError("音频为空")
    if hasattr(wav, "detach"):  # torch.Tensor, possibly still on GPU.
        wav = wav.detach().cpu().numpy()
    arr = np.asarray(wav)
    if arr.dtype.kind == "i":
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(max(info.max, -info.min))
    elif arr.dtype.kind == "u":
        mid = float(1 << (np.iinfo(arr.dtype).bits - 1))
        arr = (arr.astype(np.float32) - mid) / mid
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[1] <= 8 else arr.mean(axis=0)
    elif arr.ndim != 1:
        raise ValueError("音频维度不支持：shape=%r（应为 (n,) 或 (n, 声道)）" % (arr.shape,))
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


def _normalize_for_vad(x):
    """Normalize whole-clip RMS to -20 dBFS; near-silence returns zeros."""
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if not math.isfinite(rms) or rms <= _VAD_SILENCE_RMS:
        return np.zeros_like(x)
    return np.clip(x * (_VAD_NORM_RMS / rms), -1.0, 1.0).astype(np.float32)


class VoiceGate:
    """CAM++ voiceprint plus Silero VAD gate (pure CPU, stream-safe)."""

    def __init__(
        self,
        model_dir=None,
        threshold=DEFAULT_THRESHOLD,
        *,
        vad_threshold=DEFAULT_VAD_THRESHOLD,
        num_threads=1,
        provider="cpu",
        profile_path=None,
        require_speech_on_enroll=True,
        smooth_alpha=DEFAULT_SMOOTH_ALPHA,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - missing-dependency hint
            raise ImportError(
                "dsp.voice_gate 需要 onnxruntime（sherpa-onnx 的依赖）："
                "用 RVC runtime python 执行 python -m pip install sherpa-onnx"
            ) from exc
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "dsp.voice_gate 需要 sherpa-onnx：用 RVC runtime python 执行 "
                "python -m pip install sherpa-onnx（自带 onnxruntime）"
            ) from exc

        self.model_dir = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
        self.speaker_model = _find_onnx(self.model_dir, _SPEAKER_KEYWORDS)
        if not self.speaker_model:
            raise FileNotFoundError(
                "模型目录里没找到 CAM++ 声纹模型（文件名需含 campplus/cam++）：%s" % self.model_dir
            )
        self.vad_model = _find_onnx(self.model_dir, _VAD_KEYWORDS)
        if not self.vad_model:
            raise FileNotFoundError(
                "模型目录里没找到 Silero VAD（文件名需含 silero/vad）：%s" % self.model_dir
            )

        self.threshold = self._check_threshold(threshold)
        self.vad_threshold = self._check_vad_threshold(vad_threshold)
        self.require_speech_on_enroll = bool(require_speech_on_enroll)
        self.smooth_alpha = self._check_smooth_alpha(smooth_alpha)
        #: Smoothing state: last valid smoothed score, None when none yet.
        self._smooth_score = None
        #: Raw similarity of the latest decide call.
        self.last_raw_score = None
        #: Report dict of the latest enroll; None when never enrolled.
        self.last_enroll_report = None
        #: Raw per-segment similarities of the latest enroll.
        self._last_enroll_sims = ()
        #: Latest decide result (None when undecided); read by the GUI.
        self.last_score = None
        #: VAD probability of the latest decide (debug/display).
        self.last_vad_prob = None
        #: Profile path used by save/load.
        self.profile_path = profile_path or DEFAULT_PROFILE_PATH

        self._lock = threading.RLock()
        self._embedding = None  # Normalized voiceprint vector (dim,).
        self._emb_sum = None  # Running sum of normalized enrollment embeddings.
        self._emb_count = 0

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.speaker_model,
            num_threads=max(1, int(num_threads)),
            provider=str(provider),
            debug=False,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        if int(self._extractor.dim) <= 0:  # dim 0 means the model failed to load.
            raise RuntimeError("声纹模型加载失败：%s" % self.speaker_model)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(num_threads))
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # Errors only; silence optimizer warnings.
        self._vad = ort.InferenceSession(
            self.vad_model, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs = {node.name for node in self._vad.get_inputs()}
        if {"h", "c"} <= inputs:
            self._vad_layout = "hc"
        elif "state" in inputs:
            self._vad_layout = "state"
        else:
            raise ValueError("无法识别的 Silero VAD 模型（输入名 %s）：%s" % (sorted(inputs), self.vad_model))
        self._vad_window = VAD_WINDOW

        if self.profile_path and os.path.isfile(self.profile_path):
            try:  # Auto-load an existing profile; the caller keeps owning the threshold.
                self.load_profile(self.profile_path)
            except Exception as exc:  # Ignore corrupt profiles at startup.
                self._warn("声纹档案载入失败（忽略）：%s：%s" % (self.profile_path, exc))

    # ---------- State ----------

    def __repr__(self):
        return "<VoiceGate %s enrolled=%s threshold=%.2f last=%s>" % (
            "int8" if "int8" in os.path.basename(self.speaker_model).lower() else "fp32",
            self.is_enrolled,
            self.threshold,
            self.last_score,
        )

    @property
    def is_enrolled(self) -> bool:
        """True once enroll()/load_profile() has provided a voiceprint."""
        return self._embedding is not None

    @property
    def embedding(self):
        """Current voiceprint vector (a copy; None when unenrolled)."""
        return None if self._embedding is None else self._embedding.copy()

    @property
    def num_segments(self) -> int:
        """Segments accumulated into the mean voiceprint."""
        return int(self._emb_count)

    @property
    def embed_dim(self) -> int:
        return int(self._extractor.dim)

    @staticmethod
    def _check_threshold(value) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("声纹阈值应在 0-1 之间（cosine）：%r" % (value,))
        return value

    @staticmethod
    def _check_vad_threshold(value) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("VAD 阈值应在 0-1 之间（人声概率）：%r" % (value,))
        return value

    @staticmethod
    def _check_smooth_alpha(value) -> float:
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("平滑系数应在 0-1 之间（越大跟随越快）：%r" % (value,))
        return value

    @staticmethod
    def _warn(text):
        try:
            print("[voice_gate] %s" % text)
        except Exception:
            pass

    def set_threshold(self, value) -> None:
        """Update the match threshold (verify()/is_self() only; decide stays raw)."""
        self.threshold = self._check_threshold(value)

    def set_vad_threshold(self, value) -> None:
        """Update the activity threshold below which decide returns None."""
        self.vad_threshold = self._check_vad_threshold(value)

    def set_smooth_alpha(self, value) -> None:
        """Update the smoothing factor and drop accumulated smoothing."""
        self.smooth_alpha = self._check_smooth_alpha(value)
        with self._lock:
            self._smooth_score = None

    def reset_smooth(self) -> None:
        """Drop accumulated smoothing; the next valid decision returns raw."""
        with self._lock:
            self._smooth_score = None

    def reset(self) -> None:
        """Clear the enrolled voiceprint and smoothing; saved files untouched."""
        with self._lock:
            self._embedding = None
            self._emb_sum = None
            self._emb_count = 0
            self._smooth_score = None

    # ---------- VAD ----------

    def _vad_probs(self, x) -> np.ndarray:
        """Per-32ms-window voice probabilities; state restarts from zero each call."""
        x = _normalize_for_vad(x)
        n = int(x.size // self._vad_window)
        probs = np.zeros(max(n, 0), dtype=np.float32)
        if n <= 0:
            return probs
        sr = np.asarray(SAMPLE_RATE, dtype=np.int64)
        if self._vad_layout == "hc":
            h = np.zeros(_VAD_HC_STATE_SHAPE, dtype=np.float32)
            c = np.zeros(_VAD_HC_STATE_SHAPE, dtype=np.float32)
            for i in range(n):
                win = x[i * self._vad_window : (i + 1) * self._vad_window].reshape(1, -1)
                out, h, c = self._vad.run(None, {"input": win, "sr": sr, "h": h, "c": c})
                probs[i] = float(np.ravel(out)[0])
        else:
            state = np.zeros(_VAD_V5_STATE_SHAPE, dtype=np.float32)
            for i in range(n):
                win = x[i * self._vad_window : (i + 1) * self._vad_window].reshape(1, -1)
                out, state = self._vad.run(None, {"input": win, "state": state, "sr": sr})
                probs[i] = float(np.ravel(out)[0])
        return probs

    def vad_prob(self, wav16k) -> float:
        """Whole-clip voice probability (mean of the top 25% windows)."""
        x = _as_mono_float32(wav16k)
        with self._lock:
            probs = self._vad_probs(x)
        return self._peak_prob(probs)

    @staticmethod
    def _peak_prob(probs) -> float:
        """Mean of the top 25% window scores; 0.0 with no windows."""
        if probs.size == 0:
            return 0.0
        k = max(1, int(math.ceil(probs.size * _VAD_PEAK_RATIO)))
        return float(np.sort(probs)[-k:].mean())

    def _speech_parts(self, x):
        """Split a clip into speech segments on VAD activity."""
        return self._parts_from_probs(x, self._vad_probs(x))

    def _parts_from_probs(self, x, probs):
        """Split on precomputed window probabilities without rerunning VAD."""
        if probs.size == 0:
            return []
        act = probs >= self.vad_threshold
        gap = max(1, int(round(_SPEECH_GAP_SECONDS * SAMPLE_RATE / self._vad_window)))
        margin = int(round(_SPEECH_MARGIN_SECONDS * SAMPLE_RATE))
        parts = []
        i = 0
        total = act.size
        while i < total and len(parts) < _MAX_ENROLL_PARTS:
            if not act[i]:
                i += 1
                continue
            last = i
            j = i
            while j < total:
                if act[j]:
                    last = j
                    j += 1
                elif j - last <= gap:
                    j += 1
                else:
                    break
            start = max(0, i * self._vad_window - margin)
            end = min(x.size, (last + 1) * self._vad_window + margin)
            parts.append(x[start:end])
            i = j
        if not parts and 0 < x.size < int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
            # Short-input fallback: keep the whole clip when the peak passes but no part cut.
            if self._peak_prob(probs) >= self.vad_threshold:
                parts.append(np.ascontiguousarray(x))
        return parts

    # ---------- Embeddings ----------

    def _embed(self, x, *, allow_short=False):
        """One clip -> normalized embedding; None when too short/invalid."""
        if x.size == 0:
            return None
        if not allow_short and x.size < int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
            return None
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=x)
        stream.input_finished()
        emb = np.asarray(self._extractor.compute(stream), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if emb.size == 0 or norm <= 1e-9:
            return None
        return emb / norm

    def enroll(self, wav16k) -> dict:
        """Enroll a clip of the owner's voice; returns a report dict."""
        x = _as_mono_float32(wav16k)
        if x.size < int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
            raise ValueError("注册音频太短：至少 %.1fs（收到 %.2fs）" % (MIN_SPEECH_SECONDS, x.size / SAMPLE_RATE))
        with self._lock:
            parts = self._speech_parts(x)
            if not parts:
                if self.require_speech_on_enroll:
                    raise ValueError("注册音频里没有检测到人声（VAD 概率低于 %.2f）" % self.vad_threshold)
                parts = [x]
            embs = []
            for part in parts:
                emb = self._embed(part)
                if emb is not None:
                    embs.append(emb)
            if not embs:
                raise ValueError("注册失败：音频里没有可用的语音段（太短或全被 VAD 判为静音）")
            stack = np.stack(embs)
            center = stack.mean(axis=0)
            center_norm = float(np.linalg.norm(center))
            if not math.isfinite(center_norm) or center_norm <= 1e-9:
                raise ValueError("注册失败：语音段嵌入异常，无法形成声纹中心")
            center = center / center_norm
            scores = [float(np.dot(emb, center)) for emb in embs]
            mean_sim = float(sum(scores) / len(scores))
            keep_line = max(_ENROLL_MIN_KEEP_SIM, mean_sim - _ENROLL_OUTLIER_MARGIN)
            kept_index = [i for i, s in enumerate(scores) if s >= keep_line]
            if not kept_index:  # Keep the best segment so enrollment is never empty.
                kept_index = [int(np.argmax(np.asarray(scores, dtype=np.float64)))]
            kept = np.stack([embs[i] for i in kept_index])
            block = np.sum(kept, axis=0)
            total = block if self._emb_sum is None else self._emb_sum + block
            count = self._emb_count + len(kept_index)
            mean = total / float(count)
            self._embedding = (mean / float(np.linalg.norm(mean))).astype(np.float32)
            self._emb_sum = total
            self._emb_count = count
            self._smooth_score = None
            report = {
                "segments": len(embs),
                "kept": len(kept_index),
                "dropped": len(embs) - len(kept_index),
                "scores": [round(s, 4) for s in scores],
                "kept_index": list(kept_index),
                "keep_line": round(keep_line, 4),
            }
            self.last_enroll_report = dict(report)
            self._last_enroll_sims = tuple(float(s) for s in scores)
            return report

    def similarity(self, wav16k):
        """Raw cosine similarity without VAD gating; None when unenrolled."""
        x = _as_mono_float32(wav16k)
        with self._lock:
            ref = self._embedding
            if ref is None:
                return None
            emb = self._embed(x)
        if emb is None:
            return None
        return float(np.dot(emb, ref))

    def decide(self, wav16k):
        """Return the smoothed 0-1 similarity, None when undecidable."""
        x = _as_mono_float32(wav16k)
        with self._lock:
            probs = self._vad_probs(x)
            prob = self._peak_prob(probs)
            self.last_vad_prob = prob
            if prob < self.vad_threshold:
                self.last_score = None
                return None
            ref = self._embedding
            if ref is None:
                self.last_score = None
                return None
            seg = None
            cand = None
            parts = self._parts_from_probs(x, probs)
            if parts:
                cand = np.concatenate(parts) if len(parts) > 1 else parts[0]
                if cand.size >= int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
                    seg = cand
            emb = self._embed(seg if seg is not None else x)
            if emb is None:
                # Best effort for the first second: reuse cut activity, else the whole clip.
                best = cand if cand is not None and cand.size > 0 else x
                emb = self._embed(best, allow_short=True)
            if emb is None:
                self.last_score = None
                return None
            score = float(np.dot(emb, ref))
            score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
            self.last_raw_score = score
            if self._smooth_score is None:
                smooth = score
            else:
                smooth = self.smooth_alpha * score + (1.0 - self.smooth_alpha) * self._smooth_score
            self._smooth_score = smooth
            self.last_score = smooth
            return smooth

    def verify(self, wav16k):
        """True = owner / False = not owner / None = undecidable."""
        score = self.decide(wav16k)
        if score is None:
            return None
        return score >= self.threshold

    def is_self(self, wav16k) -> bool:
        """Boolean verify; undecidable counts as blocked."""
        return self.verify(wav16k) is True

    def suggest_threshold(self, *, margin=0.25, low=0.35, high=0.65) -> float:
        """Suggest a threshold from enrollment consistency; never mutates."""
        if self._embedding is None:
            raise RuntimeError("尚未注册声纹，无法给出阈值建议")
        margin = float(margin)
        low = float(low)
        high = float(high)
        if low > high:
            raise ValueError("阈值下界不应高于上界：low=%r high=%r" % (low, high))
        sims = []
        with self._lock:
            report = self.last_enroll_report
            cached = tuple(self._last_enroll_sims)
        if report and cached:
            kept = report.get("kept_index") or list(range(len(cached)))
            sims = [cached[i] for i in kept if 0 <= i < len(cached)]
        if not sims:
            return float(DEFAULT_THRESHOLD)
        base = sum(sims) / len(sims) - margin
        return float(min(high, max(low, base)))

    # ---------- Profiles ----------

    def _profile_dict(self, embedding=True):
        payload = {
            "version": PROFILE_VERSION,
            "sample_rate": SAMPLE_RATE,
            "dim": self.embed_dim,
            "threshold": float(self.threshold),
            "num_segments": self.num_segments,
            "model": os.path.basename(self.speaker_model),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if embedding:
            payload["embedding"] = self._embedding.tolist()
        return payload

    def save_profile(self, path=None) -> str:
        """Save the voiceprint plus threshold to .npz or .json; returns the path."""
        if self._embedding is None:
            raise RuntimeError("尚未注册声纹，无法保存档案（先 enroll 或 load_profile）")
        path = os.path.abspath(path or self.profile_path or DEFAULT_PROFILE_PATH)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(self._profile_dict(), fp, ensure_ascii=False, indent=2)
        else:
            if not path.endswith(".npz"):  # np.savez appends .npz itself; pre-append so the returned path is real.
                path += ".npz"
            np.savez(
                path,
                embedding=self._embedding,
                threshold=np.float64(self.threshold),
                sample_rate=np.int64(SAMPLE_RATE),
                dim=np.int64(self.embed_dim),
                num_segments=np.int64(self.num_segments),
                model=os.path.basename(self.speaker_model),
                version=np.int64(PROFILE_VERSION),
            )
        self.profile_path = path
        return path

    def load_profile(self, path=None, use_stored_threshold=False) -> str:
        """Load a .npz/.json profile; returns the loaded path."""
        path = os.path.abspath(path or self.profile_path or DEFAULT_PROFILE_PATH)
        if not os.path.isfile(path):
            raise FileNotFoundError("声纹档案不存在：%s" % path)
        stored_threshold = None
        num_segments = 0
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                raise ValueError("声纹档案损坏（内容不是 JSON 对象）：%s" % path)
            raw = payload.get("embedding")
            if raw is None:
                raise ValueError("声纹档案缺少 embedding 字段（文件损坏或不是本模块保存的）：%s" % path)
            emb = np.asarray(raw, dtype=np.float32).reshape(-1)
            stored_threshold = payload.get("threshold")
            num_segments = int(payload.get("num_segments") or 0)
        else:
            with np.load(path) as data:
                if "embedding" not in data.files:
                    raise ValueError("声纹档案缺少 embedding 字段（文件损坏或不是本模块保存的）：%s" % path)
                emb = np.asarray(data["embedding"], dtype=np.float32).reshape(-1)
                if "threshold" in data:
                    stored_threshold = float(data["threshold"])
                if "num_segments" in data:
                    num_segments = int(data["num_segments"])
        if emb.size != self.embed_dim:
            raise ValueError("声纹维度不匹配：档案 %d 维，模型 %d 维（%s）" % (emb.size, self.embed_dim, path))
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("声纹向量非法（范数 %.3g）：%s" % (norm, path))
        with self._lock:
            self._embedding = (emb / norm).astype(np.float32)
            count = max(1, num_segments)
            self._emb_count = count
            self._emb_sum = self._embedding * float(count)
            self._smooth_score = None
            self.last_enroll_report = None
            self._last_enroll_sims = ()
        if use_stored_threshold and stored_threshold is not None:
            self.set_threshold(stored_threshold)
        self.profile_path = path
        return path
