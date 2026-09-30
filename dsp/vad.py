# -*- coding: utf-8 -*-
"""Standalone lightweight Silero VAD: per-block voice probability for AGC.

Contract: SileroVad() finds its model under models/; process(block, sr)
takes float32 mono at any rate and returns a 0-1 voice probability;
reset() clears RNN state.

Behavior: samples accumulate to 512-sample (32 ms @16 kHz) windows with
leftovers carried over — no extra latency, window-granularity output; each
block is RMS-normalized to -20 dBFS first so level cannot bias the score.
Cost: onnxruntime CPU, ~0.1 ms per window, under 0.5 ms per 60 ms block.
"""

from __future__ import annotations

import glob
import os

import numpy as np

__all__ = ["SileroVad"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Model dir, shared with dsp.voice_gate and app.gui.
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
#: Silero analysis window at 16 kHz (32 ms).
_WINDOW = 512
#: State shapes: h/c build (2, batch, 64), v5 build (2, batch, 128).
_HC_STATE = (2, 1, 64)
_V5_STATE = (2, 1, 128)
#: Non-16k input goes through polyphase resampling.
_RESAMPLE_RATIO = {16000: None}


def _find_model(model_dir):
    """Pick a silero VAD onnx under model_dir; None when absent."""
    if not model_dir or not os.path.isdir(model_dir):
        return None
    hits = []
    for path in glob.glob(os.path.join(model_dir, "*.onnx")):
        name = os.path.basename(path).lower()
        if "silero" not in name and "vad" not in name:
            continue
        if "pyannote" in name or "segmentation" in name:
            continue
        hits.append(path)
    if not hits:
        return None
    hits.sort(key=lambda p: (0 if "int8" in os.path.basename(p).lower() else 1,
                             os.path.basename(p).lower()))
    return hits[0]


class SileroVad:
    """Streaming Silero VAD with cross-block state; any input rate, per-block probability out."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, num_threads=1):
        self._prob = None
        path = _find_model(model_dir)
        if path is None:
            raise FileNotFoundError(
                "models/ 里找不到 silero VAD 模型（找关键字 silero / vad 的 .onnx）：%s"
                % (model_dir,))
        import onnxruntime as ort

        so = ort.SessionOptions()
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
        self._sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        name = os.path.basename(path).lower()
        self._layout = "v5" if "v5" in name or "state" in name else "hc"
        # Probe input names to pick the h/c or v5 layout.
        inputs = [i.name for i in self._sess.get_inputs()]
        if "state" in inputs:
            self._layout = "v5"
        elif "h" in inputs and "c" in inputs:
            self._layout = "hc"
        self.reset()
        # Warm up: two zero windows move JIT/allocation cost into construction.
        self.process(np.zeros(_WINDOW * 2, dtype=np.float32), sr=16000)
        self.reset()

    # ------------------------------------------------------------------ #

    @property
    def available(self):
        """Always True; construction raises on failure."""
        return True

    def reset(self):
        """Clear RNN state, leftover samples, and the last probability."""
        self._prob = None
        if self._layout == "hc":
            self._h = np.zeros(_HC_STATE, dtype=np.float32)
            self._c = np.zeros(_HC_STATE, dtype=np.float32)
        else:
            self._state = np.zeros(_V5_STATE, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)

    def process(self, chunk, sr=48000):
        """One float32 mono block -> voice probability (0-1); level-normalized first."""
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return self._last_prob()
        if int(sr) != 16000:
            x = self._to_16k(x, sr)
        x = self._normalize(x)
        buf = np.concatenate([self._carry, x])
        n = int(buf.size // _WINDOW)
        self._carry = buf[n * _WINDOW:].astype(np.float32, copy=True)
        if n <= 0:
            return self._last_prob()
        sr_t = np.asarray(16000, dtype=np.int64)
        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            win = buf[i * _WINDOW:(i + 1) * _WINDOW].reshape(1, -1)
            if self._layout == "hc":
                out, self._h, self._c = self._sess.run(
                    None, {"input": win, "sr": sr_t, "h": self._h, "c": self._c})
            else:
                out, self._state = self._sess.run(
                    None, {"input": win, "state": self._state, "sr": sr_t})
            probs[i] = float(np.ravel(out)[0])
        self._prob = float(probs.mean())
        return self._prob

    # ------------------------------------------------------------------ #

    def _last_prob(self):
        """Reuse the last probability before the first full window; 0.0 initially."""
        return self._prob if self._prob is not None else 0.0

    @staticmethod
    def _normalize(x):
        """Normalize block RMS to -20 dBFS; near-silence feeds zeros."""
        n = x.size
        rms = float(np.sqrt(np.dot(x, x) / n)) if n else 0.0
        if rms <= 1e-5:
            return np.zeros_like(x)
        return np.clip(x * (0.1 / rms), -1.0, 1.0).astype(np.float32)

    def _to_16k(self, x, sr):
        """Resample any rate to 16 kHz via polyphase resampling."""
        sr = int(sr)
        if sr == 16000:
            return x
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


# --------------------------------------------------------------------------- #
# Self-test: python -X utf8 dsp/vad.py (needs a silero model in models/)
# --------------------------------------------------------------------------- #
def _selftest():
    import glob
    import time

    sr = 48000
    block = 2880  # 60 ms, matching the pipeline
    # Prefer real speech; Silero is insensitive to synthetic sines.
    speech = None
    for cand in glob.glob(os.path.join(PROJECT_ROOT, "out_dev", "voicegate_test", "*.wav")):
        speech = cand
        break
    if speech is not None:
        import wave

        with wave.open(speech, "rb") as w:
            ch, width = w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 1:  # 8-bit unsigned
            sp = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            sp = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:  # 32-bit int
            sp = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            sp = None
        if sp is not None:
            if ch > 1:
                sp = sp.reshape(-1, ch)[:, 0]
            sp = sp[: 16000 * 6]
            feed_sr = 16000
    if speech is None or sp is None or sp.size < 16000:
        t = np.arange(sr * 3) / sr
        env = 0.55 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2
        w = (np.sin(2 * np.pi * 130 * t) + 0.6 * np.sin(2 * np.pi * 260 * t)
             + 0.4 * np.sin(2 * np.pi * 520 * t) + 0.3 * np.sin(2 * np.pi * 900 * t))
        sp = (0.3 * w * env).astype(np.float32)
        feed_sr = sr
    rng = np.random.default_rng(0)
    noise = (0.02 * rng.standard_normal(sp.size)).astype(np.float32)
    blk = feed_sr * 60 // 1000  # 60 ms blocks at the clip rate

    v_n = SileroVad()
    v_s = SileroVad()
    n_probs, s_probs, times = [], [], []
    for i in range(0, sp.size - blk, blk):  # Two independent single-stream instances.
        t0 = time.perf_counter()
        n_probs.append(v_n.process(noise[i:i + blk], sr=feed_sr))
        s_probs.append(v_s.process(sp[i:i + blk], sr=feed_sr))
        times.append(time.perf_counter() - t0)
    v, n = float(np.mean(s_probs[2:])), float(np.mean(n_probs[2:]))
    ms = np.asarray(times[2:]) * 1000
    print("语音块概率均值 %.3f / 噪声块概率均值 %.3f（素材: %s）"
          % (v, n, os.path.basename(speech) if speech else "合成"))
    print("单块耗时 mean %.3f / p95 %.3f / max %.3f ms"
          % (ms.mean(), np.percentile(ms, 95), ms.max()))
    thr_v = 0.5 if speech else 0.15  # Synthetic tones only need to beat noise.
    ok = v > thr_v and n < 0.2
    print("自测: %s" % ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
