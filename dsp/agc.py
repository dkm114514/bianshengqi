# -*- coding: utf-8 -*-
"""Output-side automatic gain (AGC): runs after SOLA, before the output buffer.

Contract: Agc(sr, target_dbfs, rate_up_dbps, rate_down_dbps, knee_db);
process(block) takes float32 mono and returns equal length with zero
algorithmic latency; set_target_dbfs() updates the target live; reset()
clears cross-block state.

Behavior: speech-weighted level estimate (external voice probability, else
block RMS 6 dB above the adaptive noise floor), slew-limited gain (slow up,
fast down, capped for weak signals), then tanh soft limiting to -0.1 dBFS.
gain_db / speech_db / noise_db expose live state for the GUI.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Agc"]

#: Voice decision margin (dB): a block counts as speech this far above the noise floor.
SPEECH_MARGIN_DB = 6.0
#: Absolute floor (dBFS) for the VAD path; blocks only digital residue. A
#: relative-to-floor bound would lock out speech once the floor is lifted.
_SPEECH_MIN_DB = -70.0
#: Cold-start margin (dB): the noise floor starts this far below the first
#: block so the heuristic path still confirms speech on steady input.
_COLD_MARGIN_DB = 12.0
#: Weak-speech level (dBFS): below it the target gain is capped instead of
#: compensating fully to target_dbfs.
_LOW_SPEECH_DB = -45.0
#: Target-gain cap (dB) applied while speech is weak.
_LOW_SPEECH_MAX_GAIN_DB = 15.0
#: Level smoothing: fast attack, slow release.
_LEVEL_ATTACK = 0.5
_LEVEL_RELEASE = 0.05
#: Noise-floor tracking: instant follow down, slow leak up per block.
_NOISE_LEAK_DB_PER_BLOCK = 0.02
#: Noise-floor fall rate (dB/block): a single muted block cannot drag it down.
_NOISE_FALL_DB_PER_BLOCK = 1.0
#: Noise-floor lower bound (dBFS).
_NOISE_MIN_DB = -80.0
#: Soft-limiter ceiling and silence floor (dBFS).
_CEILING_DB = -0.1
_FLOOR_DB = -100.0


def _rms_lin(x):
    """Linear RMS of a block."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    return math.sqrt(max(float(np.dot(x, x)) / n, 0.0))


def _rms_db(x):
    """Block RMS in dBFS; near-zero clamps to -100."""
    rms = _rms_lin(x)
    if rms <= 1e-7:
        return _FLOOR_DB
    return 20.0 * math.log10(rms)


class Agc:
    """Output-side AGC: speech-weighted estimate, slew-limited gain, soft limit.

    sr: sample rate (block length derives dt). target_dbfs: target loudness.
    rate_up_dbps / rate_down_dbps: gain slew rates (slow up, fast down).
    knee_db: soft-limiter knee. max_gain_db / min_gain_db: gain clamps.
    """

    def __init__(
        self,
        sr,
        target_dbfs=-20.0,
        rate_up_dbps=5.0,
        rate_down_dbps=10.0,
        knee_db=-3.0,
        max_gain_db=30.0,
        min_gain_db=-30.0,
    ):
        sr = float(sr)
        if not math.isfinite(sr) or sr <= 0:
            raise ValueError("sr 必须是有限正数: %r" % (sr,))
        target_dbfs = float(target_dbfs)
        if not math.isfinite(target_dbfs):
            raise ValueError("target_dbfs 必须是有限数: %r" % (target_dbfs,))
        rate_up_dbps = float(rate_up_dbps)
        if not math.isfinite(rate_up_dbps) or rate_up_dbps < 0:
            raise ValueError("rate_up_dbps 必须是非负有限数: %r" % (rate_up_dbps,))
        rate_down_dbps = float(rate_down_dbps)
        if not math.isfinite(rate_down_dbps) or rate_down_dbps < 0:
            raise ValueError("rate_down_dbps 必须是非负有限数: %r" % (rate_down_dbps,))
        knee_db = float(knee_db)
        if not math.isfinite(knee_db):
            raise ValueError("knee_db 必须是有限数: %r" % (knee_db,))
        max_gain_db = float(max_gain_db)
        min_gain_db = float(min_gain_db)
        if not math.isfinite(max_gain_db) or not math.isfinite(min_gain_db):
            raise ValueError("max_gain_db/min_gain_db 必须是有限数: %r/%r"
                             % (max_gain_db, min_gain_db))
        if max_gain_db < min_gain_db:
            raise ValueError("max_gain_db 不得小于 min_gain_db: %r/%r"
                             % (max_gain_db, min_gain_db))
        self.sr = sr
        self.target_dbfs = target_dbfs
        self.rate_up_dbps = rate_up_dbps
        self.rate_down_dbps = rate_down_dbps
        self.knee_db = knee_db
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.knee = 10.0 ** (self.knee_db / 20.0)
        self.ceiling = 10.0 ** (_CEILING_DB / 20.0)
        self._span = max(self.ceiling - self.knee, 1e-6)
        self.reset()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def gain(self):
        """Current gain (linear factor)."""
        return 10.0 ** (self.gain_db / 20.0)

    def set_target_dbfs(self, value):
        """Update the target loudness live; non-finite values raise ValueError."""
        target = float(value)
        if not math.isfinite(target):
            raise ValueError("target_dbfs 必须是有限数: %r" % (value,))
        self.target_dbfs = target

    def reset(self):
        """Clear cross-block state: gain to 0 dB, estimates restart."""
        self.gain_db = 0.0
        self.speech_db = None
        self.noise_db = None
        self.blocks = 0
        self.speech_blocks = 0

    def process(self, chunk, speech_prob=None):
        """Process one float32 mono block; speech_prob selects VAD voting, None uses the energy heuristic."""
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if x.shape[0] == 0:
            return x
        self.blocks += 1
        self._track_level(_rms_db(x), speech_prob)
        self._slew_gain(x.shape[0] / self.sr)
        return self._limit(x * np.float32(self.gain))

    # ------------------------------------------------------------------ #
    # Per-block state
    # ------------------------------------------------------------------ #

    def _track_level(self, rms_db, speech_prob=None):
        """Update noise-floor and speech-level estimates (VAD vote, else energy heuristic)."""
        if self.noise_db is None:
            # Cold start: leave margin so steady input still confirms speech.
            self.noise_db = max(rms_db - _COLD_MARGIN_DB, _NOISE_MIN_DB)
        if speech_prob is not None:
            # Absolute floor blocks only digital residue; relative-to-floor would lock out speech.
            is_speech = speech_prob >= 0.5 and rms_db > _SPEECH_MIN_DB
        else:
            is_speech = rms_db > self.noise_db + SPEECH_MARGIN_DB
        if is_speech:
            self.speech_blocks += 1
            if self.speech_db is None:
                self.speech_db = rms_db  # First speech: take the value directly.
            else:
                coef = _LEVEL_ATTACK if rms_db > self.speech_db else _LEVEL_RELEASE
                self.speech_db += (rms_db - self.speech_db) * coef
            # Recovery: confirmed speech pulls an over-high floor back toward rms-6dB.
            cap = rms_db - SPEECH_MARGIN_DB
            if self.noise_db > cap:
                self.noise_db = max(cap, self.noise_db - _NOISE_FALL_DB_PER_BLOCK)
            return
        if rms_db < self.noise_db:
            self.noise_db = max(self.noise_db - _NOISE_FALL_DB_PER_BLOCK, rms_db)
        else:
            self.noise_db = min(self.noise_db + _NOISE_LEAK_DB_PER_BLOCK, rms_db)
        if self.noise_db < _NOISE_MIN_DB:
            self.noise_db = _NOISE_MIN_DB

    def _slew_gain(self, dt):
        """Slew the gain toward its target; hold still until speech is confirmed."""
        if self.speech_db is None:
            return
        target = self.target_dbfs - self.speech_db
        eff_max = self.max_gain_db
        if self.speech_db < _LOW_SPEECH_DB:
            eff_max = min(eff_max, _LOW_SPEECH_MAX_GAIN_DB)
        target = min(max(target, self.min_gain_db), eff_max)
        delta = target - self.gain_db
        step = (self.rate_up_dbps if delta > 0.0 else self.rate_down_dbps) * dt
        self.gain_db += max(-step, min(step, delta))

    def _limit(self, x):
        """Soft-limit above the knee; peaks stay below the ceiling."""
        mag = np.abs(x)
        idx = mag > self.knee
        if idx.any():
            over = mag[idx]
            x[idx] = np.sign(x[idx]) * (
                self.knee + self._span * np.tanh((over - self.knee) / self._span)
            )
        return x


# --------------------------------------------------------------------------- #
# Self-test: python -X utf8 dsp/agc.py (run at the project root)
# --------------------------------------------------------------------------- #
def _selftest():
    import time

    sr, block = 48000, 2880  # 60 ms blocks, matching the pipeline
    rng = np.random.default_rng(0)
    scene = (("大声", -12.0, 5.0), ("安静", -42.0, 10.0), ("小声", -27.0, 5.0))

    def segment(level_db, seconds):
        """Speech-like segment: harmonics with syllable envelope and word pauses."""
        n = int(seconds * sr)
        t = np.arange(n) / sr
        env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.5 * t) ** 2
        for start in range(int(0.9 * sr), n, int(0.96 * sr)):
            env[start : start + int(0.06 * sr)] = 0.0
        wave = np.sin(2.0 * np.pi * 180.0 * t) + 0.5 * np.sin(2.0 * np.pi * 360.0 * t)
        x = (wave * env).astype(np.float32)
        x *= 10.0 ** (level_db / 20.0) / max(_rms_lin(x), 1e-9)
        return x + (1e-4 * rng.standard_normal(n)).astype(np.float32)

    agc = Agc(sr=sr)
    print("%-4s %10s %10s %10s %10s" % ("段", "输入dBFS", "输出dBFS", "段末增益", "块间跳变"))
    peak, prev_gain, timings = 0.0, None, []
    for label, level_db, seconds in scene:
        x = segment(level_db, seconds)
        y = np.empty_like(x)
        seg_step = 0.0
        for start in range(0, x.shape[0], block):
            chunk = x[start : start + block]
            t0 = time.perf_counter()
            out = agc.process(chunk)
            timings.append((time.perf_counter() - t0) * 1000.0)
            if prev_gain is not None:
                seg_step = max(seg_step, abs(agc.gain_db - prev_gain))
            prev_gain = agc.gain_db
            y[start : start + out.shape[0]] = out
        peak = max(peak, float(np.max(np.abs(y))))
        print("%-4s %10.1f %10.1f %10.1f %10.2f"
              % (label, _rms_db(x), _rms_db(y), agc.gain_db, seg_step))
    ms = np.asarray(timings)
    print("峰值 %.4f（软限幅上限 %.4f）；噪声底 %.1f dBFS，语音电平 %.1f dBFS"
          % (peak, agc.ceiling, agc.noise_db, agc.speech_db))
    print("单块耗时 mean %.3f / p95 %.3f / max %.3f ms（%d 块 @ %d 采样）"
          % (ms.mean(), np.percentile(ms, 95), ms.max(), ms.size, block))


if __name__ == "__main__":
    _selftest()
