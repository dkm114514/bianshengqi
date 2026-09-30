# -*- coding: utf-8 -*-
"""VoicePipeline: realtime voice chain (capture -> RVC -> virtual mic/monitor).

Contract: VoicePipeline(profile, input/cable/monitor devices, ...);
.start()/.stop(); .set_param(key, value); .set_devices(...);
.load_model(pth, index); .process_offline(waveform); .reset_voice_gate().
Chain runs at 48000 Hz, mirrors gui_v1.py start_vc/audio_callback; model
I/O goes through engine.rvc_engine.RvcEngine. Optional stages: denoise_mode
off/torchgate (default)/dfn3; VoiceGate voice gate (1s-history decisions,
10ms/300ms gain smoothing, bounded onset protection after short pauses);
streaming input VAD, no extra audio buffering; output AGC after
RVC+SOLA. Three streams: mic input -> master ring -> virtual mic (always on)
+ monitor (switchable, headphones required). Devices are never hardcoded;
WASAPI preferred.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import os
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

from .models import rvc_root
from .rvc_engine import RvcEngine
from .defaults import DEFAULT_MODEL_PROFILE

#: Chain-wide sample rate (USB mics in WASAPI shared mode only support 48k)
SAMPLERATE = 48000

_F0_METHODS = ("pm", "harvest", "crepe", "rmvpe", "fcpe")

#: Params hot-updated through engine.rvc_engine.RvcEngine
_ENGINE_PARAMS = ("pitch", "formant", "index_rate", "f0method")
#: Hot-update params consumed by pipeline itself (per-block DSP switches / values)
_LOCAL_PARAMS = ("I_noise_reduce", "O_noise_reduce", "rms_mix_rate", "threhold")
#: Acoustic-protection hot-update keys (held by the pipeline, not via the profile dict)
_ACOUSTIC_PARAMS = ("denoise_mode", "voice_gate_threshold", "voice_gate")
#: Output-side auto gain (AGC) hot-update keys: toggle + target loudness (dBFS)
_AGC_PARAMS = ("agc", "agc_target_dbfs")
#: AGC target loudness default (dBFS)
_AGC_TARGET_DBFS = -20.0

#: Output anti-blowup: clamps on the loudness-envelope mix ratio, max +12dB lift per block
_RMS_MIX_MAX_DB = 12.0
_RMS_MIX_MAX_RATIO = 10.0 ** (_RMS_MIX_MAX_DB / 20.0)
_RMS_MIX_MIN_RATIO = 1.0 / _RMS_MIX_MAX_RATIO
#: A whole RVC output block peaking below this counts as near-silence: emit silence directly, ≈ -80dBFS
_RVC_SILENCE_PEAK = 1e-4
#: Main-output hard ceiling, peak always below 0dBFS
_OUTPUT_CEILING = 0.99

#: Input denoise modes; dfn3 = dsp.dfn.DfnDenoiser (module G2), torchgate = RVC built-in (original behavior)
_DENOISE_MODES = ("off", "torchgate", "dfn3")
_DENOISE_ALIASES = {
    "none": "off",
    "disable": "off",
    "false": "off",
    "off": "off",
    "torch_gate": "torchgate",
    "tg": "torchgate",
    "rvc": "torchgate",
    "dfn": "dfn3",
    "deepfilter": "dfn3",
    "deepfilter3": "dfn3",
}

#: Voice gate: decision interval (s) + gain smoothing time constants (s, attack opens / release closes)
_VOICE_GATE_INTERVAL = 0.25
_VOICE_GATE_ATTACK = 0.010
_VOICE_GATE_RELEASE = 0.300
#: Voice-print decision window = latest 1s of raw input (dedicated history buffer, decisions read history only)
_VOICE_GATE_WINDOW = 1.0
#: How many consecutive decision failures disable the voice gate (back to passthrough, so a bad model never mutes the mic)
_VOICE_GATE_MAX_ERRORS = 3

#: Heat-overload guard: when whole-block backlog piled up in the input callback exceeds this many blocks, drop old blocks, process only the newest
_OVERLOAD_BACKLOG_BLOCKS = 2
#: How many consecutive blocks infer_ms_avg must exceed the block budget before f0method auto-drops from rmvpe to pm
_OVERLOAD_SUSTAIN_BLOCKS = 30
#: Rearm onset protection after a normal conversational pause, including room noise.
_INPUT_PAUSE_SECONDS = 0.5
_INPUT_VAD_THRESHOLD = 0.5
#: A partial first syllable is not enough evidence to reject a new utterance.
_VOICE_GATE_MIN_ONSET_SECONDS = 0.4
#: Unknown identity only gets a bounded grace period; then restore the prior decision.
_VOICE_GATE_ONSET_GRACE_SECONDS = 0.8
#: RMS gate and both TorchGate stages skip the first two onset blocks.
_ONSET_BYPASS_BLOCKS = 2

#: Project root (parent of the engine package), the dsp package loads from here
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_DIR = os.path.join(PROJECT_ROOT, "dsp")

#: profile defaults (project-wide f0method=rmvpe; missing keys fall back here)
_PROFILE_DEFAULTS = dict(DEFAULT_MODEL_PROFILE)

_NUMERIC_PROFILE_KEYS = (
    "pitch",
    "formant",
    "index_rate",
    "block_time",
    "crossfade_time",
    "extra_time",
    "threhold",
    "rms_mix_rate",
)

#: External key aliases / case folding (lowercased for lookup)
_KEY_ALIASES = {
    "key": "pitch",
    "f0_up_key": "pitch",
    "formant_shift": "formant",
    "f0": "f0method",
    "f0_method": "f0method",
    "threshold": "threhold",
    "i_noise_reduce": "I_noise_reduce",
    "o_noise_reduce": "O_noise_reduce",
}
_KEY_LOOKUP = {key.lower(): key for key in _PROFILE_DEFAULTS}
_KEY_LOOKUP.update({key.lower(): key for key in _ENGINE_PARAMS + _LOCAL_PARAMS})
_KEY_LOOKUP["monitor"] = "monitor"
#: Restart-only keys (set_param only records): n_cpu is not in the profile defaults table,
#: registered separately for case-insensitive lookup
_KEY_LOOKUP["n_cpu"] = "n_cpu"
_KEY_LOOKUP.update({key.lower(): key for key in _ACOUSTIC_PARAMS})
_KEY_LOOKUP.update(
    {
        "noise_mode": "denoise_mode",
        "denoise": "denoise_mode",
        "gate_threshold": "voice_gate_threshold",
        "vg_threshold": "voice_gate_threshold",
        "speaker_gate": "voice_gate",
        "auto_gain": "agc",
        "agc_target": "agc_target_dbfs",
        "target_dbfs": "agc_target_dbfs",
    }
)
_KEY_LOOKUP.update({key.lower(): key for key in _AGC_PARAMS})
_KEY_LOOKUP.update(_KEY_ALIASES)

_SD = None
_TORCH = None
_RESAMPLE = None
_TORCHGATE = None


def _log(message):
    print("[pipeline] %s" % message, flush=True)


def _sd():
    global _SD
    if _SD is None:
        import sounddevice as sd

        _SD = sd
    return _SD


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _resample_cls():
    global _RESAMPLE
    if _RESAMPLE is None:
        import torchaudio.transforms as tat

        _RESAMPLE = tat.Resample
    return _RESAMPLE


def _torchgate_cls():
    """Load RVC tools/torchgate by path; the project tools/ package shadows it."""
    global _TORCHGATE
    if _TORCHGATE is not None:
        return _TORCHGATE
    root = str(rvc_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    pkg_dir = os.path.join(root, "tools", "torchgate")
    init_py = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError("找不到 RVC 的 tools/torchgate: %s" % pkg_dir)
    alias = "rvc_tools_torchgate"
    module = sys.modules.get(alias)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            alias, init_py, submodule_search_locations=[pkg_dir]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    _TORCHGATE = module.TorchGate
    return _TORCHGATE


def _dsp_attr(module_name, attr_name):
    """Fetch a dsp class by package, else by file path; None means degrade, never raise."""
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)  # append last so RVC packages such as tools keep priority
    module = None
    try:
        module = importlib.import_module("dsp.%s" % module_name)
    except Exception:
        path = os.path.join(_DSP_DIR, "%s.py" % module_name)
        if os.path.isfile(path):
            try:
                module = _load_module_by_path("bsq_dsp_%s" % module_name, path)
            except Exception:
                module = None
    if module is None:
        return None
    return getattr(module, attr_name, None)


def _load_module_by_path(alias, path):
    """Load a .py by absolute path (fallback when the dsp package cannot be imported as a whole)."""
    module = sys.modules.get(alias)
    if module is None:
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return module


def normalize_denoise_mode(value):
    """Normalize denoise_mode; return None when off the value table (caller decides degrade vs reject)."""
    mode = str(value if value is not None else "").strip().lower()
    if mode in _DENOISE_MODES:
        return mode
    return _DENOISE_ALIASES.get(mode)


def level_db(x):
    """dBFS level of a float32 waveform (-100 floor), for the on_level meter."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 1e-5:
        return -100.0
    return float(20.0 * np.log10(rms))


def resolve_device(device, kind):
    """Resolve device (index/exact name/substring) to an index; WASAPI preferred."""
    sd = _sd()
    if isinstance(device, (int, np.integer)):
        return int(device)
    name = str(device or "").strip()
    if not name:
        raise ValueError("设备名不能为空")
    exact, partial = [], []
    for index, info in enumerate(sd.query_devices()):
        if kind == "input" and info["max_input_channels"] <= 0:
            continue
        if kind == "output" and info["max_output_channels"] <= 0:
            continue
        host = sd.query_hostapis(info["hostapi"])["name"]
        rank = 0 if "WASAPI" in host else 1
        text = str(info["name"])
        if text == name:
            exact.append((rank, index))
        elif name.lower() in text.lower():
            partial.append((rank, index))
    for bucket in (exact, partial):
        if bucket:
            bucket.sort()
            return bucket[0][1]
    raise ValueError("找不到 %s 设备: %r（可传索引或名字子串）" % (kind, device))


class _MasterBuffer:
    """Single-writer ring of whole-block arrays; one _Reader cursor per output stream."""

    def __init__(self, capacity_frames, dtype=np.float32):
        self._cap = max(1, int(capacity_frames))
        self._dtype = dtype
        self._blocks = deque()  # [(absolute start, np.ndarray), ...]
        self._start = 0  # absolute index of the earliest retained sample
        self._end = 0  # total samples written
        self._lock = threading.Lock()
        #: Old blocks dropped by window rollover (only counts when dropped unconsumed, see _Reader.skips)
        self.trimmed = 0

    @property
    def capacity(self):
        return self._cap

    def clear(self):
        with self._lock:
            self._blocks.clear()
            self._start = self._end

    def write(self, data):
        block = np.asarray(data, dtype=self._dtype).reshape(-1).copy()
        if block.size == 0:
            return
        with self._lock:
            self._blocks.append((self._end, block))
            self._end += block.size
            while (
                self._blocks
                and self._end - self._blocks[0][0] - self._blocks[0][1].size
                >= self._cap
            ):
                self._blocks.popleft()
                self.trimmed += 1
            self._start = self._blocks[0][0] if self._blocks else self._end

    def reader(self):
        return _Reader(self)


class _Reader:
    """One read cursor on _MasterBuffer; first read starts at newest sample, zeros fill gaps."""

    def __init__(self, master):
        self._master = master
        self._pos = None
        self.underruns = 0
        self.skips = 0

    @property
    def position(self):
        return self._pos

    def reset(self):
        self._pos = None

    def read(self, frames):
        master = self._master
        out = np.zeros(int(frames), dtype=master._dtype)
        copied = 0
        with master._lock:
            end = master._end
            if self._pos is None:
                self._pos = end
            if self._pos < master._start:
                self._pos = master._start
                self.skips += 1
            if self._pos < end:
                need = min(out.shape[0], end - self._pos)
                for start, block in master._blocks:
                    stop = start + block.size
                    if stop <= self._pos:
                        continue
                    offset = max(self._pos - start, 0)
                    take = min(block.size - offset, need - copied)
                    if take <= 0:
                        continue
                    out[copied : copied + take] = block[offset : offset + take]
                    copied += take
                    if copied >= need:
                        break
                self._pos += copied
        if copied < out.shape[0]:
            self.underruns += 1
        return out


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


class VoicePipeline:
    """Realtime voice pipeline: RvcEngine + DSP buffers + three sounddevice streams.

    denoise_mode 'off'/'torchgate' (default)/'dfn3'; voice_gate VoiceGate or
    model dir path, None = off; agc Agc instance (passing one enables it),
    agc_enabled builds one, agc_target_dbfs sets target loudness.
    """

    def __init__(
        self,
        profile,
        input_device,
        cable_device,
        monitor_device=None,
        monitor_on=False,
        on_level=None,
        denoise_mode="torchgate",
        voice_gate=None,
        voice_gate_threshold=0.5,
        agc=None,
        agc_enabled=False,
        agc_target_dbfs=_AGC_TARGET_DBFS,
    ):
        self.profile = dict(profile or {})
        self.input_device = input_device
        self.cable_device = cable_device
        self.monitor_device = monitor_device
        self.monitor_on = bool(monitor_on)
        self.on_level = on_level

        # Input denoise mode + voice gate (decisions read history only).
        mode = normalize_denoise_mode(denoise_mode)
        if mode is None:
            _log("警告: 未知 denoise_mode=%r，回退 'torchgate'（可选 %s）"
                 % (denoise_mode, "/".join(_DENOISE_MODES)))
            mode = "torchgate"
        self.denoise_mode = mode
        self.voice_gate_threshold = float(voice_gate_threshold)
        self._voice_gate_spec = voice_gate
        self.voice_gate = voice_gate if callable(getattr(voice_gate, "decide", None)) else None
        self._dfn = None
        self._dfn_latency_ms = None
        self._gate_resampler = None
        self._input_vad = None
        self._input_vad_warned = False
        self._gate_hist = None
        self._gate_every = 4
        #: Voice-gate runtime stats for the GUI meter / self-check.
        self.voice_stats = {
            "decisions": 0,
            "blocked_blocks": 0,
            "errors": 0,
            "score": None,
            "open": False,
            "gain": 0.0,
        }
        # Decision thread: a real decide is a 20-40ms forward pass, so the
        # realtime path mails history to a background thread; the audio thread
        # only reads the cached result and never blocks.
        self._gate_thread = None
        self._gate_wake = threading.Event()
        self._gate_stop = threading.Event()
        self._gate_mailbox = None
        self._gate_mailbox_lock = threading.Lock()
        self._gate_errors = 0
        self._reset_voice_gate_state()

        # Output auto gain (AGC): runs after SOLA, before the master buffer write.
        self.agc = agc if callable(getattr(agc, "process", None)) else None
        #: Whether agc.process takes speech_prob; None = probe at call time.
        self._agc_prob = self._agc_takes_prob(self.agc)
        try:
            self.agc_target_dbfs = self._check_target_dbfs(
                self.profile.get("agc_target_dbfs", agc_target_dbfs)
            )
        except (TypeError, ValueError):
            _log("警告: agc_target_dbfs=%r 非法，回退缺省 %.1f dBFS"
                 % (self.profile.get("agc_target_dbfs", agc_target_dbfs), _AGC_TARGET_DBFS))
            self.agc_target_dbfs = float(_AGC_TARGET_DBFS)
        if self.profile.get("agc") is not None:  # explicit profile switch wins
            agc_enabled = _as_bool(self.profile["agc"])
        self.agc_enabled = bool(agc_enabled) or self.agc is not None
        if self.agc is None and self.agc_enabled:
            self.agc = self._build_agc()  # degrades to off when dsp.agc is unavailable
            if self.agc is None:
                self.agc_enabled = False
        # AGC voice-probability source (Silero VAD, cross-block state); on failure
        # AGC falls back to the energy heuristic, chain uninterrupted.
        self._agc_vad = None
        self._agc_vad_warned = False
        if self.agc is not None:
            self._apply_agc_target()
            self._build_agc_vad()
        #: AGC runtime stats for the GUI gain readout / self-check.
        self.agc_stats = {"blocks": 0, "gain_db": 0.0, "peak": 0.0}

        self.engine = RvcEngine()
        self.tgt_sr = SAMPLERATE

        self._lock = threading.RLock()
        #: Inference mutex: held by load_model/_swap_input; the callback drops blocks instead of waiting.
        self._infer_busy = threading.Lock()
        self._running = False
        self._prepared = False
        self._pending = np.zeros(0, dtype=np.float32)

        # Streams and resolved device indexes.
        self._in_stream = None
        self._cable_stream = None
        self._monitor_stream = None
        # Input stream generation: abort() never waits for callbacks, so closing /
        # swapping bumps this and stale callbacks from the old stream return early.
        self._in_gen = 0
        self._in_index = None
        self._cable_index = None
        self._monitor_index = None

        # Master output buffer + one read cursor per output stream.
        self.master = None
        self.cable_reader = None
        self.monitor_reader = None

        self.stats = {
            "blocks": 0,
            "errors": 0,
            "in_status": 0,
            "cable_blocks": 0,
            "monitor_blocks": 0,
            #: Input blocks dropped by the overload guard.
            "dropped": 0,
            "infer_ms_avg": 0.0,
            "infer_ms_max": 0.0,
        }
        #: Blocks with infer_ms_avg over budget in a row; reset rule in _check_overload.
        self._overload_hot = 0

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def is_running(self):
        return self._running

    @property
    def monitor_active(self):
        """Whether the monitor stream is actually running."""
        return self._monitor_stream is not None

    @property
    def denoise_latency_ms(self):
        """dfn3 reported algorithmic latency in ms; None when unused/unavailable."""
        if self._dfn is None:
            return self._dfn_latency_ms
        ms = self._dfn_latency_for(self._denoise_block_frames())
        if ms is None:
            return self._dfn_latency_ms
        self._dfn_latency_ms = ms
        return ms

    def reset_voice_gate(self):
        """Reset voice-gate runtime state (history/gain/cache/stats); RVC buffers untouched."""
        self._reset_voice_gate_state()
        if self._dfn is not None:
            try:
                self._dfn.reset()
            except Exception:
                pass

    @property
    def agc_gain_db(self):
        """Current AGC gain in dB; 0.0 when AGC is off."""
        return float(self.agc_stats["gain_db"])

    def reset_agc(self):
        """Reset cross-block AGC/VAD state and stats."""
        self.agc_stats = {"blocks": 0, "gain_db": 0.0, "peak": 0.0}
        reset = getattr(self.agc, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception as exc:
                _log("警告: AGC reset 失败: %r" % (exc,))
        vad_reset = getattr(self._agc_vad, "reset", None)
        if callable(vad_reset):
            try:
                vad_reset()
            except Exception as exc:
                _log("警告: VAD reset 失败: %r" % (exc,))
        self._apply_agc_target()

    def _reset_stats(self):
        """Zero run stats; every start begins a fresh count."""
        self.stats = {
            "blocks": 0,
            "errors": 0,
            "in_status": 0,
            "cable_blocks": 0,
            "monitor_blocks": 0,
            "dropped": 0,
            "infer_ms_avg": 0.0,
            "infer_ms_max": 0.0,
        }
        self._overload_hot = 0

    def _clear_buffers(self):
        """Zero resident DSP buffers so a restart starts clean."""
        for name in ("input_wav", "input_wav_denoise", "input_wav_res",
                     "sola_buffer", "nr_buffer", "output_buffer"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.zero_()
                except Exception:
                    pass
        rms = getattr(self, "rms_buffer", None)
        if rms is not None:
            rms[:] = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        # RVC retains rolling pitch history independently of the audio buffers.
        rvc = getattr(getattr(self, "engine", None), "rvc", None)
        for name in ("cache_pitch", "cache_pitchf"):
            buf = getattr(rvc, name, None)
            if buf is not None:
                buf.zero_()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Load the model (idempotent) and start streams: virtual mic, input, then monitor."""
        with self._lock:
            if self._running:
                return
            self._prepare()
            self.reset_voice_gate()
            self.reset_agc()
            self._reset_stats()
            self._clear_buffers()
            self.master = _MasterBuffer(max(8 * self.block_frame, SAMPLERATE))
            self.cable_reader = self.master.reader()
            self._open_cable()
            try:
                self._open_input()
            except Exception:
                self._close_stream("cable")
                raise
            if self.monitor_on:
                try:
                    self._open_monitor()
                except Exception as exc:  # monitor failure never affects the main output
                    _log("警告: 监听打不开，本次不启用监听: %r" % (exc,))
                    self.monitor_on = False
            self._running = True
            _log(
                "已启动: 输入=%r → index=%s, 虚拟麦=%r → index=%s, 监听=%r(%s), "
                "block=%d 帧 (%.0f ms) @ %d Hz, f0method=%s"
                % (
                    self.input_device,
                    self._in_index,
                    self.cable_device,
                    self._cable_index,
                    self.monitor_device,
                    "开" if self.monitor_active else "关",
                    self.block_frame,
                    self.block_frame * 1000.0 / SAMPLERATE,
                    SAMPLERATE,
                    self.profile.get("f0method"),
                )
            )
            _log("声学防护: denoise_mode=%s(%s), 声纹门控=%s"
                 % (
                     self.denoise_mode,
                     "有效" if self._dfn is not None else
                     ("未启用" if self.denoise_mode != "dfn3" else "降级"),
                     ("开, 阈值 %.2f, 每 %d 块判决一次（窗口 %.1fs）"
                      % (self.voice_gate_threshold, self._gate_every, _VOICE_GATE_WINDOW))
                     if self.voice_gate is not None else "关",
                 ))
            _log("输出自动增益: %s"
                 % ("开, 目标 %.1f dBFS" % self.agc_target_dbfs
                    if self.agc_enabled and self.agc is not None else "关"))

    def stop(self):
        """Abort and close all streams; the model stays loaded for the next start()."""
        with self._lock:
            was_running = self._running
            self._close_stream("monitor")
            self._close_stream("input")
            self._close_stream("cable")
            self.monitor_reader = None
            self._running = False
            self._pending = np.zeros(0, dtype=np.float32)
            self._stop_gate_worker()
            self.reset_voice_gate()
            if self.master is not None:
                self.master.clear()
            if was_running:
                _log("已停止")

    # ------------------------------------------------------------------ #
    # Hot-update / device swap / model swap
    # ------------------------------------------------------------------ #

    def set_param(self, key, value):
        """Hot-update one param; restart-only keys are recorded and take effect on rebuild."""
        name = _KEY_LOOKUP.get(str(key or "").strip().lower(), str(key or "").strip())
        if name == "monitor":
            self._set_monitor(_as_bool(value))
            return
        if name == "denoise_mode":
            self._set_denoise_mode(value)
            return
        if name == "voice_gate_threshold":
            self._set_voice_gate_threshold(value)
            return
        if name == "voice_gate":
            self._set_voice_gate(value)
            return
        if name == "agc":
            self._set_agc(value)
            return
        if name == "agc_target_dbfs":
            self._set_agc_target(value)
            return
        with self._lock:
            if name in _ENGINE_PARAMS:
                if name == "f0method":
                    value = self._check_f0method(value)
                self.engine.set_param(name, value)
                self.profile[name] = self.engine.get_param(name, value)
                return
            if name in _LOCAL_PARAMS:
                if name in ("I_noise_reduce", "O_noise_reduce"):
                    self.profile[name] = _as_bool(value)
                else:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        raise ValueError("%s 必须是数字: %r" % (name, value))
                    if not math.isfinite(number):
                        raise ValueError("%s 必须是有限数: %r" % (name, value))
                    self.profile[name] = number
                return
            if name == "n_cpu":
                try:
                    self.profile[name] = int(value)
                except (TypeError, ValueError):
                    raise ValueError("n_cpu 必须是整数: %r" % (value,))
                _log("参数 %s 不支持运行中热更（已记录，重启管线后生效）" % name)
                return
            if name in _PROFILE_DEFAULTS:
                self.profile[name] = value
                _log("参数 %s 不支持运行中热更（已记录，重启管线后生效）" % name)
                return
            raise KeyError(
                "不支持的参数 %r（可热更: %s, %s, %s, monitor）"
                % (key, ", ".join(_ENGINE_PARAMS + _LOCAL_PARAMS),
                   ", ".join(_ACOUSTIC_PARAMS), ", ".join(_AGC_PARAMS))
            )

    def get_param(self, key, default=None):
        name = _KEY_LOOKUP.get(str(key or "").strip().lower(), str(key or "").strip())
        if name == "monitor":
            return self.monitor_active
        if name == "denoise_mode":
            return self.denoise_mode
        if name == "voice_gate_threshold":
            return self.voice_gate_threshold
        if name == "voice_gate":
            return self.voice_gate
        if name == "agc":
            return self.agc_enabled
        if name == "agc_target_dbfs":
            return self.agc_target_dbfs
        return self.profile.get(name, default)

    def set_devices(self, input_device=None, monitor_device=None):
        """Refresh the whole audio chain when selecting input/monitor devices."""
        self.refresh_devices(input_device, monitor_device)

    def refresh_devices(self, input_device=None, monitor_device=None, cable_device=None):
        """Reopen all streams and reset DSP state while keeping loaded model weights."""
        with self._lock:
            previous = (self.input_device, self.monitor_device, self.cable_device, self.monitor_on)
            desired = (self.input_device if input_device is None else input_device,
                       self.monitor_device if monitor_device is None else monitor_device,
                       self.cable_device if cable_device is None else cable_device)
            if not self._running:
                self.input_device, self.monitor_device, self.cable_device = desired
                self._in_index = self._monitor_index = self._cable_index = None
                self.reset_voice_gate()
                self.reset_agc()
                self._reset_stats()
                self._clear_buffers()
                if self.master is not None:
                    self.master.clear()
                self.cable_reader = self.monitor_reader = None
                return
            # Invalid choices leave the running chain untouched.
            resolve_device(desired[0], "input")
            resolve_device(desired[2], "output")
            if self.monitor_on:
                resolve_device(desired[1], "output")
            self._infer_busy.acquire()
            try:
                self.stop()
                self.input_device, self.monitor_device, self.cable_device = desired
                try:
                    self.start()
                    if previous[3] and not self.monitor_active:
                        raise RuntimeError("监听设备未能重新打开")
                except Exception as exc:
                    self.stop()
                    (self.input_device, self.monitor_device,
                     self.cable_device, self.monitor_on) = previous
                    try:
                        self.start()
                        if previous[3] and not self.monitor_active:
                            raise RuntimeError("原监听设备未能恢复")
                    except Exception as rollback_exc:
                        self.stop()
                        raise RuntimeError("设备刷新失败且原设备无法恢复，音频已停止: %s / %s"
                                           % (exc, rollback_exc)) from exc
                    raise RuntimeError("设备刷新失败，已恢复原设备: %s" % exc) from exc
            finally:
                self._infer_busy.release()
            _log("输入、虚拟麦与监听已共同刷新，音频历史/降噪/声纹/VAD/AGC/统计已重置")

    def load_model(self, pth, index=None):
        """Hot-swap the model; input stops during the swap, outputs keep running silent."""
        with self._lock:
            self.profile["pth"] = pth
            if index is not None:
                self.profile["index"] = index
            if not self._running:
                if not self._prepared:
                    self._prepare()
                else:
                    self._reload_model(pth, index)
                return
            self._infer_busy.acquire()
            try:
                self._close_stream("input")
                try:
                    self._reload_model(pth, index)
                finally:
                    if self._in_stream is None:
                        try:
                            self._open_input()
                        except Exception:
                            traceback.print_exc()
                            self.stop()
                            raise
            finally:
                self._infer_busy.release()

    # ------------------------------------------------------------------ #
    # Offline self-check
    # ------------------------------------------------------------------ #

    def process_offline(self, waveform):
        """Run the same inference path offline on a 48k mono waveform; no devices opened."""
        with self._lock:
            if self._running:
                raise RuntimeError("管线运行中不能同时跑离线处理")
            if not self._prepared:
                self._prepare()
            x = np.asarray(waveform, dtype=np.float32).reshape(-1)
            out = np.zeros_like(x)
            for start in range(0, x.shape[0], self.block_frame):
                stop = min(start + self.block_frame, x.shape[0])
                chunk = x[start:stop]
                if chunk.shape[0] < self.block_frame:
                    chunk = np.pad(chunk, (0, self.block_frame - chunk.shape[0]))
                out[start:stop] = self._infer_block(chunk, realtime=False)[: stop - start]
            return out

    # ------------------------------------------------------------------ #
    # Prepare: model + buffers (gui_v1 start_vc 762-871)
    # ------------------------------------------------------------------ #

    def _prepare(self):
        if self._prepared:
            return
        profile = self.profile
        for key, value in _PROFILE_DEFAULTS.items():
            if profile.get(key) is None:  # missing keys and explicit nulls take defaults
                profile[key] = value
        for key in _NUMERIC_PROFILE_KEYS:
            profile[key] = float(profile[key])
        for key in ("I_noise_reduce", "O_noise_reduce"):
            profile[key] = _as_bool(profile.get(key))
        profile["f0method"] = self._check_f0method(profile.get("f0method"))
        pth = str(profile.get("pth") or "")
        if not pth or not os.path.isfile(pth):
            raise FileNotFoundError("模型 pth 不存在: %r" % pth)

        engine = self.engine
        for key in ("pitch", "formant", "index_rate", "n_cpu"):
            if profile.get(key) is not None:
                engine.set_param(key, profile[key])
        engine.set_param("f0method", profile["f0method"])
        # The engine falls rejected values back to rmvpe, but infer() reads the
        # profile per frame, so write the effective value back here.
        profile["f0method"] = engine.get_param("f0method", profile["f0method"])
        engine.load(pth, profile.get("index") or None)

        self.tgt_sr = int(engine.tgt_sr or SAMPLERATE)
        self._init_buffers()
        self._resolve_voice_gate()
        self._build_input_vad()
        if self.denoise_mode == "dfn3" and self._ensure_denoiser() is None:
            _log("警告: denoise_mode='dfn3' 不可用（dsp.dfn 未交付或初始化失败），"
                 "本次降级为 'torchgate'")
            self.denoise_mode = "torchgate"
        self._warmup_f0()
        self._prepared = True

    def _reload_model(self, pth, index):
        self.engine.load(pth, index or None)
        self.profile["pth"] = str(pth)
        self.profile["index"] = index or ""
        sr = int(self.engine.tgt_sr or SAMPLERATE)
        if sr != self.tgt_sr:
            self.tgt_sr = sr
            self._build_resampler2()
        self._warmup_f0()

    def _init_buffers(self):
        """Build buffers from block_time/crossfade_time/extra_time; mirrors gui_v1 start_vc."""
        torch = _torch()
        profile = self.profile
        device = self.engine.config.device
        sr = SAMPLERATE

        self.zc = sr // 100
        self.block_frame = int(np.round(profile["block_time"] * sr / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(profile["crossfade_time"] * sr / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(profile["extra_time"] * sr / self.zc)) * self.zc
        if self.block_frame <= 0:
            raise ValueError("block_time 太小: %r" % profile["block_time"])
        if self.crossfade_frame <= 0:
            raise ValueError("crossfade_time 太小（需 >= 0.01s）: %r" % profile["crossfade_time"])

        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=device,
            dtype=torch.float32,
        )
        need_16k = self._f0_extractor_len()
        if self.input_wav_res.shape[0] < need_16k:
            raise ValueError(
                "extra_time=%r 太小: f0 提取窗口需要 %d 个 16k 样本，当前只有 %d"
                % (profile["extra_time"], need_16k, self.input_wav_res.shape[0])
            )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = _resample_cls()(
            orig_freq=sr, new_freq=16000, dtype=torch.float32
        ).to(device)
        self._build_resampler2()
        self.tg = _torchgate_cls()(sr=sr, n_fft=4 * self.zc, prop_decrease=0.9).to(device)

        # Voice gate: decision window is the latest 1s of raw 48k input, read-only.
        block_seconds = self.block_frame / float(sr)
        self._gate_every = max(1, int(round(_VOICE_GATE_INTERVAL / block_seconds)))
        self._gate_hist = np.zeros(int(round(_VOICE_GATE_WINDOW * sr)), dtype=np.float32)
        self._gate_resampler = None  # CPU resampler for decisions; GPU inference untouched
        self._reset_voice_gate_state()

    def _build_resampler2(self):
        """Add a model-rate to 48000 resampler when the model output rate differs."""
        if self.tgt_sr != SAMPLERATE:
            self.resampler2 = _resample_cls()(
                orig_freq=self.tgt_sr, new_freq=SAMPLERATE, dtype=_torch().float32
            ).to(self.engine.config.device)
        else:
            self.resampler2 = None

    def _f0_extractor_len(self):
        """16k samples of the rtrvc f0 window (rmvpe needs 5120 alignment)."""
        frame = self.block_frame_16k + 800
        if str(self.profile.get("f0method")) == "rmvpe":
            frame = 5120 * ((frame - 1) // 5120 + 1) - 160
        return int(frame)

    def _warmup_f0(self):
        """Preload the f0 model; failures only warn and never block startup."""
        method = str(self.profile.get("f0method") or "")
        rvc = self.engine.rvc
        getter = getattr(rvc, "get_f0_%s" % method, None)
        if not callable(getter):
            return
        try:
            warm = _torch().zeros(self._f0_extractor_len())
            getter(warm, 0)
        except Exception as exc:
            _log("警告: f0 模型预载失败（首块回调里会重试）: %r" % (exc,))

    # ------------------------------------------------------------------ #
    # Acoustic protection: voice gate + DFN3 input denoise
    # ------------------------------------------------------------------ #

    def _reset_voice_gate_state(self):
        """Reset voice-gate runtime state; fail-open (audio passes until a firm reject)."""
        self._silence_seconds = 0.0
        self._input_paused = False
        self._input_noise_rms = 1e-5
        self._gate_hold_open = False
        self._gate_onset_seconds = 0.0
        self._gate_speech_seconds = 0.0
        self._gate_previous_target = 1.0
        self._onset_bypass = 0
        if self._gate_hist is not None:
            self._gate_hist[:] = 0.0
        self._gate_gain = 1.0
        self._gate_target = 1.0
        self._gate_countdown = 0
        self._gate_score = None
        self._gate_errors = 0
        self._gate_decision_epoch = None
        lock = getattr(self, "_gate_mailbox_lock", None)
        if lock is not None:
            with lock:
                self._gate_epoch = getattr(self, "_gate_epoch", 0) + 1
                self._gate_mailbox = None
                self._gate_result = None
        vad = getattr(self, "_input_vad", None)
        if vad is not None:
            vad.reset()
        stats = getattr(self, "voice_stats", None)
        if stats is not None:
            stats["decisions"] = 0
            stats["blocked_blocks"] = 0
            stats["errors"] = 0
            stats["open"] = True
            stats["gain"] = 1.0
            stats["score"] = None

    def _build_input_vad(self):
        """Detect pauses before any gate/denoise; never load a model in the callback."""
        if self._input_vad is not None:
            return
        cls = _dsp_attr("vad", "SileroVad")
        if cls is None:
            return
        try:
            model_dir = getattr(self.voice_gate, "model_dir", None)
            self._input_vad = cls(model_dir=model_dir) if model_dir else cls()
        except Exception as exc:
            _log("警告: 输入 VAD 不可用，起音保护退回电平检测: %r" % (exc,))

    def _invalidate_gate_decisions(self):
        """Discard both queued and in-flight decisions from the preceding utterance."""
        with self._gate_mailbox_lock:
            self._gate_epoch += 1
            self._gate_mailbox = None
            self._gate_result = None
        self._gate_countdown = 0
        self._gate_score = None
        self.voice_stats["score"] = None
        if self._gate_hist is not None:
            self._gate_hist[:] = 0.0

    def _track_input_activity(self, block):
        """Reopen on the first raw onset block after >=500ms without speech."""
        seconds = block.size / float(SAMPLERATE)
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        energy_onset = rms >= max(1e-4, self._input_noise_rms * 3.0)
        vad = getattr(self, "_input_vad", None)
        speech = rms >= 0.003 if vad is None else False
        if vad is not None:
            try:
                speech = float(vad.process(block, sr=SAMPLERATE)) >= _INPUT_VAD_THRESHOLD
            except Exception as exc:
                self._input_vad = None
                speech = rms >= 0.003
                if not self._input_vad_warned:
                    self._input_vad_warned = True
                    _log("警告: 输入 VAD 失败，起音保护退回电平检测: %r" % (exc,))
        # A consonant can precede Silero's first positive window. An abrupt rise
        # above the measured room floor protects the same block, without waiting.
        speech = speech or (energy_onset and (self._input_paused or self._gate_hold_open))
        if speech:
            if self._input_paused:
                self._invalidate_gate_decisions()
                self._input_paused = False
                self._gate_previous_target = self._gate_target
                self._gate_target = self._gate_gain = 1.0
                self._gate_hold_open = True
                self._gate_onset_seconds = self._gate_speech_seconds = 0.0
                self._onset_bypass = _ONSET_BYPASS_BLOCKS
                self.voice_stats["open"] = True
                self.voice_stats["gain"] = 1.0
                if getattr(self, "nr_buffer", None) is not None:
                    self.nr_buffer.zero_()
            self._silence_seconds = 0.0
            self._gate_speech_seconds += seconds
        else:
            self._silence_seconds += seconds
            # Learn the room floor only while idle, never from a new syllable.
            if self._input_paused:
                self._input_noise_rms = 0.9 * self._input_noise_rms + 0.1 * rms
            else:
                self._input_noise_rms = rms
            if self._silence_seconds >= _INPUT_PAUSE_SECONDS and not self._input_paused:
                self._input_paused = True
                self._gate_hold_open = False
                self._onset_bypass = 0
                self._invalidate_gate_decisions()
        if self._gate_hold_open:
            self._gate_onset_seconds += seconds
            if self._gate_onset_seconds > _VOICE_GATE_ONSET_GRACE_SECONDS:
                self._gate_hold_open = False
                self._gate_target = self._gate_previous_target
                self.voice_stats["open"] = bool(self._gate_target)
        protected = bool(speech and self._onset_bypass > 0)
        if protected:
            self._onset_bypass -= 1
        return protected

    # ---- Decision thread: decide stays out of the audio callback ----

    def _start_gate_worker(self):
        """Start the decision thread; False means fall back to in-callback decide."""
        if self._gate_thread is not None and self._gate_thread.is_alive():
            return True
        self._gate_stop.clear()
        self._gate_wake.clear()
        try:
            thread = threading.Thread(
                target=self._voice_gate_worker, name="voice-gate", daemon=True
            )
            thread.start()
        except Exception as exc:
            _log("警告: 声纹判决线程起不来（退回就地判决，实时性会变差）: %r" % (exc,))
            self._gate_thread = None
            return False
        self._gate_thread = thread
        return True

    def _stop_gate_worker(self):
        """Stop the decision thread; idempotent."""
        self._gate_stop.set()
        self._gate_wake.set()
        thread, self._gate_thread = self._gate_thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _voice_gate_worker(self):
        """Decide the newest mailed 1s history; drop whatever cannot keep up."""
        last_epoch = None
        while not self._gate_stop.is_set():
            self._gate_wake.wait(0.2)
            self._gate_wake.clear()
            while not self._gate_stop.is_set():
                with self._gate_mailbox_lock:
                    job, self._gate_mailbox = self._gate_mailbox, None
                if job is None:
                    break
                epoch, gate, wav = job
                if gate is None:
                    continue
                score, error = None, None
                try:
                    if epoch != last_epoch:
                        reset = getattr(gate, "reset_smooth", None)
                        if callable(reset):
                            reset()
                        last_epoch = epoch
                    score = gate.decide(self._voice_gate_to_16k(wav))
                except Exception as exc:
                    error = exc
                # Only the audio thread adopts results. A delayed rejection must
                # never race a new utterance's first-block reopening.
                with self._gate_mailbox_lock:
                    if epoch == self._gate_epoch and gate is self.voice_gate:
                        self._gate_result = (score, error)

    def _gate_applied(self, score):
        """Adopt one decision; undecidable keeps the previous target gain."""
        score = None if score is None else float(score)
        self._gate_errors = 0
        self._gate_score = score
        if self._gate_hold_open:
            # Only current-utterance decisions with enough speech reach here.
            if score is not None and math.isfinite(score):
                self._gate_hold_open = False
                self._gate_target = 1.0 if score >= self.voice_gate_threshold else 0.0
            else:
                self._gate_target = 1.0
        elif score is None or not math.isfinite(score):
            pass  # undecidable: keep the previous target
        elif score >= self.voice_gate_threshold:
            self._gate_target = 1.0
        else:
            self._gate_target = 0.0
        self.voice_stats["decisions"] += 1
        self.voice_stats["score"] = score
        self.voice_stats["open"] = bool(self._gate_target > 0.0)

    def _gate_failed(self, exc):
        """Keep the last decision on error; disable the gate after repeated failures."""
        self._gate_errors += 1
        self.voice_stats["errors"] += 1
        if self._gate_errors == 1:
            _log("警告: 声纹判决失败（保持上一次判决，不中断主链）: %r" % (exc,))
        if self._gate_errors >= _VOICE_GATE_MAX_ERRORS:
            self.voice_gate = None
            self._voice_gate_spec = None
            self._stop_gate_worker()
            self._gate_target = 1.0  # passthrough
            _log("警告: 声纹门控连续 %d 次判决失败，已停用（主链恢复直通）；"
                 "请检查声纹模型与已注册档案" % self._gate_errors)

    def _resolve_voice_gate(self):
        """Resolve the voice_gate spec to an instance; failures warn and disable the gate."""
        spec = self._voice_gate_spec
        if spec is None or spec is False:
            self.voice_gate = None
            return
        if callable(getattr(spec, "decide", None)):
            self.voice_gate = spec
            self._disable_gate_if_unenrolled()
            return
        if isinstance(spec, (str, os.PathLike)):
            cls = _dsp_attr("voice_gate", "VoiceGate")
            if cls is None:
                _log("警告: 找不到 dsp/voice_gate.py 的 VoiceGate，本次关闭声纹门控"
                     "（不影响变声主链）")
                self.voice_gate = None
                return
            try:
                self.voice_gate = cls(str(spec), self.voice_gate_threshold)
                _log("声纹门控已加载: %s" % (spec,))
                self._disable_gate_if_unenrolled()
                return
            except Exception as exc:
                _log("警告: VoiceGate(%r) 构造失败，本次关闭声纹门控: %r" % (spec, exc))
                self.voice_gate = None
                return
        _log("警告: voice_gate=%r 既不是 VoiceGate 实例也不是模型目录，忽略（门控关闭）"
             % (spec,))
        self.voice_gate = None

    def _disable_gate_if_unenrolled(self):
        """Disable an unenrolled gate instance; its decide() can never reject."""
        gate = self.voice_gate
        if gate is None:
            return
        try:
            enrolled = gate.is_enrolled
        except Exception:
            return
        if enrolled is False:
            self.voice_gate = None
            self._voice_gate_spec = None
            self._gate_target = 1.0
            _log("警告: 声纹门控未注册声纹（先注册再开门控），本次关闭门控，主链直通")

    def _ensure_denoiser(self):
        """Build dsp.dfn.DfnDenoiser on demand; None when unavailable."""
        if self._dfn is not None:
            return self._dfn
        cls = _dsp_attr("dfn", "DfnDenoiser")
        if cls is None:
            return None
        try:
            self._dfn = cls(sr=SAMPLERATE)
        except Exception as exc:
            _log("警告: DfnDenoiser 初始化失败: %r" % (exc,))
            self._dfn = None
            return None
        self._refresh_dfn_latency()
        _log("DFN3 输入降噪已加载: 算法延迟 %s"
             % ("%.1f ms" % self._dfn_latency_ms
                if self._dfn_latency_ms is not None else "未提供"))
        return self._dfn

    def _denoise_block_frames(self):
        """Frames fed to DFN: block_frame + 2*zc with the rms gate on, else block_frame."""
        block = getattr(self, "block_frame", 0)
        if not block:
            return None
        if float(self.profile.get("threhold", 0.0)) > -60:
            return int(block) + 2 * int(getattr(self, "zc", 0))
        return int(block)

    def _dfn_latency_for(self, block_frames):
        """DFN algorithmic latency in ms for the block size; None when unknown."""
        getter = getattr(self._dfn, "latency_ms", None)
        if not callable(getter):
            return None
        try:
            return float(getter(block_frames) if block_frames else getter())
        except Exception:
            return None

    def _refresh_dfn_latency(self):
        """Re-cache DFN latency for the current block size."""
        self._dfn_latency_ms = self._dfn_latency_for(self._denoise_block_frames())

    def _dfn_process(self, block):
        """Denoise one 48k block; on error disable dfn3 and return the block as-is."""
        try:
            out = np.asarray(self._dfn.process(block), dtype=np.float32).reshape(-1)
        except Exception as exc:
            _log("警告: DfnDenoiser.process 失败，关闭 dfn3 输入降噪: %r" % (exc,))
            self._dfn = None
            return block
        if out.shape[0] == block.shape[0]:
            return out
        fixed = np.zeros_like(block)
        n = min(out.shape[0], block.shape[0])
        fixed[:n] = out[:n]
        return fixed

    def _voice_gate_tick(self, block, realtime=True):
        """Append the block to the 1s history; decide on the history when due.

        History holds raw pre-gain input, so closing the gate can never lock it
        shut. realtime=True mails history to the background thread; False
        decides in place (offline path, deterministic).
        """
        hist = self._gate_hist
        if hist is None or block.shape[0] == 0:
            return
        with self._gate_mailbox_lock:
            result, self._gate_result = self._gate_result, None
        if result is not None:
            score, error = result
            if error is None:
                self._gate_applied(score)
            else:
                self._gate_failed(error)
        if self.voice_gate is None:
            return
        n = block.shape[0]
        if n >= hist.shape[0]:
            hist[:] = block[-hist.shape[0] :]
        else:
            hist[:-n] = hist[n:]
            hist[-n:] = block
        if self._input_paused:
            return
        if self._gate_hold_open and self._gate_speech_seconds < _VOICE_GATE_MIN_ONSET_SECONDS:
            return
        # Silent blocks feed history only: nothing decidable inside, skip the countdown.
        if float(np.max(np.abs(block))) < 1e-3:
            return
        if self._gate_countdown > 0:
            self._gate_countdown -= 1
            return
        self._gate_countdown = max(0, self._gate_every - 1)
        if realtime:
            if self._gate_thread is None and not self._start_gate_worker():
                realtime = False
            if realtime:
                with self._gate_mailbox_lock:
                    self._gate_mailbox = (self._gate_epoch, self.voice_gate, hist.copy())
                self._gate_wake.set()
                return
        try:
            if self._gate_decision_epoch != self._gate_epoch:
                reset = getattr(self.voice_gate, "reset_smooth", None)
                if callable(reset):
                    reset()
                self._gate_decision_epoch = self._gate_epoch
            score = self.voice_gate.decide(self._voice_gate_to_16k(hist))
        except Exception as exc:
            self._gate_failed(exc)
            return
        self._gate_applied(score)

    def _voice_gate_to_16k(self, wav48k):
        """Downsample 1s of 48k history to 16k on CPU."""
        if self._gate_resampler is None:
            self._gate_resampler = _resample_cls()(
                orig_freq=SAMPLERATE, new_freq=16000, dtype=_torch().float32
            )
        torch = _torch()
        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(wav48k, dtype=np.float32))
            return self._gate_resampler(wav).numpy()

    def _voice_gate_gain(self, n):
        """Smoothed gain for the current block (10ms attack / 300ms release); None means unity."""
        if self._gate_hold_open:
            self._gate_gain = 1.0  # protect the onset until a current decision arrives
            return None
        cur = self._gate_gain
        target = self._gate_target
        if n <= 0:
            return None
        if target >= 1.0 and cur >= 0.999:
            self._gate_gain = 1.0  # open and settled: skip multiplying
            return None
        tau = _VOICE_GATE_ATTACK if target > cur else _VOICE_GATE_RELEASE
        coef = math.exp(-1.0 / (tau * SAMPLERATE))
        gains = target + (cur - target) * np.power(
            coef, np.arange(1, n + 1, dtype=np.float64)
        )
        self._gate_gain = float(gains[-1])
        return gains.astype(np.float32)

    def _set_denoise_mode(self, value):
        """Hot-update denoise_mode; dfn3 falls back to torchgate when unavailable."""
        mode = normalize_denoise_mode(value)
        if mode is None:
            _log("警告: 未知 denoise_mode=%r（可选 %s），保持 %s"
                 % (value, "/".join(_DENOISE_MODES), self.denoise_mode))
            return
        if mode == "dfn3" and self._ensure_denoiser() is None:
            _log("警告: denoise_mode='dfn3' 不可用（dsp/dfn.py 未交付或初始化失败），"
                 "降级为 'torchgate'")
            mode = "torchgate"
        elif mode != "dfn3" and self._dfn is not None:
            try:
                self._dfn.reset()
            except Exception:
                pass
        self.denoise_mode = mode
        # Latency follows the new block layout.
        self._refresh_dfn_latency()
        _log("输入降噪模式: %s"
             % ("off（输入侧不降噪，TorchGate 也不跑）" if mode == "off" else mode))

    def _set_voice_gate_threshold(self, value):
        """Hot-update the voice-print decision threshold."""
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            raise ValueError("voice_gate_threshold 必须是数字: %r" % (value,))
        if not math.isfinite(threshold):
            raise ValueError("voice_gate_threshold 必须是有限数: %r" % (value,))
        self.voice_gate_threshold = threshold
        gate = self.voice_gate
        if gate is not None and hasattr(gate, "threshold"):
            try:
                gate.threshold = threshold
            except Exception:
                pass
        _log("声纹判决阈值: %.3f" % threshold)

    def _set_voice_gate(self, value):
        """Hot-update the voice gate: VoiceGate instance, model dir path, or None."""
        self._voice_gate_spec = value
        self._resolve_voice_gate()
        if self.voice_gate is None:
            self._stop_gate_worker()
        self.reset_voice_gate()
        _log("声纹门控: %s"
             % ("开（阈值 %.2f）" % self.voice_gate_threshold
                if self.voice_gate is not None else "关"))

    # ------------------------------------------------------------------ #
    # Output auto gain (dsp.agc.Agc)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_target_dbfs(value):
        """Validate the AGC target loudness in dBFS."""
        target = float(value)
        if not math.isfinite(target):
            raise ValueError("agc_target_dbfs 必须是有限数: %r" % (value,))
        return target

    @staticmethod
    def _agc_takes_prob(agc):
        """Whether Agc.process accepts speech_prob; None when the signature is unclear."""
        if agc is None:
            return False
        try:
            params = inspect.signature(agc.process).parameters
        except (TypeError, ValueError):
            return None
        positional = 0
        for param in params.values():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                return True
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional += 1
        return positional >= 2

    def _build_agc(self):
        """Build dsp.agc.Agc on demand; None when unavailable."""
        cls = _dsp_attr("agc", "Agc")
        if cls is None:
            _log("警告: 找不到 dsp/agc.py 的 Agc，本次不启用输出自动增益")
            return None
        try:
            agc = cls(sr=SAMPLERATE)
        except Exception as exc:
            _log("警告: Agc 初始化失败，本次不启用输出自动增益: %r" % (exc,))
            return None
        self._agc_prob = self._agc_takes_prob(agc)
        return agc

    def _build_agc_vad(self):
        """Build the SileroVad speech-probability source; AGC degrades gracefully without it."""
        if self._agc_vad is not None:
            return
        cls = _dsp_attr("vad", "SileroVad")
        if cls is None:
            self._warn_agc_vad("找不到 dsp/vad.py 的 SileroVad")
            return
        try:
            self._agc_vad = cls()
        except Exception as exc:
            self._warn_agc_vad("SileroVad 初始化失败（models/ 缺 silero 模型？）: %r" % (exc,))

    def _warn_agc_vad(self, text):
        if not self._agc_vad_warned:
            self._agc_vad_warned = True
            _log("警告: %s；AGC 语音判定退回能量启发式" % text)

    def _apply_agc_target(self):
        """Push the target loudness to the Agc instance."""
        setter = getattr(self.agc, "set_target_dbfs", None)
        if not callable(setter):
            return
        try:
            setter(self.agc_target_dbfs)
        except Exception as exc:
            _log("警告: AGC 目标响度下发失败: %r" % (exc,))

    def _set_agc(self, value):
        """Toggle AGC at runtime; True builds the instance on demand."""
        want = _as_bool(value)
        if want and self.agc is None:
            self.agc = self._build_agc()
            if self.agc is None:
                self.agc_enabled = False
                return
        self.agc_enabled = want
        if want:
            self._build_agc_vad()  # re-attach the VAD source when re-enabling
            self._apply_agc_target()
        _log("输出自动增益: %s"
             % ("开（目标 %.1f dBFS）" % self.agc_target_dbfs if want else "关（直通）"))

    def _set_agc_target(self, value):
        """Hot-update the target loudness in dBFS."""
        target = self._check_target_dbfs(value)
        self.agc_target_dbfs = target
        self._apply_agc_target()
        _log("AGC 目标响度: %.1f dBFS" % target)

    def _agc_process(self, block):
        """Run one block through AGC; on error disable AGC and return the block as-is."""
        prob = None
        if self._agc_vad is not None:
            try:
                prob = self._agc_vad.process(block)
            except Exception as exc:
                self._agc_vad = None
                self._warn_agc_vad("SileroVad process 失败: %r" % (exc,))
        try:
            if self._agc_prob is False:  # confirmed single-arg signature
                raw = self.agc.process(block)
            else:
                try:
                    raw = self.agc.process(block, prob)
                except TypeError:
                    # Signature unclear: probe once, remember single-arg implementations.
                    self._agc_prob = False
                    raw = self.agc.process(block)
                else:
                    self._agc_prob = True
            out = np.asarray(raw, dtype=np.float32).reshape(-1)
        except Exception as exc:
            _log("警告: AGC process 失败，关闭输出自动增益: %r" % (exc,))
            self.agc_enabled = False
            return block
        if out.shape[0] != block.shape[0]:
            fixed = np.zeros_like(block)
            n = min(out.shape[0], block.shape[0])
            fixed[:n] = out[:n]
            out = fixed
        gain_db = getattr(self.agc, "gain_db", None)
        if gain_db is not None:
            self.agc_stats["gain_db"] = float(gain_db)
        self.agc_stats["blocks"] += 1
        peak = float(np.max(np.abs(out))) if out.shape[0] else 0.0
        if peak > self.agc_stats["peak"]:
            self.agc_stats["peak"] = peak
        return out

    # ------------------------------------------------------------------ #
    # Inference (gui_v1 audio_callback 905-1061)
    # ------------------------------------------------------------------ #

    def _infer_block(self, indata, realtime=True):
        """Run one block_frame 48k mono block to a same-length output block."""
        torch = _torch()
        import torch.nn.functional as F
        import librosa

        profile = self.profile
        device = self.engine.config.device
        indata = np.ascontiguousarray(indata, dtype=np.float32).reshape(-1)
        if indata.shape[0] != self.block_frame:
            if indata.shape[0] > self.block_frame:
                indata = indata[: self.block_frame]
            else:
                indata = np.pad(indata, (0, self.block_frame - indata.shape[0]))

        onset_protect = self._track_input_activity(indata)

        # rms gate; below -60 counts as off.
        if profile["threhold"] > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < profile["threhold"]
            )
            if not onset_protect:
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]

        # Acoustic protection: decide on history, DFN3 denoise at 48k, then apply
        # the smoothed gain so a closed gate zeroes the block before RVC.
        gate_on = self.voice_gate is not None
        if gate_on:
            # Feed the current block only: the gate branch prepends 2*zc copied samples.
            self._voice_gate_tick(indata[-self.block_frame :], realtime=realtime)
        dfn_active = self.denoise_mode == "dfn3" and self._dfn is not None
        if dfn_active:
            indata = self._dfn_process(indata)
        # Gain applies after denoise, so a closed gate outputs 0 regardless of DFN state.
        if gate_on:
            gains = self._voice_gate_gain(indata.shape[0])
            if gains is not None:
                indata = indata * gains
                self.voice_stats["gain"] = float(gains[-1])
                if float(np.max(gains)) < 0.05:
                    self.voice_stats["blocked_blocks"] += 1

        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # Input denoise + resample to 16k.
        if profile["I_noise_reduce"] and not dfn_active and self.denoise_mode != "off":
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            if onset_protect:
                input_wav = input_wav.clone()  # onset blocks bypass TorchGate
            else:
                input_wav = self.tg(
                    input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
                ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        elif dfn_active:
            # Already denoised at 48k: keep input_wav_denoise in sync with input_wav.
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            self.input_wav_denoise[-self.block_frame :] = self.input_wav[-self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[160:]
            )
        # Inference.
        infer_wav = self.engine.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            profile["f0method"],
        )
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav)
        # Near-silent RVC output emits silence so the envelope mix cannot amplify residue.
        if float(torch.max(torch.abs(infer_wav)).item()) < _RVC_SILENCE_PEAK:
            self.sola_buffer.zero_()
            return np.zeros(self.block_frame, dtype=np.float32)
        # Output denoise.
        if profile["O_noise_reduce"]:
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            if not onset_protect:
                infer_wav = self.tg(
                    infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
                ).squeeze(0)
        # Loudness envelope mix, clamped to +-12dB per block.
        if profile["rms_mix_rate"] < 1:
            if profile["I_noise_reduce"]:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            ratio = torch.pow(
                rms1 / rms2, torch.tensor(1 - profile["rms_mix_rate"])
            )
            ratio = torch.clamp(ratio, _RMS_MIX_MIN_RATIO, _RMS_MIX_MAX_RATIO)
            infer_wav *= ratio
        # SOLA (https://github.com/yxlllc/DDSP-SVC)
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(device) or not profile.get("use_pv", False):
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        out = infer_wav[: self.block_frame].cpu().numpy()
        if self.agc_enabled and self.agc is not None:
            out = self._agc_process(out)  # output AGC: after SOLA, before the master write
        # Hard ceiling: peak always below 0dBFS before the master write.
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np.clip(out, -_OUTPUT_CEILING, _OUTPUT_CEILING, out=out)
        return out

    # ------------------------------------------------------------------ #
    # Audio callbacks
    # ------------------------------------------------------------------ #

    def _input_callback(self, indata, frames, time_info, status):
        if not self._infer_busy.acquire(blocking=False):
            return
        try:
            try:
                if status:
                    self.stats["in_status"] += 1
                import librosa

                mono = np.ascontiguousarray(librosa.to_mono(indata.T), dtype=np.float32)
                if self._pending.shape[0]:
                    mono = np.concatenate((self._pending, mono))
                if mono.shape[0] and not bool(np.isfinite(mono).all()):
                    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
                block = self.block_frame
                start = 0
                total = mono.shape[0]
                # Overload guard: only the newest block stays fresh, drop the stale backlog.
                n_blocks = total // block
                if n_blocks > _OVERLOAD_BACKLOG_BLOCKS:
                    drop = n_blocks - 1
                    start = drop * block
                    self.stats["dropped"] += drop
                while total - start >= block:
                    self._process_block(mono[start : start + block])
                    start += block
                rest = mono[start:] if start < total else np.zeros(0, dtype=np.float32)
                if rest.shape[0] > 4 * block:  # defensive cap; unreachable in theory
                    rest = rest[-block:]
                self._pending = np.ascontiguousarray(rest)
            except Exception:
                self.stats["errors"] += 1
                if self.stats["errors"] == 1:
                    traceback.print_exc()
                self._push_silence()
        finally:
            self._infer_busy.release()

    def _process_block(self, chunk):
        start_time = time.perf_counter()
        try:
            out = self._infer_block(chunk)
        except Exception:
            self.stats["errors"] += 1
            if self.stats["errors"] == 1:
                traceback.print_exc()
            self._push_silence()
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        stats = self.stats
        stats["blocks"] += 1
        stats["infer_ms_avg"] += (elapsed_ms - stats["infer_ms_avg"]) * 0.1
        if elapsed_ms > stats["infer_ms_max"]:
            stats["infer_ms_max"] = elapsed_ms
        self._check_overload()
        if self.master is not None:
            self.master.write(out)
        if self.on_level is not None:
            try:
                self.on_level(level_db(chunk), level_db(out))
            except Exception:
                pass

    def _check_overload(self):
        """Drop f0method from rmvpe to pm when inference keeps missing the block budget."""
        budget_ms = self.block_frame * 1000.0 / SAMPLERATE
        if self.stats["infer_ms_avg"] <= budget_ms:
            self._overload_hot = 0
            return
        self._overload_hot += 1
        if self._overload_hot < _OVERLOAD_SUSTAIN_BLOCKS:
            return
        if str(self.profile.get("f0method")) != "rmvpe":
            return
        try:
            self.engine.set_param("f0method", "pm")
        except Exception as exc:
            _log("警告: 过载想降 f0method 到 pm 但失败，本次保持 rmvpe: %r" % (exc,))
            return
        self.profile["f0method"] = self.engine.get_param("f0method", "pm")
        self._overload_hot = 0
        _log("警告: 推理平均耗时 %.1f ms 已连续 %d 块超过块预算 %.1f ms，"
             "疑似电脑发热降频，已把 f0method 从 rmvpe 自动降到 pm 保住实时性；"
             "负载恢复后不会自动升回，想恢复变声效果请手动切回 rmvpe"
             % (self.stats["infer_ms_avg"], _OVERLOAD_SUSTAIN_BLOCKS, budget_ms))

    def _push_silence(self):
        if self.master is not None:
            self.master.write(np.zeros(self.block_frame, dtype=np.float32))

    def _make_output_callback(self, reader, counter):
        stats = self.stats

        def callback(outdata, frames, time_info, status):
            try:
                outdata[:] = reader.read(frames)[:, None]
                stats[counter] += 1
            except Exception:
                outdata.fill(0.0)
                stats["errors"] += 1

        return callback

    def _make_input_callback(self, gen):
        """Bind the generation number; stale callbacks from older streams return early."""

        def callback(indata, frames, time_info, status):
            if gen != self._in_gen:
                return
            self._input_callback(indata, frames, time_info, status)

        return callback

    # ------------------------------------------------------------------ #
    # Stream open/close
    # ------------------------------------------------------------------ #

    @staticmethod
    def _output_channels(index):
        sd = _sd()
        return max(1, min(2, int(sd.query_devices(index)["max_output_channels"])))

    def _open_cable(self):
        sd = _sd()
        index = resolve_device(self.cable_device, "output")
        channels = self._output_channels(index)
        stream = sd.OutputStream(
            device=index,
            samplerate=SAMPLERATE,
            blocksize=self.block_frame,
            channels=channels,
            dtype="float32",
            callback=self._make_output_callback(self.cable_reader, "cable_blocks"),
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._cable_index = index
        self._cable_stream = stream

    def _open_input(self):
        sd = _sd()
        index = resolve_device(self.input_device, "input")
        stream = sd.InputStream(
            device=index,
            samplerate=SAMPLERATE,
            blocksize=self.block_frame,
            channels=1,
            dtype="float32",
            callback=self._make_input_callback(self._in_gen),
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._in_index = index
        self._in_stream = stream

    def _open_monitor(self):
        """Open the monitor stream on the same master data; headphones required."""
        if not self.monitor_device:
            raise ValueError("未配置监听设备（monitor_device 为空）")
        sd = _sd()
        index = resolve_device(self.monitor_device, "output")
        channels = self._output_channels(index)
        reader = self.master.reader()
        stream = sd.OutputStream(
            device=index,
            samplerate=SAMPLERATE,
            blocksize=self.block_frame,
            channels=channels,
            dtype="float32",
            callback=self._make_output_callback(reader, "monitor_blocks"),
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._monitor_index = index
        self._monitor_stream = stream
        self.monitor_reader = reader
        _log("监听已开启: %r → index=%d, %dch" % (self.monitor_device, index, channels))

    def _close_stream(self, name):
        if name == "input":
            # Closing input bumps the generation first so stale callbacks return early.
            self._in_gen += 1
        stream = {
            "input": self._in_stream,
            "cable": self._cable_stream,
            "monitor": self._monitor_stream,
        }[name]
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        if name == "input":
            self._in_stream = None
        elif name == "cable":
            self._cable_stream = None
        else:
            self._monitor_stream = None

    def _set_monitor(self, on):
        with self._lock:
            if not self._running:
                self.monitor_on = on
                return
            if on:
                if self._monitor_stream is None:
                    self._open_monitor()  # failure raises, state stays off
                self.monitor_on = True
            else:
                was_open = self._monitor_stream is not None
                self._close_stream("monitor")
                self.monitor_reader = None
                self.monitor_on = False
                if was_open:
                    _log("监听已关闭")

    def _swap_input(self, device):
        if not self._running:
            self.input_device = device
            self._in_index = None
            return
        sd = _sd()
        new_index = resolve_device(device, "input")  # resolve first: failure keeps the old stream
        # The new stream binds the next generation; _close_stream("input") bumps _in_gen to it.
        new_stream = sd.InputStream(
            device=new_index,
            samplerate=SAMPLERATE,
            blocksize=self.block_frame,
            channels=1,
            dtype="float32",
            callback=self._make_input_callback(self._in_gen + 1),
        )
        old_device, old_index = self.input_device, self._in_index
        self._infer_busy.acquire()
        try:
            self._close_stream("input")
            try:
                new_stream.start()
            except Exception:
                try:
                    new_stream.close()
                except Exception:
                    pass
                self.input_device, self._in_index = old_device, old_index
                try:
                    self._open_input()  # roll back to the old device
                except Exception:
                    traceback.print_exc()
                    self.stop()  # both failed: converge to stopped, never half-running
                raise
            self._in_index = new_index
            self._in_stream = new_stream
            self.input_device = device
            self._pending = np.zeros(0, dtype=np.float32)
            self.reset_voice_gate()
            vad_reset = getattr(self._agc_vad, "reset", None)
            if callable(vad_reset):
                try:
                    vad_reset()
                except Exception as exc:
                    _log("警告: VAD reset 失败: %r" % (exc,))
            _log("输入设备已切换: %r → index=%d" % (device, new_index))
        finally:
            self._infer_busy.release()

    def _swap_monitor(self, device):
        was_on = self._monitor_stream is not None
        if not self._running:
            self.monitor_device = device
            self._monitor_index = None
            return
        resolve_device(device, "output")  # resolve first: failure keeps the old stream
        old_device, old_index = self.monitor_device, self._monitor_index
        self.monitor_device = device
        if not was_on:
            return
        self._close_stream("monitor")
        self.monitor_reader = None
        try:
            self._open_monitor()
        except Exception:
            # Roll back to the old device and reopen; accept stopped state when that fails too.
            self.monitor_device, self._monitor_index = old_device, old_index
            try:
                self._open_monitor()
            except Exception:
                traceback.print_exc()
                self.monitor_on = False
            raise
        _log("监听设备已切换: %r" % (device,))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_f0method(value):
        method = str(value or "").strip().lower()
        if method not in _F0_METHODS:
            raise ValueError("f0method 必须是 %s 之一: %r" % (_F0_METHODS, value))
        return method


def phase_vocoder(a, b, fade_out, fade_in):
    """SOLA phase vocoder, same as gui_v1.py; used when profile['use_pv'] is on."""
    torch = _torch()
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result
