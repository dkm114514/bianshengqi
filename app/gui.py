"""Voice changer GUI (module E): model picker, sliders, voice params, acoustic
enhancement, device selection, start/stop/self-check, meters, load line.

Contract: ModelRegistry/VoicePipeline are called lazily by public contract;
VoicePipeline ctor kwargs pass by introspected name; monitor/VoiceGate
fallback shims cover older engine.pipeline builds. VoiceGate is imported
lazily for enrollment only.
profiles.json is written on every control event (plus once on close);
top-level keys are global switches, per-model params live under models[key].
Voice params: index_rate/threhold apply live via set_param; block_time/
crossfade_time/extra_time/n_cpu apply on restart (pipeline rebuild). f0method
is fixed to rmvpe and hidden from the UI.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback

import FreeSimpleGUI as sg

from engine.defaults import DEFAULT_ACTIVE_MODEL, DEFAULT_MODEL_PROFILE, DEFAULT_TOP_CONFIG
from app.runtime import DATA_DIR, log_error
from app.storage import atomic_write_json
from app.version import VERSION

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(str(DATA_DIR), "profiles.json")
SELFCHECK_PATH = os.path.join(PROJECT_ROOT, "selfcheck.py")

HOSTAPI_NAME = "Windows WASAPI"
LEVEL_REFRESH_MS = 100  # Meter refresh period.
SELFCHECK_TIMEOUT_S = 600
MODEL_PLACEHOLDER = "（未发现模型）"
DEVICE_PLACEHOLDER = "（无可用设备）"
CABLE_MISSING_TEXT = "未发现 VB-CABLE，请运行文件夹里的“安装或卸载虚拟声卡.bat”"
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Default device keyword hints (case-insensitive): input / monitor / virtual mic.
# Virtual mic prefers the renamed CABLE render endpoint, then any CABLE entry.
INPUT_HINTS = ("usbaudio",)
MONITOR_HINTS = ("hecate", "realtek")
CABLE_HINTS = ("变声器输出", "cable")

# Fallback params for a missing profiles.json or missing keys.
DEFAULT_PROFILE = dict(DEFAULT_MODEL_PROFILE)

SLIDERS = {
    "-PITCH-": "pitch",
    "-RMS-": "rms_mix_rate",
    "-FORMANT-": "formant",
}
SLIDER_LABELS = {
    "-PITCH-": "-PITCH-VAL-",
    "-RMS-": "-RMS-VAL-",
    "-FORMANT-": "-FORMANT-VAL-",
}
SWITCHES = {
    "-I-NR-": "I_noise_reduce",
    "-O-NR-": "O_noise_reduce",
}
# Hot-swappable (set_param) params visible here, re-pushed on model switch.
LIVE_KEYS = (
    "pitch",
    "formant",
    "rms_mix_rate",
    "index_rate",
    "threhold",
    "I_noise_reduce",
    "O_noise_reduce",
)

#: Voice params: block-time and thread-count combos (same names as upstream RVC).
BLOCK_TIME_CHOICES = (0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
N_CPU_CHOICES = (1, 2, 3, 4)
DEFAULT_N_CPU = 4
BLOCK_TIME_LABELS = tuple("%.2f（%d ms）" % (t, round(t * 1000)) for t in BLOCK_TIME_CHOICES)
BLOCK_TIME_BY_LABEL = dict(zip(BLOCK_TIME_LABELS, BLOCK_TIME_CHOICES))
BLOCK_TIME_LABEL_BY_VALUE = dict(zip(BLOCK_TIME_CHOICES, BLOCK_TIME_LABELS))

#: Advanced sliders: control -> profile key / value label / (lo, hi, step).
ADV_SLIDERS = {
    "-CROSSFADE-": "crossfade_time",
    "-EXTRA-": "extra_time",
    "-THREHOLD-": "threhold",
    "-INDEX-RATE-": "index_rate",
}
ADV_SLIDER_LABELS = {
    "-CROSSFADE-": "-CROSSFADE-VAL-",
    "-EXTRA-": "-EXTRA-VAL-",
    "-THREHOLD-": "-THREHOLD-VAL-",
    "-INDEX-RATE-": "-INDEX-RATE-VAL-",
}
ADV_SLIDER_RANGES = {
    "-CROSSFADE-": (0.01, 0.08, 0.005),
    "-EXTRA-": (0.2, 5.0, 0.1),
    "-THREHOLD-": (-60, -30, 1),
    "-INDEX-RATE-": (0.0, 1.0, 0.01),
}
#: Display names for the status line.
ADV_LABELS = {
    "block_time": "采样块长",
    "crossfade_time": "交叉淡出",
    "extra_time": "额外推理",
    "threhold": "响应阈值",
    "index_rate": "检索占比",
    "n_cpu": "推理线程",
}
#: Advanced params applied live via set_param.
ADV_HOT_KEYS = ("index_rate", "threhold")
#: Advanced params applied on restart (pipeline rebuild).
ADV_RESTART_KEYS = ("block_time", "crossfade_time", "extra_time", "n_cpu")

#: Bundled model assets and private user data are kept separately.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VOICE_PROFILE_PATH = os.path.join(str(DATA_DIR), "voice_profile.npz")

#: Denoise-mode combo (label <-> profile value) and one-line hints.
DENOISE_MODES = (("关", "off"), ("TorchGate", "torchgate"), ("DFN3", "dfn3"))
DENOISE_LABELS = {mode: label for label, mode in DENOISE_MODES}
DENOISE_MODE_BY_LABEL = {label: mode for label, mode in DENOISE_MODES}
DENOISE_HINTS = {
    "off": "关闭降噪",
    "torchgate": "谱减降噪（原 I 降噪）",
    "dfn3": "CPU 深度降噪 · +37ms",
}
DEFAULT_DENOISE_MODE = DEFAULT_TOP_CONFIG["denoise_mode"]

#: Voice-gate default and threshold slider range.
DEFAULT_VOICE_GATE = DEFAULT_TOP_CONFIG["voice_gate"]
DEFAULT_VOICE_GATE_THRESHOLD = DEFAULT_TOP_CONFIG["voice_gate_threshold"]
VOICE_GATE_THRESHOLD_RANGE = (0.0, 0.9)

#: Output AGC default and target-loudness slider range (dBFS).
DEFAULT_AGC = DEFAULT_TOP_CONFIG["agc"]
DEFAULT_AGC_TARGET_DBFS = DEFAULT_TOP_CONFIG["agc_target_dbfs"]
AGC_TARGET_RANGE = (-26.0, -8.0)

#: Enrollment: record seconds / sample rate / block size.
ENROLL_SECONDS = 25
ENROLL_SR = 48000
ENROLL_BLOCK = 4800

#: Verification: record seconds, score offline against the threshold.
VERIFY_SECONDS = 3

#: Top-level acoustic keys of profiles.json (defaults for missing/null).
PROFILE_TOP_DEFAULTS = dict(DEFAULT_TOP_CONFIG)


def _as_bool(value):
    """Loose bool parse: "0"/"false"/"no"/"off" are false."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _as_float(value):
    """Loose float parse; None on missing/invalid input."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def normalize_denoise_mode(value):
    """Clamp a profile value to off/torchgate/dfn3; default torchgate."""
    mode = str(value or "").strip().lower()
    return mode if mode in DENOISE_LABELS else DEFAULT_DENOISE_MODE


def normalize_threshold(value):
    """Clamp the match threshold into [0, 0.9]; default 0.26."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_VOICE_GATE_THRESHOLD
    if not math.isfinite(number):
        return DEFAULT_VOICE_GATE_THRESHOLD
    low, high = VOICE_GATE_THRESHOLD_RANGE
    return round(min(max(number, low), high), 2)


def nearest_choice(value, choices, default):
    """Snap a profile value to the nearest legal choice."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(choices, key=lambda choice: abs(float(choice) - number))


def clamp_value(value, low, high, default):
    """Clamp a value into [low, high]; default on invalid input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, low), high)


def normalize_agc_target(value):
    """Clamp the AGC target into [-26, -8] dBFS; default -13."""
    low, high = AGC_TARGET_RANGE
    return clamp_value(value, low, high, DEFAULT_AGC_TARGET_DBFS)


def signature_params(fn):
    """Parameter names of a callable; None when introspection fails."""
    try:
        return tuple(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


#: Sentinel: voice_gate spec not yet pushed to the current pipeline.
_UNSET = object()


def _insert_project_root():
    """Ensure the engine/app packages are importable."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


# profiles.json
# --------------------------------------------------------------------------- #


def load_profiles(path=PROFILES_PATH):
    """Read profiles.json; return (data, error). Missing/corrupt file yields defaults."""
    data = {"active": DEFAULT_ACTIVE_MODEL, "models": {}}
    for key, value in PROFILE_TOP_DEFAULTS.items():
        data[key] = value
    if not os.path.isfile(path):
        return data, "首次使用，已加载默认参数"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return data, "profiles.json 读取失败: %s" % exc
    if not isinstance(raw, dict):
        return data, "profiles.json 顶层不是对象"
    models = raw.get("models")
    merged = dict(raw)
    for key, value in PROFILE_TOP_DEFAULTS.items():
        if merged.get(key) is None:
            merged[key] = value
    merged["active"] = str(raw.get("active") or DEFAULT_ACTIVE_MODEL)
    merged["models"] = models if isinstance(models, dict) else {}
    merged["denoise_mode"] = normalize_denoise_mode(merged.get("denoise_mode"))
    merged["voice_gate"] = _as_bool(merged.get("voice_gate"))
    merged["voice_gate_threshold"] = normalize_threshold(merged.get("voice_gate_threshold"))
    merged["agc"] = _as_bool(merged.get("agc"))
    merged["agc_target_dbfs"] = normalize_agc_target(merged.get("agc_target_dbfs"))
    for _dev_key in ("input_device", "monitor_device"):
        _dev_value = merged.get(_dev_key)
        merged[_dev_key] = _dev_value if isinstance(_dev_value, str) else ""
    merged["monitor_on"] = _as_bool(merged.get("monitor_on"))
    return merged, ""


def save_profiles(profiles, path=PROFILES_PATH):
    """A failed save leaves the old settings intact and cleans up its own temporary file."""
    atomic_write_json(path, profiles)


# Model / device enumeration
# --------------------------------------------------------------------------- #


def scan_models():
    """Call ModelRegistry.scan(); return (list[{pth,index,display}], error)."""
    _insert_project_root()
    try:
        from engine.models import ModelRegistry
    except Exception as exc:
        return [], "engine.models 不可用: %s" % exc
    scan = getattr(ModelRegistry, "scan", None)
    if scan is None:
        return [], "ModelRegistry.scan 不存在"
    try:
        raw_models = scan()
    except TypeError:
        raw_models = ModelRegistry().scan()
    except Exception as exc:
        return [], "ModelRegistry.scan() 失败: %s" % exc
    models = []
    for m in raw_models or []:
        if not isinstance(m, dict):
            continue
        pth = str(m.get("pth") or "")
        index = str(m.get("index") or "")
        display = str(m.get("display") or "") or os.path.splitext(os.path.basename(pth))[0]
        if not display:
            continue
        models.append({"pth": pth, "index": index, "display": display})
    return models, ""


def list_devices(hostapi_name=HOSTAPI_NAME):
    """Enumerate input/output device names and index maps for a host API."""
    import sounddevice as sd

    devices = [dict(d) for d in sd.query_devices()]
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    names = [h["name"] for h in hostapis]
    if hostapi_name not in names and names:
        hostapi_name = names[0]
    inputs, outputs, in_map, out_map = [], [], {}, {}
    for d in devices:
        if d.get("hostapi_name") != hostapi_name:
            continue
        name = d["name"]
        if d["max_input_channels"] > 0 and name not in in_map:
            inputs.append(name)
            in_map[name] = d["index"]
        if d["max_output_channels"] > 0 and name not in out_map:
            outputs.append(name)
            out_map[name] = d["index"]
    return {
        "hostapi": hostapi_name,
        "input": inputs,
        "output": outputs,
        "input_map": in_map,
        "output_map": out_map,
    }


def pick_default(names, hints):
    """Pick a default device by keyword hints; fall back to the first."""
    for hint in hints:
        hint = hint.lower()
        for name in names:
            if hint in str(name).lower():
                return name
    return names[0] if names else ""


def pick_cable(names):
    """Pick the virtual-mic output by keyword hit only; "" when nothing matches.

    Never falls back to the first device: misrouting would cross wires.
    """
    for hint in CABLE_HINTS:
        hint = hint.lower()
        for name in names:
            if hint in str(name).lower():
                return name
    return ""


# VoicePipeline compat layer
# --------------------------------------------------------------------------- #


def _closure_values(callback):
    """Values captured by a callback closure."""
    values = []
    for cell in getattr(callback, "__closure__", None) or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:
            continue
    return values


def find_monitor_stream(pipeline):
    """Locate the monitor output stream in the pipeline; None when unknown."""
    stream = getattr(pipeline, "monitor_stream", None) or getattr(pipeline, "_monitor_stream", None)
    if stream is not None:
        return stream
    streams = list(getattr(pipeline, "_streams", None) or [])
    if not streams:
        return None
    try:
        import sounddevice as sd

        real = [s for s in streams if isinstance(s, sd.OutputStream)]
    except Exception:
        real = []
    streams = real if len(real) >= 2 else [
        s for s in streams if hasattr(s, "start") and hasattr(s, "abort")
    ]
    if len(streams) < 2:
        return None
    ring = getattr(pipeline, "ring_monitor", None)
    if ring is not None:
        for s in streams:
            if any(v is ring for v in _closure_values(getattr(s, "callback", None))):
                return s
    try:
        from engine.pipeline import resolve_device

        index = resolve_device(pipeline.monitor_device, "output")
        matched = [s for s in streams if getattr(s, "device", None) == index]
        if len(matched) == 1:
            return matched[0]
    except Exception:
        pass
    return streams[-1]


# Main window
# --------------------------------------------------------------------------- #


class App:
    """Main window; event errors pop up and continue, stderr in smoke mode."""

    def __init__(self, profiles=None, profiles_error="", profiles_path=PROFILES_PATH, smoke=False):
        self.profiles_path = profiles_path
        self.smoke = bool(smoke) or bool(os.environ.get("BSMOKE"))
        if profiles is None:
            profiles, profiles_error = load_profiles(profiles_path)
        self.profiles = profiles if isinstance(profiles, dict) else {}
        self.profiles.setdefault("active", "")
        if not isinstance(self.profiles.get("models"), dict):
            self.profiles["models"] = {}
        self.profiles_error = profiles_error or ""
        self.dirty = False
        self.pipeline = None
        self.monitor_on = _as_bool(self.profiles.get("monitor_on"))
        self._monitor_native = False
        self._shutdown_done = False
        self._level_lock = threading.Lock()
        self._levels = (None, None)
        self._meter_texts = ("", "")
        self._load_text = ""
        self._load_overload = False
        self._devices_error = ""

        # Acoustic enhancement: denoise mode / voice gate (top-level keys).
        self.denoise_mode = normalize_denoise_mode(self.profiles.get("denoise_mode"))
        self.voice_gate_threshold = normalize_threshold(self.profiles.get("voice_gate_threshold"))
        self.voice_profile_ok = os.path.isfile(VOICE_PROFILE_PATH)
        stored_gate = _as_bool(self.profiles.get("voice_gate"))
        self.voice_gate_on = bool(stored_gate and self.voice_profile_ok)
        if stored_gate and not self.voice_profile_ok:
            self._set_top_value("voice_gate", False)
        self._enrolling = False
        self._verifying = False
        self._match_lock = threading.Lock()
        self._match_score = None
        self._score_text = ""
        self._gate_spec_sent = _UNSET

        # Output AGC (top-level keys).
        self.agc_on = _as_bool(self.profiles.get("agc"))
        self.agc_target_dbfs = normalize_agc_target(self.profiles.get("agc_target_dbfs"))
        self._agc_gain_text = ""

        self.models, self.models_error = scan_models()
        self.model_map = {}
        for m in self.models:
            self.model_map[m["display"]] = m
        self.devices = self._enumerate_devices()

        init = self._initial_state()
        self.current_model = init["model"]
        self.cable_device = init["cable_dev"]
        # Slider has no get(); this dict is the value source of truth.
        self.slider_values = {
            "-PITCH-": init["profile"]["pitch"],
            "-RMS-": init["profile"]["rms_mix_rate"],
            "-FORMANT-": init["profile"]["formant"],
            "-VG-THR-": init["voice_gate_threshold"],
            "-AGC-TARGET-": init["agc_target_dbfs"],
            "-CROSSFADE-": self._clamp_adv("-CROSSFADE-", init["profile"]["crossfade_time"]),
            "-EXTRA-": self._clamp_adv("-EXTRA-", init["profile"]["extra_time"]),
            "-THREHOLD-": self._clamp_adv("-THREHOLD-", init["profile"]["threhold"]),
            "-INDEX-RATE-": self._clamp_adv("-INDEX-RATE-", init["profile"]["index_rate"]),
        }
        self.window = sg.Window("变声器 · v%s" % VERSION, self._build_layout(init), finalize=True, resizable=False)
        try:
            self.window["-MONITOR-"].update(value=self.monitor_on)
        except Exception:
            pass
        self._update_slider_labels()
        self._refresh_meter(force=True)
        self._refresh_load(force=True)
        self._refresh_voice_profile_state()
        self._refresh_voice_score(force=True)
        self._sync_input_denoise_ui()
        if self.profiles_error:
            self._set_status(self.profiles_error)
        elif self.models_error:
            self._set_status(self.models_error)
        elif self._devices_error:
            self._set_status(self._devices_error)
        else:
            self._set_status("已停止")

    # Init helpers

    def _enumerate_devices(self):
        try:
            return list_devices()
        except Exception as exc:
            self._devices_error = "设备枚举失败: %s" % exc
            return {"hostapi": HOSTAPI_NAME, "input": [], "output": [], "input_map": {}, "output_map": {}}

    def _resolve_key(self, display):
        """Map a model display name to its profiles["models"] key; None if unknown."""
        if not display:
            return None
        models = self.profiles["models"]
        if isinstance(models.get(display), dict):
            return display
        model = self.model_map.get(display)
        if not model:
            return None
        pth = model.get("pth") or ""
        if pth:
            target = os.path.normcase(os.path.abspath(pth))
            for key, entry in models.items():
                if isinstance(entry, dict) and entry.get("pth"):
                    if os.path.normcase(os.path.abspath(str(entry["pth"]))) == target:
                        return key
        stem = os.path.splitext(os.path.basename(pth))[0]
        if stem and isinstance(models.get(stem), dict):
            return stem
        return None

    @staticmethod
    def _sanitize_profile(merged):
        """Heal illegal profile values back into legal ranges."""
        merged["pitch"] = int(round(clamp_value(merged.get("pitch"), -24, 24,
                                                DEFAULT_PROFILE["pitch"])))
        merged["rms_mix_rate"] = round(clamp_value(merged.get("rms_mix_rate"), 0.0, 1.0,
                                                   DEFAULT_PROFILE["rms_mix_rate"]), 3)
        merged["formant"] = round(clamp_value(merged.get("formant"), -1.0, 1.0,
                                              DEFAULT_PROFILE["formant"]), 3)
        merged["index_rate"] = round(clamp_value(merged.get("index_rate"), 0.0, 1.0,
                                                 DEFAULT_PROFILE["index_rate"]), 3)
        merged["threhold"] = int(round(clamp_value(merged.get("threhold"), -60, -30,
                                                   DEFAULT_PROFILE["threhold"])))
        merged["crossfade_time"] = clamp_value(merged.get("crossfade_time"), 0.01, 0.08,
                                               DEFAULT_PROFILE["crossfade_time"])
        merged["extra_time"] = clamp_value(merged.get("extra_time"), 0.2, 5.0,
                                           DEFAULT_PROFILE["extra_time"])
        merged["block_time"] = nearest_choice(merged.get("block_time"), BLOCK_TIME_CHOICES,
                                              DEFAULT_PROFILE["block_time"])
        merged["n_cpu"] = nearest_choice(merged.get("n_cpu"), N_CPU_CHOICES, DEFAULT_N_CPU)
        merged["I_noise_reduce"] = _as_bool(merged.get("I_noise_reduce"))
        merged["O_noise_reduce"] = _as_bool(merged.get("O_noise_reduce"))
        if not merged.get("f0method"):
            merged["f0method"] = DEFAULT_PROFILE["f0method"]
        return merged

    def _profile_view(self, display):
        """Return (key, merged params) for a model, creating the entry if needed."""
        models = self.profiles["models"]
        key = self._resolve_key(display)
        if key is None:
            key = display or ""
        entry = models.get(key) if key else None
        if not isinstance(entry, dict):
            entry = {}
            if key:
                models[key] = entry
                self._mark_dirty()
        model = self.model_map.get(display) or {}
        if not entry.get("pth") and model.get("pth"):
            entry["pth"] = model["pth"]
            self._mark_dirty()
        if not entry.get("index") and model.get("index"):
            entry["index"] = model["index"]
            self._mark_dirty()
        merged = dict(DEFAULT_PROFILE)
        for k, v in entry.items():
            if v is not None:
                merged[k] = v
        # The folder can move between drives; resolved model paths always come from this scan.
        if model.get("pth"):
            merged["pth"] = model["pth"]
            merged["index"] = model.get("index") or ""
        return key, self._sanitize_profile(merged)

    def _set_profile_value(self, display, value_key, value):
        """Write one param into the profile and flush; only on change."""
        key, _profile = self._profile_view(display)
        if not key:
            return
        models = self.profiles["models"]
        entry = models.get(key)
        if not isinstance(entry, dict):
            entry = {}
            models[key] = entry
        if entry.get(value_key) != value:
            entry[value_key] = value
            self._mark_dirty()

    @staticmethod
    def _recall_device(saved, pool, hints):
        """Restore the saved device when still listed, else pick by hints."""
        pool = list(pool or [])
        if saved and saved in pool:
            return saved
        return pick_default(pool, hints) if pool else ""

    def _initial_state(self):
        displays = [m["display"] for m in self.models]
        selected = displays[0] if displays else MODEL_PLACEHOLDER
        active_key = str(self.profiles.get("active") or "")
        if active_key:
            for d in displays:
                if self._resolve_key(d) == active_key or d == active_key:
                    selected = d
                    break
        if displays:
            _key, profile = self._profile_view(selected)
        else:
            profile = dict(DEFAULT_PROFILE)
        dev = self.devices
        cable_dev = pick_cable(dev["output"]) if dev["output"] else ""
        return {
            "model": selected,
            "profile": profile,
            "in_values": list(dev["input"]) or [DEVICE_PLACEHOLDER],
            "in_dev": self._recall_device(self.profiles.get("input_device"), dev["input"], INPUT_HINTS) or DEVICE_PLACEHOLDER,
            "out_values": list(dev["output"]) or [DEVICE_PLACEHOLDER],
            "mon_dev": self._recall_device(self.profiles.get("monitor_device"), dev["output"], MONITOR_HINTS) or DEVICE_PLACEHOLDER,
            "cable_dev": cable_dev,
            "cable_text": cable_dev or CABLE_MISSING_TEXT,
            "denoise_mode": self.denoise_mode,
            "denoise_label": DENOISE_LABELS.get(self.denoise_mode, DENOISE_LABELS[DEFAULT_DENOISE_MODE]),
            "denoise_hint": DENOISE_HINTS.get(self.denoise_mode, ""),
            "agc": self.agc_on,
            "agc_target_dbfs": self.agc_target_dbfs,
            "agc_hint": "按语音电平自动补 / 压增益" if self.agc_on else "关：输出直通",
            "voice_gate": self.voice_gate_on,
            "voice_gate_threshold": self.voice_gate_threshold,
            "voice_profile_ok": self.voice_profile_ok,
            "voice_hint": "声纹档案已就绪" if self.voice_profile_ok else "未注册声纹，先点「注册声纹」",
        }

    def _build_layout(self, init):
        # Two-column layout; control keys, events and defaults unchanged.
        label = (8, 1)
        slider_size = (22, 15)
        combo_size = (44, 1)
        hint_size = (26, 2)
        hint_font = ("Microsoft YaHei UI", 8)
        hint_color = "#808080"
        adv = init["profile"]
        block_label = BLOCK_TIME_LABEL_BY_VALUE[
            nearest_choice(adv.get("block_time"), BLOCK_TIME_CHOICES,
                           DEFAULT_PROFILE["block_time"])
        ]
        n_cpu_label = str(nearest_choice(adv.get("n_cpu"), N_CPU_CHOICES, DEFAULT_N_CPU))
        left_col = sg.Column(
            [
                [
                    sg.Frame(
                        "模型与参数",
                        [
                            [
                                sg.Text("模型", size=label),
                                sg.Combo(
                                    list(self.model_map) or [MODEL_PLACEHOLDER],
                                    default_value=init["model"],
                                    key="-MODEL-",
                                    readonly=True,
                                    enable_events=True,
                                    size=(36, 1),
                                ),
                            ],
                            [
                                sg.Text("音调", size=label),
                                sg.Slider(range=(-24, 24), resolution=1, orientation="h",
                                          default_value=init["profile"]["pitch"], key="-PITCH-",
                                          enable_events=True, size=slider_size,
                                          disable_number_display=True),
                                sg.Text("", key="-PITCH-VAL-", size=(7, 1)),
                            ],
                            [
                                sg.Text("响度混合", size=label),
                                sg.Slider(range=(0.0, 1.0), resolution=0.01, orientation="h",
                                          default_value=init["profile"]["rms_mix_rate"], key="-RMS-",
                                          enable_events=True, size=slider_size,
                                          disable_number_display=True),
                                sg.Text("", key="-RMS-VAL-", size=(7, 1)),
                            ],
                            [
                                sg.Text("性别因子", size=label),
                                sg.Slider(range=(-1.0, 1.0), resolution=0.01, orientation="h",
                                          default_value=init["profile"]["formant"], key="-FORMANT-",
                                          enable_events=True, size=slider_size,
                                          disable_number_display=True),
                                sg.Text("", key="-FORMANT-VAL-", size=(7, 1)),
                            ],
                            [
                                sg.Text("降噪", size=label),
                                sg.Checkbox("输入降噪 (I)", key="-I-NR-", enable_events=True,
                                            default=bool(init["profile"]["I_noise_reduce"])),
                                sg.Checkbox("输出降噪 (O)", key="-O-NR-", enable_events=True,
                                            default=bool(init["profile"]["O_noise_reduce"])),
                            ],
                        ],
                        expand_x=True,
                    )
                ],
                [
                    sg.Frame(
                        "变声参数",
                        [
                            [
                                sg.Text("采样块长", size=label),
                                sg.Combo(list(BLOCK_TIME_LABELS), default_value=block_label,
                                         key="-BLOCK-TIME-", readonly=True, enable_events=True,
                                         size=(14, 1),
                                         tooltip="每块送进 RVC 的音频长度：块越大延迟越高但越扛发热"),
                                sg.Text("", key="-BLOCK-TIME-VAL-", size=(9, 1)),
                                sg.Text("块越大延迟越高但越扛发热 · 重启管线生效", key="-BLOCK-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                            [
                                sg.Text("交叉淡出", size=label),
                                sg.Slider(range=ADV_SLIDER_RANGES["-CROSSFADE-"][:2],
                                          resolution=ADV_SLIDER_RANGES["-CROSSFADE-"][2], orientation="h",
                                          default_value=self.slider_values["-CROSSFADE-"],
                                          key="-CROSSFADE-", enable_events=True, size=slider_size,
                                          disable_number_display=True,
                                          tooltip="相邻推理块拼接时的交叉淡化时长（SOLA）"),
                                sg.Text("", key="-CROSSFADE-VAL-", size=(9, 1)),
                                sg.Text("SOLA 拼接淡化 · 重启管线生效", key="-CROSSFADE-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                            [
                                sg.Text("额外推理", size=label),
                                sg.Slider(range=ADV_SLIDER_RANGES["-EXTRA-"][:2],
                                          resolution=ADV_SLIDER_RANGES["-EXTRA-"][2], orientation="h",
                                          default_value=self.slider_values["-EXTRA-"],
                                          key="-EXTRA-", enable_events=True, size=slider_size,
                                          disable_number_display=True,
                                          tooltip="迭代上下文（RVC 的 extra_time）：给音高提取留的历史窗口；"
                                                  "越大越稳但启动预热越久，底延迟也更大"),
                                sg.Text("", key="-EXTRA-VAL-", size=(9, 1)),
                                sg.Text("迭代上下文：越大越稳但启动预热越久 · 重启管线生效", key="-EXTRA-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                            [
                                sg.Text("响应阈值", size=label),
                                sg.Slider(range=ADV_SLIDER_RANGES["-THREHOLD-"][:2],
                                          resolution=ADV_SLIDER_RANGES["-THREHOLD-"][2], orientation="h",
                                          default_value=self.slider_values["-THREHOLD-"],
                                          key="-THREHOLD-", enable_events=True, size=slider_size,
                                          disable_number_display=True,
                                          tooltip="低于该电平（dB）的块当静音、不送推理；-60 = 门限关闭"),
                                sg.Text("", key="-THREHOLD-VAL-", size=(9, 1)),
                                sg.Text("低于它当静音 · 即时生效", key="-THREHOLD-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                            [
                                sg.Text("检索占比", size=label),
                                sg.Slider(range=ADV_SLIDER_RANGES["-INDEX-RATE-"][:2],
                                          resolution=ADV_SLIDER_RANGES["-INDEX-RATE-"][2], orientation="h",
                                          default_value=self.slider_values["-INDEX-RATE-"],
                                          key="-INDEX-RATE-", enable_events=True, size=slider_size,
                                          disable_number_display=True,
                                          tooltip="特征检索占比：0 = 不检索（延迟最低），越大越贴近索引音色"),
                                sg.Text("", key="-INDEX-RATE-VAL-", size=(9, 1)),
                                sg.Text("索引特征占比 · 即时生效", key="-INDEX-RATE-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                            [
                                sg.Text("推理线程", size=label),
                                sg.Combo([str(v) for v in N_CPU_CHOICES], default_value=n_cpu_label,
                                         key="-N-CPU-", readonly=True, enable_events=True,
                                         size=(14, 1),
                                         tooltip="CPU 侧推理线程数（1~4）：机器忙 / 发烫就调小，CPU 空闲可调大"),
                                sg.Text("", key="-N-CPU-VAL-", size=(9, 1)),
                                sg.Text("CPU 线程数 · 重启管线生效", key="-N-CPU-HINT-",
                                        size=hint_size, font=hint_font, text_color=hint_color),
                            ],
                        ],
                        expand_x=True,
                    )
                ],
            ],
            vertical_alignment="top",
            pad=(0, 0),
        )
        right_col = sg.Column(
            [
                [
                    sg.Frame(
                        "声学增强",
                        [
                            [
                                sg.Text("降噪模式", size=label),
                                sg.Combo(list(DENOISE_MODE_BY_LABEL), default_value=init["denoise_label"],
                                         key="-DENOISE-MODE-", readonly=True, enable_events=True,
                                         size=(14, 1),
                                         tooltip="关 = 不降噪；TorchGate = 原输入降噪；"
                                                 "DFN3 = CPU 深度降噪（实测约 +37ms 延迟）"),
                                sg.Text(init["denoise_hint"], key="-DENOISE-HINT-", size=(24, 1)),
                            ],
                            [
                                sg.Text("声纹门", size=label),
                                sg.Checkbox("只变本人的声音", key="-VOICE-GATE-", enable_events=True,
                                            default=bool(init["voice_gate"]),
                                            disabled=not init["voice_profile_ok"],
                                            tooltip="打开后先判声纹再送 RVC：不是本人的声音不参与变声（输出静音）"),
                                sg.Button("注册声纹", key="-ENROLL-",
                                          tooltip="从当前输入设备录 %d 秒本人语音，存 models/voice_profile.npz"
                                                  % ENROLL_SECONDS),
                                sg.Button("删除声纹", key="-VOICE-DELETE-",
                                          disabled=not init["voice_profile_ok"],
                                          tooltip="先自动备份，再删除当前声纹；声纹门自动关闭"),
                            ],
                            [
                                sg.Text("声纹文件", size=label),
                                sg.Button("导入声纹", key="-VOICE-IMPORT-",
                                          tooltip="导入 .npz / .json，无需重新录制；覆盖前自动备份"),
                                sg.Button("导出声纹", key="-VOICE-EXPORT-",
                                          disabled=not init["voice_profile_ok"],
                                          tooltip="保存当前声纹到 .npz / .json，换机或恢复时可直接导入"),
                            ],
                            [
                                sg.Text("档案状态", size=label),
                                sg.Text("", key="-VOICE-PROFILE-", size=(46, 1)),
                            ],
                            [
                                sg.Text("匹配阈值", size=label),
                                sg.Slider(range=VOICE_GATE_THRESHOLD_RANGE, resolution=0.01, orientation="h",
                                          default_value=init["voice_gate_threshold"], key="-VG-THR-",
                                          enable_events=True, size=slider_size, disable_number_display=True,
                                          disabled=not init["voice_profile_ok"],
                                          tooltip="匹配分数 ≥ 阈值 → 放行，否则静音"),
                                sg.Text("", key="-VG-THR-VAL-", size=(7, 1)),
                            ],
                            [
                                sg.Text("匹配分数", size=label),
                                sg.Text("--", key="-VOICE-SCORE-", size=(22, 1),
                                        font=("Consolas", 11), text_color="#808080"),
                                sg.Text(init["voice_hint"], key="-VOICE-HINT-", size=(24, 1)),
                            ],
                            [
                                sg.Text("声纹验证", size=label),
                                sg.Button("验证", key="-VERIFY-",
                                          disabled=not init["voice_profile_ok"],
                                          tooltip="从当前输入设备录 %d 秒，离线算一次匹配分数并与阈值比较"
                                                  % VERIFY_SECONDS),
                                sg.Text("", key="-VERIFY-TEXT-", size=(38, 1)),
                            ],
                            [
                                sg.ProgressBar(100, orientation="h", size=(30, 10), key="-ENROLL-BAR-",
                                               visible=False),
                                sg.Text("", key="-ENROLL-TEXT-", size=(24, 1)),
                            ],
                            [
                                sg.Text("自动增益", size=label),
                                sg.Checkbox("稳定输出响度", key="-AGC-", enable_events=True,
                                            default=bool(init["agc"]),
                                            tooltip="按输出语音电平自动补 / 压增益，响度稳在目标值（关 = 直通）"),
                                sg.Text(init["agc_hint"], key="-AGC-HINT-", size=(24, 1)),
                            ],
                            [
                                sg.Text("目标响度", size=label),
                                sg.Slider(range=AGC_TARGET_RANGE, resolution=1.0, orientation="h",
                                          default_value=init["agc_target_dbfs"], key="-AGC-TARGET-",
                                          enable_events=True, size=slider_size,
                                          disable_number_display=True,
                                          disabled=not init["agc"],
                                          tooltip="语音段的输出电平目标（dBFS，-26 ~ -8）"),
                                sg.Text("", key="-AGC-TARGET-VAL-", size=(7, 1)),
                            ],
                            [
                                sg.Text("当前增益", size=label),
                                sg.Text("--", key="-AGC-GAIN-", size=(22, 1),
                                        font=("Consolas", 11), text_color="#808080"),
                            ],
                        ],
                        expand_x=True,
                    )
                ],
                [
                    sg.Frame(
                        "设备（%s）" % self.devices["hostapi"],
                        [
                            [sg.Text("输入", size=label),
                             sg.Combo(init["in_values"], default_value=init["in_dev"], key="-IN-DEV-",
                                      readonly=True, enable_events=True, size=combo_size),
                             sg.Button("一起刷新", key="-DEV-REFRESH-",
                                       tooltip="重新打开输入、虚拟麦和监听并清除旧声音状态；保留当前模型参数")],
                            [sg.Text("监听", size=label),
                             sg.Combo(init["out_values"], default_value=init["mon_dev"], key="-MON-DEV-",
                                      readonly=True, enable_events=True, size=combo_size)],
                            [sg.Text("虚拟麦", size=label),
                             sg.Text(init["cable_text"], key="-CABLE-DEV-", size=combo_size)],
                        ],
                        expand_x=True,
                    )
                ],
            ],
            vertical_alignment="top",
            pad=(0, 0),
        )
        return [
            [sg.Text("变声器 · 实时语音转换", font=("Microsoft YaHei UI", 13, "bold"))],
            [left_col, right_col],
            [
                sg.Button("启动", key="-START-", button_color=("white", "#2E7D32")),
                sg.Button("停止", key="-STOP-", disabled=True),
                sg.Button("自检", key="-SELFCHECK-"),
                sg.Checkbox("监听", key="-MONITOR-", default=False, enable_events=True,
                            tooltip="打开后把变声结果同时送到监听设备（默认关）"),
                sg.Text("开监听请戴耳机，否则易啸叫", key="-MONITOR-HINT-",
                        font=("Microsoft YaHei UI", 8), text_color="#808080"),
            ],
            [sg.Text("", key="-STATUS-", size=(70, 1))],
            [sg.Text("负载: 未运行", key="-LOAD-", size=(70, 1))],
            [sg.Text("输入电平:      -- dB", key="-LEVEL-IN-", size=(26, 1), font=("Consolas", 11)),
             sg.Text("输出电平:      -- dB", key="-LEVEL-OUT-", size=(26, 1), font=("Consolas", 11))],
        ]

    # State / meters

    def _set_status(self, text):
        try:
            self.window["-STATUS-"].update(str(text)[:160])
        except Exception:
            pass

    def _set_running_ui(self, running):
        """Toggle start/stop buttons; device combos stay enabled."""
        try:
            self.window["-START-"].update(disabled=running)
            self.window["-STOP-"].update(disabled=not running)
        except Exception:
            pass

    def _on_level(self, in_db, out_db):
        """on_level callback, possibly from the audio thread; stores values only."""
        try:
            in_db = None if in_db is None else float(in_db)
            out_db = None if out_db is None else float(out_db)
        except (TypeError, ValueError):
            in_db = out_db = None
        with self._level_lock:
            self._levels = (in_db, out_db)

    @staticmethod
    def _fmt_db(value):
        if value is None:
            return "--"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "--"
        if not math.isfinite(value):
            return "--"
        return "%.1f" % value

    def _refresh_meter(self, force=False):
        with self._level_lock:
            in_db, out_db = self._levels
        texts = (
            "输入电平: %7s dB" % self._fmt_db(in_db),
            "输出电平: %7s dB" % self._fmt_db(out_db),
        )
        try:
            if force or texts[0] != self._meter_texts[0]:
                self.window["-LEVEL-IN-"].update(texts[0])
            if force or texts[1] != self._meter_texts[1]:
                self.window["-LEVEL-OUT-"].update(texts[1])
        except Exception:
            return
        self._meter_texts = texts

    def _reset_meter(self):
        """Clear level state and repaint the meters."""
        with self._level_lock:
            self._levels = (None, None)
        self._refresh_meter(force=True)

    # Acoustic enhancement: read / refresh

    def _flush_profiles(self):
        """Flush dirty profiles to disk; skipped in smoke mode."""
        if not self.dirty or self.smoke:
            return
        try:
            save_profiles(self.profiles, self.profiles_path)
        except Exception:
            log_error("[profiles] 写盘失败，保留旧配置，稍后重试: %s" % traceback.format_exc())
            return
        self.dirty = False

    def _mark_dirty(self):
        """Mark profiles dirty and flush."""
        self.dirty = True
        self._flush_profiles()

    def _set_top_value(self, key, value):
        """Write one top-level global key and flush on change."""
        if self.profiles.get(key) != value:
            self.profiles[key] = value
            self._mark_dirty()

    def _threshold_value(self):
        """Current threshold slider value; default only on None."""
        value = self.slider_values.get("-VG-THR-")
        return float(DEFAULT_VOICE_GATE_THRESHOLD if value is None else value)

    @staticmethod
    def _coerce_score(value):
        """Fold float/0-d-array/queue scores into float; None when unreadable."""
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        except Exception:
            return None
        try:
            return float(value[-1])
        except Exception:
            return None

    def _on_match(self, *args):
        """Match-score callback; stores the newest readable score."""
        for arg in reversed(args):
            score = self._coerce_score(arg)
            if score is not None:
                with self._match_lock:
                    self._match_score = score
                return

    def _poll_match_score(self):
        """Read the newest match score from the pipeline; None when unreadable."""
        pipeline = self.pipeline
        if pipeline is None:
            return None
        stats = getattr(pipeline, "voice_stats", None)
        if isinstance(stats, dict):
            for key in ("score", "match_score", "voice_score"):
                score = self._coerce_score(stats.get(key))
                if score is not None:
                    return score
        for name in ("match_score", "voice_match_score", "last_match_score", "voice_score",
                     "voice_gate_score", "speaker_score"):
            score = self._coerce_score(getattr(pipeline, name, None))
            if score is not None:
                return score
        for name in ("get_match_score", "get_voice_match_score", "get_voice_score"):
            fn = getattr(pipeline, name, None)
            if callable(fn):
                try:
                    score = self._coerce_score(fn())
                except Exception:
                    score = None
                if score is not None:
                    return score
        for owner_name in ("voice_gate", "_voice_gate", "gate"):
            gate = getattr(pipeline, owner_name, None)
            if gate is None or isinstance(gate, bool):
                continue
            for name in ("last_score", "score", "last_prob", "prob",
                         "last_similarity", "similarity"):
                score = self._coerce_score(getattr(gate, name, None))
                if score is not None:
                    return score
        return None

    def _refresh_voice_score(self, force=False):
        """Repaint the match-score text; warn when the gate is on but missing."""
        pipeline = self.pipeline
        score = self._poll_match_score()
        with self._match_lock:
            if score is None:
                score = self._match_score
            else:
                self._match_score = score
        gate_missing = pipeline is not None and getattr(pipeline, "voice_gate", None) is None
        if self.voice_gate_on and gate_missing:
            text, color = "声纹门未生效", "#C62828"
        elif score is None:
            text, color = "--", "#808080"
        else:
            passed = score >= self._threshold_value()
            if not self.voice_gate_on:
                mark = "声纹门关"
            else:
                mark = "通过" if passed else "拦截"
            text = "%5.2f  %s" % (score, mark)
            color = "#2E7D32" if passed else "#C62828"
        if force or text != self._score_text:
            try:
                self.window["-VOICE-SCORE-"].update(text, text_color=color)
            except Exception:
                return
            self._score_text = text

    @staticmethod
    def _voice_profile_info():
        """Voice profile status: (exists, segments, saved time from mtime)."""
        if not os.path.isfile(VOICE_PROFILE_PATH):
            return False, 0, ""
        segments = 0
        try:
            import numpy as np

            with np.load(VOICE_PROFILE_PATH) as data:
                if "num_segments" in data.files:
                    segments = int(data["num_segments"])
        except Exception:
            segments = 0
        try:
            saved = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(os.path.getmtime(VOICE_PROFILE_PATH)))
        except OSError:
            saved = ""
        return True, segments, saved

    def _refresh_voice_profile_state(self):
        """Sync gate controls and the profile row with the npz on disk."""
        exists, segments, saved = self._voice_profile_info()
        self.voice_profile_ok = exists
        hint = "声纹档案已就绪" if exists else "未注册声纹，先点「注册声纹」"
        if exists:
            profile_text = "已注册 · 累计 %d 段 · 保存 %s" % (segments, saved) if saved \
                else "已注册 · 累计 %d 段" % segments
        else:
            profile_text = "未注册"
        try:
            self.window["-VOICE-GATE-"].update(disabled=not exists)
            self.window["-VG-THR-"].update(disabled=not exists)
            self.window["-VOICE-HINT-"].update(hint)
            self.window["-VOICE-PROFILE-"].update(profile_text)
            self.window["-VOICE-DELETE-"].update(disabled=not exists)
            self.window["-VOICE-EXPORT-"].update(disabled=not exists)
            if not self._verifying:
                self.window["-VERIFY-"].update(disabled=not exists)
        except Exception:
            pass
        return exists

    # Events

    def _handle_event(self, event, values):
        if event in (None, sg.WINDOW_CLOSED):
            return
        if event == sg.TIMEOUT_EVENT:
            self._refresh_meter()
            self._refresh_voice_score()
            self._refresh_agc_gain()
            self._refresh_load()
        elif event == "-MODEL-":
            self._on_model_change(values.get("-MODEL-") or "")
        elif event in SLIDERS:
            self._on_slider(event, values.get(event))
        elif event in ADV_SLIDERS:
            self._on_adv_slider(event, values.get(event))
        elif event in ("-BLOCK-TIME-", "-N-CPU-"):
            self._on_adv_combo(event)
        elif event in SWITCHES:
            self._on_switch(event, bool(values.get(event)))
        elif event == "-MONITOR-":
            self._on_monitor_switch(bool(values.get(event)))
        elif event in ("-IN-DEV-", "-MON-DEV-"):
            self._on_device_combo(values)
        elif event == "-DEV-REFRESH-":
            self._on_device_refresh()
        elif event == "-START-":
            self._on_start(values)
        elif event == "-STOP-":
            self._on_stop()
        elif event == "-SELFCHECK-":
            self._on_selfcheck()
        elif event == "-SELFCHECK-DONE-":
            self._on_selfcheck_done(values.get(event) or {})
        elif event == "-DENOISE-MODE-":
            self._on_denoise_mode(values.get(event))
        elif event == "-VOICE-GATE-":
            self._on_voice_gate(values.get(event))
        elif event == "-VG-THR-":
            self._on_threshold(values.get(event))
        elif event == "-AGC-":
            self._on_agc(values.get(event))
        elif event == "-AGC-TARGET-":
            self._on_agc_target(values.get(event))
        elif event == "-ENROLL-":
            self._on_enroll()
        elif event == "-ENROLL-PROGRESS-":
            self._on_enroll_progress(values.get(event))
        elif event == "-ENROLL-DONE-":
            self._on_enroll_done(values.get(event) or {})
        elif event == "-VOICE-DELETE-":
            self._on_voice_delete()
        elif event == "-VOICE-IMPORT-":
            self._on_voice_import()
        elif event == "-VOICE-EXPORT-":
            self._on_voice_export()
        elif event == "-VERIFY-":
            self._on_verify()
        elif event == "-VERIFY-DONE-":
            self._on_verify_done(values.get(event) or {})

    @staticmethod
    def _format_slider_value(profile_key, value):
        if profile_key == "pitch":
            return str(int(round(float(value))))
        return "%g" % round(float(value), 2)

    @staticmethod
    def _format_adv_value(profile_key, value):
        """Display text for advanced params (ms/s, dB, thread count)."""
        if profile_key == "threhold":
            return "%d dB" % int(round(float(value)))
        if profile_key == "agc_target_dbfs":
            return "%.0f dB" % round(float(value))
        if profile_key == "n_cpu":
            return "%d 线程" % int(round(float(value)))
        if profile_key == "extra_time":
            return "%.1f s" % float(value)
        if profile_key in ("crossfade_time", "block_time"):
            return "%.0f ms" % (float(value) * 1000.0)
        return "%.2f" % float(value)

    @staticmethod
    def _clamp_adv(slider_key, value):
        """Clamp a profile value into an advanced-slider range."""
        low, high, _step = ADV_SLIDER_RANGES[slider_key]
        return clamp_value(value, low, high, DEFAULT_PROFILE[ADV_SLIDERS[slider_key]])

    def _update_slider_labels(self):
        for slider_key, profile_key in SLIDERS.items():
            value = self.slider_values.get(slider_key)
            if value is None:
                continue
            try:
                text = self._format_slider_value(profile_key, value)
            except (TypeError, ValueError):
                continue
            try:
                self.window[SLIDER_LABELS[slider_key]].update(text)
            except Exception:
                pass
        for slider_key, profile_key in ADV_SLIDERS.items():
            value = self.slider_values.get(slider_key)
            if value is None:
                continue
            try:
                text = self._format_adv_value(profile_key, value)
            except (TypeError, ValueError):
                continue
            try:
                self.window[ADV_SLIDER_LABELS[slider_key]].update(text)
            except Exception:
                pass
        value = self.slider_values.get("-VG-THR-")
        if value is not None:
            try:
                self.window["-VG-THR-VAL-"].update(
                    self._format_slider_value("voice_gate_threshold", value)
                )
            except Exception:
                pass
        value = self.slider_values.get("-AGC-TARGET-")
        if value is not None:
            try:
                self.window["-AGC-TARGET-VAL-"].update(
                    self._format_adv_value("agc_target_dbfs", value)
                )
            except Exception:
                pass
        try:
            self.window["-BLOCK-TIME-VAL-"].update(
                self._format_adv_value("block_time", self._block_time_value())
            )
            self.window["-N-CPU-VAL-"].update(
                self._format_adv_value("n_cpu", self._n_cpu_value())
            )
        except Exception:
            pass

    def _block_time_value(self):
        """Current block-time combo value in seconds."""
        try:
            label = str(self.window["-BLOCK-TIME-"].get() or "")
        except Exception:
            return DEFAULT_PROFILE["block_time"]
        return BLOCK_TIME_BY_LABEL.get(label, DEFAULT_PROFILE["block_time"])

    def _n_cpu_value(self):
        """Current inference-thread combo value."""
        try:
            raw = self.window["-N-CPU-"].get()
        except Exception:
            return DEFAULT_N_CPU
        try:
            value = int(round(float(raw)))
        except (TypeError, ValueError):
            return DEFAULT_N_CPU
        return nearest_choice(value, N_CPU_CHOICES, DEFAULT_N_CPU)

    def _push_adv_hot(self, name, value):
        """Push a hot param to the pipeline; return keys needing restart."""
        pipeline = self.pipeline
        if pipeline is None:
            return []
        try:
            pipeline.set_param(name, value)
        except KeyError:
            return [name]
        except Exception:
            print("[pipeline] set_param(%r) 失败: %s" % (name, traceback.format_exc()),
                  file=sys.stderr)
            return [name]
        return []

    def _on_adv_slider(self, event, raw):
        """Voice-param sliders: hot keys apply live, the rest only persist."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return
        name = ADV_SLIDERS[event]
        value = int(round(value)) if name == "threhold" else round(value, 3)
        self.slider_values[event] = value
        self._set_profile_value(self.current_model, name, value)
        try:
            self.window[ADV_SLIDER_LABELS[event]].update(self._format_adv_value(name, value))
        except Exception:
            pass
        text = "%s %s" % (ADV_LABELS[name], self._format_adv_value(name, value))
        if name in ADV_HOT_KEYS:
            pending = self._push_adv_hot(name, value)
            text += "（%s）" % ("重启管线后生效" if pending else "即时生效")
        else:
            text += "（重启管线生效）"
        self._set_status(text)

    def _on_adv_combo(self, event):
        """Block-time/thread combos: persist only, apply on restart."""
        if event == "-BLOCK-TIME-":
            name, value = "block_time", self._block_time_value()
        else:
            name, value = "n_cpu", self._n_cpu_value()
        self._set_profile_value(self.current_model, name, value)
        self._update_slider_labels()
        self._set_status("%s %s（重启管线生效）"
                         % (ADV_LABELS[name], self._format_adv_value(name, value)))

    def _on_slider(self, event, raw):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return
        name = SLIDERS[event]
        value = int(round(value)) if name == "pitch" else round(value, 3)
        self.slider_values[event] = value
        self._set_profile_value(self.current_model, name, value)
        try:
            self.window[SLIDER_LABELS[event]].update(self._format_slider_value(name, value))
        except Exception:
            pass
        if self.pipeline is not None:
            self.pipeline.set_param(name, value)

    def _on_switch(self, event, value):
        name = SWITCHES[event]
        self._set_profile_value(self.current_model, name, value)
        if self.pipeline is not None:
            self.pipeline.set_param(name, value)
        if event == "-I-NR-":
            self._follow_input_denoise_switch(bool(value))

    def _follow_input_denoise_switch(self, value):
        """Keep the legacy input-denoise checkbox consistent with denoise_mode."""
        mode = self.denoise_mode
        if value and mode == "off":
            target = "torchgate"
        elif not value and mode == "torchgate":
            target = "off"
        else:
            return
        self.denoise_mode = target
        self._set_top_value("denoise_mode", target)
        try:
            self.window["-DENOISE-MODE-"].update(value=DENOISE_LABELS[target])
            self.window["-DENOISE-HINT-"].update(DENOISE_HINTS.get(target, ""))
        except Exception:
            pass
        pending = self._push_acoustic_params()
        text = "降噪模式：%s（跟「输入降噪 (I)」联动）" % DENOISE_LABELS[target]
        if pending:
            text += "（%s 重启管线后生效）" % ", ".join(pending)
        self._set_status(text)

    def _sync_input_denoise_ui(self, write_profile=True):
        """Mirror denoise_mode onto the input-denoise checkbox."""
        mode = self.denoise_mode
        want = mode != "off"
        try:
            current = bool(self.window["-I-NR-"].get())
        except Exception:
            current = want
        try:
            self.window["-I-NR-"].update(value=want, disabled=(mode != "torchgate"))
        except Exception:
            pass
        if write_profile and current != want:
            self._on_switch("-I-NR-", want)

    def _on_monitor_switch(self, value):
        """Monitor checkbox: persist, then apply via contract or fallback."""
        self.monitor_on = bool(value)
        self._set_top_value("monitor_on", bool(value))
        if self.pipeline is None:
            self._set_status("监听%s（启动后生效）" % ("开" if self.monitor_on else "关"))
            return
        self._apply_monitor_state()
        self._sync_monitor_from_pipeline()
        self._set_status("运行中 · 监听%s" % ("开" if self.monitor_on else "关"))

    def _on_model_change(self, display):
        if not display or display not in self.model_map:
            return
        prev_model = self.current_model
        prev_sliders = dict(self.slider_values)
        prev_active = self.profiles.get("active")
        self.current_model = display
        self._sync_input_denoise_ui(write_profile=False)
        key, profile = self._profile_view(display)
        profile["I_noise_reduce"] = self.denoise_mode != "off"
        self.slider_values["-PITCH-"] = profile["pitch"]
        self.slider_values["-RMS-"] = profile["rms_mix_rate"]
        self.slider_values["-FORMANT-"] = profile["formant"]
        self.slider_values["-CROSSFADE-"] = self._clamp_adv("-CROSSFADE-", profile["crossfade_time"])
        self.slider_values["-EXTRA-"] = self._clamp_adv("-EXTRA-", profile["extra_time"])
        self.slider_values["-THREHOLD-"] = self._clamp_adv("-THREHOLD-", profile["threhold"])
        self.slider_values["-INDEX-RATE-"] = self._clamp_adv("-INDEX-RATE-", profile["index_rate"])
        try:
            self.window["-PITCH-"].update(profile["pitch"])
            self.window["-RMS-"].update(profile["rms_mix_rate"])
            self.window["-FORMANT-"].update(profile["formant"])
            self.window["-O-NR-"].update(bool(profile["O_noise_reduce"]))
            for slider_key in ADV_SLIDERS:
                self.window[slider_key].update(self.slider_values[slider_key])
            self.window["-BLOCK-TIME-"].update(
                value=BLOCK_TIME_LABEL_BY_VALUE[
                    nearest_choice(profile.get("block_time"), BLOCK_TIME_CHOICES,
                                   DEFAULT_PROFILE["block_time"])
                ]
            )
            self.window["-N-CPU-"].update(
                value=str(nearest_choice(profile.get("n_cpu"), N_CPU_CHOICES, DEFAULT_N_CPU))
            )
        except Exception:
            pass
        self._update_slider_labels()
        if key and self.profiles.get("active") != key:
            self.profiles["active"] = key
            self._mark_dirty()
        if self.pipeline is not None:
            try:
                pth = profile.get("pth") or ""
                if not pth:
                    raise RuntimeError("模型 %s 缺少权重路径（pth）" % display)
                self.pipeline.load_model(pth, profile.get("index") or "")
                for k in LIVE_KEYS:
                    self.pipeline.set_param(k, profile[k])
                self._apply_monitor_state()
            except Exception:
                self._rollback_model_ui(prev_model, prev_sliders, prev_active)
                raise
        self._set_status(
            ("运行中 · %s" % display) if self.pipeline is not None else ("已停止 · 当前模型 %s" % display)
        )

    def _rollback_model_ui(self, prev_model, prev_sliders, prev_active):
        """Roll the UI and the active key back after a failed live model swap."""
        self.current_model = prev_model
        self.slider_values.update(prev_sliders)
        _, prev_profile = self._profile_view(prev_model)
        self.profiles["active"] = prev_active
        self._mark_dirty()
        try:
            self.window["-MODEL-"].update(value=prev_model)
            self.window["-PITCH-"].update(self.slider_values["-PITCH-"])
            self.window["-RMS-"].update(self.slider_values["-RMS-"])
            self.window["-FORMANT-"].update(self.slider_values["-FORMANT-"])
            self.window["-O-NR-"].update(bool(prev_profile["O_noise_reduce"]))
            for slider_key in ADV_SLIDERS:
                self.window[slider_key].update(self.slider_values[slider_key])
            self.window["-BLOCK-TIME-"].update(
                value=BLOCK_TIME_LABEL_BY_VALUE[
                    nearest_choice(prev_profile.get("block_time"), BLOCK_TIME_CHOICES,
                                   DEFAULT_PROFILE["block_time"])
                ]
            )
            self.window["-N-CPU-"].update(
                value=str(nearest_choice(prev_profile.get("n_cpu"), N_CPU_CHOICES, DEFAULT_N_CPU))
            )
        except Exception:
            pass
        self._update_slider_labels()
        self._set_status("换模型失败，已回滚 %s" % (prev_model or ""))

    # Start / stop / device switching

    def _refresh_device_choices(self):
        """Re-enumerate devices and refresh combos, keeping the current selection."""
        self.devices = self._enumerate_devices()
        self.cable_device = pick_cable(self.devices["output"])
        try:
            self.window["-CABLE-DEV-"].update(self.cable_device or CABLE_MISSING_TEXT)
        except Exception:
            pass
        for key, pool in (("-IN-DEV-", self.devices["input"]),
                          ("-MON-DEV-", self.devices["output"])):
            values = list(pool) or [DEVICE_PLACEHOLDER]
            try:
                current = self.window[key].get()
                self.window[key].update(values=values, value=current)
            except Exception:
                pass

    def _on_device_refresh(self):
        """Reload devices and refresh the complete running audio chain together."""
        if self._enrolling or self._verifying:
            self._set_status("声纹录制/验证进行中，请结束后再刷新设备")
            return
        try:
            old_in = self._combo_value(self.window, "-IN-DEV-", "")
            old_mon = self._combo_value(self.window, "-MON-DEV-", "")
        except Exception:
            old_in, old_mon = "", ""
        try:
            devices = list_devices()
        except Exception as exc:
            self._set_status("设备刷新失败：%s" % exc)
            return
        self.devices = devices
        self._devices_error = ""
        self.cable_device = pick_cable(devices["output"])
        try:
            self.window["-CABLE-DEV-"].update(self.cable_device or CABLE_MISSING_TEXT)
        except Exception:
            pass
        for key, pool, old in (("-IN-DEV-", devices["input"], old_in),
                               ("-MON-DEV-", devices["output"], old_mon)):
            values = list(pool)
            if old and old != DEVICE_PLACEHOLDER and old not in values:
                values = [old] + values
            if not values:
                values = [DEVICE_PLACEHOLDER]
            try:
                self.window[key].update(values=values, value=old or values[0])
            except Exception:
                pass
        if self.pipeline is not None:
            pipeline = self.pipeline
            in_dev = old_in or self._combo_value(self.window, "-IN-DEV-", "")
            mon_dev = old_mon or self._combo_value(self.window, "-MON-DEV-", "")
            cable_dev = self.cable_device
            if not cable_dev:
                raise RuntimeError("虚拟麦不可用，未刷新正在运行的音频链路")
            self._set_status("正在共同刷新输入、虚拟麦与监听…")
            try:
                refresher = getattr(pipeline, "refresh_devices", None)
                if callable(refresher):
                    refresher(input_device=in_dev, monitor_device=mon_dev, cable_device=cable_dev)
                else:
                    pipeline.stop()
                    pipeline.input_device, pipeline.monitor_device, pipeline.cable_device = in_dev, mon_dev, cable_dev
                    pipeline.start()
            except Exception:
                if not getattr(pipeline, "is_running", False):
                    self.pipeline = None
                    self._set_running_ui(False)
                self._set_status("设备共同刷新失败，请检查设备连接")
                raise
            self._remember_devices(in_dev, mon_dev)
            self._reset_meter()
            with self._match_lock:
                self._match_score = None
            self._refresh_voice_score(force=True)
            self._refresh_agc_gain(force=True)
            self._refresh_load(force=True)
            self._set_status("输入、虚拟麦与监听已共同刷新，处理缓存和自动增益已重置")
        else:
            self._set_status("设备列表已刷新，启动后使用所选设备")

    @staticmethod
    def _combo_value(window, key, fallback):
        """Read a combo's display value; fall back to the event value."""
        try:
            value = window[key].get()
        except Exception:
            value = None
        return value or fallback

    def _on_start(self, values):
        if self.pipeline is not None:
            self._set_status("已在运行中")
            return
        self._refresh_device_choices()
        display = self._combo_value(self.window, "-MODEL-", values.get("-MODEL-") or "")
        if display not in self.model_map:
            raise RuntimeError(
                "未发现可用模型（engine.models.ModelRegistry.scan 为空或不可用）\n%s"
                % (self.models_error or "")
            )
        in_dev = self._combo_value(self.window, "-IN-DEV-", values.get("-IN-DEV-") or "")
        mon_dev = self._combo_value(self.window, "-MON-DEV-", values.get("-MON-DEV-") or "")
        cable_dev = self.cable_device
        if not cable_dev or cable_dev not in self.devices["output_map"]:
            raise RuntimeError("虚拟麦克风（固定的 CABLE 输出）不可用：%s" % CABLE_MISSING_TEXT)
        checks = [("输入设备", in_dev, self.devices["input_map"])]
        if self.monitor_on:
            checks.append(("监听设备", mon_dev, self.devices["output_map"]))
        for label, dev, pool in checks:
            if dev not in pool:
                raise RuntimeError("%s 不可用: %r" % (label, dev))
        self._remember_devices(in_dev, mon_dev)

        key, profile = self._profile_view(display)
        profile = dict(profile)
        profile["pitch"] = int(round(float(values.get("-PITCH-"))))
        profile["rms_mix_rate"] = round(float(values.get("-RMS-")), 3)
        profile["formant"] = round(float(values.get("-FORMANT-")), 3)
        profile["I_noise_reduce"] = bool(values.get("-I-NR-"))
        profile["O_noise_reduce"] = bool(values.get("-O-NR-"))
        profile["monitor"] = bool(self.monitor_on)
        self._refresh_voice_profile_state()
        self.denoise_mode = self._denoise_mode_from(values)
        self.voice_gate_threshold = normalize_threshold(values.get("-VG-THR-"))
        self.voice_gate_on = bool(values.get("-VOICE-GATE-")) and self.voice_profile_ok
        profile["denoise_mode"] = self.denoise_mode
        profile["voice_gate"] = self.voice_gate_on
        profile["voice_gate_threshold"] = self.voice_gate_threshold
        self.agc_on = bool(values.get("-AGC-"))
        self.agc_target_dbfs = normalize_agc_target(values.get("-AGC-TARGET-"))
        profile["agc"] = self.agc_on
        profile["agc_target_dbfs"] = self.agc_target_dbfs
        for top_key, top_value in (
            ("denoise_mode", self.denoise_mode),
            ("voice_gate", self.voice_gate_on),
            ("voice_gate_threshold", self.voice_gate_threshold),
            ("agc", self.agc_on),
            ("agc_target_dbfs", self.agc_target_dbfs),
        ):
            self._set_top_value(top_key, top_value)
        self.slider_values["-PITCH-"] = profile["pitch"]
        self.slider_values["-RMS-"] = profile["rms_mix_rate"]
        self.slider_values["-FORMANT-"] = profile["formant"]
        self.slider_values["-AGC-TARGET-"] = self.agc_target_dbfs
        # Voice params: hot keys (index_rate/threhold) apply via set_param at
        # runtime; restart keys fix buffer/thread layout on pipeline rebuild.
        for slider_key, profile_key in ADV_SLIDERS.items():
            value = self.slider_values.get(slider_key, profile[profile_key])
            profile[profile_key] = int(round(float(value))) if profile_key == "threhold" else round(float(value), 3)
        profile["block_time"] = self._block_time_value()
        profile["n_cpu"] = self._n_cpu_value()
        for k in LIVE_KEYS + ADV_RESTART_KEYS:
            self._set_profile_value(display, k, profile[k])
        if key and self.profiles.get("active") != key:
            self.profiles["active"] = key
            self._mark_dirty()

        _insert_project_root()
        from engine.pipeline import VoicePipeline

        # First model load takes seconds; warn because events stall meanwhile.
        self._set_status("正在启动…（加载模型中，请稍候）")
        try:
            self.window.refresh()
        except Exception:
            pass
        pipeline, monitor_native = self._construct_pipeline(
            VoicePipeline, profile, in_dev, cable_dev, mon_dev, self.monitor_on
        )
        self._monitor_native = monitor_native
        try:
            pipeline.start()
        except Exception as exc:
            if not monitor_native and self._fallback_monitor_restart(pipeline, cable_dev):
                mon_dev = cable_dev
                self._remember_devices(in_dev, cable_dev)
                self._set_status("监听设备打不开，已按“监听关”回退（无监听）：%s" % exc)
            else:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                self._set_status("启动失败")
                raise
        self.pipeline = pipeline
        self._gate_spec_sent = MODELS_DIR if self.voice_gate_on else None
        self._apply_monitor_state()
        self._sync_monitor_from_pipeline()
        self._set_running_ui(True)
        self._refresh_voice_score(force=True)
        text = "运行中 · %s · 监听%s · 降噪%s · 声纹门%s · 自动增益%s" % (
            display, "开" if self.monitor_on else "关",
            DENOISE_LABELS.get(self.denoise_mode, "?"),
            "开" if self.voice_gate_on else "关",
            "开 %d dB" % round(self.agc_target_dbfs) if self.agc_on else "关")
        try:
            latency_ms = getattr(pipeline, "denoise_latency_ms", None)
            if latency_ms:
                text += " · DFN3 +%.0fms" % float(latency_ms)
        except Exception:
            pass
        self._set_status(text)

    def _on_stop(self):
        pipeline, self.pipeline = self.pipeline, None
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            self._gate_spec_sent = _UNSET
            with self._match_lock:
                self._match_score = None
            self._reset_meter()
            self._refresh_voice_score(force=True)
            self._refresh_agc_gain(force=True)
            self._refresh_load(force=True)
            self._set_running_ui(False)
            self._set_status("已停止")

    def _remember_devices(self, in_dev, mon_dev):
        """Persist a device selection together instead of replacing the file twice in a row."""
        changed = False
        for key, value in (("input_device", in_dev), ("monitor_device", mon_dev)):
            if value and value != DEVICE_PLACEHOLDER and self.profiles.get(key) != value:
                self.profiles[key] = value
                changed = True
        if changed:
            self._mark_dirty()

    def _on_device_combo(self, values):
        """Swap input/monitor devices live; only remember while stopped."""
        in_dev = values.get("-IN-DEV-") or ""
        mon_dev = values.get("-MON-DEV-") or ""
        if self.pipeline is None:
            self._remember_devices(in_dev, mon_dev)
            self._set_status("已停止 · 设备已选，启动后生效")
            return
        if (in_dev, mon_dev) == (
            getattr(self.pipeline, "input_device", None),
            getattr(self.pipeline, "monitor_device", None),
        ):
            return
        self._restart_with_devices(in_dev, mon_dev)

    def _restart_with_devices(self, in_dev, mon_dev):
        """Swap devices via set_devices() or stream restart; roll back on failure."""
        pipeline = self.pipeline
        prev = (getattr(pipeline, "input_device", None), getattr(pipeline, "monitor_device", None))
        self._set_status("切换设备…")
        try:
            self._apply_device_change(pipeline, in_dev, mon_dev)
        except Exception as exc:
            try:
                self._apply_device_change(pipeline, prev[0], prev[1])
                self._show_devices(prev[0], prev[1])
                self._remember_devices(prev[0], prev[1])
                text = "切换设备失败，已回滚原设备: %s" % exc
            except Exception:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                self._reset_meter()
                self._set_running_ui(False)
                text = "切换设备失败，流水线已停止: %s" % exc
            self._set_status(text)
            raise RuntimeError(text)
        self._remember_devices(in_dev, mon_dev)
        self._apply_monitor_state()
        self._set_status("运行中 · 输入=%s · 监听=%s" % (in_dev, mon_dev))

    def _show_devices(self, in_dev, mon_dev):
        """Reset combo display values; ignore empty/missing controls."""
        for key, dev in (("-IN-DEV-", in_dev), ("-MON-DEV-", mon_dev)):
            if not dev:
                continue
            try:
                self.window[key].update(value=dev)
            except Exception:
                pass

    @staticmethod
    def _apply_device_change(pipeline, in_dev, mon_dev):
        setter = getattr(pipeline, "set_devices", None)
        if callable(setter):
            try:
                setter(input_device=in_dev, monitor_device=mon_dev)
                return
            except TypeError:
                setter(in_dev, mon_dev)
                return
        pipeline.input_device = in_dev
        pipeline.monitor_device = mon_dev
        pipeline.stop()
        pipeline.start()

    @staticmethod
    def _init_param_names(cls):
        try:
            return tuple(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            return None

    def _construct_pipeline(self, cls, profile, in_dev, cable_dev, mon_dev, monitor_on):
        """Build VoicePipeline by introspected signature; return (instance, native).

        Only kwargs the signature declares are passed, so old and new
        signatures both work; native tells whether the monitor switch is built in.
        """
        params = self._init_param_names(cls)
        if params is None:
            return cls(profile=profile, input_device=in_dev, cable_device=cable_dev,
                       monitor_device=mon_dev, on_level=self._on_level), False
        kwargs = {}
        if "profile" in params:
            kwargs["profile"] = profile
        if "input_device" in params:
            kwargs["input_device"] = in_dev
        if "cable_device" in params:
            kwargs["cable_device"] = cable_dev
        if "monitor_device" in params:
            kwargs["monitor_device"] = mon_dev
        if "on_level" in params:
            kwargs["on_level"] = self._on_level
        if "denoise_mode" in params:
            kwargs["denoise_mode"] = self.denoise_mode
        if "voice_gate" in params:
            kwargs["voice_gate"] = MODELS_DIR if self.voice_gate_on else None
        if "voice_gate_threshold" in params:
            kwargs["voice_gate_threshold"] = self._threshold_value()
        if "agc_enabled" in params:
            kwargs["agc_enabled"] = bool(self.agc_on)
        if "agc_target_dbfs" in params:
            kwargs["agc_target_dbfs"] = float(self.agc_target_dbfs)
        for name in ("on_match", "on_voice_match"):
            if name in params:
                kwargs[name] = self._on_match
        native = "monitor_on" in params
        if native:
            kwargs["monitor_on"] = bool(monitor_on)
        pipeline = cls(**kwargs)
        if not native:
            native = callable(getattr(pipeline, "set_monitor", None)) or hasattr(pipeline, "monitor_on")
        return pipeline, native

    def _fallback_monitor_restart(self, pipeline, cable_dev):
        """With monitor off, retry a failed start routed to CABLE, then mute it."""
        if self.monitor_on:
            return False
        try:
            if getattr(pipeline, "monitor_device", None) == cable_dev:
                return False
            pipeline.monitor_device = cable_dev
            pipeline.start()
            return True
        except Exception:
            return False

    def _apply_monitor_state(self):
        """Apply the monitor switch: set_param('monitor'), else pause/resume the stream."""
        pipeline = self.pipeline
        if pipeline is None:
            return
        want = bool(self.monitor_on)
        try:
            pipeline.set_param("monitor", want)
        except Exception:
            print("[monitor] set_param('monitor') 失败: %s" % traceback.format_exc(), file=sys.stderr)
        if self._monitor_native:
            return
        setter = getattr(pipeline, "set_monitor", None)
        if callable(setter):
            try:
                setter(want)
                return
            except Exception:
                pass
        if hasattr(pipeline, "monitor_on"):
            try:
                pipeline.monitor_on = want
                return
            except Exception:
                pass
        stream = find_monitor_stream(pipeline)
        if stream is None:
            return
        try:
            if want:
                ring = getattr(pipeline, "ring_monitor", None)
                clear = getattr(ring, "clear", None)
                if not stream.active and callable(clear):
                    clear()
                if not stream.active:
                    stream.start()
            elif stream.active:
                stream.abort()
        except Exception:
            print("[monitor] 监听流 暂停/恢复 失败: %s" % traceback.format_exc(), file=sys.stderr)

    def _sync_monitor_from_pipeline(self):
        """Reflect the pipeline's real monitor state back onto the checkbox."""
        pipeline = self.pipeline
        if pipeline is None:
            return
        state = getattr(pipeline, "monitor_active", None)
        if state is None:
            state = getattr(pipeline, "monitor_on", None)
        if state is None or bool(state) == self.monitor_on:
            return
        self.monitor_on = bool(state)
        try:
            self.window["-MONITOR-"].update(value=self.monitor_on)
        except Exception:
            pass
        self._set_status("监听%s（已按流水线实际状态同步）" % ("开" if self.monitor_on else "关"))

    # Acoustic enhancement: denoise mode / voice gate / enrollment

    def _denoise_mode_from(self, values):
        """Combo label to profile value; keep the current mode when unknown."""
        label = str(values.get("-DENOISE-MODE-") or "").strip()
        return DENOISE_MODE_BY_LABEL.get(label, self.denoise_mode)

    def _push_acoustic_params(self, reload_voice_profile=False):
        """Push denoise/voice-gate params; return keys the pipeline cannot hot-apply.

        voice_gate is only resent on spec change: resending rebuilds the VoiceGate.
        """
        pipeline = self.pipeline
        if pipeline is None:
            return []
        spec = MODELS_DIR if self.voice_gate_on else None
        if reload_voice_profile and spec is not None:
            if not self._reload_voice_gate_profile(pipeline):
                self._gate_spec_sent = _UNSET
        jobs = [("denoise_mode", self.denoise_mode)]
        if spec != self._gate_spec_sent:
            jobs.append(("voice_gate", spec))
        jobs.append(("voice_gate_threshold", self._threshold_value()))
        pending = []
        for key, value in jobs:
            try:
                pipeline.set_param(key, value)
            except KeyError:
                pending.append(key)
                continue
            except Exception:
                print("[pipeline] set_param(%r) 失败: %s" % (key, traceback.format_exc()),
                      file=sys.stderr)
                pending.append(key)
                continue
            if key == "voice_gate":
                self._gate_spec_sent = spec
        return pending

    @staticmethod
    def _reload_voice_gate_profile(pipeline):
        """Hot-reload a fresh profile into the running gate; False if unsupported."""
        for name in ("voice_gate", "_voice_gate", "gate"):
            gate = getattr(pipeline, name, None)
            if gate is None or isinstance(gate, bool) or isinstance(gate, (str, os.PathLike)):
                continue
            loader = getattr(gate, "load_profile", None)
            if not callable(loader):
                continue
            try:
                loader(VOICE_PROFILE_PATH)
            except TypeError:
                try:
                    loader()
                except Exception:
                    continue
            except Exception:
                continue
            return True
        return False

    def _enable_voice_gate(self, value):
        """Set the voice-gate switch (checkbox, disk, pipeline)."""
        want = bool(value)
        if want and not os.path.isfile(VOICE_PROFILE_PATH):
            want = False
        self.voice_gate_on = want
        self._set_top_value("voice_gate", want)
        try:
            self.window["-VOICE-GATE-"].update(value=want)
        except Exception:
            pass
        pending = self._push_acoustic_params(reload_voice_profile=want)
        text = "声纹门%s" % ("开" if want else "关")
        if self.pipeline is None:
            text += "（启动后生效）"
        if pending:
            text += "（%s 重启管线后生效）" % ", ".join(pending)
        self._set_status(text)
        self._refresh_voice_score(force=True)

    def _on_denoise_mode(self, label):
        """Denoise-mode combo: persist, mirror the checkbox, push to pipeline."""
        mode = DENOISE_MODE_BY_LABEL.get(str(label or "").strip())
        if mode is None:
            return
        self.denoise_mode = mode
        self._set_top_value("denoise_mode", mode)
        try:
            self.window["-DENOISE-HINT-"].update(DENOISE_HINTS.get(mode, ""))
        except Exception:
            pass
        self._sync_input_denoise_ui()
        pending = self._push_acoustic_params()
        text = "降噪模式：%s" % DENOISE_LABELS[mode]
        if pending:
            text += "（%s 重启管线后生效）" % ", ".join(pending)
        self._set_status(text)

    def _on_voice_gate(self, value):
        """Gate checkbox: refuse to open without a profile on disk."""
        want = bool(value)
        if want and not os.path.isfile(VOICE_PROFILE_PATH):
            self.voice_gate_on = False
            self._set_top_value("voice_gate", False)
            try:
                self.window["-VOICE-GATE-"].update(value=False)
            except Exception:
                pass
            self._refresh_voice_profile_state()
            self._set_status("还没有声纹档案：先点「注册声纹」录 %d 秒" % ENROLL_SECONDS)
            return
        self._enable_voice_gate(want)

    def _on_threshold(self, raw):
        """Threshold slider: persist and push to the pipeline."""
        value = normalize_threshold(raw)
        self.slider_values["-VG-THR-"] = value
        self._set_top_value("voice_gate_threshold", value)
        try:
            self.window["-VG-THR-VAL-"].update(
                self._format_slider_value("voice_gate_threshold", value)
            )
        except Exception:
            pass
        pending = self._push_acoustic_params()
        self._refresh_voice_score(force=True)
        if pending:
            self._set_status("匹配阈值 %.2f（%s 重启管线后生效）" % (value, ", ".join(pending)))

    # Output AGC

    def _on_agc(self, value):
        """AGC checkbox: persist, toggle the target slider, push to pipeline."""
        want = bool(value)
        self.agc_on = want
        self._set_top_value("agc", want)
        try:
            self.window["-AGC-TARGET-"].update(disabled=not want)
            self.window["-AGC-HINT-"].update("按语音电平自动补 / 压增益" if want else "关：输出直通")
        except Exception:
            pass
        pending = self._push_agc_params()
        text = "自动增益%s（目标 %.0f dB）" % ("开" if want else "关", self.agc_target_dbfs)
        if self.pipeline is None:
            text += " · 启动后生效"
        if pending:
            text += "（%s 重启管线后生效）" % ", ".join(pending)
        self._set_status(text)
        self._refresh_agc_gain(force=True)

    def _on_agc_target(self, raw):
        """Target-loudness slider: persist and hot-apply."""
        value = normalize_agc_target(raw)
        self.agc_target_dbfs = value
        self.slider_values["-AGC-TARGET-"] = value
        self._set_top_value("agc_target_dbfs", value)
        try:
            self.window["-AGC-TARGET-VAL-"].update(
                self._format_adv_value("agc_target_dbfs", value)
            )
        except Exception:
            pass
        pending = self._push_agc_params()
        self._set_status("目标响度 %.0f dB（%s）"
                         % (value, "重启管线后生效" if pending else "即时生效"))

    def _push_agc_params(self):
        """Push the AGC switch/target; return keys needing restart."""
        pipeline = self.pipeline
        if pipeline is None:
            return []
        pending = []
        for key, value in (("agc", bool(self.agc_on)),
                           ("agc_target_dbfs", float(self.agc_target_dbfs))):
            try:
                pipeline.set_param(key, value)
            except KeyError:
                pending.append(key)
            except Exception:
                print("[pipeline] set_param(%r) 失败: %s" % (key, traceback.format_exc()),
                      file=sys.stderr)
                pending.append(key)
        return pending

    def _poll_agc_gain(self):
        """Current AGC gain in dB; None when off or unreadable."""
        pipeline = self.pipeline
        if pipeline is None or not self.agc_on:
            return None
        stats = getattr(pipeline, "agc_stats", None)
        if isinstance(stats, dict):
            gain = _as_float(stats.get("gain_db"))
            if gain is not None:
                return gain
        for name in ("agc_gain_db", "agc_gain"):
            gain = _as_float(getattr(pipeline, name, None))
            if gain is not None:
                return gain
        agc = getattr(pipeline, "agc", None)
        if agc is None or isinstance(agc, bool):
            return None
        return _as_float(getattr(agc, "gain_db", None))

    def _refresh_agc_gain(self, force=False):
        """Repaint the current-gain text (10 Hz, with the meters)."""
        gain = self._poll_agc_gain()
        if gain is None:
            text, color = ("关" if not self.agc_on else "--"), "#808080"
        else:
            text = "%+6.1f dB" % gain
            color = "#2E7D32" if abs(gain) >= 0.5 else "#808080"
        if force or text != self._agc_gain_text:
            try:
                self.window["-AGC-GAIN-"].update(text, text_color=color)
            except Exception:
                return
            self._agc_gain_text = text

    # Load line: mean inference ms / block budget / dropped blocks.

    def _poll_load_stats(self):
        """Load stats from the pipeline; None when stopped or unreadable."""
        pipeline = self.pipeline
        if pipeline is None:
            return None
        stats = getattr(pipeline, "stats", None)
        if not isinstance(stats, dict):
            return None
        try:
            infer_ms = float(stats.get("infer_ms_avg", 0.0))
            dropped = int(stats.get("dropped", 0))
        except (TypeError, ValueError):
            return None
        try:
            budget_ms = float(getattr(pipeline, "block_frame", 0)) * 1000.0 / 48000.0
        except (TypeError, ValueError):
            budget_ms = 0.0
        if not budget_ms:
            budget_ms = float(self._block_time_value()) * 1000.0
        profile = getattr(pipeline, "profile", None)
        f0 = profile.get("f0method") if isinstance(profile, dict) else ""
        return {"infer_ms": infer_ms, "dropped": dropped,
                "budget_ms": budget_ms, "f0": str(f0 or "")}

    def _refresh_load(self, force=False):
        """Repaint the load line; overload means mean inference beats the budget."""
        data = self._poll_load_stats()
        if data is None:
            text, color = "负载: 未运行", "#808080"
            overloaded = False
        else:
            infer_ms = data["infer_ms"]
            budget_ms = data["budget_ms"]
            dropped = data["dropped"]
            overloaded = budget_ms > 0 and infer_ms > budget_ms
            if overloaded:
                text = ("过载：推理均值 %.1f ms 超块预算 %.0f ms，丢块 %d。"
                        "建议采样块长调大、关占 GPU 程序、检查散热"
                        % (infer_ms, budget_ms, dropped))
                if data["f0"] and data["f0"] != "rmvpe":
                    text += "（已自动降级保实时）"
                color = "#C62828"
            elif dropped > 0:
                text = ("负载偏高：推理均值 %.1f ms / 块预算 %.0f ms，累计丢块 %d。"
                        "建议采样块长调大、关占 GPU 程序、检查散热"
                        % (infer_ms, budget_ms, dropped))
                color = "#EF6C00"
            else:
                text = ("负载正常：推理均值 %.1f ms / 块预算 %.0f ms，丢块 %d"
                        % (infer_ms, budget_ms, dropped))
                color = "#2E7D32"
        if force or text != self._load_text:
            try:
                self.window["-LOAD-"].update(text, text_color=color)
            except Exception:
                return
            self._load_text = text
        if overloaded and not self._load_overload:
            self._set_status("过载：推理跟不上，建议采样块长调大、关占 GPU 程序、检查散热")
        self._load_overload = overloaded

    # Enrollment: record, resample to 16 k, enroll, save profile.

    def _on_enroll(self):
        if self._enrolling:
            return
        in_dev = self.window["-IN-DEV-"].get() or ""
        index = self.devices["input_map"].get(in_dev)
        if index is None:
            raise RuntimeError("输入设备不可用：%r（先在「输入设备」里选一个麦克风）" % in_dev)
        if not self.smoke:
            answer = sg.popup_ok_cancel(
                "接下来从「%s」录制 %d 秒。\n\n"
                "请用平时说话的音量和语气连续朗读（随便一段文字即可），\n"
                "中途不要长时间停顿，保持周围安静。\n\n"
                "点「OK」开始录制。" % (in_dev, ENROLL_SECONDS),
                title="注册声纹 · 录制 %d 秒" % ENROLL_SECONDS,
                keep_on_top=True,
            )
            if str(answer).strip().upper() != "OK":
                return
        self._enrolling = True
        try:
            self.window["-ENROLL-"].update(disabled=True)
            self.window["-ENROLL-BAR-"].update(current_count=0, visible=True)
            self.window["-ENROLL-TEXT-"].update("准备录音…")
        except Exception:
            pass
        self._set_status("注册声纹：录制 %d 秒…" % ENROLL_SECONDS)
        threading.Thread(target=self._enroll_worker, args=(index,), daemon=True).start()

    def _enroll_worker(self, device_index):
        """Enroll thread: record, resample, enroll, save; post the result back."""
        payload = {"ok": False, "message": "", "kept": None, "dropped": None,
                   "suggested_threshold": None}
        try:
            import numpy as np

            wav = self._record_mono(device_index, ENROLL_SECONDS, on_progress=self._post_enroll_progress)
            if wav.size < ENROLL_SR * 3:
                raise RuntimeError("录到的音频太短（%.1f 秒）" % (wav.size / float(ENROLL_SR)))
            rms = float(np.sqrt(np.mean(np.square(wav))))
            if rms < 1e-3:
                raise RuntimeError("录音电平过低（rms=%.5f）：检查麦克风是否选对 / 被静音 / 增益太低" % rms)
            import librosa

            wav16 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=ENROLL_SR,
                                     target_sr=16000)
            gate = self._make_voice_gate(for_enroll=True)
            report = self._call_enroll(gate, wav16)
            try:
                if isinstance(report, dict):
                    if report.get("kept") is not None:
                        payload["kept"] = int(report.get("kept"))
                    if report.get("dropped") is not None:
                        payload["dropped"] = int(report.get("dropped"))
            except (TypeError, ValueError):
                payload["kept"] = payload["dropped"] = None
            try:
                suggest = getattr(gate, "suggest_threshold", None)
                if callable(suggest):
                    number = float(suggest())
                    if math.isfinite(number):
                        payload["suggested_threshold"] = number
            except Exception:
                pass
            self._save_voice_gate_profile(gate)
            payload["ok"] = True
            payload["message"] = "声纹注册成功：%s（%.1f 秒语音，电平 %.1f dB）" % (
                os.path.basename(VOICE_PROFILE_PATH), wav.size / float(ENROLL_SR),
                20.0 * math.log10(max(rms, 1e-6)),
            )
        except Exception as exc:
            message = str(exc) or repr(exc)
            payload["message"] = message[:300]
        try:
            self.window.write_event_value("-ENROLL-DONE-", payload)
        except Exception:
            pass

    def _post_enroll_progress(self, percent):
        try:
            self.window.write_event_value("-ENROLL-PROGRESS-", int(percent))
        except Exception:
            pass

    @staticmethod
    def _stream_mono(data):
        """Fold a captured block to mono float32."""
        import numpy as np

        block = np.asarray(data, dtype=np.float32)
        if block.ndim == 1:
            return block
        return block.mean(axis=1, dtype=np.float32)

    def _record_mono(self, device_index, seconds, on_progress=None):
        """Record seconds of 48 kHz mono float32 (blocking, worker threads)."""
        import numpy as np
        import sounddevice as sd

        total = int(seconds * ENROLL_SR)

        def open_stream(channels):
            stream = sd.InputStream(
                device=device_index,
                samplerate=ENROLL_SR,
                channels=channels,
                dtype="float32",
                blocksize=ENROLL_BLOCK,
            )
            stream.start()
            return stream

        try:
            stream = open_stream(1)
        except Exception:
            stream = open_stream(2)
        chunks = []
        got = 0
        try:
            while got < total:
                data, _overflowed = stream.read(ENROLL_BLOCK)
                mono = self._stream_mono(data)
                chunks.append(mono)
                got += mono.shape[0]
                if on_progress is not None:
                    try:
                        on_progress(min(100, int(got * 100 / total)))
                    except Exception:
                        pass
        finally:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)[:total]

    def _make_voice_gate(self, for_enroll=False):
        """Build a VoiceGate (lazy import) by runtime signature.

        for_enroll starts from a blank template so the old profile cannot
        dilute the new recording via segment-count averaging.
        """
        _insert_project_root()
        try:
            from dsp.voice_gate import VoiceGate
        except Exception as exc:
            raise RuntimeError(
                "声纹门模块不可用（dsp/voice_gate.py 缺失或依赖未装）: %s" % exc
            )
        threshold = self._threshold_value()
        params = signature_params(VoiceGate.__init__)
        if params is None:
            gate = VoiceGate(MODELS_DIR, threshold)
        else:
            kwargs = {}
            for name in ("model_dir", "models_dir", "models_root", "model_root"):
                if name in params:
                    kwargs[name] = MODELS_DIR
                    break
            if "threshold" in params:
                kwargs["threshold"] = threshold
            if "profile_path" in params:
                kwargs["profile_path"] = VOICE_PROFILE_PATH
            gate = VoiceGate(MODELS_DIR, threshold) if not kwargs else VoiceGate(**kwargs)
        if for_enroll:
            clearer = getattr(gate, "reset", None)
            if callable(clearer):
                clearer()
        return gate

    @staticmethod
    def _call_enroll(gate, wav16):
        """Call VoiceGate.enroll(wav16k) and return its report dict as-is."""
        enroll = getattr(gate, "enroll", None)
        if not callable(enroll):
            raise RuntimeError("VoiceGate 缺少 enroll()")
        params = signature_params(enroll)
        if params is not None and len(params) > 1:
            return enroll(wav16, 16000)
        return enroll(wav16)

    @staticmethod
    def _save_voice_gate_profile(gate):
        """Save the voice profile to the contract path."""
        saver = getattr(gate, "save_profile", None)
        if not callable(saver):
            raise RuntimeError("VoiceGate 缺少 save_profile()")
        os.makedirs(os.path.dirname(VOICE_PROFILE_PATH), exist_ok=True)
        from app.voice_profiles import backup_profile
        backup_profile(VOICE_PROFILE_PATH)
        try:
            saver(VOICE_PROFILE_PATH)
        except TypeError:
            saver()
        if not os.path.isfile(VOICE_PROFILE_PATH):
            raise RuntimeError(
                "声纹档案没有落到 %s（检查 VoiceGate.save_profile 的写入路径）" % VOICE_PROFILE_PATH
            )
        return VOICE_PROFILE_PATH

    def _on_enroll_progress(self, percent):
        try:
            percent = int(percent)
        except (TypeError, ValueError):
            return
        try:
            self.window["-ENROLL-BAR-"].update(current_count=percent, visible=True)
            self.window["-ENROLL-TEXT-"].update("录制中 %d%%" % percent)
        except Exception:
            pass

    def _on_enroll_done(self, payload):
        self._enrolling = False
        try:
            self.window["-ENROLL-"].update(disabled=False)
            self.window["-ENROLL-BAR-"].update(current_count=0, visible=False)
            self.window["-ENROLL-TEXT-"].update("")
        except Exception:
            pass
        if not bool(payload.get("ok")):
            self._set_status("声纹注册失败")
            self._report_error("注册声纹失败", str(payload.get("message") or "未知原因"))
            return
        self._refresh_voice_profile_state()
        self._enable_voice_gate(True)
        text = str(payload.get("message") or "声纹注册成功")
        try:
            parts = []
            if payload.get("kept") is not None and payload.get("dropped") is not None:
                parts.append("保留 %d 段 / 丢掉 %d 段"
                             % (int(payload.get("kept")), int(payload.get("dropped"))))
            if payload.get("suggested_threshold") is not None:
                number = float(payload.get("suggested_threshold"))
                if math.isfinite(number):
                    parts.append("建议阈值 %.2f" % number)
            if parts:
                text += "（%s）" % "，".join(parts)
        except (TypeError, ValueError):
            pass
        self._set_status(text)

    # Delete profile / verification (record 3 s, compare offline).

    def _on_voice_delete(self):
        """Delete the profile file; the gate switch closes with it."""
        if self._voice_file_busy():
            return
        if not os.path.isfile(VOICE_PROFILE_PATH):
            self._refresh_voice_profile_state()
            self._set_status("没有声纹档案可删")
            return
        if not self.smoke:
            answer = sg.popup_ok_cancel(
                "确定删除声纹档案吗？\n%s\n删除前会自动备份，可随时导入恢复。\n删除后声纹门自动关闭。"
                % VOICE_PROFILE_PATH,
                title="删除声纹档案",
                keep_on_top=True,
            )
            if str(answer).strip().upper() != "OK":
                return
        try:
            from app.voice_profiles import delete_profile
            backup = delete_profile(VOICE_PROFILE_PATH)
        except OSError as exc:
            self._report_error("删除声纹档案失败", str(exc))
            return
        self._refresh_voice_profile_state()
        self._enable_voice_gate(False)
        self._set_status("声纹已删除并备份，声纹门已关闭：%s" % backup)

    def _voice_file_busy(self):
        if self._enrolling or self._verifying:
            self._set_status("声纹录制/验证进行中，请结束后再管理声纹文件")
            return True
        return False

    def _on_voice_import(self):
        if self._voice_file_busy():
            return
        path = sg.popup_get_file("选择要导入的声纹档案", title="导入声纹", no_window=True,
                                 initial_folder=MODELS_DIR,
                                 file_types=(("声纹档案", "*.npz *.json"), ("所有文件", "*.*")))
        if not path:
            return
        from app.voice_profiles import import_profile
        backup = import_profile(path, VOICE_PROFILE_PATH, MODELS_DIR, self.voice_gate_threshold)
        self._refresh_voice_profile_state()
        # Rebuild the live gate so a queued decision from the previous voiceprint is discarded.
        self._gate_spec_sent = _UNSET
        self._enable_voice_gate(True)
        self._set_status("声纹已导入，无需重新录制" + ("；原声纹已备份：%s" % backup if backup else ""))

    def _on_voice_export(self):
        if self._voice_file_busy():
            return
        if not os.path.isfile(VOICE_PROFILE_PATH):
            self._set_status("没有可导出的声纹，先导入已有档案或注册声纹")
            return
        path = sg.popup_get_file("保存当前声纹", title="导出声纹", no_window=True,
                                 save_as=True, default_extension=".npz", default_path="我的声纹.npz",
                                 initial_folder=MODELS_DIR,
                                 file_types=(("声纹档案 NPZ", "*.npz"), ("声纹档案 JSON", "*.json")))
        if not path:
            return
        from app.voice_profiles import export_profile
        saved = export_profile(VOICE_PROFILE_PATH, path, MODELS_DIR, self.voice_gate_threshold)
        self._set_status("声纹已导出：%s" % saved)

    def _on_verify(self):
        """Verification: record 3 s and score once against the threshold."""
        if self._verifying:
            return
        if not os.path.isfile(VOICE_PROFILE_PATH):
            self._set_status("还没有声纹档案：先点「注册声纹」")
            return
        try:
            in_dev = self.window["-IN-DEV-"].get() or ""
        except Exception:
            in_dev = ""
        index = self.devices["input_map"].get(in_dev)
        if index is None:
            raise RuntimeError("输入设备不可用：%r（先在「输入设备」里选一个麦克风）" % in_dev)
        self._verifying = True
        try:
            self.window["-VERIFY-"].update(disabled=True)
            self.window["-VERIFY-TEXT-"].update("录制中…")
        except Exception:
            pass
        self._set_status("声纹验证：录制 %d 秒…" % VERIFY_SECONDS)
        threading.Thread(target=self._verify_worker, args=(index,), daemon=True).start()

    def _verify_worker(self, device_index):
        """Verify thread: record, resample, decide, post the result back."""
        payload = {"ok": False, "score": None, "passed": None, "message": ""}
        try:
            import numpy as np

            wav = self._record_mono(device_index, VERIFY_SECONDS)
            if wav.size < ENROLL_SR * 1:
                raise RuntimeError("录到的音频太短（%.1f 秒）" % (wav.size / float(ENROLL_SR)))
            import librosa

            wav16 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=ENROLL_SR,
                                     target_sr=16000)
            gate = self._make_voice_gate()
            score = gate.decide(wav16)
            threshold = self._threshold_value()
            if score is None:
                payload["message"] = "判不出（无人声或太短），对麦说话再试"
            else:
                passed = bool(score >= threshold)
                payload["ok"] = True
                payload["score"] = float(score)
                payload["passed"] = passed
                payload["message"] = "分数 %.2f，阈值 %.2f：%s" % (
                    score, threshold, "通过" if passed else "拦截")
        except Exception as exc:
            payload["message"] = (str(exc) or repr(exc))[:300]
        try:
            self.window.write_event_value("-VERIFY-DONE-", payload)
        except Exception:
            pass

    def _on_verify_done(self, payload):
        """Show the verification score and verdict; re-enable the button."""
        self._verifying = False
        try:
            self.window["-VERIFY-"].update(disabled=not self.voice_profile_ok)
            if payload.get("ok"):
                color = "#2E7D32" if payload.get("passed") else "#C62828"
            else:
                color = "#808080"
            self.window["-VERIFY-TEXT-"].update(
                str(payload.get("message") or ""), text_color=color)
        except Exception:
            pass
        if payload.get("ok"):
            self._set_status("声纹验证：%s" % payload.get("message"))
        else:
            self._set_status("声纹验证失败")
            if payload.get("message"):
                self._report_error("声纹验证失败", str(payload.get("message")))

    # Self-check

    def _on_selfcheck(self):
        if not os.path.isfile(SELFCHECK_PATH):
            raise RuntimeError("未找到自检脚本: %s" % SELFCHECK_PATH)
        try:
            self.window["-SELFCHECK-"].update(disabled=True)
        except Exception:
            pass
        self._set_status("自检运行中…")
        threading.Thread(target=self._selfcheck_worker, daemon=True).start()

    def _selfcheck_worker(self):
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", SELFCHECK_PATH],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SELFCHECK_TIMEOUT_S,
                creationflags=_CREATE_NO_WINDOW,
            )
            payload = {"rc": proc.returncode, "out": (proc.stdout or "") + (proc.stderr or "")}
        except Exception as exc:
            payload = {"rc": -1, "out": "启动 selfcheck.py 失败: %s" % exc}
        try:
            self.window.write_event_value("-SELFCHECK-DONE-", payload)
        except Exception:
            pass

    def _on_selfcheck_done(self, payload):
        try:
            self.window["-SELFCHECK-"].update(disabled=False)
        except Exception:
            pass
        rc = payload.get("rc")
        out = str(payload.get("out") or "").strip()
        if len(out) > 20000:
            out = out[-20000:]
        passed = rc == 0
        self._set_status("自检%s（退出码 %s）" % ("通过" if passed else "失败", rc))
        if self.smoke:
            print("[selfcheck] rc=%s\n%s" % (rc, out), file=sys.stderr)
            return
        sg.popup_scrolled(
            out or "(无输出)",
            title="自检结果 · %s（退出码 %s）" % ("通过" if passed else "失败", rc),
            size=(100, 26),
        )

    # Run / teardown

    def _report_error(self, title, message):
        log_error("[%s]\n%s" % (title, message))
        if self.smoke:
            return
        try:
            sg.popup_error(message, title=title)
        except Exception:
            pass

    def run(self):
        """Run the event loop; per-event errors pop up without exiting."""
        try:
            if self.smoke:
                event, _values = self.window.read(timeout=200)
                print("[smoke] window tick event=%r" % (event,))
                return 0
            while True:
                try:
                    event, values = self.window.read(timeout=LEVEL_REFRESH_MS)
                    if event in (None, sg.WINDOW_CLOSED):
                        break
                    self._handle_event(event, values)
                except Exception:
                    self._report_error("变声器运行异常", traceback.format_exc())
            return 0
        finally:
            self.shutdown()

    def shutdown(self):
        """Stop the pipeline, flush profiles, close the window; idempotent."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self._on_stop()
        except Exception:
            self._report_error("停止流水线失败", traceback.format_exc())
        if self.dirty and not self.smoke:
            try:
                save_profiles(self.profiles, self.profiles_path)
                self.dirty = False
            except Exception:
                self._report_error("保存 profiles.json 失败", traceback.format_exc())
        elif self.dirty and self.smoke:
            print("[smoke] 有档案改动（冒烟模式不写盘）: active=%r" % (self.profiles.get("active"),))
        try:
            self.window.close()
        except Exception:
            pass


def run_app(profiles=None, profiles_error="", smoke=False):
    """Build the App and run it (the main.py entry)."""
    app = App(profiles=profiles, profiles_error=profiles_error, smoke=smoke)
    if app.smoke:
        print(
            "[smoke] hostapi=%s models=%d in_devices=%d out_devices=%d"
            % (
                app.devices["hostapi"],
                len(app.models),
                len(app.devices["input"]),
                len(app.devices["output"]),
            )
        )
        if app.models_error:
            print("[smoke] models_error: %s" % app.models_error, file=sys.stderr)
        if app._devices_error:
            print("[smoke] devices_error: %s" % app._devices_error, file=sys.stderr)
        print(
            "[smoke] model=%r in=%r mon=%r cable=%r monitor_on=%r"
            % (
                app.current_model,
                app.window["-IN-DEV-"].get(),
                app.window["-MON-DEV-"].get(),
                app.window["-CABLE-DEV-"].get(),
                app.monitor_on,
            )
        )
        try:
            gate_disabled = bool(app.window["-VOICE-GATE-"].Disabled)
        except Exception:
            gate_disabled = None
        print(
            "[smoke] denoise=%r(下拉 %r) voice_gate=%r threshold=%s voice_profile=%r "
            "gate_checkbox_disabled=%r hint=%r"
            % (
                app.denoise_mode,
                app.window["-DENOISE-MODE-"].get(),
                app.voice_gate_on,
                app.slider_values.get("-VG-THR-"),
                app.voice_profile_ok,
                gate_disabled,
                app.window["-VOICE-HINT-"].get(),
            )
        )
        print("[smoke] score_text=%r enroll_button=%r" % (app._score_text, app.window["-ENROLL-"].get_text()))
        print(
            "[smoke] agc=%r target=%r 增益显示=%r hint=%r"
            % (
                app.agc_on,
                app.slider_values.get("-AGC-TARGET-"),
                app.window["-AGC-GAIN-"].get(),
                app.window["-AGC-HINT-"].get(),
            )
        )
        print(
            "[smoke] 变声参数: block=%s(%r ms) crossfade=%.3f extra=%.1f threhold=%s "
            "index_rate=%.2f n_cpu=%s  [热更 %s / 重启 %s]"
            % (
                app.window["-BLOCK-TIME-"].get(),
                app.window["-BLOCK-TIME-VAL-"].get(),
                float(app.slider_values.get("-CROSSFADE-", 0.0)),
                float(app.slider_values.get("-EXTRA-", 0.0)),
                app.window["-THREHOLD-VAL-"].get(),
                float(app.slider_values.get("-INDEX-RATE-", 0.0)),
                app.window["-N-CPU-"].get(),
                "/".join(ADV_HOT_KEYS),
                "/".join(ADV_RESTART_KEYS),
            )
        )
    rc = app.run()
    if app.smoke:
        print("[smoke] OK: window built, single non-blocking tick done, no main loop")
    return rc
