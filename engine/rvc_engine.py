"""RvcEngine: loader + hot-update wrapper for RVC realtime inference.

Mirrors gui_v1.py start_vc: Config()/rtrvc.RVC() built with cwd at the RVC
root (relative paths like assets/hubert depend on it); the RVC root goes
first on sys.path; inp_q/opt_q are multiprocessing queues; hot-swap passes
the previous instance as last_rvc to reuse hubert/rmvpe caches.
Contract: RvcEngine(config_json_path=None); .load(pth, index) -> self;
.set_param(key, value); .tgt_sr; .rvc/.config/.inp_q/.opt_q/.params.
Params: project configs/inuse/config.json (or BSQ_CONFIG_JSON) is the only
active source; the official GUI legacy config donates machine-local info
only (paths, devices). GPU pitch is rmvpe only; pm/harvest are CPU
fallbacks. Call load() with the audio callback stopped.
"""

import contextlib
import json
import math
import os
import sys
import threading
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, Optional

from .models import PROJECT_ROOT, rvc_root
from .defaults import DEFAULT_MODEL_PROFILE

#: Default active project params (tuned for the bb48k.pth + guanguanV1.index combo)
#: overridden item by item when project configs/inuse/config.json exists; the legacy
#: config from the RVC official GUI never overrides them (see PROJECT_OWNED_PARAMS).
DEFAULT_PARAMS: Dict[str, Any] = {
    "pth_path": "",
    "index_path": "",
    "pitch": DEFAULT_MODEL_PROFILE["pitch"],
    "formant": 0.0,
    "index_rate": 0.0,
    "block_time": DEFAULT_MODEL_PROFILE["block_time"],
    "crossfade_length": DEFAULT_MODEL_PROFILE["crossfade_time"],
    "extra_time": DEFAULT_MODEL_PROFILE["extra_time"],
    "threhold": DEFAULT_MODEL_PROFILE["threhold"],
    "I_noise_reduce": True,
    "O_noise_reduce": DEFAULT_MODEL_PROFILE["O_noise_reduce"],
    "rms_mix_rate": 0.0,
    "f0method": "rmvpe",
    "n_cpu": DEFAULT_MODEL_PROFILE["n_cpu"],
    "sr_type": "sr_device",
    "use_jit": False,
    "use_pv": False,
}

#: Numeric params forwarded to the RVC object (set_param coerces them to float)
NUMERIC_PARAMS = (
    "pitch",
    "formant",
    "index_rate",
    "n_cpu",
    "block_time",
    "crossfade_length",
    "extra_time",
    "threhold",
    "rms_mix_rate",
)

#: This project keeps only rmvpe as the GPU pitch algorithm (pm / harvest are CPU fallbacks)
GPU_F0METHOD = "rmvpe"
CPU_F0METHODS = ("pm", "harvest")
#: Supported f0 algorithm set for this project
PROJECT_F0METHODS = (GPU_F0METHOD,) + CPU_F0METHODS
#: Algorithms rtrvc knows but this project does not use: rejected back to rmvpe with a warning
REJECTED_F0METHODS = ("crepe", "fcpe")

#: Legal ranges for numeric params, clamped to the bounds with a warning; defaults are all in range
PARAM_RANGES: Dict[str, tuple] = {
    "pitch": (-24.0, 24.0),
    "formant": (-8.0, 8.0),
    "index_rate": (0.0, 1.0),
    "rms_mix_rate": (0.0, 1.0),
    "block_time": (0.01, 1.0),
    "crossfade_length": (0.0, 1.0),
    "extra_time": (0.0, 5.0),
    "threhold": (-60.0, 0.0),
    "n_cpu": (0.0, 32.0),
}

#: Known keys that are only recorded, never applied: paths and machine-local info
#: from config files, no unknown-param warning
RECORD_ONLY_PARAMS = frozenset(
    {
        "pth",
        "index",
        "sg_hostapi",
        "sg_wasapi_exclusive",
        "sg_input_device",
        "sg_output_device",
    }
)

#: The project's own params (active params), higher priority than the legacy
#: configs/inuse/config.json left by the RVC official GUI
PROJECT_OWNED_PARAMS = (
    "pitch",
    "formant",
    "index_rate",
    "block_time",
    "crossfade_length",
    "extra_time",
    "threhold",
    "I_noise_reduce",
    "O_noise_reduce",
    "rms_mix_rate",
    "f0method",
)

#: External key aliases / case folding (lowercased for lookup; unknown keys kept as-is)
_LOWER_TO_CANON = {key.lower(): key for key in DEFAULT_PARAMS}
_LOWER_TO_CANON.update(
    {
        "key": "pitch",
        "f0_up_key": "pitch",
        "formant_shift": "formant",
        "f0_method": "f0method",
        "crossfade_time": "crossfade_length",
        "threshold": "threhold",
    }
)

#: Serial lock for cwd switches (os.chdir is process-global state)
_CWD_LOCK = threading.RLock()


def _log(message: str) -> None:
    print("[engine] %s" % message, flush=True)


def canonical_key(key) -> str:
    """Normalize an external key (aliases/case) to the canonical params key; unknown keys returned as-is."""
    text = str(key).strip()
    return _LOWER_TO_CANON.get(text.lower(), text)


def _norm_path(path) -> Optional[str]:
    """Normalize an index path, only used to tell whether it is a different file."""
    if not path:
        return None
    return os.path.normcase(os.path.abspath(str(path)))


@contextlib.contextmanager
def _chdir(path):
    """Temporarily switch cwd, restore on exit (use with _CWD_LOCK)."""
    previous = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _safe_argv():
    """Config() parses sys.argv and exits on unknown args; keep only known ones."""
    original = sys.argv
    argv = [original[0] if original else "python"]
    index = 1
    while index < len(original):
        token = original[index]
        option = token.split("=", 1)[0]
        if option in ("--port", "--pycmd"):
            argv.append(token)
            if (
                "=" not in token
                and index + 1 < len(original)
                and not original[index + 1].startswith("-")
            ):
                argv.append(original[index + 1])
                index += 1
        elif option in ("--colab", "--noparallel", "--noautoopen", "--dml"):
            argv.append(token)
        index += 1
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def _default_config_path():
    """Active config path: env > project configs > official GUI legacy config."""
    value = os.environ.get("BSQ_CONFIG_JSON")
    if value:
        return Path(value), False
    local = PROJECT_ROOT / "configs" / "inuse" / "config.json"
    if local.is_file():
        return local, False
    legacy = rvc_root() / "configs" / "inuse" / "config.json"
    if legacy.is_file():
        return legacy, True
    return None, False


def _read_config_json(path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError("找不到配置文件: %s" % file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("配置文件不是合法 JSON: %s (%s)" % (file_path, exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("配置文件内容必须是 JSON 对象: %s" % file_path)
    return data


class RvcEngine:
    """RVC realtime inference engine; load() then infer via .rvc, hot-update via set_param()."""

    def __init__(self, config_json_path=None):
        self.inp_q = Queue()
        self.opt_q = Queue()

        if config_json_path is not None:
            config_path: Optional[Path] = Path(config_json_path)
            legacy_config = False
            if not config_path.is_file():
                raise FileNotFoundError("配置文件不存在: %s" % config_path)
        else:
            config_path, legacy_config = _default_config_path()
        self.raw_config: Dict[str, Any] = (
            _read_config_json(config_path) if config_path is not None else {}
        )
        self.config_path: Optional[str] = (
            str(config_path) if config_path is not None else None
        )

        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        self.ignored_config: Dict[str, Any] = {}
        unknown = []
        for key, value in self.raw_config.items():
            name = canonical_key(key)
            if legacy_config and name in PROJECT_OWNED_PARAMS:
                self.ignored_config[name] = value
                continue
            self.params[name] = value
            if name not in DEFAULT_PARAMS and name not in RECORD_ONLY_PARAMS:
                unknown.append(name)
        if unknown:
            _log(
                "警告: 配置里有未知参数, 只记录不生效: %s"
                % ", ".join(sorted(set(unknown)))
            )
        self.params["f0method"] = self._check_f0method(self.params.get("f0method"))
        if self.ignored_config:
            _log(
                "现役参数以项目为准 (profiles.json), 已忽略 RVC 官方 GUI 遗留 config 的"
                "项目参数: %s"
                % ", ".join(
                    "%s=%r" % item for item in sorted(self.ignored_config.items())
                )
            )

        self._rvc = None
        self._config = None
        self._pth_path: Optional[str] = None
        self._index_path: Optional[str] = None
        self._lock = threading.RLock()
        self._harvest_warned = False
        self._index_warned = None
        self._index_zero_warned = None
        self._desired_index_rate: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def rvc(self):
        """Underlying rtrvc.RVC instance (None before load)."""
        return self._rvc

    @property
    def config(self):
        """RVC configs.config.Config instance (singleton, created on first access)."""
        return self._ensure_config()

    @property
    def is_loaded(self) -> bool:
        return self._rvc is not None and hasattr(self._rvc, "tgt_sr")

    @property
    def tgt_sr(self) -> Optional[int]:
        """Model output sample rate; None before a model is loaded."""
        rvc = self._rvc
        return getattr(rvc, "tgt_sr", None) if rvc is not None else None

    @property
    def pth_path(self) -> Optional[str]:
        """Loaded model pth path."""
        return self._pth_path

    @property
    def index_path(self) -> Optional[str]:
        """Loaded/effective index path (None when indexless)."""
        return self._index_path

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load(self, pth, index=None):
        """Load/hot-swap a model, return self; missing index or index/model dim mismatch degrades index_rate to 0."""
        if not pth:
            raise ValueError("load() 需要模型 pth 路径")
        pth_path = str(Path(pth).resolve())
        if not os.path.isfile(pth_path):
            raise FileNotFoundError("模型文件不存在: %s" % pth_path)
        index_path = None
        if index:
            candidate = str(Path(index).resolve())
            if os.path.isfile(candidate):
                index_path = candidate
            else:
                _log("警告: 索引文件不存在, 已忽略: %s" % candidate)

        root = rvc_root()
        with self._lock:
            params = self.params
            pitch = self._sanitize_number("pitch", DEFAULT_PARAMS["pitch"])
            formant = self._sanitize_number("formant", DEFAULT_PARAMS["formant"])
            index_rate = self._sanitize_number("index_rate", 0.0)
            if index_rate == 0 and self._desired_index_rate and index_path is not None:
                index_rate = self._desired_index_rate
            if index_rate != 0 and index_path is None:
                _log("警告: 没有可用索引, index_rate 由 %.2f 降级为 0" % index_rate)
                self._desired_index_rate = index_rate
                index_rate = 0.0
            if index_rate != 0:
                self._desired_index_rate = index_rate
            n_cpu = self._sanitize_number("n_cpu", DEFAULT_PARAMS["n_cpu"])
            self._prepare_sys_path(root)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # surface torch errors at rtrvc import instead
                pass

            with _CWD_LOCK, _chdir(root):
                try:
                    from infer.lib import rtrvc
                except ImportError as exc:
                    raise ImportError(
                        "无法导入 RVC 实时推理模块 (RVC 根目录: %s): %s" % (root, exc)
                    ) from exc
                config = self._ensure_config()
                previous = None
                if self._rvc is not None and getattr(self._rvc, "net_g", None) is not None:
                    previous = self._rvc
                new_rvc = rtrvc.RVC(
                    pitch,
                    formant,
                    pth_path,
                    index_path,
                    index_rate,
                    n_cpu,
                    self.inp_q,
                    self.opt_q,
                    config,
                    previous,
                )

            if getattr(new_rvc, "net_g", None) is None or not hasattr(new_rvc, "tgt_sr"):
                raise RuntimeError(
                    "RVC 初始化失败 (pth=%s, index=%s), 请检查上方 RVC 打印的 "
                    "traceback 以及模型/索引文件是否匹配" % (pth_path, index_path)
                )

            self._wrap_rmvpe(new_rvc, root)
            self._clear_index_warn()
            self._harvest_warned = False
            self._rvc = new_rvc
            self._pth_path = pth_path
            self._index_path = index_path
            params["pth_path"] = pth_path
            params["index_path"] = index_path or ""
            params["index_rate"] = index_rate
            self._check_index_compat(new_rvc)  # mismatch drops index_rate to 0
            self._apply_harvest_guard(new_rvc)
            _log(
                "模型已加载: %s | index=%s | tgt_sr=%s"
                % (
                    Path(pth_path).name,
                    Path(index_path).name if index_path else "无",
                    self.tgt_sr,
                )
            )
        return self

    # ------------------------------------------------------------------ #
    # Hot-update
    # ------------------------------------------------------------------ #

    def set_param(self, key, value):
        """Hot-update one param; pitch/formant/index_rate/n_cpu/index_path apply to the RVC object, rest are recorded."""
        name = canonical_key(key)
        if name not in DEFAULT_PARAMS and name not in RECORD_ONLY_PARAMS:
            _log("警告: 未知参数 %r, 只记录不生效" % (name,))
        with self._lock:
            had_old = name in self.params
            old_value = self.params.get(name)
            old_index_path = self._index_path
            old_marks = (
                self._index_warned,
                self._index_zero_warned,
                self._desired_index_rate,
            )
            rvc = None
            old_rvc = None
            try:
                if name in NUMERIC_PARAMS:
                    value = self._clamp_number(value, name)
                if name == "f0method":
                    value = self._check_f0method(value)
                elif name == "index_rate":
                    value = self._coerce_index_rate(value)
                elif name == "index_path":
                    old_path = self._index_path
                    value = self._resolve_index_path(value) or ""
                    changed = _norm_path(old_path) != _norm_path(value or None)
                    self._index_path = value or None
                    self._clear_index_warn()

                self.params[name] = value

                rvc = self._rvc
                if rvc is None:
                    return
                old_rvc = (
                    getattr(rvc, "f0_up_key", None),
                    getattr(rvc, "formant_shift", None),
                    getattr(rvc, "index_rate", None),
                    getattr(rvc, "n_cpu", None),
                )
                if name == "pitch":
                    rvc.change_key(value)
                elif name == "formant":
                    rvc.change_formant(value)
                elif name == "index_rate":
                    rvc.change_index_rate(value)
                    self._check_index_compat(rvc)
                elif name == "n_cpu":
                    rvc.n_cpu = value
                    self._apply_harvest_guard(rvc)
                elif name == "index_path":
                    self._reload_index(rvc, value or None, changed)
                elif name == "f0method":
                    self._apply_harvest_guard(rvc)
            except Exception:
                if had_old:
                    self.params[name] = old_value
                else:
                    self.params.pop(name, None)
                self._index_path = old_index_path
                (
                    self._index_warned,
                    self._index_zero_warned,
                    self._desired_index_rate,
                ) = old_marks
                if rvc is not None and old_rvc is not None:
                    # Rollback touches attrs only, never change_index_rate (avoids re-reading the index).
                    try:
                        for attr, val in zip(
                            ("f0_up_key", "formant_shift", "index_rate", "n_cpu"),
                            old_rvc,
                        ):
                            if val is not None and hasattr(rvc, attr):
                                setattr(rvc, attr, val)
                    except Exception:
                        pass
                raise

    def get_param(self, key, default=None):
        """Read a param (keys support aliases/case)."""
        return self.params.get(canonical_key(key), default)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_sys_path(root: Path) -> None:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    def _ensure_config(self):
        """RVC Config singleton; construction needs the RVC-root cwd, created on first call."""
        if self._config is not None:
            return self._config
        root = rvc_root()
        with _CWD_LOCK:
            self._prepare_sys_path(root)
            with _chdir(root), _safe_argv():
                try:
                    from configs.config import Config
                except ImportError as exc:
                    raise ImportError(
                        "无法导入 RVC 配置模块 configs.config (RVC 根目录: %s): %s"
                        % (root, exc)
                    ) from exc
                self._config = Config()
        return self._config

    @staticmethod
    def _wrap_rmvpe(rvc, root: Path) -> None:
        """rmvpe loads assets/rmvpe/rmvpe.pt by relative path on first infer; wrap it to guarantee cwd."""
        original = getattr(rvc, "get_f0_rmvpe", None)
        if not callable(original):
            return

        def get_f0_rmvpe(x, f0_up_key):
            with _CWD_LOCK, _chdir(root):
                return original(x, f0_up_key)

        rvc.get_f0_rmvpe = get_f0_rmvpe

    @staticmethod
    def _coerce_number(value, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s 需要是数字: %r" % (name, value))
        if not math.isfinite(number):
            raise ValueError("%s 需要是有限数字: %r" % (name, value))
        return number

    @staticmethod
    def _clamp_number(value, name: str) -> float:
        number = RvcEngine._coerce_number(value, name)
        bounds = PARAM_RANGES.get(name)
        if bounds is None:
            return number
        low, high = bounds
        clamped = min(max(number, low), high)
        if clamped != number:
            _log(
                "警告: %s=%r 超出范围 [%r, %r], 已夹到 %r"
                % (name, value, low, high, clamped)
            )
        return clamped

    def _sanitize_number(self, name: str, default: float) -> float:
        try:
            return self._clamp_number(self.params.get(name, default), name)
        except ValueError:
            _log(
                "警告: params[%s]=%r 非法, 用缺省 %r"
                % (name, self.params.get(name), default)
            )
            return float(default)

    @staticmethod
    def _check_f0method(value) -> str:
        """Normalize f0method: only rmvpe (GPU) and pm/harvest (CPU fallbacks), everything else falls back to rmvpe."""
        method = str(value or "").strip().lower()
        if method == GPU_F0METHOD or method in CPU_F0METHODS:
            return method
        if method in REJECTED_F0METHODS:
            _log("警告: 本项目 GPU 音高算法只用 rmvpe, 已忽略 f0method=%r" % (value,))
            return GPU_F0METHOD
        _log("警告: 未知 f0method %r, 回退为 rmvpe" % (value,))
        return GPU_F0METHOD

    def _coerce_index_rate(self, value) -> float:
        # Stash non-0 requests; only the effective value stays 0 on mismatch/missing index.
        rate = self._clamp_number(value, "index_rate")
        if rate == 0:
            self._desired_index_rate = None
            return rate
        self._desired_index_rate = rate
        if self._rvc is None:
            return rate
        if self._index_warned is not None:
            # Current index already judged incompatible: zero directly, skip faiss re-read.
            return 0.0
        if not (self._index_path and os.path.isfile(self._index_path)):
            _log("警告: 当前模型没有可用索引, index_rate 保持 0")
            return 0.0
        return rate

    @staticmethod
    def _resolve_index_path(value) -> Optional[str]:
        if not value:
            return None
        path = str(Path(value).resolve())
        if not os.path.isfile(path):
            _log("警告: 索引文件不存在, 已忽略: %s" % path)
            return None
        return path

    def _reload_index(
        self, rvc, path: Optional[str], retry_desired: bool = False
    ) -> None:
        old_path = rvc.index_path
        old_rate = float(getattr(rvc, "index_rate", 0.0) or 0.0)
        rvc.index_path = path
        try:
            target = float(self.params.get("index_rate", 0.0) or 0.0)
            if target == 0 and retry_desired and self._desired_index_rate:
                target = self._desired_index_rate
            if target == 0:
                return
            if path is None:
                _log("警告: 索引不可用, index_rate 降级为 0")
                self.params["index_rate"] = 0.0
                rvc.change_index_rate(0.0)
                return
            rvc.index_rate = 0.0  # force change_index_rate to re-read the faiss index
            rvc.change_index_rate(target)
            if self._check_index_compat(rvc):
                self.params["index_rate"] = target
        except Exception:
            rvc.index_path = old_path
            try:
                rvc.change_index_rate(old_rate)
            except Exception:
                rvc.index_rate = 0.0
                _log("警告: 索引回滚失败, index_rate 置 0 保出声")
            raise

    def _clear_index_warn(self) -> None:
        """Forget the previous combo's incompatibility verdict and warning throttle on model/index change."""
        self._index_warned = None
        self._index_zero_warned = None

    def _check_index_compat(self, rvc) -> bool:
        """Drop index_rate to 0 on index/model dim mismatch; each warning fires once per combo."""
        index = getattr(rvc, "index", None)
        if index is None:
            return True
        version = getattr(rvc, "version", "v1")
        expected = 256 if version == "v1" else 768
        dim = getattr(index, "d", None)
        if dim is None or dim == expected:
            return True
        marker = (dim, version)
        if marker != self._index_warned:
            self._index_warned = marker
            _log(
                "警告: 索引维度 %s 与模型版本 %s (期望 %s 维) 不匹配, "
                "RVC 会跳过索引检索" % (dim, version, expected)
            )
        if float(getattr(rvc, "index_rate", 0.0) or 0.0) != 0:
            rvc.index_rate = 0.0
            self.params["index_rate"] = 0.0
            if marker != self._index_zero_warned:
                self._index_zero_warned = marker
                _log("警告: 索引与模型不匹配, index_rate 已降为 0")
        return False

    def _apply_harvest_guard(self, rvc) -> None:
        """harvest is only safe with n_cpu == 1; other values are lowered to 1 in params."""
        method = str(self.params.get("f0method", "")).lower()
        raw = self.params.get("n_cpu", 0)
        try:
            n_cpu = float(raw)
        except (TypeError, ValueError):
            n_cpu = float("nan")
        if method != "harvest":
            self._harvest_warned = False
            return
        if n_cpu == 1:
            return
        rvc.n_cpu = 1
        self.params["n_cpu"] = 1
        if not self._harvest_warned:
            _log(
                "警告: f0method=harvest 且 n_cpu=%r 不是 rtrvc 的安全单进程分支, "
                "本引擎未启动 Harvest 工作进程, 已将 n_cpu 降为 1" % (raw,)
            )
            self._harvest_warned = True
