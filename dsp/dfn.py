"""Module G2: DeepFilterNet3 streaming denoise (ONNX/CPU) for input denoise.

Contract: DfnDenoiser(sr=48000); process(chunk) takes any-length float32
mono and returns equal length (state kept across blocks, short tails
zero-padded); reset() clears state; latency_ms(block=None) reports the
algorithmic latency (32.0 ms native, plus block-alignment padding).

Key facts: 48 kHz native with 512-sample frames; ~3.8 ms per 60 ms block
single-threaded (num_threads=1 default); the model auto-downloads to
models/dfn3-512-v1/ with sha256 checks. DFN3 is a speech enhancer: steady
tones are suppressed as non-speech, and other voices are kept (that case
belongs to the voiceprint gate).
"""

from __future__ import annotations

import hashlib
import math
import os
import pathlib
import shutil
import socket
import threading
import urllib.request

import numpy as np

__all__ = ["DfnDenoiser"]

#: Model version dir name (matches the deepfilter-stream release tag).
MODEL_VERSION = "dfn3-512-v1"
#: Project model dir: <project root>/models/dfn3-512-v1/.
_PROJECT_MODEL_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "models" / MODEL_VERSION
)
#: Model-dir env var honored by deepfilter-stream assets.
_ENV_MODEL_DIR = "DEEPFILTER_STREAM_MODEL_DIR"
_MODEL_FILES = ("denoiser_model.onnx", "initial_states.npz", "meta.json")

#: Proxy mirrors tried in order when the official release is unreachable.
_MIRRORS = ("https://gh-proxy.com/", "https://ghfast.top/")
_OFFICIAL_TIMEOUT_S = 8.0
_MIRROR_TIMEOUT_S = 90.0

#: DFN3 native latency (ms); latency_ms(block) adds alignment padding.
LATENCY_MS = 32.0
#: STFT frame length (samples) for alignment-padding math.
_FRAME = 512


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_complete(model_dir) -> bool:
    """True when all three model files exist and are non-empty."""
    if not model_dir:
        return False
    path = pathlib.Path(model_dir)
    return all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in _MODEL_FILES
    )


def _package_cache_dir():
    """deepfilter-stream's own cache dir, or None when unavailable."""
    try:
        from platformdirs import user_cache_dir
        from deepfilter_stream import _meta

        return pathlib.Path(user_cache_dir("deepfilter-stream")) / _meta.MODEL_VERSION
    except Exception:
        return None


def _urlretrieve(url: str, dst, timeout: float) -> None:
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        urllib.request.urlretrieve(url, dst)
    finally:
        socket.setdefaulttimeout(old)


def _copy_model(src, dst) -> None:
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in _MODEL_FILES:
        source, target = pathlib.Path(src) / name, dst / name
        if source.is_file() and not target.is_file():
            shutil.copy2(source, target)


def _download_model(dst) -> None:
    """Download missing model files with per-file sha256 checks."""
    from deepfilter_stream import _meta

    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    sources = [(_meta.RELEASE_BASE_URL, _OFFICIAL_TIMEOUT_S)]
    sources += [(mirror + _meta.RELEASE_BASE_URL, _MIRROR_TIMEOUT_S) for mirror in _MIRRORS]

    for name, want in _meta.ASSETS.items():
        target = dst / name
        if target.is_file() and _sha256(target) == want:
            continue
        errors = []
        while sources:
            base, timeout = sources[0]
            part = target.with_name(target.name + ".part")
            try:
                _urlretrieve("%s/%s" % (base, name), part, timeout)
                digest = _sha256(part)
                if digest != want:
                    raise RuntimeError("sha256 校验失败: %s" % digest)
                os.replace(part, target)
                break
            except Exception as exc:  # noqa: BLE001 - per-source fallback, single error at the end
                errors.append("  %s -> %r" % (base, exc))
                try:
                    part.unlink()
                except OSError:
                    pass
                sources.pop(0)  # Dead source: skip it for the remaining files too.
        else:
            raise RuntimeError(
                "模型 %s 下载失败（官方源与镜像 %s 都不可用）：\n%s\n"
                "可手动下载 deepfilter-stream release %s 的三个文件放到 %s 后重试。"
                % (name, list(_MIRRORS), "\n".join(errors), MODEL_VERSION, dst)
            )


def _resolve_model_dir(model_dir=None) -> pathlib.Path:
    """Locate (downloading if needed) the model dir and export the env var."""
    # Offline releases must never repair missing assets over the network.
    offline = _PROJECT_MODEL_DIR.parent.parent / "offline_bundle.json"
    if offline.is_file() or os.environ.get("BSQ_OFFLINE") == "1":
        path = pathlib.Path(model_dir) if model_dir is not None else _PROJECT_MODEL_DIR
        if not _dir_complete(path):
            raise FileNotFoundError("离线安装包缺少 DFN3 模型，请重新安装完整发行包：%s" % path)
        os.environ[_ENV_MODEL_DIR] = str(path)
        return path
    if model_dir is not None:
        path = pathlib.Path(model_dir)
        if not _dir_complete(path):
            raise FileNotFoundError(
                "model_dir 缺少模型文件 %s：%s" % (list(_MODEL_FILES), path)
            )
        os.environ[_ENV_MODEL_DIR] = str(path)
        return path

    env_dir = os.environ.get(_ENV_MODEL_DIR)
    if _dir_complete(env_dir):
        return pathlib.Path(env_dir)

    project = _PROJECT_MODEL_DIR
    if not _dir_complete(project):
        cache = _package_cache_dir()
        if _dir_complete(cache):
            _copy_model(cache, project)
    if not _dir_complete(project):
        _download_model(project)
    os.environ[_ENV_MODEL_DIR] = str(project)
    return project


class DfnDenoiser:
    """DeepFilterNet3 (ONNX/CPU) streaming denoise with equal-block I/O.

    sr: sample rate, 48000 native (other rates go through streaming resample).
    model_dir: override the auto locate/download order. num_threads: ONNX
    intra-op threads, 1 by default (fastest measured). atten_lim_db: optional
    attenuation cap in dB, None means unlimited.
    """

    def __init__(self, sr: int = 48000, *, model_dir=None, num_threads: int = 1,
                 atten_lim_db=None):
        self.sr = int(sr)
        if self.sr <= 0:
            raise ValueError("sr 必须是正整数")
        resolved = _resolve_model_dir(model_dir)
        try:
            from deepfilter_stream import DeepFilterModel
        except ImportError as exc:
            raise ImportError(
                "dsp.dfn 需要 deepfilter-stream（依赖 onnxruntime）：用 RVC runtime python 执行 "
                "python -m pip install deepfilter-stream onnxruntime==1.19.2"
            ) from exc
        try:
            self._model = DeepFilterModel(
                model_path=str(resolved / "denoiser_model.onnx"),
                intra_op_num_threads=num_threads,
            )
        except Exception as exc:
            raise RuntimeError(
                "DeepFilterNet3 ONNX 会话加载失败（%s）。本模型声明了 ai.onnx.ml opset 4，"
                "需要 onnxruntime>=1.17（本机实测 1.15/1.16 加载即报 opset 错误，1.19.2 可用）："
                "python -m pip install onnxruntime==1.19.2" % (exc,)
            ) from exc
        self._stream = self._model.new_stream(atten_lim_db=atten_lim_db)
        self._frame = int(self._stream.frame_size)
        self._lock = threading.Lock()
        self._fifo = np.zeros(0, dtype=np.float32)

    # -- Contract ----------------------------------------------------------
    @staticmethod
    def latency_ms(block_size: int = None) -> float:
        """Algorithmic latency in ms; with block_size, alignment padding included."""
        if block_size is None:
            return LATENCY_MS
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size 必须是正整数")
        padding = _FRAME - math.gcd(block_size, _FRAME)  # Upper bound: 512 - gcd(block, 512).
        return LATENCY_MS + 1000.0 * padding / 48000.0

    def process(self, chunk) -> np.ndarray:
        """Denoise one block; returns equal-length float32 mono."""
        x = np.asarray(chunk)
        if x.ndim >= 3:
            raise ValueError("只收 mono/stereo，维度不支持：shape=%r" % (x.shape,))
        if x.ndim == 2:  # Fold any (n, ch)/(ch, n) stereo to mono.
            x = x.mean(axis=1 if x.shape[1] <= x.shape[0] else 0)
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            y = np.asarray(self._stream.process(x, sr=self.sr), dtype=np.float32).reshape(-1)
            if self._fifo.size:
                y = np.concatenate((self._fifo, y))
                self._fifo = np.zeros(0, dtype=np.float32)
            if y.size >= n:
                out = y[:n].copy()
                self._fifo = y[n:].copy()
            else:  # Short of one frame: zero-pad to equal length, debt repaid next call.
                out = np.concatenate((y, np.zeros(n - y.size, dtype=np.float32)))
        return out

    def reset(self) -> None:
        """Clear stream state and the internal FIFO."""
        with self._lock:
            self._stream.reset()
            self._fifo = np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Self-test: python -X utf8 dsp/dfn.py (run at the project root)
# ---------------------------------------------------------------------------
def _selftest() -> int:
    sr = 48000
    block = 2880  # 60 ms, matching engine.pipeline.
    rng = np.random.default_rng(20260917)

    def run(denoiser, signal, size=None):
        size = size or block
        return np.concatenate(
            [denoiser.process(signal[i:i + size]) for i in range(0, signal.size, size)]
        )

    def speech_like(dur=10.0):
        """Speech-like probe: harmonic stack with a syllable envelope."""
        t = np.arange(int(sr * dur)) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
        sig = np.zeros_like(t)
        for h in range(1, 25):
            freq = 150.0 * h
            if freq > 4000.0:
                break
            sig += (
                (1.0 / h)
                * np.exp(-((freq - 500.0) ** 2) / (2 * 800.0 ** 2))
                * np.sin(2 * np.pi * freq * t + 0.3 * h)
            )
        return (0.6 * env * sig / np.abs(sig).max()).astype(np.float32)

    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

    def xcorr_lag(x, y, max_lag=4096):
        n = 1 << 17
        spec = np.fft.rfft(y[:n], n) * np.conj(np.fft.rfft(x[:n], n))
        cc = np.fft.irfft(spec, n)
        return int(np.argmax(np.abs(cc[:max_lag])))

    def snr_gain(clean, mix, out, delay):
        """Input/output SNR in dB after alignment and least-squares gain."""
        n = int(min(clean.size, out.size - delay))
        trim = int(0.2 * sr)
        s = clean[:n][trim:n - trim].astype(np.float64)
        y = out[delay:delay + n][trim:n - trim].astype(np.float64)
        noise = mix[:n][trim:n - trim].astype(np.float64) - s
        gain = float(np.dot(y, s) / np.dot(s, s))
        residual = y - gain * s
        snr_in = 10 * np.log10(np.sum(s ** 2) / np.sum(noise ** 2))
        snr_out = 10 * np.log10((gain * gain) * np.sum(s ** 2) / np.sum(residual ** 2))
        return snr_in, snr_out

    ok = True
    denoiser = DfnDenoiser(sr=sr)
    print("模型目录 : %s" % os.environ[_ENV_MODEL_DIR])
    print("帧长     : %d 采样 @%d Hz；DFN 自身延迟 %.2f ms"
          % (denoiser._frame, sr, DfnDenoiser.latency_ms()))

    # 1) Latency: impulse peak vs declared value.
    impulse = np.zeros(int(sr * 1.5), dtype=np.float32)
    impulse[sr // 2] = 1.0
    for size in (block, 2560):
        declared = DfnDenoiser.latency_ms(size)
        denoiser.reset()
        sig = run(denoiser, impulse, size)
        lag = int(np.argmax(np.abs(sig))) - sr // 2
        lag_ms = lag * 1000.0 / sr
        passed = abs(lag_ms - declared) <= 1.0
        ok &= passed
        print("延迟(脉冲)     : 块长 %4d -> %d 采样 = %5.2f ms（声明 %5.2f ms）-> %s"
              % (size, lag, lag_ms, declared, "OK" if passed else "FAIL"))

    # 2) Denoise: speech-like probe plus white noise, expect >= 10 dB gain.
    clean = speech_like()
    mix = (clean + (rng.standard_normal(clean.size) * 0.095).astype(np.float32)).astype(np.float32)
    denoiser.reset()
    out = run(denoiser, mix)
    lag = xcorr_lag(mix, out)
    lag_ms = lag * 1000.0 / sr
    declared = DfnDenoiser.latency_ms(block)
    snr_in, snr_out = snr_gain(clean, mix, out, lag)
    gain_db = snr_out - snr_in
    lag_ok = abs(lag_ms - declared) <= 1.0
    snr_ok = gain_db >= 10.0
    ok &= lag_ok and snr_ok
    print("延迟(互相关)   : %d 采样 = %.2f ms（声明 %.2f ms）-> %s"
          % (lag, lag_ms, declared, "OK" if lag_ok else "FAIL"))
    print("信噪比         : 输入 %.1f dB -> 输出 %.1f dB（提升 %.1f dB）-> %s"
          % (snr_in, snr_out, gain_db, "OK" if snr_ok else "FAIL"))

    # 3) Reference only: steady sines read as non-speech, no SNR verdict.
    t = np.arange(int(sr * 3.0)) / sr
    sine = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    denoiser.reset()
    sine_out = run(denoiser, sine)
    print("纯 1 kHz 正弦  : 输出电平 %.1f dB（稳态纯音被判非语音压制，不作 SNR 判据）"
          % (20 * np.log10(max(rms(sine_out), 1e-12) / rms(sine))))

    print("自测结果       : %s" % ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
