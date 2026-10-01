"""Keep application state and dependency caches beside the portable program."""
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROFILES_PATH = DATA_DIR / "profiles.json"
VOICE_PROFILE_PATH = DATA_DIR / "voice_profile.npz"
LOG_PATH = ROOT / "logs" / "运行日志.log"
_log_stream = None
_log_lock = threading.RLock()


def configure_runtime(root=ROOT):
    """Configure this process and its children before loading any audio/ML libraries."""
    root = Path(root).resolve()
    folders = {
        "TMP": root / "cache" / "tmp",
        "TEMP": root / "cache" / "tmp",
        "TMPDIR": root / "cache" / "tmp",
        "XDG_CACHE_HOME": root / "cache",
        "TORCH_HOME": root / "cache" / "torch",
        "HF_HOME": root / "cache" / "huggingface",
        "HF_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "cache" / "huggingface" / "transformers",
        "NUMBA_CACHE_DIR": root / "cache" / "numba",
        "CUDA_CACHE_PATH": root / "cache" / "cuda",
        "MPLCONFIGDIR": root / "cache" / "matplotlib",
    }
    try:
        for folder in {root / "data", root / "logs", *folders.values()}:
            folder.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".write-check-", dir=str(root / "data"))
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        raise OSError("程序文件夹无法写入。请把整个变声器文件夹移到自己的桌面或其他可写目录后再启动。") from exc
    os.environ.update({name: str(folder) for name, folder in folders.items()})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(root / "cache" / "tmp")
    if (root / "offline_bundle.json").is_file():
        os.environ.update(BSQ_OFFLINE="1", BSQ_RVC_ROOT=str(root / "RVC"),
                          RVC_ROOT=str(root / "RVC"),
                          RVC_RUNTIME=str(root / "RVC" / "runtime" / "python.exe"),
                          BSQ_WEIGHTS_DIR=str(root / "RVC" / "assets" / "weights"),
                          BSQ_INDEX_DIR=str(root / "RVC" / "logs"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        os.environ.pop("BSQ_CONFIG_JSON", None)
    return root


def migrate_legacy_data(root=ROOT):
    """Copy older local data once; never remove originals or overwrite new data."""
    root = Path(root)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    for source in (root / "profiles.json", root / "models" / "voice_profile.npz"):
        target = data / source.name
        if source.is_file() and not target.exists():
            from app.storage import replace_with_retry
            fd, staged = tempfile.mkstemp(prefix=".migrate-", dir=str(data))
            os.close(fd)
            try:
                shutil.copy2(source, staged)
                replace_with_retry(staged, str(target))
            finally:
                if os.path.exists(staged):
                    os.unlink(staged)


class AlreadyRunningError(RuntimeError):
    pass


class InstanceLock:
    """The OS releases the folder-local lock even if the app crashes."""
    def __init__(self, path=DATA_DIR / ".instance.lock"):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                self.stream.write(b"\0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise AlreadyRunningError("这个文件夹里的变声器已经在运行，请切回原来的窗口。") from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class LogStream(io.TextIOBase):
    """A missing/invalid console handle must never turn logging into another error."""
    def __init__(self, mirror, log):
        self.mirror = mirror
        self.log = log

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):
        return self.log.fileno()

    def write(self, text):
        text = str(text)
        with _log_lock:
            try:
                self.log.write(text)
                self.log.flush()
            except (OSError, ValueError):
                pass
            if self.mirror is not None:
                try:
                    self.mirror.write(text)
                    self.mirror.flush()
                except (OSError, ValueError):
                    self.mirror = None
        return len(text)

    def flush(self):
        with _log_lock:
            for stream in (self.log, self.mirror):
                if stream is not None:
                    try:
                        stream.flush()
                    except (OSError, ValueError):
                        pass


def install_logging(path=LOG_PATH):
    global _log_stream
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 2 * 1024 * 1024:
        try:
            os.replace(str(path), str(path.with_suffix(".previous.log")))
        except OSError:
            pass
    _log_stream = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = LogStream(sys.stdout, _log_stream)
    sys.stderr = LogStream(sys.stderr, _log_stream)


def log_error(message):
    """Safe even before startup completes or when a caller imports the GUI directly."""
    try:
        print(message, file=sys.stderr)
    except (OSError, ValueError, AttributeError):
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as log:
                log.write(str(message) + "\n")
        except OSError:
            pass
