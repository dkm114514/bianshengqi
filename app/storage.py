"""Atomic local writes that tolerate brief Windows file locks."""
import errno
import json
import os
from pathlib import Path
import tempfile
import time


def replace_with_retry(source, target):
    """Keep the original file intact if a scanner or another reader locks it."""
    delays = (0.03, 0.06, 0.12, 0.24, 0.24)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            transient = (isinstance(exc, PermissionError)
                         or exc.errno in (errno.EACCES, errno.EPERM, errno.EBUSY)
                         or getattr(exc, "winerror", None) in (5, 32, 33))
            if not transient or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_write_json(path, content):
    """Use a different temporary file for every save, in the destination folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, str(path))
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
