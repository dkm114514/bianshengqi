"""Validated voiceprint transfer with atomic writes and automatic local backups."""
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def backup_profile(path, backup_dir=None):
    """Copy a profile byte-for-byte; verify the copy before any replacement/deletion."""
    if not os.path.isfile(path):
        return None
    backup_dir = backup_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "voice_backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    name, ext = os.path.splitext(os.path.basename(path))
    target = os.path.join(backup_dir, "%s_%s%s" % (name, stamp, ext))
    shutil.copy2(path, target)
    if _hash(path) != _hash(target):
        raise OSError("声纹备份校验失败，已取消操作")
    return target


def _validate(path, model_dir=None, threshold=0.2):
    if os.path.splitext(path)[1].lower() not in (".npz", ".json"):
        raise ValueError("声纹档案仅支持 .npz 或 .json")
    from dsp.voice_gate import VoiceGate

    gate = VoiceGate(model_dir=model_dir, threshold=threshold, profile_path=path)
    gate.load_profile(path)  # Explicit validation; constructor ignores corrupt auto-loads.
    return gate


def _atomic_transfer(source, target, gate=None):
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(target)[1].lower()
    fd, temporary = tempfile.mkstemp(prefix=".voice_transfer_", suffix=suffix, dir=parent)
    os.close(fd)
    try:
        if gate is None:
            shutil.copy2(source, temporary)
        else:
            gate.save_profile(temporary)
        os.replace(temporary, target)
    finally:
        if os.path.isfile(temporary):
            os.remove(temporary)


def import_profile(source, target, model_dir=None, threshold=0.2):
    """Validate first, preserve the old profile, then atomically install an NPZ."""
    source, target = os.path.abspath(source), os.path.abspath(target)
    gate = _validate(source, model_dir, threshold)
    if _same_path(source, target):
        return None
    # Stage before backing up/replacing, so a failed conversion leaves the active file intact.
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=".voice_import_", suffix=".npz", dir=parent)
    os.close(fd)
    try:
        if source.lower().endswith(".npz"):
            shutil.copy2(source, staged)
        else:
            gate.save_profile(staged)
        backup = backup_profile(target)
        os.replace(staged, target)
        return backup
    finally:
        if os.path.isfile(staged):
            os.remove(staged)


def export_profile(source, target, model_dir=None, threshold=0.2):
    """NPZ export preserves exact bytes; JSON carries the current voiceprint and threshold."""
    source, target = os.path.abspath(source), os.path.abspath(target)
    extension = os.path.splitext(target)[1].lower()
    if not extension:
        target += ".npz"
        extension = ".npz"
    if extension not in (".npz", ".json"):
        raise ValueError("导出格式仅支持 .npz 或 .json")
    gate = _validate(source, model_dir, threshold)
    if not _same_path(source, target):
        _atomic_transfer(source, target, gate if extension == ".json" else None)
    return target


def delete_profile(path):
    """Deletion always preserves a verified backup first."""
    backup = backup_profile(path)
    if backup is not None:
        os.remove(path)
    return backup
