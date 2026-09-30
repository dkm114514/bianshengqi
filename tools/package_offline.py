"""Build a relocatable offline application from a working RVC installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.package_source import source_files
from engine.models import rvc_root

VERSION = "0.1.0"
SKIP_DIRS = {"__pycache__", ".git", ".cache"}
PRIVATE_NAMES = {"profiles.json", "voice_profile.npz", "runtime.local.txt", "config.json"}


def copy_tree(source, target, excluded_files=()):
    """Copy runtime/source trees without caches, bytecode, symlinks or local settings."""
    source, target = Path(source), Path(target)
    for directory, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not (Path(directory) / d).is_symlink())
        relative = Path(directory).relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            origin = Path(directory) / name
            if origin.is_symlink() or name.endswith((".pyc", ".pyo", ".log")):
                continue
            if (relative / name).as_posix() in excluded_files:
                continue
            shutil.copy2(origin, target / relative / name)


def build_bundle(target, rvc=None, default_model=None):
    target = Path(target).resolve()
    rvc = Path(rvc or rvc_root()).resolve()
    default_model = Path(default_model or (rvc / "assets" / "weights" / "bb48k.pth")).resolve()
    if target.exists():
        raise FileExistsError("Use a new staging directory: %s" % target)
    if target == ROOT or target == rvc or target.is_relative_to(rvc):
        raise ValueError("Bundle output must not overwrite a working installation")
    required = [rvc / "runtime" / "python.exe", rvc / "infer" / "lib" / "rtrvc.py",
                rvc / "configs" / "config.py", rvc / "tools" / "torchgate" / "torchgate.py",
                rvc / "assets" / "hubert" / "hubert_base.pt",
                rvc / "assets" / "rmvpe" / "rmvpe.pt", rvc / "LICENSE", default_model,
                ROOT / "models" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                ROOT / "models" / "silero_vad.onnx"]
    required += [ROOT / "models" / "dfn3-512-v1" / name for name in
                 ("denoiser_model.onnx", "initial_states.npz", "meta.json")]
    required += [ROOT / "downloads" / "vbcable" / "VBCABLE_Setup_x64.exe"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Offline build assets missing: %s" % ", ".join(missing))
    target.mkdir(parents=True)
    for path in source_files(ROOT):
        destination = target / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print("Copying the validated Python/CUDA runtime...", flush=True)
    copy_tree(rvc / "runtime", target / "RVC" / "runtime")
    for name in ("infer", "configs", "tools/torchgate"):
        print("Copying RVC %s..." % name, flush=True)
        copy_tree(rvc / name, target / "RVC" / name,
                  excluded_files=("inuse/config.json",))
    for filename in ("LICENSE", "MIT协议暨相关引用库协议"):
        path = rvc / filename
        if path.is_file():
            destination = "LICENSE" if filename == "LICENSE" else "THIRD_PARTY_LICENSES.txt"
            shutil.copy2(path, target / "RVC" / destination)
    for relative in ("assets/hubert/hubert_base.pt", "assets/rmvpe/rmvpe.pt"):
        destination = target / "RVC" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rvc / relative, destination)
    weights = target / "RVC" / "assets" / "weights"
    weights.mkdir(parents=True)
    shutil.copy2(default_model, weights / default_model.name)
    # The default index rate is zero; no private/training indices are required.
    (target / "RVC" / "logs").mkdir(parents=True)
    for path in required:
        if path.is_relative_to(ROOT / "models"):
            destination = target / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    copy_tree(ROOT / "downloads" / "vbcable", target / "drivers" / "vbcable")
    if (ROOT / "downloads" / "repair_deps" / "pycaw").is_dir():
        for path in (ROOT / "downloads" / "repair_deps").glob("pycaw*"):
            if path.is_dir():
                copy_tree(path, target / "RVC" / "runtime" / "Lib" / "site-packages" / path.name)
    marker = {"version": VERSION, "edition": "windows-x64-cu118-offline",
              "default_model": default_model.name, "network_required": False,
              "includes_personal_voiceprint": False}
    (target / "offline_bundle.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    for path in target.rglob("*"):
        if path.is_file() and path.name in PRIVATE_NAMES:
            # RVC package/model definitions named config.json are expected except the GUI settings.
            if path.name == "config.json" and path != target / "RVC" / "configs" / "inuse" / "config.json":
                continue
            raise ValueError("Private local file found in bundle: %s" % path)
    manifest = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            manifest[path.relative_to(target).as_posix()] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
    (target / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Offline bundle ready: %s (%d files, %.2f GiB)" %
          (target, len(manifest), sum(item["bytes"] for item in manifest.values()) / 1024**3), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "offline-app")
    parser.add_argument("--rvc-root", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    build_bundle(args.output, args.rvc_root, args.model)
