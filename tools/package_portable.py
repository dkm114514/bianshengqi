"""Build the full portable extractor and a small offline application update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.version import VERSION
from tools.package_offline import build_bundle
from tools.package_source import source_files
from app.storage import atomic_write_json
import shutil


def refresh_source(bundle):
    """Refresh small source files while retaining hashes of untouched bundled binaries."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for path in source_files(ROOT):
        relative = path.relative_to(ROOT)
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        content = path.read_bytes()
        manifest[relative.as_posix()] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    marker_path = bundle / "offline_bundle.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.update(version=VERSION, portable=True, user_data_directory="data", cache_directory="cache")
    atomic_write_json(marker_path, marker)
    content = marker_path.read_bytes()
    manifest["offline_bundle.json"] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    atomic_write_json(bundle / "bundle_manifest.json", manifest)


def compile_packages(bundle, release, compiler=None):
    bundle, release = Path(bundle).resolve(), Path(release).resolve()
    if not (bundle / "offline_bundle.json").is_file():
        raise FileNotFoundError("Build the offline bundle before compiling")
    if any((bundle / name).exists() for name in ("data", "cache", "profiles.json", "models/voice_profile.npz")):
        raise ValueError("Compile a fresh staging folder without user data or caches")
    refresh_source(bundle)
    compiler = Path(compiler or (Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Inno Setup 6" / "ISCC.exe"))
    if not compiler.is_file():
        raise FileNotFoundError("Inno Setup compiler not found: %s" % compiler)
    release.mkdir(parents=True, exist_ok=True)
    log = release.parent / ("portable-build-%s.log" % VERSION)
    with log.open("w", encoding="utf-8") as output:
        for patch in (True, False):
            print("Compiling %s; build log: %s" % ("small update" if patch else "full portable package", log), flush=True)
            command = [str(compiler), "/Qp", "/DBundleDir=" + str(bundle),
                       "/DReleaseDir=" + str(release), "/DAppVersion=" + VERSION,
                       "/DPatchOnly=" + str(int(patch)), str(ROOT / "installer" / "portable.iss")]
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            output.flush()
            if result.returncode:
                raise RuntimeError("Package compiler failed with %s; inspect %s" % (result.returncode, log))
    checksums = []
    for path in sorted(release.glob("bianshengqi-%s-windows-x64-*" % VERSION)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums.append("%s  %s" % (digest.hexdigest(), path.name))
        print("%s: %d bytes" % (path.name, path.stat().st_size), flush=True)
    (release / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    expected = ["bianshengqi-%s-windows-x64-%s" % (VERSION, suffix) for suffix in
                ("portable.exe", "portable-1.bin", "portable-2.bin", "update.exe")]
    if any(not (release / name).is_file() for name in expected):
        raise RuntimeError("The expected full package and update files were not generated")
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / ("portable-app-%s" % VERSION))
    parser.add_argument("--release", type=Path, default=ROOT / "dist" / "portable-release")
    parser.add_argument("--rvc-root", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only:
        build_bundle(args.bundle, args.rvc_root)
    compile_packages(args.bundle, args.release, args.compiler)


if __name__ == "__main__":
    main()
