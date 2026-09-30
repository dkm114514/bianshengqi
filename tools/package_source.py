"""Build a source-only ZIP from an explicit allowlist, without Git history."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = (
    ".gitignore", ".gitattributes", "README.md", "requirements.txt", "requirements-devsetup.txt",
    "profiles.example.json", "启动变声器.bat", "main.py", "selfcheck.py",
    "smoke_test.py", "LICENSE", "THIRD_PARTY_NOTICES.md",
)
SOURCE_DIRS = ("app", "dsp", "engine", "tools", "tests", "docs", "installer")
SOURCE_SUFFIXES = {".py", ".md", ".iss", ".isl"}


def source_files(root=ROOT):
    root = Path(root)
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in SOURCE_SUFFIXES
                    and not any(part.startswith(".") or part == "__pycache__"
                                for part in path.relative_to(root).parts)):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_archive(target, root=ROOT):
    root, target = Path(root).resolve(), Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Source must stay inside the workspace: %s" % path)
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            manifest[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            archive.writestr("bianshengqi/" + relative, content)
        archive.writestr("bianshengqi/source_manifest.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise OSError("ZIP integrity check failed")
        for name, info in manifest.items():
            content = archive.read("bianshengqi/" + name)
            if hashlib.sha256(content).hexdigest() != info["sha256"]:
                raise OSError("Source checksum mismatch: %s" % name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "bianshengqi-source.zip")
    args = parser.parse_args()
    manifest = build_archive(args.output)
    print("%s (%d source files, %d bytes)" % (args.output.resolve(), len(manifest), args.output.stat().st_size))


if __name__ == "__main__":
    main()
