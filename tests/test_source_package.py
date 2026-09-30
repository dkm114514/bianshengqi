"""Portable runtime selection and exclusion of private files from publication."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import models
from tools.package_source import build_archive


class SourcePublicationTests(unittest.TestCase):
    def test_runtime_selection_and_local_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / "installation" / "runtime" / "python.exe"
            (root / "runtime.local.txt").write_text(str(exe), encoding="utf-8")
            with patch.object(models, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(models.rvc_root(), exe.parent.parent)
                os.environ["RVC_RUNTIME"] = str(root / "other" / "runtime" / "python.exe")
                self.assertEqual(models.rvc_root(), root / "other")
                os.environ["RVC_ROOT"] = str(root / "root")
                self.assertEqual(models.rvc_root(), root / "root")
                os.environ["BSQ_RVC_ROOT"] = str(root / "preferred")
                self.assertEqual(models.rvc_root(), root / "preferred")

    def test_package_excludes_local_files_and_verifies_source_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "README.md": "public instructions",
                "app/gui.py": "# public source",
                "app/__pycache__/cached.py": "private cache",
                "app/.local/private.py": "private source",
                "models/voice_profile.npz": "private voice",
                "models/voice_backups/original.npz": "private backup",
                "profiles.json": "private model paths",
                "runtime.local.txt": "private runtime path",
                ".git/config": "private remote",
                "downloads/driver.exe": "binary",
                "out/results.json": "private output",
            }
            for name, content in fixtures.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            target = root / "dist" / "source.zip"
            manifest = build_archive(target, root)
            self.assertEqual(set(manifest), {"README.md", "app/gui.py"})
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "bianshengqi/README.md", "bianshengqi/app/gui.py",
                    "bianshengqi/source_manifest.json",
                })
                self.assertEqual(json.loads(archive.read("bianshengqi/source_manifest.json")), manifest)
                self.assertEqual(hashlib.sha256(archive.read("bianshengqi/app/gui.py")).hexdigest(),
                                 manifest["app/gui.py"]["sha256"])


if __name__ == "__main__":
    unittest.main()
