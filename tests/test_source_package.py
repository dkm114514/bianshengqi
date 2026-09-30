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
    def test_offline_dfn_exports_bundled_directory_to_the_third_party_loader(self):
        from dsp import dfn
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "models" / "dfn3-512-v1"
            assets.mkdir(parents=True)
            for name in dfn._MODEL_FILES:
                (assets / name).write_bytes(b"fixture")
            with patch.object(dfn, "_PROJECT_MODEL_DIR", assets), patch.dict(os.environ, {"BSQ_OFFLINE": "1"}):
                self.assertEqual(dfn._resolve_model_dir(), assets)
                self.assertEqual(os.environ[dfn._ENV_MODEL_DIR], str(assets))

    def test_offline_marker_prefers_bundled_runtime_over_machine_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "offline_bundle.json").write_text("{}", encoding="utf-8")
            with patch.object(models, "PROJECT_ROOT", root), patch.dict(os.environ, {"RVC_ROOT": "old-install"}):
                self.assertEqual(models.rvc_root(), root / "RVC")

    def test_missing_offline_dfn_models_never_start_a_download(self):
        from dsp import dfn
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "models" / "dfn3-512-v1"
            with patch.object(dfn, "_PROJECT_MODEL_DIR", missing), patch.dict(os.environ, {"BSQ_OFFLINE": "1"}), \
                    patch.object(dfn, "_download_model") as download:
                with self.assertRaisesRegex(FileNotFoundError, "离线安装包"):
                    dfn._resolve_model_dir()
                download.assert_not_called()

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
