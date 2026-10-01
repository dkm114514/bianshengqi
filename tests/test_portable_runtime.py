"""Regression coverage for locked settings, broken consoles and folder-local state."""
import ctypes
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import gui, runtime
from app.storage import atomic_write_json


class BrokenConsole:
    def write(self, text):
        raise OSError(6, "invalid Windows console handle")

    def flush(self):
        raise OSError(6, "invalid Windows console handle")


class SettingsPersistenceTests(unittest.TestCase):
    def test_temporary_replace_denial_is_retried_without_losing_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text('{"old": true}', encoding="utf-8")
            real_replace = os.replace
            attempts = []

            def locked_twice(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError(13, "sharing violation")
                real_replace(source, target)

            with patch("app.storage.os.replace", side_effect=locked_twice), patch("app.storage.time.sleep"):
                atomic_write_json(path, {"input_device": "麦克风", "monitor_device": "耳机"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["monitor_device"], "耳机")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_permanent_denial_keeps_the_old_file_and_removes_the_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            previous = b'{"old": true}'
            path.write_bytes(previous)
            with patch("app.storage.os.replace", side_effect=PermissionError(13, "locked")), patch("app.storage.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_write_json(path, {"new": True})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_two_device_choices_are_saved_together_once(self):
        app = object.__new__(gui.App)
        app.profiles = {}
        app._mark_dirty = Mock()
        app._remember_devices("microphone", "headphones")
        self.assertEqual(app.profiles, {"input_device": "microphone", "monitor_device": "headphones"})
        app._mark_dirty.assert_called_once_with()
        app._remember_devices("microphone", "headphones")
        app._mark_dirty.assert_called_once_with()

    def test_failed_save_and_invalid_stderr_do_not_abort_the_gui_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            app = object.__new__(gui.App)
            app.profiles, app.profiles_path = {}, str(Path(directory) / "profiles.json")
            app.dirty, app.smoke = True, False
            log = Path(directory) / "error.log"
            with patch("app.gui.save_profiles", side_effect=PermissionError(13, "locked")), \
                    patch.object(sys, "stderr", BrokenConsole()), patch.object(runtime, "LOG_PATH", log):
                app._flush_profiles()
            self.assertTrue(app.dirty)
            self.assertIn("PermissionError", log.read_text(encoding="utf-8"))
            app._flush_profiles()
            self.assertFalse(app.dirty)
            self.assertTrue(Path(app.profiles_path).is_file())

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_real_windows_reader_lock_is_retried_until_reader_releases_it(self):
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text("{}", encoding="utf-8")
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            released = threading.Event()

            def release():
                try:
                    kernel.CloseHandle(handle)
                finally:
                    released.set()

            timer = threading.Timer(0.08, release)
            timer.start()
            try:
                atomic_write_json(path, {"saved_after_reader_closed": True})
                self.assertTrue(released.is_set())
                self.assertTrue(json.loads(path.read_text())["saved_after_reader_closed"])
            finally:
                timer.join()


class PortableStateTests(unittest.TestCase):
    def test_cache_and_temp_paths_ignore_machine_paths_and_stay_in_the_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "portable"
            root.mkdir()
            (root / "offline_bundle.json").write_text("{}")
            with patch.dict(os.environ, {"TORCH_HOME": "outside", "BSQ_WEIGHTS_DIR": "outside"}), \
                    patch.object(tempfile, "tempdir", None), patch.object(sys, "dont_write_bytecode"):
                runtime.configure_runtime(root)
                for name in ("TORCH_HOME", "HF_HOME", "NUMBA_CACHE_DIR", "CUDA_CACHE_PATH", "TEMP", "TMP"):
                    self.assertTrue(Path(os.environ[name]).is_relative_to(root))
                self.assertEqual(tempfile.gettempdir(), str(root / "cache" / "tmp"))
                self.assertEqual(os.environ["BSQ_WEIGHTS_DIR"], str(root / "RVC" / "assets" / "weights"))
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_migration_preserves_original_voice_bytes_and_never_overwrites_new_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            original = b"personal voice fixture"
            (root / "models" / "voice_profile.npz").write_bytes(original)
            (root / "profiles.json").write_text('{"pitch": 16}', encoding="utf-8")
            runtime.migrate_legacy_data(root)
            self.assertEqual((root / "data" / "voice_profile.npz").read_bytes(), original)
            self.assertEqual((root / "models" / "voice_profile.npz").read_bytes(), original)
            (root / "data" / "voice_profile.npz").write_bytes(b"new voice")
            runtime.migrate_legacy_data(root)
            self.assertEqual((root / "data" / "voice_profile.npz").read_bytes(), b"new voice")

    def test_invalid_console_writes_still_reach_the_file_log(self):
        log = io.StringIO()
        stream = runtime.LogStream(BrokenConsole(), log)
        self.assertEqual(stream.write("保存失败\n"), 5)
        stream.flush()
        stream.write("第二条\n")
        self.assertEqual(log.getvalue(), "保存失败\n第二条\n")

    def test_duplicate_instance_is_rejected_and_the_lock_is_released_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".instance.lock"
            with runtime.InstanceLock(lock):
                with self.assertRaises(runtime.AlreadyRunningError):
                    with runtime.InstanceLock(lock):
                        pass
            with runtime.InstanceLock(lock):
                pass

    def test_moving_folder_uses_current_scanned_model_instead_of_saved_absolute_path(self):
        app = object.__new__(gui.App)
        app.profiles = {"models": {"bb48k": {"pth": "old-folder/bb48k.pth", "pitch": 11}}}
        app.model_map = {"bb48k": {"pth": "new-folder/bb48k.pth", "index": None}}
        app._mark_dirty = Mock()
        _, profile = app._profile_view("bb48k")
        self.assertEqual(profile["pth"], "new-folder/bb48k.pth")
        self.assertEqual(profile["pitch"], 11)


if __name__ == "__main__":
    unittest.main()
