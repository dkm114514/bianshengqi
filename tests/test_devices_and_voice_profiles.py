"""Full audio refresh and recoverable voiceprint file operations."""
import json
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import gui
from app.voice_profiles import delete_profile, export_profile, import_profile
from engine.defaults import DEFAULT_MODEL_PROFILE
from engine.pipeline import VoicePipeline, _MasterBuffer


class DeviceRefreshTests(unittest.TestCase):
    def make_pipeline(self, running=True):
        p = VoicePipeline({}, "mic", "cable", "headphones", monitor_on=True)
        p._running, p._prepared, p.block_frame = running, True, 7200
        p._prepare = Mock()  # Model stays resident; audio drivers are replaced below.
        for name in ("input_wav", "input_wav_denoise", "input_wav_res", "sola_buffer", "nr_buffer", "output_buffer"):
            setattr(p, name, torch.ones(16))
        p.rms_buffer = np.ones(16)
        p._pending = np.ones(16)
        p.engine._rvc = SimpleNamespace(cache_pitch=torch.ones(16), cache_pitchf=torch.ones(16))
        p.agc = SimpleNamespace(reset=Mock(), target_dbfs=-20)
        p.agc_stats["gain_db"] = 12
        p._agc_vad = SimpleNamespace(reset=Mock())
        p._input_vad = SimpleNamespace(reset=Mock())
        p._dfn = SimpleNamespace(reset=Mock())
        p._gate_hist = np.ones(48000, dtype=np.float32)
        p._gate_target, p._gate_gain = 0.0, 0.0
        p.stats["errors"], p._overload_hot = 7, 29
        p.master = _MasterBuffer(48000)
        p.master.write(np.ones(16))
        p.cable_reader, p.monitor_reader = p.master.reader(), p.master.reader()
        p._in_stream = p._cable_stream = p._monitor_stream = object()
        p._in_index = p._cable_index = p._monitor_index = 3

        def close_stream(name):
            setattr(p, "_%s_stream" % ("in" if name == "input" else name), None)

        def open_stream(name):
            setattr(p, "_%s_stream" % ("in" if name == "input" else name), object())
            if name == "monitor":
                p.monitor_reader = p.master.reader()

        p._close_stream = Mock(side_effect=close_stream)
        p._open_cable = Mock(side_effect=lambda: open_stream("cable"))
        p._open_input = Mock(side_effect=lambda: open_stream("input"))
        p._open_monitor = Mock(side_effect=lambda: open_stream("monitor"))
        return p

    def assert_clean(self, p):
        for name in ("input_wav", "input_wav_denoise", "input_wav_res", "sola_buffer", "nr_buffer", "output_buffer"):
            self.assertEqual(getattr(p, name).abs().sum().item(), 0)
        self.assertEqual(p.engine.rvc.cache_pitch.abs().sum().item(), 0)
        self.assertEqual(p.engine.rvc.cache_pitchf.abs().sum().item(), 0)
        self.assertEqual(float(p.rms_buffer.sum()), 0)
        self.assertEqual(p._pending.size, 0)
        self.assertEqual(p._gate_target, 1.0)
        self.assertEqual(float(p._gate_hist.sum()), 0)
        self.assertEqual(p.agc_stats["gain_db"], 0)
        self.assertEqual(p.stats["errors"], 0)
        self.assertEqual(p._overload_hot, 0)
        p.agc.reset.assert_called()
        p._dfn.reset.assert_called()
        p._input_vad.reset.assert_called()
        p._agc_vad.reset.assert_called()

    @patch("engine.pipeline.resolve_device", return_value=3)
    def test_same_devices_still_refresh_every_stream_and_state(self, resolve):
        p = self.make_pipeline()
        old_master, old_rvc = p.master, p.engine.rvc
        p.refresh_devices()
        self.assert_clean(p)
        self.assertIsNot(p.master, old_master)
        self.assertIs(p.engine.rvc, old_rvc)
        for opener in (p._open_input, p._open_cable, p._open_monitor):
            opener.assert_called_once()
        self.assertTrue(p.is_running)

    @patch("engine.pipeline.resolve_device", return_value=3)
    def test_monitor_selection_also_refreshes_input_and_cable(self, resolve):
        p = self.make_pipeline()
        p.set_devices(monitor_device="new headphones")
        self.assertEqual(p.monitor_device, "new headphones")
        p._open_input.assert_called_once()
        p._open_cable.assert_called_once()
        self.assert_clean(p)

    @patch("engine.pipeline.resolve_device", side_effect=ValueError("missing endpoint"))
    def test_invalid_device_does_not_disturb_running_chain(self, resolve):
        p = self.make_pipeline()
        with self.assertRaises(ValueError):
            p.refresh_devices(input_device="missing")
        p._close_stream.assert_not_called()
        self.assertTrue(p.is_running)
        self.assertEqual(p.input_device, "mic")

    @patch("engine.pipeline.resolve_device", return_value=3)
    def test_input_open_failure_restores_original_devices(self, resolve):
        p = self.make_pipeline()
        original_open = p._open_input.side_effect
        attempts = []

        def open_input():
            attempts.append(p.input_device)
            if p.input_device == "broken":
                raise RuntimeError("driver failed")
            original_open()

        p._open_input.side_effect = open_input
        with self.assertRaisesRegex(RuntimeError, "已恢复原设备"):
            p.refresh_devices(input_device="broken")
        self.assertEqual(attempts, ["broken", "mic"])
        self.assertEqual(p.input_device, "mic")
        self.assertTrue(p.is_running)
        self.assertFalse(p._infer_busy.locked())

    def test_stopped_refresh_clears_state_without_starting_audio(self):
        p = self.make_pipeline(running=False)
        p.refresh_devices(input_device="other mic")
        self.assert_clean(p)
        p._open_input.assert_not_called()
        self.assertFalse(p.is_running)

    @patch("engine.pipeline.resolve_device", return_value=3)
    def test_monitor_open_failure_restores_original_monitor(self, resolve):
        p = self.make_pipeline()
        original_open = p._open_monitor.side_effect

        def open_monitor():
            if p.monitor_device == "broken headphones":
                raise RuntimeError("monitor driver failed")
            original_open()

        p._open_monitor.side_effect = open_monitor
        with self.assertRaisesRegex(RuntimeError, "已恢复原设备"):
            p.refresh_devices(monitor_device="broken headphones")
        self.assertTrue(p.is_running)
        self.assertTrue(p.monitor_active)
        self.assertEqual(p.monitor_device, "headphones")

    @patch("engine.pipeline.resolve_device", return_value=3)
    def test_failed_rollback_converges_to_stopped(self, resolve):
        p = self.make_pipeline()
        p._open_input.side_effect = RuntimeError("all microphones unavailable")
        with self.assertRaisesRegex(RuntimeError, "音频已停止"):
            p.refresh_devices(input_device="broken")
        self.assertFalse(p.is_running)
        self.assertIsNone(p._in_stream)
        self.assertIsNone(p._cable_stream)
        self.assertIsNone(p._monitor_stream)
        self.assertFalse(p._infer_busy.locked())


class VoiceProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.active = os.path.join(self.temporary.name, "voice_profile.npz")
        # Synthetic fixture: public tests never read a user's biometric profile.
        embedding = np.arange(1, 193, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        np.savez(self.active, embedding=embedding, threshold=0.2, sample_rate=16000,
                 dim=192, num_segments=3, version=1,
                 model="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
        with open(self.active, "rb") as f:
            self.original = f.read()
        self.model_dir = os.path.join(ROOT, "models")
        independent = {"test_missing_config_uses_current_defaults_without_clamping_extra_time",
                       "test_refresh_button_refreshes_unchanged_running_devices"}
        if self._testMethodName not in independent:
            from dsp.voice_gate import _find_onnx, _SPEAKER_KEYWORDS, _VAD_KEYWORDS
            if not os.path.isdir(self.model_dir) or not (
                _find_onnx(self.model_dir, _SPEAKER_KEYWORDS)
                and _find_onnx(self.model_dir, _VAD_KEYWORDS)
            ):
                self.skipTest("声纹文件集成测试需要 models/ 下的 CAM++ 和 Silero ONNX")

    def test_npz_export_delete_import_restores_exact_voiceprint(self):
        exported = export_profile(self.active, os.path.join(self.temporary.name, "my_voice.npz"), self.model_dir)
        with open(exported, "rb") as f:
            self.assertEqual(f.read(), self.original)
        backup = delete_profile(self.active)
        self.assertFalse(os.path.exists(self.active))
        with open(backup, "rb") as f:
            self.assertEqual(f.read(), self.original)
        import_profile(exported, self.active, self.model_dir)
        with open(self.active, "rb") as f:
            self.assertEqual(f.read(), self.original)

    def test_json_roundtrip_and_overwrite_backup(self):
        exported = export_profile(self.active, os.path.join(self.temporary.name, "my_voice.json"), self.model_dir, 0.2)
        backup = import_profile(exported, self.active, self.model_dir, 0.2)
        with open(backup, "rb") as f:
            self.assertEqual(f.read(), self.original)
        with np.load(self.active) as restored, np.load(backup) as original:
            np.testing.assert_allclose(restored["embedding"], original["embedding"], atol=1e-6)
            self.assertEqual(int(restored["num_segments"]), int(original["num_segments"]))
        with open(exported, encoding="utf8") as f:
            self.assertEqual(json.load(f)["threshold"], 0.2)

    def test_invalid_import_preserves_active_voiceprint(self):
        invalid = os.path.join(self.temporary.name, "corrupt.json")
        with open(invalid, "w") as f:
            json.dump({"embedding": [0, 0, 0]}, f)
        with self.assertRaises(ValueError):
            import_profile(invalid, self.active, self.model_dir)
        with open(self.active, "rb") as f:
            self.assertEqual(f.read(), self.original)
        self.assertFalse(os.path.exists(os.path.join(self.temporary.name, "voice_backups")))

    def test_failed_replacement_preserves_active_and_a_recoverable_backup(self):
        exported = export_profile(self.active, os.path.join(self.temporary.name, "my_voice.npz"), self.model_dir)
        with patch("app.voice_profiles.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                import_profile(exported, self.active, self.model_dir)
        with open(self.active, "rb") as f:
            self.assertEqual(f.read(), self.original)
        backups = os.listdir(os.path.join(self.temporary.name, "voice_backups"))
        self.assertEqual(len(backups), 1)
        self.assertFalse(any(name.startswith(".voice_import_") for name in os.listdir(self.temporary.name)))

    def test_missing_config_uses_current_defaults_without_clamping_extra_time(self):
        data, error = gui.load_profiles(os.path.join(self.temporary.name, "absent.json"))
        self.assertTrue(error)
        self.assertEqual(data["active"], "bb48k")
        self.assertEqual(data["denoise_mode"], "dfn3")
        self.assertEqual(data["voice_gate_threshold"], 0.2)
        self.assertEqual(data["agc_target_dbfs"], -20)
        self.assertTrue(data["monitor_on"])
        view = gui.App._sanitize_profile(dict(DEFAULT_MODEL_PROFILE))
        self.assertEqual((view["pitch"], view["block_time"], view["extra_time"]), (16, 0.15, 4.0))
        self.assertEqual(view["crossfade_time"], 0.01)
        self.assertEqual(view["threhold"], -60)
        self.assertTrue(view["O_noise_reduce"])

    def make_app(self):
        app = gui.App.__new__(gui.App)
        app._enrolling = app._verifying = False
        app.smoke, app.voice_gate_threshold = True, 0.2
        app._gate_spec_sent = gui.MODELS_DIR
        app.window = defaultdict(Mock)
        app._set_status = Mock()
        app._refresh_voice_profile_state = Mock()
        app._enable_voice_gate = Mock()
        return app

    def test_import_button_backs_up_and_rebuilds_live_gate(self):
        exported = export_profile(self.active, os.path.join(self.temporary.name, "import.npz"), self.model_dir)
        app = self.make_app()
        with patch("app.gui.VOICE_PROFILE_PATH", self.active), patch("app.gui.sg.popup_get_file", return_value=exported):
            app._handle_event("-VOICE-IMPORT-", {})
        app._enable_voice_gate.assert_called_once_with(True)
        self.assertIs(app._gate_spec_sent, gui._UNSET)
        self.assertEqual(len(os.listdir(os.path.join(self.temporary.name, "voice_backups"))), 1)
        with open(self.active, "rb") as f:
            self.assertEqual(f.read(), self.original)

    def test_export_button_saves_the_current_voiceprint(self):
        app = self.make_app()
        exported = os.path.join(self.temporary.name, "exported.npz")
        with patch("app.gui.VOICE_PROFILE_PATH", self.active), patch("app.gui.sg.popup_get_file", return_value=exported):
            app._handle_event("-VOICE-EXPORT-", {})
        with open(exported, "rb") as f:
            self.assertEqual(f.read(), self.original)
        self.assertTrue(os.path.isfile(self.active))

    def test_delete_button_preserves_backup_and_turns_gate_off(self):
        app = self.make_app()
        with patch("app.gui.VOICE_PROFILE_PATH", self.active):
            app._handle_event("-VOICE-DELETE-", {})
        app._enable_voice_gate.assert_called_once_with(False)
        self.assertFalse(os.path.exists(self.active))
        folder = os.path.join(self.temporary.name, "voice_backups")
        with open(os.path.join(folder, os.listdir(folder)[0]), "rb") as f:
            self.assertEqual(f.read(), self.original)

    def test_refresh_button_refreshes_unchanged_running_devices(self):
        app = self.make_app()
        app.window["-IN-DEV-"].get.return_value = "mic"
        app.window["-MON-DEV-"].get.return_value = "headphones"
        app.pipeline = SimpleNamespace(refresh_devices=Mock(), is_running=True)
        app._remember_devices = Mock()
        app._reset_meter = Mock()
        app._match_lock, app._match_score = threading.Lock(), 0.1
        app._refresh_voice_score = app._refresh_agc_gain = app._refresh_load = Mock()
        devices = dict(input=["mic"], output=["headphones", "CABLE Input"],
                       input_map={"mic": 0}, output_map={"headphones": 1, "CABLE Input": 2},
                       hostapi="Windows WASAPI")
        with patch("app.gui.list_devices", return_value=devices):
            app._handle_event("-DEV-REFRESH-", {})
        app.pipeline.refresh_devices.assert_called_once_with(input_device="mic", monitor_device="headphones",
                                                            cable_device="CABLE Input")
        self.assertIsNone(app._match_score)


if __name__ == "__main__":
    unittest.main()
