"""Verify a disposable extracted package through the normal GUI start/stop handlers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.request


def verify(root, audio=False, verify_files=False):
    root = Path(root).resolve()
    if not (root / "offline_bundle.json").is_file():
        raise ValueError("Select an extracted offline package")
    if any((root / path).exists() for path in ("data/voice_profile.npz", "models/voice_profile.npz")):
        raise ValueError("Use a disposable package without a personal voiceprint for this check")
    sys.path.insert(0, str(root))
    report = {"root": str(root), "normal_gui_mode": True}
    if verify_files:
        manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
        for relative, entry in manifest.items():
            digest = hashlib.sha256()
            with (root / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == entry["sha256"], "Bundle file changed: %s" % relative
        report["verified_package_files"] = len(manifest)

    original_connect = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("Unexpected remote connection: %s" % (address,))
        return original_connect(sock, address)
    def deny_network(*args, **kwargs):
        raise AssertionError("Unexpected remote download")
    socket.socket.connect = connect
    socket.create_connection = deny_network
    urllib.request.urlopen = deny_network
    urllib.request.urlretrieve = deny_network
    report["remote_python_network_blocked"] = True

    from app import runtime
    assert runtime.ROOT == root, "The check imported an app from a different folder"
    runtime.configure_runtime(root)
    os.environ.pop("BSMOKE", None)
    writes = set()
    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                return
        elif event in ("os.mkdir", "os.remove"):
            path = args[0]
        else:
            return
        if isinstance(path, int) or str(path).lower() in ("nul", "\\\\.\\nul", "/dev/null"):
            return
        path = Path(os.fsdecode(path)).resolve()
        assert path.is_relative_to(root), "Write outside portable folder: %s" % path
        writes.add(str(path.relative_to(root)))
    sys.addaudithook(audit)

    with runtime.InstanceLock(root / "data" / ".instance.lock"):
        runtime.install_logging(root / "logs" / "运行日志.log")
        runtime.migrate_legacy_data(root)
        from app import gui
        profiles, error = gui.load_profiles()
        app = gui.App(profiles, error, smoke=False)
        assert not app.smoke
        app.window.TKroot.withdraw()
        try:
            _, values = app.window.read(timeout=50)
            assert app.model_map and not app.models_error, "No local voice model"
            app._remember_devices(values["-IN-DEV-"], values["-MON-DEV-"])
            if audio:
                app._handle_event("-START-", values)
                assert app.pipeline is not None and app.pipeline.is_running
                pipeline = app.pipeline
                end = time.monotonic() + 3.0
                while time.monotonic() < end:
                    event, values = app.window.read(timeout=100)
                    app._handle_event(event, values)
                report["audio_stats"] = dict(pipeline.stats)
                assert pipeline.stats["blocks"] > 0, "No audio blocks processed"
                assert pipeline.stats["cable_blocks"] > 0, "Virtual microphone output did not run"
                assert pipeline.stats["errors"] == 0, "Audio callback failed"
                report["real_audio_started"] = True
                app._handle_event("-STOP-", values)
                assert not pipeline.is_running
            app._flush_profiles()
            assert not app.dirty
            saved = json.loads(Path(gui.PROFILES_PATH).read_text(encoding="utf-8"))
            assert saved["input_device"] == app.window["-IN-DEV-"].get()
            assert saved["monitor_device"] == app.window["-MON-DEV-"].get()
            assert not list((root / "data").glob("*.tmp"))
            report["settings_saved"] = True
        finally:
            app.shutdown()

        # Reopen the normal GUI and prove the choices were actually saved to disk.
        reopened = gui.App(smoke=False)
        reopened.window.TKroot.withdraw()
        try:
            assert reopened.window["-IN-DEV-"].get() == saved["input_device"]
            assert reopened.window["-MON-DEV-"].get() == saved["monitor_device"]
            report["settings_restored"] = True
        finally:
            reopened.shutdown()

    report["writes_inside_portable_folder"] = len(writes)
    report["user_data_directory"] = str(Path(gui.PROFILES_PATH).relative_to(root).parent)
    report["cache_directory"] = str(Path(os.environ["TORCH_HOME"]).relative_to(root).parent)
    target = root / "cache" / "portable-validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print("PORTABLE GUI CHECK PASS", flush=True)
    return report


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    verify(args.app_root, args.audio, args.verify_files)
