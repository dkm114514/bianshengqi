"""Exercise the relocated app with network access blocked at Python socket/URL APIs."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def deny_network(*args, **kwargs):
    raise AssertionError("Offline bundle attempted a network connection")


def main(full=False):
    if not (ROOT / "offline_bundle.json").is_file():
        raise RuntimeError("Run this check from the staged/installed offline application")
    from app.runtime import configure_runtime
    configure_runtime(ROOT)
    # Keep local sockets available for multiprocessing; block connections to remote hosts.
    original_connect = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            return deny_network(address)
        return original_connect(sock, address)
    socket.socket.connect = connect
    socket.create_connection = deny_network
    urllib.request.urlopen = deny_network
    urllib.request.urlretrieve = deny_network
    os.environ["BSQ_RVC_ROOT"] = str(ROOT / "RVC")
    os.environ["RVC_ROOT"] = str(ROOT / "RVC")
    os.environ["BSQ_OFFLINE"] = "1"
    # Ensure no caller's DFN cache/environment masks missing bundled models.
    os.environ.pop("DEEPFILTER_STREAM_MODEL_DIR", None)
    import numpy as np
    import torch
    from engine.defaults import DEFAULT_MODEL_PROFILE
    from engine.models import ModelRegistry, rvc_root
    from dsp.dfn import DfnDenoiser
    from dsp.vad import SileroVad
    from dsp.voice_gate import VoiceGate
    assert rvc_root().resolve() == (ROOT / "RVC").resolve()
    assert not (ROOT / "models" / "voice_profile.npz").exists()
    assert not (ROOT / "data" / "voice_profile.npz").exists()
    entries = ModelRegistry.scan()
    assert entries, "No bundled voice model found"
    marker = json.loads((ROOT / "offline_bundle.json").read_text(encoding="utf-8"))
    entry = next(item for item in entries if Path(item["pth"]).name == marker["default_model"])
    rng = np.random.default_rng(42)
    x = rng.normal(0, 0.01, 14400).astype(np.float32)
    vad = SileroVad()
    probability = vad.process(x)
    gate = VoiceGate(profile_path=str(ROOT / "models" / "unused_test_profile.npz"))
    assert gate.embed_dim == 192
    dfn = DfnDenoiser()
    y = dfn.process(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    report = {"runtime": str(Path(sys.executable).relative_to(ROOT)),
              "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "voice_model": Path(entry["pth"]).name, "vad_probability": probability,
              "speaker_dimensions": gate.embed_dim, "dfn_finite": True,
              "portable_cache": os.environ["TORCH_HOME"],
              "remote_python_network_blocked": True}
    if full:
        from engine.pipeline import VoicePipeline
        profile = dict(DEFAULT_MODEL_PROFILE, pth=entry["pth"], index=entry["index"] or "")
        pipeline = VoicePipeline(profile, "offline", "offline", denoise_mode="dfn3", agc_enabled=True)
        try:
            # Enough blocks to exercise HuBERT, RMVPE, synthesis, SOLA, denoise and AGC.
            t = np.arange(96000, dtype=np.float32) / 48000
            signal = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
            converted = pipeline.process_offline(signal)
            assert np.isfinite(converted).all() and converted.size == signal.size
            assert pipeline._dfn is not None and pipeline._input_vad is not None
            report.update(rvc_inference=True, output_samples=converted.size,
                          output_peak=float(np.max(np.abs(converted))),
                          rvc_device=str(pipeline.engine.config.device))
        finally:
            pipeline._stop_gate_worker()
            pipeline.stop()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print("OFFLINE BUNDLE CHECK PASS", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(args.full)
