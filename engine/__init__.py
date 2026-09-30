"""Voice changer engine package: model registry + RVC realtime inference wrapper.

Public interface:
    engine.models.ModelRegistry  scan weights/index dirs, output loadable model list
    engine.rvc_engine.RvcEngine  RVC realtime inference wrapper (load model / hot-update params / tgt_sr)
    engine.pipeline.VoicePipeline realtime audio pipeline (capture->RVC->virtual mic/monitor, with denoise/voice gate/AGC)

Project conventions:
    - model entry = {"pth", "index", "display"}; active combo = bb48k.pth + logs/guanguanV1.index;
    - GPU pitch algorithm is rmvpe only (engine.rvc_engine.GPU_F0METHOD), pm / harvest are CPU fallbacks.
"""

from .models import ModelRegistry
from .rvc_engine import RvcEngine

__all__ = ["ModelRegistry", "RvcEngine"]
