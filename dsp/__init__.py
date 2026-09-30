"""dsp: real-time DSP layer (gate/denoise/voice_fx ported from gui_v1.py).

Gate, NRdenoiser, and rms_mix import lazily via module __getattr__ so the
package import never pulls in torch or the RVC source tree; dsp.agc,
dsp.vad, dsp.dfn, and dsp.voice_gate stay dependency-free.
"""
import importlib

__all__ = ["Gate", "NRdenoiser", "rms_mix"]

#: Lazy export name -> submodule.
_LAZY_ATTRS = {"Gate": "gate", "NRdenoiser": "denoise", "rms_mix": "voice_fx"}


def __getattr__(name):
    """Lazy import so from dsp import Gate works without torch/RVC."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(importlib.import_module("%s.%s" % (__name__, module_name)), name)
    globals()[name] = value  # Cache as a regular attribute afterwards.
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
