"""Input/output denoise: NRdenoiser wraps TorchGate with a SOLA crossfade.

Contract: NRdenoiser(samplerate, zc, device); apply() denoises input
segments into the rolling buffer; apply_output() denoises inference output
against the rolling noise reference; reset() clears rolling state.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

from engine.models import rvc_root


def _candidate_rvc_roots():
    """Candidate RVC roots: sys.path entries, env vars, then the default path."""
    roots = [entry for entry in sys.path if entry]
    for name in ("BSQ_RVC_ROOT", "RVC_ROOT"):
        value = os.environ.get(name)
        if value:
            roots.append(value)
    roots.append(str(rvc_root()))
    return roots


def _load_torch_gate():
    """Load RVC's TorchGate without colliding with this project's tools package."""
    try:
        from tools.torchgate import TorchGate

        return TorchGate
    except ImportError:
        pass
    for root in _candidate_rvc_roots():
        pkg_dir = os.path.join(root, "tools", "torchgate")
        if not os.path.isfile(os.path.join(pkg_dir, "torchgate.py")):
            continue
        if root not in sys.path:
            sys.path.append(root)  # torchgate.py itself needs infer.lib.rmvpe importable
        name = "_rvc_torchgate"
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module.TorchGate
    raise ImportError(
        "未找到 RVC 的 tools/torchgate（试过 sys.path / BSQ_RVC_ROOT / RVC_ROOT / %s）"
        % rvc_root()
    )


TorchGate = _load_torch_gate()


class NRdenoiser:
    """TorchGate input/output denoise with SOLA crossfade state."""

    def __init__(self, samplerate: int, zc: int, device):
        """Build TorchGate(sr, n_fft=4*zc, prop_decrease=0.9); windows build lazily."""
        self.samplerate = int(samplerate)
        self.zc = int(zc)
        self.device = torch.device(device)
        self.tg = TorchGate(
            sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.device)
        self.fade_in_window: torch.Tensor = None
        self.fade_out_window: torch.Tensor = None
        self.nr_buffer: torch.Tensor = None
        self.input_wav_denoise: torch.Tensor = None
        self.output_buffer: torch.Tensor = None

    def reset(self) -> None:
        """Clear rolling state; windows and buffers rebuild on next apply."""
        self.fade_in_window = None
        self.fade_out_window = None
        self.nr_buffer = None
        self.input_wav_denoise = None
        self.output_buffer = None

    def _ensure_input_state(self, sola_buffer_frame: int, total_len: int) -> None:
        if self.fade_in_window is None or self.fade_in_window.numel() != sola_buffer_frame:
            self.fade_in_window = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=sola_buffer_frame,
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window = 1 - self.fade_in_window
            self.nr_buffer = torch.zeros(
                sola_buffer_frame, device=self.device, dtype=torch.float32
            )
        if self.input_wav_denoise is None or self.input_wav_denoise.numel() != total_len:
            self.input_wav_denoise = torch.zeros(
                total_len, device=self.device, dtype=torch.float32
            )

    def apply(
        self, input_wav_full: torch.Tensor, sola_buffer_frame: int, block_frame: int
    ) -> torch.Tensor:
        """Denoise the input segment and crossfade it into the rolling buffer."""
        sola_buffer_frame = int(sola_buffer_frame)
        block_frame = int(block_frame)
        if block_frame <= 0:
            raise ValueError("block_frame 必须是正整数: %r" % (block_frame,))
        if input_wav_full.numel() < sola_buffer_frame + block_frame:
            raise ValueError(
                "input_wav_full 至少要 sola_buffer_frame + block_frame 个采样点"
            )
        self._ensure_input_state(sola_buffer_frame, input_wav_full.numel())
        self.input_wav_denoise[:-block_frame] = self.input_wav_denoise[
            block_frame:
        ].clone()
        input_wav = input_wav_full[-sola_buffer_frame - block_frame :]
        input_wav = self.tg(
            input_wav.unsqueeze(0), input_wav_full.unsqueeze(0)
        ).squeeze(0)
        input_wav[:sola_buffer_frame] *= self.fade_in_window
        input_wav[:sola_buffer_frame] += self.nr_buffer * self.fade_out_window
        self.input_wav_denoise[-block_frame:] = input_wav[:block_frame]
        self.nr_buffer[:] = input_wav[block_frame:]
        return self.input_wav_denoise

    def apply_output(
        self, infer_wav: torch.Tensor, block_frame: int, buffer_len: int = None
    ) -> torch.Tensor:
        """Denoise inference output against the rolling noise reference."""
        block_frame = int(block_frame)
        if block_frame <= 0:
            raise ValueError("block_frame 必须是正整数: %r" % (block_frame,))
        if buffer_len is None:
            buffer_len = (
                self.input_wav_denoise.numel()
                if self.input_wav_denoise is not None
                else max(int(infer_wav.numel()), block_frame)
            )
        elif int(buffer_len) < block_frame:
            raise ValueError(
                "buffer_len 至少要 block_frame（gui 的 output_buffer 为整段输入缓冲长度）"
            )
        n = int(buffer_len)
        if self.output_buffer is None or self.output_buffer.numel() != n:
            self.output_buffer = torch.zeros(n, device=self.device, dtype=torch.float32)
        self.output_buffer[:-block_frame] = self.output_buffer[block_frame:].clone()
        self.output_buffer[-block_frame:] = infer_wav[-block_frame:]
        return self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
