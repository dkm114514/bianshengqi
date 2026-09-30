"""Loudness-envelope mixing: scale inference output back to input loudness."""
from __future__ import annotations

import math

import librosa
import torch
import torch.nn.functional as F


def rms_mix(
    infer_wav: torch.Tensor,
    input_wav: torch.Tensor,
    zc: int,
    rate: float,
    device,
) -> torch.Tensor:
    """Pull inference output toward the input RMS envelope; rate >= 1 returns it unchanged."""
    rate = float(rate)
    if not math.isfinite(rate):
        raise ValueError("rate 必须是有限数: %r" % (rate,))
    if rate >= 1:
        return infer_wav
    zc = int(zc)
    rms1 = librosa.feature.rms(
        y=input_wav[: infer_wav.shape[0]].detach().cpu().numpy(),
        frame_length=4 * zc,
        hop_length=zc,
    )
    rms1 = torch.from_numpy(rms1).to(device)
    rms1 = F.interpolate(
        rms1.unsqueeze(0),
        size=infer_wav.shape[0] + 1,
        mode="linear",
        align_corners=True,
    )[0, 0, :-1]
    rms2 = librosa.feature.rms(
        y=infer_wav.detach().cpu().numpy(), frame_length=4 * zc, hop_length=zc
    )
    rms2 = torch.from_numpy(rms2).to(device)
    rms2 = F.interpolate(
        rms2.unsqueeze(0),
        size=infer_wav.shape[0] + 1,
        mode="linear",
        align_corners=True,
    )[0, 0, :-1]
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
    return infer_wav * torch.pow(
        rms1 / rms2,
        torch.tensor(1.0 - float(rate), device=rms2.device, dtype=rms2.dtype),
    )
