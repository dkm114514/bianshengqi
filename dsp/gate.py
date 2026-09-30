"""RMS gate (silence gate): zeroes sub-threshold frames of a mono block.

Contract: Gate(zc) holds the rms_buffer state; apply(indata, zc, threhold)
returns block_frame + 2*zc samples; threhold <= -60 bypasses the gate.
Pure numpy, no audio-device dependency.
"""
from __future__ import annotations

import librosa
import numpy as np

DEFAULT_ZC = 480  # zc = samplerate // 100 at 48kHz


class Gate:
    """RMS threshold gate with cross-block rms_buffer state."""

    def __init__(self, zc: int = DEFAULT_ZC):
        """Build a 4*zc rms_buffer."""
        self.zc = int(zc)
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")

    def apply(self, indata: np.ndarray, zc: int, threhold: float) -> np.ndarray:
        """Gate one mono block; threhold <= -60 returns the input unchanged."""
        zc = int(zc)
        if self.rms_buffer.shape[0] != 4 * zc:
            self.zc = zc
            self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        if threhold <= -60:
            return indata
        indata = np.append(self.rms_buffer, indata)
        rms = librosa.feature.rms(y=indata, frame_length=4 * zc, hop_length=zc)[:, 2:]
        self.rms_buffer[:] = indata[-4 * zc :]
        indata = indata[2 * zc - zc // 2 :]
        db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < threhold
        for i in range(db_threhold.shape[0]):
            if db_threhold[i]:
                indata[i * zc : (i + 1) * zc] = 0
        return indata[zc // 2 :]
