"""Pause/onset regressions using the real pipeline and deterministic DSP doubles."""
import os
import sys
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.pipeline import SAMPLERATE, VoicePipeline


class EnergyVad:
    def reset(self):
        pass

    def process(self, block, sr=SAMPLERATE):
        return float(np.max(np.abs(block)) > 0.02)


class SpeakerGate:
    def __init__(self):
        self.score = None
        self.resets = 0

    def reset_smooth(self):
        self.resets += 1

    def decide(self, wav):
        return self.score


def make_pipe(block_time=0.06, output_denoise=False):
    p = VoicePipeline.__new__(VoicePipeline)
    block = int(block_time * SAMPLERATE)
    zc, sola, search, extra = 480, 480, 480, 48000
    size = extra + sola + search + block
    p.profile = dict(threhold=-60, I_noise_reduce=True,
                     O_noise_reduce=output_denoise, rms_mix_rate=1.0,
                     f0method="rmvpe")
    p.block_frame, p.block_frame_16k, p.zc = block, block // 3, zc
    p.sola_buffer_frame, p.sola_search_frame = sola, search
    p.extra_frame, p.skip_head = extra, extra // zc
    p.return_length = (block + sola + search) // zc
    for name in ("input_wav", "input_wav_denoise", "output_buffer"):
        setattr(p, name, torch.zeros(size))
    p.input_wav_res = torch.zeros(size // 3)
    p.sola_buffer, p.nr_buffer = torch.zeros(sola), torch.zeros(sola)
    p.rms_buffer = np.zeros(4 * zc, dtype=np.float32)
    p.fade_in_window = torch.linspace(0, 1, sola)
    p.fade_out_window = 1 - p.fade_in_window
    p.resampler = lambda x: torch.full((x.numel() // 3,), float(x.abs().mean()))
    p.resampler2 = None
    p.tg = lambda x, noise: torch.zeros_like(x)
    p.engine = SimpleNamespace(config=SimpleNamespace(device="cpu"),
                              rvc=SimpleNamespace(infer=lambda x, *args:
                                  torch.full((block + sola + search,), float(x[-block // 3:].abs().max()))))
    p.voice_gate, p.voice_gate_threshold = SpeakerGate(), 0.2
    p.denoise_mode, p._dfn = "torchgate", None
    p._input_vad, p._input_vad_warned = EnergyVad(), False
    p._gate_hist = np.zeros(SAMPLERATE, dtype=np.float32)
    p._gate_every = max(1, round(0.25 / block_time))
    p._gate_resampler = None
    p._voice_gate_to_16k = lambda wav: wav[::3].copy()
    p._gate_mailbox_lock = threading.Lock()
    p._gate_mailbox, p._gate_thread = None, None
    p._gate_wake, p._gate_stop = threading.Event(), threading.Event()
    p.voice_stats = dict(decisions=0, blocked_blocks=0, errors=0,
                        score=None, open=False, gain=0.0)
    p.agc_enabled, p.agc = False, None
    p._reset_voice_gate_state()
    return p


def voice(p):
    t = np.arange(p.block_frame) / SAMPLERATE
    return (0.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def pause(p, seconds=2.0, noise=0.0):
    rng = np.random.RandomState(123)
    for _ in range(int(np.ceil(seconds * SAMPLERATE / p.block_frame))):
        block = (rng.randn(p.block_frame) * noise).astype(np.float32)
        p._infer_block(block, realtime=False)


class PauseOnsetTests(unittest.TestCase):
    def test_two_second_pause_does_not_keep_old_rejection(self):
        for block_time in (0.04, 0.06, 0.15):
            with self.subTest(block_time=block_time):
                p = make_pipe(block_time)
                p._gate_target = p._gate_gain = 0.0
                pause(p)
                out = p._infer_block(voice(p), realtime=False)
                self.assertEqual(p._gate_target, 1.0)
                self.assertGreater(float(np.max(np.abs(out))), 0.03)

    def test_room_noise_pause_also_preserves_onset(self):
        p = make_pipe(0.15)
        p._gate_target = p._gate_gain = 0.0
        pause(p, noise=0.004)
        out = p._infer_block(voice(p), realtime=False)
        self.assertEqual(p._gate_target, 1.0)
        self.assertGreater(float(np.max(np.abs(out))), 0.05)

    def test_uncertain_noise_vad_does_not_prevent_pause(self):
        p = make_pipe(0.04)
        p._input_vad.process = lambda block, sr: 1.0 if np.max(np.abs(block)) > 0.02 else 0.25
        p._gate_target = p._gate_gain = 0.0
        pause(p, noise=0.004)
        p._infer_block(voice(p), realtime=False)
        self.assertEqual(p._gate_target, 1.0)

    def test_output_denoise_cannot_erase_protected_onset(self):
        p = make_pipe(output_denoise=True)
        pause(p)
        out = p._infer_block(voice(p), realtime=False)
        self.assertGreater(float(np.max(np.abs(out))), 0.05)

    def test_new_speaker_can_still_be_rejected(self):
        p = make_pipe()
        pause(p)
        p.voice_gate.score = 0.05
        p._infer_block(voice(p), realtime=False)
        self.assertEqual(p._gate_target, 1.0, "partial onset is not enough to reject")
        for _ in range(30):
            p._infer_block(voice(p), realtime=False)
        self.assertEqual(p._gate_target, 0.0)
        self.assertLess(p._gate_gain, 0.02)

    def test_short_word_pause_keeps_existing_speaker_decision(self):
        p = make_pipe()
        p._gate_target = p._gate_gain = 0.0
        pause(p, seconds=0.18)
        p.voice_gate.score = 0.05
        p._infer_block(voice(p), realtime=False)
        self.assertEqual(p._gate_target, 0.0)

    def test_undecidable_onset_grace_is_bounded(self):
        p = make_pipe()
        p._gate_target = p._gate_gain = 0.0
        pause(p)
        for _ in range(16):
            p._infer_block(voice(p), realtime=False)
        self.assertFalse(p._gate_hold_open)
        self.assertEqual(p._gate_target, 0.0)

    def test_input_vad_failure_keeps_audio_path_working(self):
        p = make_pipe()

        def failed_vad(*args, **kwargs):
            raise RuntimeError("unavailable VAD")

        p._input_vad.process = failed_vad
        p._gate_target = p._gate_gain = 0.0
        pause(p)
        out = p._infer_block(voice(p), realtime=False)
        self.assertIsNone(p._input_vad)
        self.assertGreater(float(np.max(np.abs(out))), 0.05)

    def test_speaker_smoothing_is_reset_for_new_utterance(self):
        p = make_pipe()
        p.voice_gate.score = 0.05
        for _ in range(10):
            p._infer_block(voice(p), realtime=False)
        resets = p.voice_gate.resets
        pause(p)
        p.voice_gate.score = 0.8
        for _ in range(10):
            p._infer_block(voice(p), realtime=False)
        self.assertGreater(p.voice_gate.resets, resets)
        self.assertEqual(p._gate_target, 1.0)

    def test_inflight_old_rejection_or_error_cannot_close_new_onset(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                p = make_pipe()
                entered, release = threading.Event(), threading.Event()

                def delayed_decide(wav):
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test did not release the decision")
                    if fail:
                        raise ValueError("old utterance failure")
                    return 0.01

                p.voice_gate.decide = delayed_decide
                try:
                    p._voice_gate_tick(voice(p), realtime=True)
                    self.assertTrue(entered.wait(1), "decision worker did not start")
                    p._gate_target = p._gate_gain = 0.0
                    pause(p)
                    p._infer_block(voice(p), realtime=True)
                    p._gate_stop.set()
                    release.set()
                    p._gate_thread.join(2)
                    self.assertFalse(p._gate_thread.is_alive())
                    self.assertIsNone(p._gate_result)
                    self.assertEqual(p._gate_target, 1.0)
                    self.assertEqual(p.voice_stats["errors"], 0)
                finally:
                    release.set()
                    p._stop_gate_worker()


if __name__ == "__main__":
    unittest.main()
