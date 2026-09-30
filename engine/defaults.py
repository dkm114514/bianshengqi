"""Project defaults captured from the user's active bb48k profile."""

DEFAULT_MODEL_PROFILE = {
    "pth": "",
    "index": "",
    "pitch": 16,
    "formant": 0.0,
    "index_rate": 0.0,
    "block_time": 0.15,
    "crossfade_time": 0.01,
    "extra_time": 4.0,
    "threhold": -60,
    "I_noise_reduce": True,
    "O_noise_reduce": True,
    "rms_mix_rate": 0.0,
    "f0method": "rmvpe",
    "n_cpu": 4,
}

DEFAULT_TOP_CONFIG = {
    "denoise_mode": "dfn3",
    "voice_gate": True,
    "voice_gate_threshold": 0.2,
    "agc": True,
    "agc_target_dbfs": -20.0,
    "input_device": "",
    "monitor_device": "",
    "monitor_on": True,
}

DEFAULT_ACTIVE_MODEL = "bb48k"
