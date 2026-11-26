# Derived from pipecat-ai/pipecat src/pipecat/audio/turn/smart_turn/_whisper_features.py
# (Copyright (c) 2024-2026, Daily; BSD 2-Clause License), itself derived from Hugging Face Transformers'
# WhisperFeatureExtractor (Copyright 2022-2023 The HuggingFace Inc. team; Apache License 2.0).
"""Whisper's log-mel features, in numpy, for Smart Turn (`modelship.utils.smart_turn`): what
`transformers.WhisperFeatureExtractor(chunk_length=8)` gives for 8 s of 16 kHz audio with `do_normalize=True`,
without importing transformers."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000
# the features cover this many samples: 8 s
SAMPLES = 8 * SAMPLE_RATE
_N_FFT = 400
_HOP = 160
_N_MELS = 80
_MEL_FLOOR = 1e-10
_VARIANCE_EPS = 1e-7


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """The Slaney mel scale: linear below 1 kHz, logarithmic above."""
    mels = 3.0 * hz / 200.0
    above = hz >= 1000.0
    mels[above] = 15.0 + np.log(hz[above] / 1000.0) * (27.0 / np.log(6.4))
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    hz = 200.0 * mels / 3.0
    above = mels >= 15.0
    hz[above] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[above] - 15.0))
    return hz


def _mel_filters() -> np.ndarray:
    """The Slaney-normalized triangular filters, (frequency bins, mels)."""
    low, high = _hz_to_mel(np.array([0.0, SAMPLE_RATE / 2]))
    edges = _mel_to_hz(np.linspace(low, high, _N_MELS + 2))
    bins = np.linspace(0, SAMPLE_RATE // 2, _N_FFT // 2 + 1)
    slopes = edges[None, :] - bins[:, None]
    widths = np.diff(edges)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (edges[2:] - edges[:-2]))[None, :]


_FILTERS = _mel_filters()
# periodic Hann, as torch.hann_window
_WINDOW = np.hanning(_N_FFT + 1)[:-1]


def log_mel(audio: np.ndarray) -> np.ndarray:
    """The (80, 800) float32 features of *audio*: exactly SAMPLES float samples at 16 kHz."""
    if audio.shape != (SAMPLES,):
        raise ValueError(f"expected {SAMPLES} samples, got shape {audio.shape}")
    x = audio.astype(np.float32)
    # zero mean, unit variance, in float32 as the reference
    x = (x - x.mean()) / np.sqrt(x.var() + _VARIANCE_EPS)
    padded = np.pad(x.astype(np.float64), _N_FFT // 2, mode="reflect")
    frames = sliding_window_view(padded, _N_FFT)[::_HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=-1)) ** 2
    mel = np.log10(np.maximum(_MEL_FLOOR, power @ _FILTERS)).T[:, :-1]
    mel = np.maximum(mel, mel.max() - 8.0)
    return ((mel + 4.0) / 4.0).astype(np.float32)
