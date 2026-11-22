"""Audio utilities shared by TTS/STT engines and serving wrappers."""

from __future__ import annotations

import io
import struct

import librosa
import numpy as np
import soundfile as sf
import soxr


def from_pcm16(pcm: bytes) -> np.ndarray:
    """Decode signed 16-bit little-endian PCM bytes as float32 in [-1, 1)."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class StreamResampler:
    """Resamples mono float32 audio chunk by chunk, keeping the filter state between
    chunks, so the output matches resampling the whole stream at once."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        self._stream = None if from_rate == to_rate else soxr.ResampleStream(from_rate, to_rate, 1, dtype="float32")

    def __call__(self, chunk: np.ndarray, last: bool = False) -> np.ndarray:
        """*last* flushes the filter's tail; the stream takes no chunk after it."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if self._stream is None:
            return chunk
        return self._stream.resample_chunk(chunk, last=last)


def to_pcm16(audio: np.ndarray) -> bytes:
    """Encode a numpy audio array as signed 16-bit little-endian PCM bytes.
    Float arrays are clipped to [-1, 1] and scaled; int arrays pass through."""
    if audio.ndim > 1:
        audio = audio.squeeze()
    if np.issubdtype(audio.dtype, np.floating):
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    return audio.tobytes()


def wrap_pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap signed 16-bit mono PCM bytes with a WAV header."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        len(pcm),
    )
    return header + pcm


def to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a numpy audio array as a 16-bit PCM mono WAV file."""
    return wrap_pcm16_wav(to_pcm16(audio), sample_rate)


def decode_audio(data: bytes, target_sr: int) -> tuple[np.ndarray, float]:
    """Decode audio bytes and resample to `target_sr`. Returns (samples, duration_seconds)."""
    samples, source_sr = sf.read(io.BytesIO(data), dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    duration_seconds = len(samples) / source_sr
    if source_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=source_sr, target_sr=target_sr)
    return samples, duration_seconds
