"""Audio encodings a session converts between, all pure functions. Input: G.711 decoded to 16-bit PCM, and PCM wrapped
in the WAV a committed turn goes to the speech-to-text model as; the loaders resample whatever rate they get. Output:
the text-to-speech model's WAV in the client's format, resampled and G.711-encoded."""

import functools
import io
import wave

import numpy as np

G711_RATE = 8000


def _ulaw(code: int) -> int:
    code = ~code & 0xFF
    sample = (((code & 0x0F) << 3) + 0x84) << ((code >> 4) & 0x07)
    return 0x84 - sample if code & 0x80 else sample - 0x84


def _alaw(code: int) -> int:
    code ^= 0x55
    exponent, mantissa = (code >> 4) & 0x07, code & 0x0F
    sample = (mantissa << 4) + 8 if exponent == 0 else ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if code & 0x80 else -sample


_G711 = {
    "audio/pcmu": [_ulaw(code).to_bytes(2, "little", signed=True) for code in range(256)],
    "audio/pcma": [_alaw(code).to_bytes(2, "little", signed=True) for code in range(256)],
}


def to_pcm16(audio: bytes, kind: str) -> bytes:
    """*audio* of format *kind* as 16-bit little-endian PCM."""
    table = _G711.get(kind)
    return audio if table is None else b"".join(table[code] for code in audio)


# the top of each G.711 segment, on 14-bit (mu-law) and 13-bit (A-law) samples
_ULAW_SEGMENTS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_ALAW_SEGMENTS = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def _encode_ulaw(pcm: np.ndarray) -> np.ndarray:
    value = pcm >> 2
    mask = np.where(value < 0, 0x7F, 0xFF)
    value = np.minimum(np.abs(value), 8159) + (0x84 >> 2)
    segment = np.searchsorted(_ULAW_SEGMENTS, value)
    code = (segment << 4) | ((value >> (segment + 1)) & 0x0F)
    return (np.where(segment >= 8, 0x7F, code) ^ mask).astype(np.uint8)


def _encode_alaw(pcm: np.ndarray) -> np.ndarray:
    value = pcm >> 3
    mask = np.where(value >= 0, 0xD5, 0x55)
    value = np.where(value >= 0, value, -value - 1)
    segment = np.searchsorted(_ALAW_SEGMENTS, value)
    shift = np.maximum(segment, 1)
    code = (segment << 4) | ((value >> shift) & 0x0F)
    return (np.where(segment >= 8, 0x7F, code) ^ mask).astype(np.uint8)


# every 16-bit sample's code, indexed by the sample + 32768
_ALL_SAMPLES = np.arange(-32768, 32768, dtype=np.int32)
_G711_CODES = {"audio/pcmu": _encode_ulaw(_ALL_SAMPLES), "audio/pcma": _encode_alaw(_ALL_SAMPLES)}


@functools.cache
def _lowpass(ratio: float, taps: int = 63) -> np.ndarray:
    """A windowed-sinc filter that keeps what a rate *ratio* times the source's can hold."""
    cutoff = 0.45 * ratio
    n = np.arange(taps) - (taps - 1) / 2
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(taps)
    return kernel / kernel.sum()


def resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    """16-bit *samples* at rate *source*, at rate *target*."""
    if source == target or not len(samples):
        return samples
    signal = samples.astype(np.float64)
    if target < source:
        signal = np.convolve(signal, _lowpass(target / source), mode="same")
    count = round(len(signal) * target / source)
    resampled = np.interp(np.arange(count) * (source / target), np.arange(len(signal)), signal)
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """A 16-bit PCM WAV's samples, mixed to mono, and its rate."""
    with wave.open(io.BytesIO(data)) as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"the WAV has {wav_file.getsampwidth() * 8}-bit samples, not 16-bit")
        channels, rate = wav_file.getnchannels(), wav_file.getframerate()
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, rate


def output_rate(kind: str, pcm_rate: int) -> int:
    """The sample rate of output audio of format *kind*: G.711 is always 8 kHz."""
    return G711_RATE if kind in _G711_CODES else pcm_rate


def from_wav(data: bytes, kind: str, pcm_rate: int) -> bytes:
    """The WAV *data* in format *kind*: 16-bit little-endian PCM at *pcm_rate*, or G.711 at 8 kHz."""
    samples, rate = read_wav(data)
    samples = resample(samples, rate, output_rate(kind, pcm_rate))
    codes = _G711_CODES.get(kind)
    if codes is None:
        return samples.astype("<i2").tobytes()
    return codes[samples.astype(np.int32) + 32768].tobytes()


def bytes_per_second(kind: str, pcm_rate: int) -> int:
    """How many bytes a second of audio of format *kind* takes: one per sample for G.711, two for PCM."""
    return G711_RATE if kind in _G711_CODES else pcm_rate * 2


def wav(pcm16: bytes, rate: int) -> bytes:
    """Mono 16-bit *pcm16* at *rate*, as a WAV file."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm16)
    return out.getvalue()
