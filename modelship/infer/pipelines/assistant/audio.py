"""A session's audio. Input: the turn the client hasn't committed yet, as 16-bit mono PCM, and the WAV a committed
turn goes to the speech-to-text model as; G.711 is decoded here, and the loaders resample whatever rate they get.
Output: the text-to-speech model's WAV in the client's format, resampled and G.711-encoded here."""

import functools
import io
import wave

import numpy as np

# OpenAI's least audio for a commit
MIN_SECONDS = 0.1
# the most PCM the input audio buffer holds: an hour of 24 kHz 16-bit audio
MAX_BYTES = 3600 * 24000 * 2
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
    with wave.open(io.BytesIO(data)) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"the WAV has {wav.getsampwidth() * 8}-bit samples, not 16-bit")
        channels, rate = wav.getnchannels(), wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, rate


def output_rate(kind: str, pcm_rate: int) -> int:
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
    return G711_RATE if kind in _G711_CODES else pcm_rate * 2


def wav(pcm16: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm16)
    return out.getvalue()


class InputAudioBuffer:
    """The client's audio not committed yet, at one sample rate, placed on the session's timeline: the ms of
    audio the client has sent since the session opened."""

    def __init__(self) -> None:
        self._pcm = bytearray()
        self.rate: int | None = None
        # where the buffer's first sample is on the timeline
        self.start_ms = 0.0

    @property
    def seconds(self) -> float:
        return len(self._pcm) // 2 / self.rate if self.rate else 0.0

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.seconds * 1000

    def append(self, pcm16: bytes, rate: int) -> str | None:
        """Adds *pcm16* at *rate*; why it can't, or None."""
        if self.rate is not None and self._pcm and rate != self.rate:
            return "The audio format changed with audio in the buffer; clear it first."
        if len(self._pcm) + len(pcm16) > MAX_BYTES:
            return f"The input audio buffer holds at most {MAX_BYTES / 1e6:g} MB of PCM; commit or clear it."
        self.rate = rate
        self._pcm += pcm16
        return None

    def _offset(self, at_ms: float) -> int:
        """The byte offset of *at_ms* in the buffer, clamped to it."""
        if self.rate is None:
            return 0
        samples = round((at_ms - self.start_ms) * self.rate / 1000)
        return min(max(samples, 0), len(self._pcm) // 2) * 2

    def pcm(self, from_ms: float, to_ms: float) -> bytes:
        """The audio from *from_ms* to *to_ms*, as much of it as the buffer holds."""
        return bytes(self._pcm[self._offset(from_ms) : self._offset(to_ms)])

    def drop_before(self, at_ms: float) -> None:
        """Drops the audio before *at_ms*."""
        cut = self._offset(at_ms)
        if cut and self.rate is not None:
            del self._pcm[:cut]
            self.start_ms += cut // 2 * 1000 / self.rate

    def clear(self) -> None:
        self.drop_before(self.end_ms)

    def take(self, from_ms: float | None = None, to_ms: float | None = None) -> tuple[bytes, float]:
        """The audio from *from_ms* (the buffer's start by default) to *to_ms* (its end) as a WAV, and its length
        in seconds; the buffer then starts at *to_ms*."""
        assert self.rate is not None
        start = self.start_ms if from_ms is None else from_ms
        end = self.end_ms if to_ms is None else to_ms
        pcm = self.pcm(start, end)
        self.drop_before(end)
        return wav(pcm, self.rate), len(pcm) // 2 / self.rate
