"""A session's input audio: the turn the client hasn't committed yet, as 16-bit mono PCM, and the WAV a committed turn
goes to the speech-to-text model as. G.711 is decoded here; the loaders resample whatever rate they get."""

import io
import wave

# OpenAI's limit for a commit, and the most audio one turn may hold
MIN_SECONDS = 0.1
MAX_SECONDS = 120.0
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


class InputAudioBuffer:
    """The audio of the turn the client hasn't committed, at one sample rate."""

    def __init__(self) -> None:
        self._pcm = bytearray()
        self.rate: int | None = None

    @property
    def seconds(self) -> float:
        return len(self._pcm) // 2 / self.rate if self.rate else 0.0

    def append(self, pcm16: bytes, rate: int) -> str | None:
        """Adds *pcm16* at *rate*; why it can't, or None."""
        if self.rate is not None and self._pcm and rate != self.rate:
            return "The audio format changed with audio in the buffer; clear it first."
        if (len(self._pcm) + len(pcm16)) // 2 / rate > MAX_SECONDS:
            return f"The input audio buffer holds at most {MAX_SECONDS:.0f} s; commit or clear it."
        self.rate = rate
        self._pcm += pcm16
        return None

    def clear(self) -> None:
        self._pcm = bytearray()

    def take(self) -> tuple[bytes, float]:
        """The buffered turn as a WAV and its length in seconds; the buffer is then empty."""
        assert self.rate is not None
        pcm = bytes(self._pcm[: len(self._pcm) // 2 * 2])
        seconds = self.seconds
        self.clear()
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            wav.writeframes(pcm)
        return out.getvalue(), seconds
