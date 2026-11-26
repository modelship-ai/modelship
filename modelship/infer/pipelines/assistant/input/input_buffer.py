"""The input audio buffer: the client's audio the session hasn't committed as a turn yet."""

from modelship.infer.pipelines.assistant.codecs import wav

# OpenAI's least audio for a commit
MIN_SECONDS = 0.1
# the most PCM the input audio buffer holds: an hour of 24 kHz 16-bit audio
MAX_BYTES = 3600 * 24000 * 2


class InputAudioBuffer:
    """The client's audio not committed yet, as 16-bit mono PCM at one sample rate, placed on the session's
    timeline: the ms of audio the client has sent since the session opened. In half duplex the buffer holds silence
    where the client's audio was the answer's echo; once `keep_sent` is called it keeps the audio as sent beside it."""

    def __init__(self) -> None:
        self._pcm = bytearray()
        # once kept: the audio as the client sent it, where the buffer holds silence in its place
        self._sent: bytearray | None = None
        self.rate: int | None = None
        # where the buffer's first sample is on the timeline
        self.start_ms = 0.0

    @property
    def seconds(self) -> float:
        return len(self._pcm) // 2 / self.rate if self.rate else 0.0

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.seconds * 1000

    @property
    def keeps_sent(self) -> bool:
        return self._sent is not None

    def keep_sent(self) -> None:
        """From now on, keeps the audio as the client sent it beside the buffer's."""
        if self._sent is None:
            self._sent = bytearray(self._pcm)

    def append(self, pcm16: bytes, rate: int, sent: bytes | None = None) -> str | None:
        """Adds *pcm16* at *rate*, where the client sent *sent* if it differs; why it can't, or None."""
        if self.rate is not None and self._pcm and rate != self.rate:
            return "The audio format changed with audio in the buffer; clear it first."
        if len(self._pcm) + len(pcm16) > MAX_BYTES:
            return f"The input audio buffer holds at most {MAX_BYTES / 1e6:g} MB of PCM; commit or clear it."
        self.rate = rate
        self._pcm += pcm16
        if self._sent is not None:
            self._sent += pcm16 if sent is None else sent
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

    def sent_pcm(self, from_ms: float, to_ms: float) -> bytes:
        """The same audio as the client sent it."""
        if self._sent is None:
            return self.pcm(from_ms, to_ms)
        return bytes(self._sent[self._offset(from_ms) : self._offset(to_ms)])

    def drop_before(self, at_ms: float) -> None:
        """Drops the audio before *at_ms*."""
        cut = self._offset(at_ms)
        if cut and self.rate is not None:
            del self._pcm[:cut]
            if self._sent is not None:
                del self._sent[:cut]
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
