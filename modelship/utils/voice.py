"""`VoiceDetector`: what the realtime pipeline's server VAD runs over one stream of a session's audio. It finds the
speech in it with Silero (`modelship.utils.vad.SpeechStream`) and, when the session has wake words, the wake words in
the same audio (`modelship.utils.wake.WakeStream`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelship.utils.vad import SAMPLE_RATE, SileroModel, SpeechSegment, SpeechStream

if TYPE_CHECKING:
    import numpy as np

    from modelship.utils.audio import StreamResampler
    from modelship.utils.wake import WakeStream


class VoiceDetector:
    """The speech, and with *wake* the wake words, in one stream of 16-bit mono PCM at *rate*, on a timeline in ms
    that starts at *start_ms*. Both detectors run on the same audio resampled to 16 kHz once. When the caller gives
    the audio as the client sent it too (half duplex mutes the answer's echo in the buffer), the wake words are heard
    in that, resampled on its own. One call at a time."""

    def __init__(self, model: SileroModel, rate: int, start_ms: float, wake: WakeStream | None = None) -> None:
        # modelship.utils.audio imports librosa, which only a replica running server VAD needs
        from modelship.utils.audio import StreamResampler

        self.rate = rate
        self._resample = StreamResampler(rate, SAMPLE_RATE)
        self._speech = SpeechStream(model, start_ms)
        self._wake = wake
        # resamples the audio as sent, once the caller gives it
        self._sent_resample: StreamResampler | None = None
        # where a wake word ended in the last call's audio, each time one was heard
        self.wake_ends_ms: list[float] = []

    def detect(
        self, pcm16: bytes, threshold: float, silence_duration_ms: int, sent_pcm16: bytes | None = None
    ) -> list[SpeechSegment]:
        """The speech segments that closed in *pcm16*, then the one still open; the wake words heard go to
        `wake_ends_ms`. *sent_pcm16*, when given, must be given on every call."""
        from modelship.utils.audio import from_pcm16, to_pcm16

        samples = self._resample(from_pcm16(pcm16))
        if self._wake is not None:
            self.wake_ends_ms = self._wake.detect(to_pcm16(self._wake_audio(samples, sent_pcm16)))
        return self._speech.detect(samples, threshold, silence_duration_ms)

    def _wake_audio(self, samples: np.ndarray, sent_pcm16: bytes | None) -> np.ndarray:
        """The 16 kHz audio the wake words are heard in: what Silero hears, or the audio as sent."""
        from modelship.utils.audio import StreamResampler, from_pcm16

        if sent_pcm16 is None:
            return samples
        if self._sent_resample is None:
            self._sent_resample = StreamResampler(self.rate, SAMPLE_RATE)
        return self._sent_resample(from_pcm16(sent_pcm16))
