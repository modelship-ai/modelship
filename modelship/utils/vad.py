"""Voice activity detection with Silero.

`Vad` runs sherpa-onnx's detector, one per audio stream, timed the way OpenAI's Realtime
`server_vad` reports it; its audio is float32, mono, 16 kHz. `SileroModel` runs the same
model on onnxruntime, one for every stream, and each `SpeechStream` keeps only the model's
state and where its speech is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from modelship.utils.sources import FileSource, check_file_source

if TYPE_CHECKING:
    from modelship.utils.wake import WakeStream

SAMPLE_RATE = 16000

# k2-fsa/sherpa-onnx asr-models silero_vad_v5.onnx, mirrored unmodified.
_SILERO_URL = "https://github.com/modelship-ai/model-bundles/releases/download/vad/silero/silero_vad_v5-6b99cbfd.onnx"
_SILERO_SHA256 = "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396"
_SILERO_DEST = "vad/silero_vad_v5-6b99cbfd.onnx"


def silero_vad_source() -> FileSource:
    """The pinned Silero VAD model, for `download_model_source`."""
    return check_file_source(_SILERO_URL, _SILERO_SHA256, _SILERO_DEST)


@dataclass(frozen=True)
class VadSettings:
    """OpenAI's `server_vad` parameters, with its defaults."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


@dataclass(frozen=True)
class SpeechStarted:
    # where the turn's audio starts, prefix padding included
    audio_start_ms: int


@dataclass(frozen=True)
class SpeechStopped:
    audio_start_ms: int
    # where speech was found to have stopped, the trailing silence included
    audio_end_ms: int
    # the turn's audio from audio_start_ms to audio_end_ms
    audio: np.ndarray


VadEvent = SpeechStarted | SpeechStopped


def _ms(samples: int) -> int:
    return samples * 1000 // SAMPLE_RATE


class Vad:
    def __init__(
        self,
        model_path: str,
        settings: VadSettings | None = None,
        *,
        min_speech_ms: int = 250,
        max_speech_s: float = 300.0,
    ) -> None:
        import sherpa_onnx

        self.settings = settings or VadSettings()
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = self.settings.threshold
        config.silero_vad.min_silence_duration = self.settings.silence_duration_ms / 1000
        config.silero_vad.min_speech_duration = min_speech_ms / 1000
        config.silero_vad.max_speech_duration = max_speech_s
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        self._detector: Any = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max_speech_s + 5)
        self._window: int = config.silero_vad.window_size
        self._min_speech = min_speech_ms * SAMPLE_RATE // 1000
        self._padding = self.settings.prefix_padding_ms * SAMPLE_RATE // 1000
        self._unfed = np.empty(0, dtype=np.float32)
        # samples the detector has seen
        self._fed = 0
        # audio from sample `_kept_from` on, so a turn can include its prefix padding
        self._kept = np.empty(0, dtype=np.float32)
        self._kept_from = 0
        self._speaking = False

    def feed(self, samples: np.ndarray) -> list[VadEvent]:
        samples = np.asarray(samples, dtype=np.float32)
        self._kept = np.concatenate([self._kept, samples])
        self._unfed = np.concatenate([self._unfed, samples])
        events: list[VadEvent] = []
        while len(self._unfed) >= self._window:
            window, self._unfed = self._unfed[: self._window], self._unfed[self._window :]
            self._detector.accept_waveform(window)
            self._fed += self._window
            events += self._events()
        if not self._speaking:
            self._trim()
        return events

    def flush(self) -> list[VadEvent]:
        """Ends the stream: a turn still in progress stops at the last sample fed."""
        if len(self._unfed):
            self._detector.accept_waveform(self._unfed)
            self._fed += len(self._unfed)
            self._unfed = np.empty(0, dtype=np.float32)
        self._detector.flush()
        return self._events()

    def _events(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        speaking = bool(self._detector.is_speech_detected())
        if speaking and not self._speaking:
            # the detector reports speech once it has lasted min_speech_ms
            start = max(0, self._fed - self._min_speech - self._window - self._padding)
            events.append(SpeechStarted(_ms(start)))
        while not self._detector.empty():
            segment = self._detector.front
            start = max(0, segment.start - self._padding)
            events.append(SpeechStopped(_ms(start), _ms(self._fed), self._slice(start, self._fed)))
            self._detector.pop()
            self._trim()
        self._speaking = speaking and not any(isinstance(event, SpeechStopped) for event in events[-1:])
        return events

    def _slice(self, start: int, end: int) -> np.ndarray:
        begin = max(start, self._kept_from) - self._kept_from
        return self._kept[begin : end - self._kept_from].copy()

    def _trim(self) -> None:
        # a segment can start before the detector reports it; a second's margin covers that
        keep_from = self._fed - self._padding - self._min_speech - self._window - SAMPLE_RATE
        if keep_from > self._kept_from:
            self._kept = self._kept[keep_from - self._kept_from :]
            self._kept_from = keep_from


# Silero v5 at 16 kHz: windows of 512 samples, each run with the 64 samples before it
WINDOW = 512
CONTEXT = 64
# Silero's recurrent state for one stream
STATE_SHAPE = (2, 1, 128)
# below the threshold less this, a window is silence; between the two, speech goes on but doesn't start
HYSTERESIS = 0.15
# speech that lasts this long opens a segment
MIN_SPEECH_MS = 250


class SileroModel:
    """Silero VAD v5 on onnxruntime, one model for every stream. Thread-safe."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self._rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def run(self, window: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        """The speech probability of *window* (CONTEXT + WINDOW samples), and the state after it."""
        probability, after = self._session.run(None, {"input": window[None], "state": state, "sr": self._rate})
        return float(np.asarray(probability).item()), np.asarray(after)


@dataclass(frozen=True)
class SpeechSegment:
    start_ms: float
    # None while the silence after its speech is shorter than the silence duration
    end_ms: float | None
    # where the speech heard in it ends: its end once closed
    speech_end_ms: float


class SpeechStream:
    """The speech in one stream of 16-bit mono PCM at *rate*, found as it arrives, on a timeline in ms that starts
    at *start_ms*. A segment opens once speech lasts MIN_SPEECH_MS, dated back to where it started, and closes once
    silence lasts the silence duration. With *wake*, the wake words in the same audio too. One feed at a time."""

    def __init__(self, model: SileroModel, rate: int, start_ms: float, wake: WakeStream | None = None) -> None:
        from modelship.utils.audio import StreamResampler

        self.rate = rate
        self._model = model
        self._resample = StreamResampler(rate, SAMPLE_RATE)
        self._wake = wake
        # the audio the wake words are heard in, when it isn't what Silero hears
        self._wake_resample: StreamResampler | None = None
        # where a wake word ended in the last feed's audio, at each step it was heard
        self.woke: list[float] = []
        self._start_ms = start_ms
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._window = np.zeros(CONTEXT + WINDOW, dtype=np.float32)
        self._unfed = np.empty(0, dtype=np.float32)
        # samples run through the model
        self._fed = 0
        # where speech not long enough to open a segment yet started
        self._speech: int | None = None
        # the open segment's start, and where the silence in it started
        self._open: int | None = None
        self._silence: int | None = None

    def feed(
        self, pcm16: bytes, threshold: float, silence_duration_ms: int, wake_pcm16: bytes | None = None
    ) -> list[SpeechSegment]:
        """The segments that closed in *pcm16*, then the one still open; the wake words go to `woke`, heard in
        *wake_pcm16*, the same stretch of audio as *pcm16*, when it is given on every feed."""
        from modelship.utils.audio import StreamResampler, from_pcm16, to_pcm16

        samples = self._resample(from_pcm16(pcm16))
        if self._wake is not None:
            if wake_pcm16 is not None:
                if self._wake_resample is None:
                    self._wake_resample = StreamResampler(self.rate, SAMPLE_RATE)
                heard = self._wake_resample(from_pcm16(wake_pcm16))
            else:
                heard = samples
            self.woke = self._wake.feed(to_pcm16(heard))
        self._unfed = np.concatenate([self._unfed, samples])
        quiet = max(threshold - HYSTERESIS, 0.01)
        min_speech = MIN_SPEECH_MS * SAMPLE_RATE // 1000
        min_silence = silence_duration_ms * SAMPLE_RATE // 1000
        found: list[SpeechSegment] = []
        count = len(self._unfed) // WINDOW
        for index in range(count):
            self._window[:CONTEXT] = self._window[-CONTEXT:]
            self._window[CONTEXT:] = self._unfed[index * WINDOW : (index + 1) * WINDOW]
            probability, self._state = self._model.run(self._window, self._state)
            start, self._fed = self._fed, self._fed + WINDOW
            if self._open is None:
                if probability >= threshold and self._speech is None:
                    self._speech = start
                elif probability < quiet:
                    self._speech = None
                if self._speech is not None and self._fed - self._speech >= min_speech:
                    self._open, self._speech, self._silence = self._speech, None, None
                continue
            if probability >= threshold:
                self._silence = None
            elif probability < quiet and self._silence is None:
                self._silence = start
            if self._silence is not None and self._fed - self._silence >= min_silence:
                end = self._ms(self._silence)
                found.append(SpeechSegment(self._ms(self._open), end, end))
                self._open = self._silence = None
        self._unfed = self._unfed[count * WINDOW :]
        if self._open is not None:
            speech_end = self._fed if self._silence is None else self._silence
            found.append(SpeechSegment(self._ms(self._open), None, self._ms(speech_end)))
        return found

    def _ms(self, samples: int) -> float:
        return self._start_ms + samples * 1000 / SAMPLE_RATE
