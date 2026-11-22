"""Voice activity detection with Silero (through sherpa-onnx) over a stream of
audio, timed the way OpenAI's Realtime `server_vad` reports it.

One `Vad` per audio stream: it keeps state between `feed` calls. Audio is float32,
mono, 16 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modelship.utils.sources import FileSource, check_file_source

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
