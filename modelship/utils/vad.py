"""Voice activity detection with Silero v5 on onnxruntime, for the realtime pipeline's server VAD. `SileroModel` is
one model shared by every stream of a replica, and each `SpeechStream` keeps only the model's state and where its
speech is. The offline speech-to-text loader's detector is `modelship.infer.sherpa_onnx.vad.Vad`."""

from __future__ import annotations

from dataclasses import dataclass

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
    """The speech in one stream of 16 kHz float32 mono samples, found as it arrives, on a timeline in ms that starts
    at *start_ms*. A segment opens once speech lasts MIN_SPEECH_MS, dated back to where it started, and closes once
    silence lasts the silence duration. One call at a time; `modelship.utils.voice.VoiceDetector` feeds it PCM at
    any rate."""

    def __init__(self, model: SileroModel, start_ms: float) -> None:
        self._model = model
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

    def detect(self, samples: np.ndarray, threshold: float, silence_duration_ms: int) -> list[SpeechSegment]:
        """The segments that closed in *samples*, then the one still open."""
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
