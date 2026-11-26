"""Wake words with microWakeWord, through pymicro-wakeword: a pipeline's replica checks its words once, and each
session's server VAD runs its own detector over the 16 kHz audio it feeds Silero. A detector's state lives in its
TFLite interpreter, so every session has its own."""

import json
import os
from pathlib import Path

from modelship.utils.config_schema import BUILTIN_WAKE_WORDS, WakeWord

# the TFLite C library the bootstrapper provisions on macOS; unset, the package's own
TFLITE_LIB_ENV = "MSHIP_TFLITE_C_LIB"
# a feature covers this much audio, and one follows every _FRAME_MS
_WINDOW_MS = 30
_FRAME_MS = 10


def _manifest(word: WakeWord) -> Path:
    if word.model in BUILTIN_WAKE_WORDS:
        import pymicro_wakeword

        return Path(pymicro_wakeword.__path__[0]) / "models" / f"{word.model}.json"
    return Path(word.model)


def _check(manifest: Path) -> None:
    """Raises ValueError unless *manifest* is a microWakeWord v2 manifest with its .tflite beside it."""
    try:
        config = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"wake word manifest {str(manifest)!r} can't be read: {e}") from e
    if not isinstance(config, dict) or config.get("type") != "micro" or config.get("version") != 2:
        raise ValueError(f"wake word manifest {str(manifest)!r} is not a microWakeWord v2 manifest")
    model = config.get("model")
    if not isinstance(model, str) or not (manifest.parent / model).is_file():
        raise ValueError(f"wake word manifest {str(manifest)!r} names model {model!r}, which isn't beside it")


class WakeStream:
    """The wake words heard in one stream of 16 kHz 16-bit mono PCM, found as it arrives, on a timeline in ms that
    starts at *start_ms*. One call at a time."""

    def __init__(self, words: list[tuple[Path, float | None]], library: str | None, start_ms: float) -> None:
        from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures

        self._models = []
        for manifest, cutoff in words:
            model = MicroWakeWord.from_config(manifest, libtensorflowlite_c_path=library)
            if cutoff is not None:
                model.probability_cutoff = cutoff
            self._models.append(model)
        self._features = MicroWakeWordFeatures()
        self._start_ms = start_ms
        # features made so far
        self._frames = 0

    def detect(self, pcm16: bytes) -> list[float]:
        """Where a wake word ends at each step its score is above its cutoff in *pcm16*: the end of that step's
        audio."""
        found = []
        for features in self._features.process_streaming(pcm16):
            at = self._start_ms + _WINDOW_MS + self._frames * _FRAME_MS
            self._frames += 1
            for model in self._models:
                if model.process_streaming(features):
                    found.append(at)
        return found


class WakeWords:
    """A pipeline's wake words, checked and loaded once in its replica; raises ValueError when one isn't usable."""

    def __init__(self, words: list[WakeWord]) -> None:
        self._words = [(_manifest(word), word.cutoff) for word in words]
        for manifest, _ in self._words:
            _check(manifest)
        self._library = os.environ.get(TFLITE_LIB_ENV) or None
        # every word loads here, so a broken one fails the replica, not a session
        self.stream(0.0)

    def stream(self, start_ms: float) -> WakeStream:
        return WakeStream(self._words, self._library, start_ms)
