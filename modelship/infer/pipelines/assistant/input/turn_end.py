"""How a turn whose speech paused ends, for `TurnDetector`: `SilenceEnd` (server_vad) once the silence lasts
`silence_duration_ms`; `SemanticEnd` (semantic_vad) at once when a classifier hears the speech finished, else when
the user speaks again or after the longest wait the session's `eagerness` allows."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.openai.protocol.realtime import SemanticVad, ServerVad

# semantic_vad: the silence that closes speech for the classifier to hear
SEMANTIC_PAUSE_MS = 200
# semantic_vad hears speech as server_vad does by default
SEMANTIC_THRESHOLD = 0.5
SEMANTIC_PREFIX_PADDING_MS = 300
# above this, the classifier hears the turn finished
COMPLETE = 0.5
# the longest a turn heard as unfinished waits for more speech, by eagerness (openai-python's SemanticVad)
EAGERNESS_WAIT_MS = {"low": 8000, "medium": 4000, "high": 2000, "auto": 4000}
# the classifier hears this much of a turn's end
_HEARD_MS = 8000


class TurnClassifier(Protocol):
    """How likely a turn's speaker is done (`modelship.utils.smart_turn.SmartTurnModel`). Thread-safe."""

    def end_probability(self, pcm16: bytes, rate: int) -> float: ...


class TurnClassifierLoader(Protocol):
    """Builds the replica's turn classifier when a session first asks for semantic VAD
    (`modelship.utils.smart_turn.SmartTurnLoader`); raises when it can't."""

    async def load(self) -> TurnClassifier: ...


class TurnClassifierError(Exception):
    """The classifier failed on a turn."""


class TurnEnd(Protocol):
    """What a kind of turn detection decides: how it hears speech, and where a turn whose speech paused ends."""

    @property
    def threshold(self) -> float:
        """The speech probability that is speech."""
        ...

    @property
    def prefix_padding_ms(self) -> int:
        """The audio before the speech a turn starts with."""
        ...

    @property
    def pause_ms(self) -> int:
        """The silence that closes speech."""
        ...

    async def ends_at(self, buffer: InputAudioBuffer, turn_start_ms: float, speech_end_ms: float) -> float | None:
        """Where the turn starting at *turn_start_ms*, whose speech stopped at *speech_end_ms* and stayed quiet for
        `pause_ms`, ends; None while it may go on. Raises `TurnClassifierError` when it can't tell."""
        ...

    def gives_up_ms(self, speech_end_ms: float) -> float:
        """Where a turn that may go on, whose speech stopped at *speech_end_ms*, ends if no speech follows."""
        ...


@dataclass(frozen=True)
class SilenceEnd:
    """server_vad: a turn ends once its silence lasts `silence_duration_ms`."""

    settings: ServerVad

    @property
    def threshold(self) -> float:
        return self.settings.threshold

    @property
    def prefix_padding_ms(self) -> int:
        return self.settings.prefix_padding_ms

    @property
    def pause_ms(self) -> int:
        return self.settings.silence_duration_ms

    async def ends_at(self, buffer: InputAudioBuffer, turn_start_ms: float, speech_end_ms: float) -> float | None:
        return speech_end_ms + self.settings.silence_duration_ms

    def gives_up_ms(self, speech_end_ms: float) -> float:
        return speech_end_ms + self.settings.silence_duration_ms


@dataclass(frozen=True)
class SemanticEnd:
    """semantic_vad: a turn ends after a short pause when *classifier* hears it finished, else after the wait its
    eagerness allows."""

    settings: SemanticVad
    classifier: TurnClassifier

    threshold: float = SEMANTIC_THRESHOLD
    prefix_padding_ms: int = SEMANTIC_PREFIX_PADDING_MS
    pause_ms: int = SEMANTIC_PAUSE_MS

    async def ends_at(self, buffer: InputAudioBuffer, turn_start_ms: float, speech_end_ms: float) -> float | None:
        at_ms = speech_end_ms + self.pause_ms
        rate = buffer.rate
        assert rate is not None
        pcm = buffer.pcm(max(turn_start_ms, at_ms - _HEARD_MS), at_ms)
        try:
            probability = await asyncio.to_thread(self.classifier.end_probability, pcm, rate)
        except Exception as e:
            raise TurnClassifierError from e
        return at_ms if probability > COMPLETE else None

    def gives_up_ms(self, speech_end_ms: float) -> float:
        return speech_end_ms + EAGERNESS_WAIT_MS[self.settings.eagerness]


def turn_end(settings: ServerVad | SemanticVad, classifier: TurnClassifier | None) -> TurnEnd:
    """The end of turn *settings* ask for. A semantic one needs *classifier*."""
    if isinstance(settings, ServerVad):
        return SilenceEnd(settings)
    if classifier is None:
        raise RuntimeError("semantic_vad needs a turn classifier")
    return SemanticEnd(settings, classifier)
