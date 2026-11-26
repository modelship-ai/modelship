"""When speech may start a turn (`TurnGate`): barge-in's min_interruption_ms, and with wake words the follow-up's
window; and `VoiceDetector` running Silero and the wake words over the same audio."""

import numpy as np

from modelship.infer.pipelines.assistant.input.gate import TurnGate, asks
from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.infer.pipelines.assistant.input.turns import ANYWHERE, SHUT, Window
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.openai.protocol.realtime import AssistantMessage, OutputAudio, OutputText, RealtimeResponse
from modelship.utils.config_schema import AssistantConfig
from modelship.utils.vad import CONTEXT
from modelship.utils.voice import VoiceDetector


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _response(said: str, status: str = "completed", spoken: bool = False) -> RealtimeResponse:
    part = OutputAudio(type="output_audio", transcript=said) if spoken else OutputText(type="output_text", text=said)
    message = AssistantMessage(type="message", role="assistant", content=[part])
    return RealtimeResponse(
        id="r", status=status, output=[message], output_modalities=["text"], max_output_tokens="inf"
    )  # type: ignore[arg-type]


def _gate(follow_up: str | None = None, duplex: str = "full", answering: bool = False):
    config: dict = {"stt": "w", "llm": "l", "duplex": duplex, "min_interruption_ms": 400}
    if follow_up is not None:
        config["wake"] = {"words": ["hey_jarvis"], "listen_s": 5, "follow_up": {"mode": follow_up, "listen_s": 8}}
    assistant = AssistantConfig.model_validate(config)
    clock, buffer = _Clock(), InputAudioBuffer()
    buffer.append(bytes(16000 * 2), 16000)
    playback = Playback(clock)
    gate = TurnGate(assistant, assistant.wake, buffer, playback, lambda: answering)
    return gate, playback, clock


class TestMinSpeech:
    def test_speech_must_last_while_an_answer_plays_in_full_duplex(self):
        gate, playback, _ = _gate()
        assert gate.min_speech_ms() == 0
        playback.sent(2.0)
        assert gate.min_speech_ms() == 400

    def test_half_duplex_mutes_the_answer_instead(self):
        gate, playback, _ = _gate(duplex="half")
        playback.sent(2.0)
        assert gate.min_speech_ms() == 0


class TestWindow:
    def test_without_wake_words_speech_starts_a_turn_anywhere(self):
        gate, _, _ = _gate()
        assert gate.window() == ANYWHERE and gate.listen_ms is None

    def test_with_wake_words_only_a_wake_word_starts_one(self):
        gate, _, _ = _gate(follow_up="off")
        assert gate.window() == SHUT and gate.listen_ms == 5000
        gate.answered(_response("Done?"), in_conversation=True)
        assert gate.window() == SHUT

    def test_a_question_opens_the_follow_up_once_its_answer_has_played(self):
        gate, playback, _ = _gate(follow_up="question")
        playback.sent(2.0)
        gate.answered(_response("Which room?", spoken=True), in_conversation=True)
        # the buffer ends at 1000 ms; the answer plays 2 s more
        assert gate.window() == Window(3000, 11000)
        gate.turn_started()
        assert gate.window() == SHUT

    def test_an_answer_that_asks_nothing_opens_no_follow_up_in_question_mode(self):
        gate, _, _ = _gate(follow_up="question")
        gate.answered(_response("Lights on."), in_conversation=True)
        assert gate.window() == SHUT

    def test_only_a_completed_answer_in_the_conversation_opens_one(self):
        gate, _, _ = _gate(follow_up="always")
        gate.answered(_response("Hm?", status="cancelled"), in_conversation=True)
        gate.answered(_response("Hm?"), in_conversation=False)
        assert gate.window() == SHUT
        gate.answered(_response("Lights on."), in_conversation=True)
        assert gate.window() == Window(1000, 9000)

    def test_always_lets_speech_start_a_turn_while_answering(self):
        gate, _, _ = _gate(follow_up="always", answering=True)
        assert gate.window() == ANYWHERE

    def test_asks_reads_the_last_message(self):
        assert asks(_response("Is that all? "))
        assert asks(_response("好吗？"))
        assert not asks(_response("Done."))


class _Silero:
    def run(self, window: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        return min(1.0, 2 * float(np.abs(window[CONTEXT:]).mean())), state


class _Wake:
    """A wake word stream recording the 16 kHz PCM it hears, and hearing one end per call."""

    def __init__(self) -> None:
        self.heard: list[int] = []

    def detect(self, pcm16: bytes) -> list[float]:
        self.heard.append(len(pcm16))
        return [42.0]


class TestVoiceDetector:
    def test_the_wake_words_hear_the_same_16k_audio_as_silero(self):
        wake = _Wake()
        detector = VoiceDetector(_Silero(), 24000, 0.0, wake)  # type: ignore[arg-type]
        detector.detect(bytes(24000 * 2), 0.5, 500)
        assert detector.wake_ends_ms == [42.0]
        # a second at 16 kHz, less the few ms the streaming resampler holds back
        assert 16000 * 2 - 1600 <= sum(wake.heard) <= 16000 * 2

    def test_the_wake_words_hear_the_audio_as_sent_when_given(self):
        wake = _Wake()
        detector = VoiceDetector(_Silero(), 16000, 0.0, wake)  # type: ignore[arg-type]
        loud = (np.full(16000, 0.4) * 32768).astype("<i2").tobytes()
        assert detector.detect(bytes(len(loud)), 0.5, 500, sent_pcm16=loud) == []
        assert wake.heard == [len(loud)]

    def test_without_wake_words_none_are_heard(self):
        detector = VoiceDetector(_Silero(), 16000, 0.0)  # type: ignore[arg-type]
        detector.detect(bytes(16000 * 2), 0.5, 500)
        assert detector.wake_ends_ms == []
