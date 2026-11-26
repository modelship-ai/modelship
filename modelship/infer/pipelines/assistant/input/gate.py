"""`TurnGate`: when speech may start a turn, the session's barge-in and wake-word rules in one place. Server VAD's
`TurnDetector` asks it before it starts a turn."""

from collections.abc import Callable

from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.infer.pipelines.assistant.input.turns import ANYWHERE, SHUT, Window
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.openai.protocol.realtime import AssistantMessage, OutputText, RealtimeResponse
from modelship.utils.config_schema import AssistantConfig, Wake

# an answer that ends in one of these asks something (Home Assistant's continue_conversation)
_ASKS = ("?", ";", "？")


def asks(response: RealtimeResponse) -> bool:
    """Whether *response*'s last message ends in a question."""
    for item in reversed(response.output):
        if isinstance(item, AssistantMessage) and item.content:
            part = item.content[-1]
            said = part.text if isinstance(part, OutputText) else part.transcript
            return said.rstrip().endswith(_ASKS)
    return False


class TurnGate:
    """The session's rules for when speech starts a turn:

    - while an answer plays in full duplex, speech must last `min_interruption_ms`, since the turn cuts it off;
    - without *wake* words, speech starts a turn anywhere;
    - with them, a wake word starts one, and other speech only in a follow-up's window. The window opens once a
      completed answer the follow-up mode covers has played, lasts `follow_up.listen_s`, and the turn it lets start
      shuts it. Under `follow_up.mode: always` speech also starts a turn while an answer runs.

    *answering* says whether a response is in progress; *buffer* and *playback* place the window on the timeline."""

    def __init__(
        self,
        assistant: AssistantConfig,
        wake: Wake | None,
        buffer: InputAudioBuffer,
        playback: Playback,
        answering: Callable[[], bool],
    ):
        self._assistant = assistant
        self._wake = wake
        self._buffer = buffer
        self._playback = playback
        self._answering = answering
        self.listen_ms: float | None = wake.listen_s * 1000 if wake is not None else None
        # with wake words: where speech may start a turn without one
        self._follow_up = SHUT

    def min_speech_ms(self) -> int:
        if self._assistant.duplex == "full" and self._playback.playing():
            return self._assistant.min_interruption_ms
        return 0

    def window(self) -> Window:
        if self._wake is None:
            return ANYWHERE
        answering = self._answering() or self._playback.playing()
        if self._wake.follow_up.mode == "always" and answering:
            return ANYWHERE
        return self._follow_up

    def turn_started(self) -> None:
        """A turn started: a follow-up's window lets one turn start."""
        self._follow_up = SHUT

    def answered(self, response: RealtimeResponse, in_conversation: bool) -> None:
        """*response* ended: with wake words, opens the follow-up's window once its answer has played, when the
        follow-up mode covers it."""
        if self._wake is None or not in_conversation or response.status != "completed":
            return
        follow_up = self._wake.follow_up
        if follow_up.mode == "off" or (follow_up.mode == "question" and not asks(response)):
            return
        opens = self._buffer.end_ms + self._playback.remaining() * 1000
        self._follow_up = Window(opens, opens + follow_up.listen_s * 1000)
