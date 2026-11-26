"""How the audio of a session's turns reaches its speech-to-text model. `RecordedInput` takes a turn's audio whole
when the turn is committed. `StreamedInput` streams it to the model as it arrives, so the commit only finishes the
stream. The session calls the same methods on either, as audio arrives and as server VAD finds turns."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.infer.pipelines.assistant.input.stt_stream import StreamTarget, TurnStream
from modelship.infer.pipelines.assistant.input.transcripts import RecordedTurn, StreamedTurn, SttContext, TurnAudio
from modelship.infer.pipelines.assistant.input.turns import TurnDetector
from modelship.infer.pipelines.assistant.tasks import BackgroundTasks
from modelship.openai.protocol import SttStreamOptions
from modelship.openai.protocol.realtime import server_event
from modelship.utils import random_uuid

Send = Callable[[dict], Awaitable[None]]


class TurnInput(Protocol):
    """How the audio of the session's turns reaches its speech-to-text model."""

    # whether the model hears a turn as it arrives; a streamed turn costs the model no more as it grows
    streams: bool

    def appended(self) -> None:
        """The client appended audio to the buffer."""
        ...

    def vad_progressed(self) -> None:
        """Server VAD has heard more of the buffer."""
        ...

    def turn_started(self, turn: str, start_ms: float, woken: bool) -> None:
        """Server VAD found turn *turn* starting at *start_ms*, where its wake word ended when *woken*."""
        ...

    def take(self, start_ms: float, end_ms: float, turn: str | None) -> TurnAudio:
        """Takes the audio from *start_ms* to *end_ms* out of the buffer, as the committed turn *turn* (None for the
        buffer the client commits)."""
        ...

    def drop(self) -> None:
        """Forgets the audio heard so far, which won't be committed."""
        ...

    async def wait_closed(self, timeout: float) -> None:
        """Waits, at most *timeout* seconds, for what the session leaves unfinished to close."""
        ...


class RecordedInput:
    """The turn's audio goes to the model whole, as a WAV, once the turn is committed."""

    streams = False

    def __init__(self, buffer: InputAudioBuffer):
        self._buffer = buffer

    def appended(self) -> None:
        pass

    def vad_progressed(self) -> None:
        pass

    def turn_started(self, turn: str, start_ms: float, woken: bool) -> None:
        pass

    def take(self, start_ms: float, end_ms: float, turn: str | None) -> TurnAudio:
        wav, seconds = self._buffer.take(start_ms, end_ms)
        return RecordedTurn(wav, seconds)

    def drop(self) -> None:
        pass

    async def wait_closed(self, timeout: float) -> None:
        pass


class StreamedInput:
    """The turn's audio goes to a streaming model as it arrives. One stream is open at a time: the turn server VAD
    has open, from its start, or the buffer from its start when the client commits. A turn a wake word opened streams
    once speech past the wake word is heard, so the silence after it never reaches the model. *detector* gives the
    session's server VAD, None while it is off, and *send* sends each turn's transcription deltas."""

    streams = True

    def __init__(
        self,
        context: SttContext,
        buffer: InputAudioBuffer,
        detector: Callable[[], TurnDetector | None],
        send: Send,
    ):
        self._context = context
        self._buffer = buffer
        self._detector = detector
        self._send = send
        self._target = StreamTarget(context.calls, context.model, context.capabilities.chunk_ms or 0, asyncio.Lock())
        # the stream open now, if any
        self._stream: TurnStream | None = None
        # streams closing, which the session waits for when it closes
        self._closing = BackgroundTasks()

    def appended(self) -> None:
        if self._detector() is None and self._stream is None:
            # the client commits: the buffer streams from its start
            self._stream = self._open(self._buffer.start_ms)
        self._push()

    def vad_progressed(self) -> None:
        detector = self._detector()
        opens = self._stream is None and detector is not None and detector.turn is not None and not detector.waiting
        if opens and detector is not None and detector.turn is not None:
            # a woken turn now holds speech past its wake word
            turn, start = detector.turn
            self._stream = self._open(start, turn)
        self._push()

    def turn_started(self, turn: str, start_ms: float, woken: bool) -> None:
        # the turn's deltas go out after its speech_started
        self.drop()
        if not woken:
            self._stream = self._open(start_ms, turn)
            self._push()

    def take(self, start_ms: float, end_ms: float, turn: str | None) -> TurnAudio:
        stream = self._stream if self._stream is not None else self._open(start_ms, turn)
        self._stream = None
        rest = self._buffer.pcm(stream.sent_ms, end_ms)
        self._buffer.drop_before(end_ms)
        return StreamedTurn(stream, rest, (end_ms - stream.start_ms) / 1000, self._closing)

    def drop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            self._closing.spawn(stream.close())

    async def wait_closed(self, timeout: float) -> None:
        await self._closing.wait(timeout)

    def _push(self) -> None:
        if self._stream is not None:
            self._stream.push_ready()

    def _open(self, start_ms: float, turn: str | None = None) -> TurnStream:
        """A stream of the buffer from *start_ms*: of the turn *turn* server VAD found, which gets deltas and no
        audio past what server VAD has heard, or of the buffer the client commits."""
        key = f"{self._context.session_id}:{turn or f'buffer_{random_uuid()[:12]}'}"
        options = SttStreamOptions(language=self._context.language())
        detector = self._detector()
        if turn is None or detector is None:
            return TurnStream(self._target, key, self._buffer, start_ms, options)

        async def delta(text: str) -> None:
            event = "conversation.item.input_audio_transcription.delta"
            await self._send(server_event(event, item_id=turn, content_index=0, delta=text))

        # audio server VAD hasn't heard yet may be past the turn's end
        return TurnStream(self._target, key, self._buffer, start_ms, options, delta, lambda: detector.heard_ms)
