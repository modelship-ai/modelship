"""Transcribing a session's committed turns. Each committed turn's audio is a `TurnAudio`: recorded whole, streamed
to the model as it arrived, or silent. The session's `TranscriptionQueue` transcribes the turns one after another,
in the order they were committed, and tells the client each transcript."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.input.stt_stream import TurnStream
from modelship.infer.pipelines.assistant.tasks import BackgroundTasks
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    SttCapabilities,
    SttStreamText,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.openai.protocol.realtime import InputAudio, RealtimeSession, UserMessage, server_event

logger = get_logger("realtime")

Send = Callable[[dict], Awaitable[None]]

# what a model that couldn't tell the language reports
_NO_LANGUAGE = ("", "unknown")


class SttContext:
    """What every speech-to-text call of a session shares: the model, what it can do, the session's transcription
    settings, and the language the model detected on the session's first turn."""

    def __init__(self, calls: PipelineCalls, model: str, capabilities: SttCapabilities, session: RealtimeSession):
        self.calls = calls
        self.model = model
        self.capabilities = capabilities
        self._session = session
        # the language the model detected on the session's first turn, used when the settings name none
        self.detected: str | None = None

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def streams(self) -> bool:
        """Whether the model takes a turn's audio as it arrives."""
        return self.capabilities.streaming

    def language(self) -> str | None:
        """The language to transcribe in: the session's setting, else the one detected."""
        settings = self._session.audio.input.transcription
        return (settings.language if settings is not None else None) or self.detected

    def prompt(self) -> str:
        settings = self._session.audio.input.transcription
        return (settings.prompt or "") if settings is not None else ""

    def detects_language(self) -> bool:
        """Whether the next transcription should report its language, to be remembered."""
        return self.language() is None and self.capabilities.reports_language

    def remember_language(self, found: str | None) -> None:
        """Keeps the language a transcription reported, when none was detected before."""
        if self.detected is None and self.capabilities.reports_language and found and found not in _NO_LANGUAGE:
            self.detected = found


@dataclass(frozen=True)
class Transcript:
    """A turn's transcription: its *text*, or the *error* the client's failed event carries."""

    text: str | None = None
    error: dict | None = None


class TurnAudio(Protocol):
    """A committed turn's audio, ready to be transcribed."""

    # how long the turn is, for the transcript's usage
    seconds: float

    async def transcribe(self, context: SttContext) -> Transcript: ...

    def abandon(self) -> None:
        """The session closed before the transcription finished."""
        ...


@dataclass
class RecordedTurn:
    """A turn recorded whole, sent to the model as one WAV."""

    wav: bytes
    seconds: float

    async def transcribe(self, context: SttContext) -> Transcript:
        detect = context.detects_language()
        request = TranscriptionRequest.model_construct(
            file=None,
            model=context.model,
            language=context.language(),
            prompt=context.prompt(),
            response_format="verbose_json" if detect else "json",
            timestamp_granularities=[],
            stream=False,
            temperature=0.0,
        )
        try:
            result = await context.calls.transcribe(context.model, self.wav, request, context.session_id)
        except Exception:
            logger.exception("Realtime session %s: transcribing a turn failed", context.session_id)
            result = None
        if isinstance(result, TranscriptionResponse | TranscriptionResponseVerbose):
            if detect and isinstance(result, TranscriptionResponseVerbose):
                context.remember_language(result.language)
            return Transcript(text=result.text)
        if isinstance(result, ErrorResponse):
            return Transcript(error=result.error.model_dump(include={"type", "code", "message", "param"}))
        message = f"Model {context.model!r} didn't answer."
        return Transcript(error={"type": "server_error", "code": "server_error", "message": message})

    def abandon(self) -> None:
        pass


@dataclass
class StreamedTurn:
    """A turn whose audio was streamed to the model as it arrived; *rest* is the audio not pushed yet, which
    finishes the stream. *closing* holds the stream's close when the session leaves it unfinished."""

    stream: TurnStream
    rest: bytes
    seconds: float
    closing: BackgroundTasks

    async def transcribe(self, context: SttContext) -> Transcript:
        result = await self.stream.finish(self.rest)
        if isinstance(result, SttStreamText):
            context.remember_language(result.language)
            return Transcript(text=result.text)
        return Transcript(error=result)

    def abandon(self) -> None:
        self.closing.spawn(self.stream.close())


@dataclass
class SilentTurn:
    """A turn with no speech, after a wake word nobody followed: its transcript is empty, and no model hears it."""

    seconds: float

    async def transcribe(self, context: SttContext) -> Transcript:
        return Transcript(text="")

    def abandon(self) -> None:
        pass


class TranscriptionQueue:
    """Transcribes the session's committed turns one after another, in the order they were committed, and tells
    the client each transcript through *send*. Each turn is a user message of *conversation*."""

    def __init__(self, context: SttContext, conversation: Conversation, send: Send):
        self._context = context
        self._conversation = conversation
        self._send = send
        self._tasks = BackgroundTasks()
        # the transcription queued last; a response waits for it, so it answers every turn committed before it
        self.last: asyncio.Task | None = None

    def add(self, item: UserMessage, audio: TurnAudio) -> asyncio.Task:
        """Queues the transcription of *audio* into *item*'s transcript; the task that does it."""
        task = self.last = self._tasks.spawn(self._run(item, audio, self.last))
        return task

    def cancel(self) -> None:
        self._tasks.cancel()

    async def wait(self, timeout: float) -> None:
        await self._tasks.wait(timeout)

    async def _run(self, item: UserMessage, audio: TurnAudio, previous: asyncio.Task | None) -> None:
        try:
            if previous is not None:
                await asyncio.wait([previous])
            transcript = await audio.transcribe(self._context)
        except asyncio.CancelledError:
            audio.abandon()
            raise
        await self._publish(item, audio.seconds, transcript)

    async def _publish(self, item: UserMessage, seconds: float, transcript: Transcript) -> None:
        """Gives *item* its transcript and tells the client, or tells it the transcription failed."""
        if transcript.text is not None:
            part = item.content[0]
            assert isinstance(part, InputAudio)
            part.transcript = transcript.text
            usage = {"type": "duration", "seconds": round(seconds, 3)}
            event = server_event(
                "conversation.item.input_audio_transcription.completed",
                item_id=item.id,
                content_index=0,
                transcript=transcript.text,
                usage=usage,
            )
        else:
            event = server_event(
                "conversation.item.input_audio_transcription.failed",
                item_id=item.id,
                content_index=0,
                error=transcript.error,
            )
        await self._send(event)
        if self._conversation.get(item.id) is not None:
            before = self._conversation.before(item.id)
            await self._send(server_event("conversation.item.done", previous_item_id=before, item=item))
