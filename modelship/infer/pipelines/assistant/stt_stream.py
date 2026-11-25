"""A turn's audio streamed to a streaming speech-to-text model as it arrives: each push carries at least the model's
chunk, one push is in flight at a time, and the turn's commit finishes the stream. A failed call, or a stream that
holds less audio than was sent to it (opened again, on another replica), fails the turn: the stream is closed and
gets nothing more. A session's calls to the model go one at a time, through its *lane*."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from modelship.infer.pipelines.assistant.audio import InputAudioBuffer
from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, SttStreamOptions, SttStreamText

logger = get_logger("realtime")

# how far a stream's count of the audio it holds may be from the audio sent, by rounding
_AUDIO_SLACK_MS = 1.0

# gets the text a push added to what the stream said before
Delta = Callable[[str], Awaitable[None]]
Push = Callable[[str, str, bytes, int, SttStreamOptions], Awaitable[SttStreamText | ErrorResponse | None]]


class TurnStream:
    """The audio of *buffer* from *start_ms* on, streamed to *model* as stream *key*, opened with *options*; each
    push carries at least *chunk_ms*, and no audio past *bound*, when given. *delta* gets each push's new text, when
    the text only grew."""

    def __init__(
        self,
        calls: PipelineCalls,
        model: str,
        key: str,
        buffer: InputAudioBuffer,
        start_ms: float,
        chunk_ms: int,
        options: SttStreamOptions,
        lane: asyncio.Lock,
        delta: Delta | None = None,
        bound: Callable[[], float] | None = None,
    ):
        assert buffer.rate is not None
        self._calls = calls
        self._model = model
        self.key = key
        self._buffer = buffer
        self.rate = buffer.rate
        self.start_ms = start_ms
        # where the audio sent so far ends, on the session's timeline
        self.sent_ms = start_ms
        self._sent_bytes = 0
        self._chunk_ms = chunk_ms
        self._options = options
        self._lane = lane
        self._delta = delta
        self._bound = bound
        # the text sent as deltas so far
        self._said = ""
        # why the turn can't be transcribed, as an error event's error; None while it can
        self.failure: dict | None = None
        self._push_task: asyncio.Task | None = None
        self._closed = False

    def feed(self) -> None:
        """Pushes the buffer's audio since the last push, once it holds a chunk and no push is in flight."""
        if self._closed or self.failure is not None:
            return
        if self._push_task is not None and not self._push_task.done():
            return
        end = self._buffer.end_ms if self._bound is None else min(self._buffer.end_ms, self._bound())
        if end - self.sent_ms < self._chunk_ms:
            return
        pcm = self._buffer.pcm(self.sent_ms, end)
        if not pcm:
            return
        self.sent_ms = end
        self._push_task = asyncio.ensure_future(self._push(pcm))

    async def _push(self, pcm: bytes) -> None:
        result = await self._call(self._calls.transcribe_push, pcm)
        if result is None:
            await self.close()
            return
        if self._delta is not None and result.text.startswith(self._said) and len(result.text) > len(self._said):
            added, self._said = result.text[len(self._said) :], result.text
            await self._delta(added)
        self.feed()

    async def finish(self, pcm: bytes) -> SttStreamText | dict:
        """Ends the stream with *pcm*, the turn's audio not pushed yet: its final text, or the error it failed with."""
        if self._push_task is not None:
            await asyncio.wait([self._push_task])
        if self.failure is None:
            result = await self._call(self._calls.transcribe_finish, pcm)
            if result is not None:
                # a finished stream is gone from its replica
                self._closed = True
                return result
        await self.close()
        assert self.failure is not None
        return self.failure

    async def close(self) -> None:
        """Drops the stream; a push in flight completes first."""
        if self._closed:
            return
        self._closed = True
        task = self._push_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.wait([task])
        try:
            async with self._lane:
                await self._calls.transcribe_close(self._model, self.key)
        except Exception as e:
            logger.debug("Closing stream %s failed: %s", self.key, e)

    async def _call(self, call: Push, pcm: bytes) -> SttStreamText | None:
        """*call* with *pcm*; its answer, or None once the stream failed."""
        self._sent_bytes += len(pcm)
        try:
            async with self._lane:
                result = await call(self._model, self.key, pcm, self.rate, self._options)
        except Exception as e:
            return self._fail(f"the call failed: {e!r}")
        if isinstance(result, ErrorResponse):
            return self._fail(result.error.message, result.error.model_dump(include={"type", "code", "message"}))
        if result is None:
            return self._fail("nothing serves the model")
        sent_ms = self._sent_bytes / 2 / self.rate * 1000
        if abs(result.audio_ms - sent_ms) > _AUDIO_SLACK_MS:
            return self._fail(f"it holds {result.audio_ms:.0f} ms of the {sent_ms:.0f} ms sent")
        return result

    def _fail(self, cause: str, error: dict | None = None) -> None:
        logger.warning("Streamed transcription %s on %s failed: %s", self.key, self._model, cause)
        self.failure = error or {
            "type": "server_error",
            "code": "server_error",
            "message": f"Model {self._model!r} failed to transcribe the turn as it streamed.",
        }


class Streamed(NamedTuple):
    """A committed turn's stream, and the turn's audio it hasn't been sent yet."""

    stream: TurnStream
    rest: bytes
