"""A turn's audio streamed to a streaming speech-to-text model as it arrives: each push carries at least the model's
chunk, one push is in flight at a time, and the turn's commit finishes the stream. A failed call, or a stream that
holds less audio than was sent to it (opened again, on another replica), fails the turn: the stream is closed and
gets nothing more."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, SttStreamOptions, SttStreamText

logger = get_logger("realtime")

# how far a stream's count of the audio it holds may be from the audio sent, by rounding
_AUDIO_SLACK_MS = 1.0

# gets the text a push added to what the stream said before
Delta = Callable[[str], Awaitable[None]]
Push = Callable[[str, str, bytes, int, SttStreamOptions], Awaitable[SttStreamText | ErrorResponse | None]]


@dataclass(frozen=True)
class StreamTarget:
    """Where a session's streams go: its streaming speech-to-text *model*, through *calls*. Each push carries at
    least *chunk_ms*, and the session's calls to the model go one at a time through *lane*, since Serve routes no
    second call of a request while one is in flight."""

    calls: PipelineCalls
    model: str
    chunk_ms: int
    lane: asyncio.Lock


class TurnStream:
    """The audio of *buffer* from *start_ms* on, streamed to *target* as stream *key*, opened with *options*. No push
    carries audio past *bound*, when given. *delta* gets each push's new text, when the text only grew."""

    def __init__(
        self,
        target: StreamTarget,
        key: str,
        buffer: InputAudioBuffer,
        start_ms: float,
        options: SttStreamOptions,
        delta: Delta | None = None,
        bound: Callable[[], float] | None = None,
    ):
        assert buffer.rate is not None
        self._target = target
        self.key = key
        self._buffer = buffer
        self.rate = buffer.rate
        self.start_ms = start_ms
        # where the audio sent so far ends, on the session's timeline
        self.sent_ms = start_ms
        self._sent_bytes = 0
        self._options = options
        self._delta = delta
        self._bound = bound
        # the text sent as deltas so far
        self._said = ""
        # why the turn can't be transcribed, as an error event's error; None while it can
        self.failure: dict | None = None
        self._push_task: asyncio.Task | None = None
        self._closed = False

    def push_ready(self) -> None:
        """Pushes the buffer's audio since the last push, once it holds a chunk and no push is in flight."""
        if self._closed or self.failure is not None:
            return
        if self._push_task is not None and not self._push_task.done():
            return
        end = self._buffer.end_ms if self._bound is None else min(self._buffer.end_ms, self._bound())
        if end - self.sent_ms < self._target.chunk_ms:
            return
        pcm = self._buffer.pcm(self.sent_ms, end)
        if not pcm:
            return
        self.sent_ms = end
        self._push_task = asyncio.ensure_future(self._push(pcm))

    async def _push(self, pcm: bytes) -> None:
        result = await self._call(self._target.calls.transcribe_push, pcm)
        if result is None:
            await self.close()
            return
        if self._delta is not None and result.text.startswith(self._said) and len(result.text) > len(self._said):
            added, self._said = result.text[len(self._said) :], result.text
            await self._delta(added)
        self.push_ready()

    async def finish(self, pcm: bytes) -> SttStreamText | dict:
        """Ends the stream with *pcm*, the turn's audio not pushed yet: its final text, or the error it failed with."""
        if self._push_task is not None:
            await asyncio.wait([self._push_task])
        if self.failure is None:
            result = await self._call(self._target.calls.transcribe_finish, pcm)
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
            async with self._target.lane:
                await self._target.calls.transcribe_close(self._target.model, self.key)
        except Exception as e:
            logger.debug("Closing stream %s failed: %s", self.key, e)

    async def _call(self, call: Push, pcm: bytes) -> SttStreamText | None:
        """*call* with *pcm*; its answer, or None once the stream failed."""
        self._sent_bytes += len(pcm)
        try:
            async with self._target.lane:
                result = await call(self._target.model, self.key, pcm, self.rate, self._options)
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
        logger.warning("Streamed transcription %s on %s failed: %s", self.key, self._target.model, cause)
        self.failure = error or {
            "type": "server_error",
            "code": "server_error",
            "message": f"Model {self._target.model!r} failed to transcribe the turn as it streamed.",
        }
