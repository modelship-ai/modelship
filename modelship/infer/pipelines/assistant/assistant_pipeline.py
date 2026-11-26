"""The `assistant` pipeline in its replica: each `/v1/realtime` session it holds runs here, over a `RelayChannel`
the gateway feeds with the client's frames and drains of the session's."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

from modelship.infer.base_pipeline import BasePipeline
from modelship.infer.infer_config import get_disconnect_registry
from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.infer.pipelines.assistant.input.turns import NewFinder
from modelship.infer.pipelines.assistant.model_calls import ModelCalls
from modelship.infer.pipelines.assistant.session import refuse, serve_session
from modelship.infer.pipelines.relay import (
    CLOSE_FAILED,
    ClientFrame,
    FrameQueue,
    RealtimeClosed,
    RealtimeFrame,
    RealtimeOpened,
    RealtimeOpening,
)
from modelship.infer.pipelines.routes import ModelRoutes
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import RealtimeError
from modelship.utils.config_schema import AssistantPipelineConfig
from modelship.utils.vad import SileroModel, SpeechStream, silero_vad_source
from modelship.utils.wake import WakeWords

logger = get_logger("realtime")

# how long a session that lost its client has to wind down
_WIND_DOWN_S = 5.0


class RelayChannel:
    """A session's `Channel`: the client's frames as the gateway hands them over, and the session's frames until the
    gateway takes them."""

    def __init__(self) -> None:
        self._inbound = FrameQueue()
        self._outbound: asyncio.Queue[RealtimeFrame] = asyncio.Queue()
        self._closed = False

    async def push(self, frames: list[ClientFrame]) -> None:
        """Queues the client's *frames*, returning once at most MAX_QUEUED_BYTES wait again."""
        await self._inbound.put(frames)

    def leave(self) -> None:
        """The client left: once its queued frames are read, `receive` answers None."""
        self._inbound.close()

    async def receive(self) -> ClientFrame | None:
        return await self._inbound.get()

    async def send(self, text: str) -> None:
        if not self._closed:
            self._outbound.put_nowait(text)

    async def close(self, code: int) -> None:
        if not self._closed:
            self._closed = True
            self._outbound.put_nowait(RealtimeClosed(code))

    async def frames(self, session: asyncio.Future) -> AsyncIterator[RealtimeFrame]:
        """The session's frames until it closes the WebSocket or ends; a session that failed ends them with a
        RealtimeClosed(CLOSE_FAILED)."""
        while True:
            taken = asyncio.ensure_future(self._outbound.get())
            try:
                await asyncio.wait({taken, session}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                got = taken.done() and not taken.cancelled()
                if not got:
                    taken.cancel()
            if got:
                frame = taken.result()
                yield frame
                if isinstance(frame, RealtimeClosed):
                    return
                continue
            while not self._outbound.empty():
                frame = self._outbound.get_nowait()
                yield frame
                if isinstance(frame, RealtimeClosed):
                    return
            if not session.cancelled() and session.exception() is not None:
                yield RealtimeClosed(CLOSE_FAILED)
            return


class AssistantPipeline(BasePipeline):
    config: AssistantPipelineConfig

    def __init__(self, config: AssistantPipelineConfig, routes: ModelRoutes):
        super().__init__(config)
        self.routes = routes
        # Silero, shared by every session's server VAD; None without an stt
        self._silero: SileroModel | None = None
        # the wake words, checked once; None without
        self._wake: WakeWords | None = None
        # session key -> its channel
        self._channels: dict[str, RelayChannel] = {}

    async def start(self) -> None:
        assistant = self.config.assistant_config
        if assistant.stt is not None:
            from modelship.utils.downloads import locked_download

            path = await locked_download(silero_vad_source(), self.config.name)
            self._silero = await asyncio.to_thread(SileroModel, path)
        if assistant.wake is not None:
            self._wake = await asyncio.to_thread(WakeWords, assistant.wake.words)

    def _finder(self) -> NewFinder | None:
        """How a session's server VAD opens its speech finder, which hears the wake words too when asked; None
        without speech detection."""
        model, wake = self._silero, self._wake
        if model is None:
            return None

        def open_finder(rate: int, start_ms: float, woken: bool) -> SpeechStream:
            words = wake.stream(start_ms) if woken and wake is not None else None
            return SpeechStream(model, rate, start_ms, words)

        return open_finder

    async def realtime(self, opening: RealtimeOpening) -> AsyncIterator[RealtimeFrame]:
        channel = RelayChannel()
        self._channels[opening.key] = channel
        session = asyncio.ensure_future(self._serve(channel, opening))
        try:
            yield RealtimeOpened()
            async for frame in channel.frames(session):
                yield frame
        finally:
            self._channels.pop(opening.key, None)
            channel.leave()
            await self._wind_down(session)

    async def _serve(self, channel: RelayChannel, opening: RealtimeOpening) -> None:
        if not await self.routes.ready():
            error = RealtimeError(message=f"Pipeline {self.config.name!r} isn't ready yet.", type="server_error")
            await refuse(channel, error)
            return
        await serve_session(channel, self.config, self._calls(opening), self._finder())

    def _calls(self, opening: RealtimeOpening) -> PipelineCalls:
        """How the session *opening* opens calls the pipeline's models."""
        return ModelCalls(self.routes, opening.headers, get_disconnect_registry(), opening.identity)

    async def _wind_down(self, session: asyncio.Future) -> None:
        """Lets *session* finish once its client left, cancelling it after a while."""
        done, _ = await asyncio.wait({session}, timeout=_WIND_DOWN_S)
        if not done:
            session.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session
        elif not session.cancelled() and (error := session.exception()) is not None:
            logger.error("Realtime session on pipeline %s failed", self.config.name, exc_info=error)

    async def realtime_event(self, key: str, frames: list[ClientFrame]) -> bool:
        channel = self._channels.get(key)
        if channel is None:
            return False
        await channel.push(frames)
        return True

    async def realtime_close(self, key: str) -> None:
        if (channel := self._channels.get(key)) is not None:
            channel.leave()
