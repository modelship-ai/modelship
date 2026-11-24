"""`/v1/realtime` in the gateway: it finds the pipeline the client names, and relays the client's WebSocket to the
pipeline replica that holds the session. Frames that arrive while a hand-over is in flight go together in the next
one, so they keep their order, and the gateway stops reading once MAX_QUEUED_BYTES wait. Experimental."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Protocol

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from modelship.infer.base_pipeline import CLOSE_REFUSED
from modelship.infer.pipelines.relay import (
    CLOSE_FAILED,
    MAX_QUEUED_BYTES,
    RealtimeClosed,
    RealtimeFrame,
    RealtimeOpened,
    RealtimeOpening,
)
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import RealtimeError, error_event
from modelship.utils import random_uuid
from modelship.utils.config_schema import ModelUsecase, parse_pipeline

logger = get_logger("realtime")

# how long a session whose client left has to end before its stream is stopped
_WIND_DOWN_S = 10.0


class PipelineLink(Protocol):
    """A session's calls to its pipeline's app, all routed to one replica."""

    def open(self, opening: RealtimeOpening) -> AsyncIterator[RealtimeFrame]:
        """Opens the session; its frames, while it lasts."""
        ...

    async def send(self, key: str, frames: list[str | bytes]) -> bool:
        """Hands the client's *frames* to session *key*; False when its replica doesn't hold it."""
        ...

    async def leave(self, key: str) -> None:
        """Tells session *key* its client left."""
        ...

    async def stop(self) -> None:
        """Stops the session's stream."""
        ...


def _refusal(name: str | None, pipelines: Mapping[str, dict]) -> RealtimeError | None:
    """Why the client can't open a session on pipeline *name*, or None; *pipelines* is the gateway's table."""
    if not name:
        return RealtimeError(
            message="Pass the pipeline to use as /v1/realtime?model=<name>.", code="missing_model", param="model"
        )
    route = pipelines.get(name)
    if route is None:
        return RealtimeError(message=f"No pipeline {name!r} on this gateway.", code="model_not_found", param="model")
    if parse_pipeline(route["config"]).USECASE is not ModelUsecase.realtime:
        return RealtimeError(
            message=f"Pipeline {name!r} doesn't serve /v1/realtime.", code="model_not_supported", param="model"
        )
    if not route["ready"]:
        return RealtimeError(message=f"Pipeline {name!r} isn't ready yet.", code="model_not_ready")
    return None


async def _refuse(websocket: WebSocket, error: RealtimeError) -> None:
    await websocket.send_text(json.dumps(error_event(error)))
    await websocket.close(code=CLOSE_REFUSED)


async def serve_relay(
    websocket: WebSocket,
    pipelines: Mapping[str, dict],
    link: Callable[[str], PipelineLink | None],
    headers: dict[str, str],
    identity: str,
) -> None:
    """Relays one client's session until it ends. *pipelines* is the gateway's pipeline table, and *link* the way to
    a pipeline's app, None while it can't be reached."""
    await websocket.accept()
    name = websocket.query_params.get("model")
    refusal = _refusal(name, pipelines)
    found = None
    if refusal is None and name is not None:
        found = link(name)
        if found is None:
            refusal = RealtimeError(message=f"Pipeline {name!r} isn't ready yet.", code="model_not_ready")
    if refusal is not None:
        await _refuse(websocket, refusal)
        return
    assert found is not None
    await _Relay(websocket, found, RealtimeOpening(key=random_uuid(), headers=headers, identity=identity)).run()


class _Relay:
    def __init__(self, websocket: WebSocket, link: PipelineLink, opening: RealtimeOpening):
        self.websocket = websocket
        self.link = link
        self.key = opening.key
        self.opening = opening
        # the client's frames not handed over yet
        self._waiting: list[str | bytes] = []
        self._queued = 0
        self._arrived = asyncio.Event()
        self._room = asyncio.Event()
        self._room.set()
        # the client left
        self._read_all = False
        # the session heard that its client left
        self._left = False

    async def run(self) -> None:
        stream = self.link.open(self.opening)
        tasks: list[asyncio.Future] = []
        code = CLOSE_FAILED
        try:
            first = await anext(stream)
            if not isinstance(first, RealtimeOpened):
                raise RuntimeError(f"a session's stream began with {first!r}")
            pump = asyncio.ensure_future(self._pump(stream))
            forwarder = asyncio.ensure_future(self._forward())
            tasks = [pump, forwarder, asyncio.ensure_future(self._read())]
            await asyncio.wait({pump, forwarder}, return_when=asyncio.FIRST_COMPLETED)
            if pump.done():
                code = pump.result()
            elif forwarder.result():
                # the client left, and the session heard it: it ends on its own
                done, _ = await asyncio.wait({pump}, timeout=_WIND_DOWN_S)
                code = pump.result() if done else 1000
        except Exception:
            logger.exception("Realtime session %s: its pipeline replica failed", self.key)
        finally:
            for task in tasks:
                task.cancel()
            if not self._left:
                with contextlib.suppress(Exception):
                    await self._leave()
            with contextlib.suppress(Exception):
                await self.link.stop()
            with contextlib.suppress(Exception):
                await self.websocket.close(code=code)

    async def _pump(self, stream: AsyncIterator[RealtimeFrame]) -> int:
        """Sends the session's frames to the client until the session ends; the close code it ended with."""
        try:
            async for frame in stream:
                if isinstance(frame, RealtimeClosed):
                    return frame.code
                if isinstance(frame, str):
                    try:
                        await self.websocket.send_text(frame)
                    except (WebSocketDisconnect, RuntimeError, OSError):
                        # the client left
                        return 1000
            return 1000
        except Exception:
            logger.exception("Realtime session %s: its pipeline replica failed", self.key)
            return CLOSE_FAILED

    async def _leave(self) -> None:
        self._left = True
        await self.link.leave(self.key)

    async def _read(self) -> None:
        """Takes the client's frames as they arrive, until MAX_QUEUED_BYTES wait to be handed over."""
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                frame = message.get("text")
                if frame is None:
                    frame = message.get("bytes") or b""
                self._waiting.append(frame)
                self._queued += len(frame)
                self._arrived.set()
                while self._queued > MAX_QUEUED_BYTES:
                    self._room.clear()
                    await self._room.wait()
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            self._read_all = True
            self._arrived.set()

    async def _forward(self) -> bool:
        """Hands the waiting frames to the session, all of them in one call, each time the last call returns. True
        once the client left and the session heard it; False when the session's replica no longer holds it."""
        while True:
            await self._arrived.wait()
            self._arrived.clear()
            frames, self._waiting = self._waiting, []
            self._queued = 0
            self._room.set()
            try:
                held = not frames or await self.link.send(self.key, frames)
            except Exception:
                logger.warning("Realtime session %s: its pipeline replica failed", self.key, exc_info=True)
                return False
            if not held:
                logger.warning("Realtime session %s: its pipeline replica doesn't hold it any more", self.key)
                return False
            if self._read_all and not self._waiting:
                await self._leave()
                return True
