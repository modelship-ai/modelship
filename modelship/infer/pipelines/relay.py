"""What crosses Ray between the gateway and a pipeline replica for a `/v1/realtime` session. The gateway relays the
client's WebSocket: its frames go to the replica in order through `realtime_event`, pinned to the replica that took
the session's `realtime` stream, and the stream carries the replica's frames back. Both sides hold the client's
frames in a `FrameQueue`."""

import asyncio
from collections import deque
from dataclasses import dataclass, field

# per session, the client frames a side holds before it stops taking more, so TCP slows the client down
MAX_QUEUED_BYTES = 1024 * 1024
# RFC 6455 internal error: the session or its replica failed
CLOSE_FAILED = 1011


@dataclass(frozen=True)
class RealtimeOpening:
    """A session the gateway opens on a pipeline replica."""

    # routes every call of the session to one replica, and names the session there
    key: str
    # the client's, for the session's model calls
    headers: dict[str, str] = field(default_factory=dict)
    identity: str = ""


@dataclass(frozen=True)
class RealtimeOpened:
    """The stream's first item: the session is open on the replica, and takes the client's frames."""


@dataclass(frozen=True)
class RealtimeClosed:
    """The stream's last item when the session closes the client's WebSocket, with *code*."""

    code: int


# what the stream carries: server events as JSON text, after one RealtimeOpened, until a RealtimeClosed
RealtimeFrame = str | RealtimeOpened | RealtimeClosed

ClientFrame = str | bytes


class FrameQueue:
    """The client's frames waiting to be taken, in order. `put` returns only once at most MAX_QUEUED_BYTES wait, so
    the side putting them stops reading the client and backpressure reaches it. Once closed, the frames still waiting
    can be taken, and then the takers get none."""

    def __init__(self) -> None:
        self._limit = MAX_QUEUED_BYTES
        self._frames: deque[ClientFrame] = deque()
        self._bytes = 0
        self._arrived = asyncio.Event()
        self._room = asyncio.Event()
        self._room.set()
        self.closed = False

    @property
    def empty(self) -> bool:
        return not self._frames

    async def put(self, frames: list[ClientFrame]) -> None:
        """Queues *frames*, then waits until at most the limit waits or the queue is closed."""
        for frame in frames:
            self._frames.append(frame)
            self._bytes += len(frame)
        self._arrived.set()
        while self._bytes > self._limit and not self.closed:
            self._room.clear()
            await self._room.wait()

    def close(self) -> None:
        """No more frames come; waits on the queue end."""
        self.closed = True
        self._arrived.set()
        self._room.set()

    async def get(self) -> ClientFrame | None:
        """The next frame, once one waits; None once the queue is closed and empty."""
        if not await self._wait():
            return None
        frame = self._frames.popleft()
        self._bytes -= len(frame)
        if self._bytes <= self._limit:
            self._room.set()
        return frame

    async def get_all(self) -> list[ClientFrame]:
        """Every frame waiting, once one waits; none once the queue is closed and empty."""
        if not await self._wait():
            return []
        frames = list(self._frames)
        self._frames.clear()
        self._bytes = 0
        self._room.set()
        return frames

    async def _wait(self) -> bool:
        """Waits for a frame; False once the queue is closed and empty."""
        while not self._frames:
            if self.closed:
                return False
            self._arrived.clear()
            await self._arrived.wait()
        return True
