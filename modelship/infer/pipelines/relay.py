"""What crosses Ray between the gateway and a pipeline replica for a `/v1/realtime` session. The gateway relays the
client's WebSocket: its frames go to the replica in order through `realtime_event`, pinned to the replica that took
the session's `realtime` stream, and the stream carries the replica's frames back."""

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
