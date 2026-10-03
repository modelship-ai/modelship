"""Conversation state for ``/v1/responses`` — one snapshot per response id.

A stored snapshot is **self-contained**: it holds the full conversation as of that
response, so continuing from a ``previous_response_id`` is a single read (O(1))
rather than a walk back down a chain of pointers. Each turn mints a fresh response
id and therefore a fresh key, which is also what makes branching work — two
requests may continue from the same ``previous_response_id`` without colliding.

The cost of that shape: snapshot *N* embeds turns 1..*N*, so a conversation of *n*
turns costs O(n²) total storage. Deliberate — reads happen every turn, and TTL
bounds the total.

Keys are scoped by caller identity (``responses/<identity>/<response_id>``), never
by response id alone: a bare id would let any caller fetch another's conversation
by guessing or replaying one. A read for the wrong identity simply builds a
different key and misses, so isolation needs no comparison logic.

This is the OpenAI-domain layer over the generic ``modelship.state`` store: it takes
a store and never builds one, exactly as ``deploy.ledger`` is the deploy
domain's layer over the same store. The store stays generic and knows nothing about
Responses.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import cast

import ray
from ray.exceptions import RayActorError

from modelship.logging import get_logger
from modelship.state import StateStore
from modelship.utils import head_node_options

logger = get_logger("api")

# State-store namespace; one key per response: "responses/<identity>/<response_id>".
_NAMESPACE = "responses"

# How long a stored conversation lives. Each turn writes a new key with a fresh TTL,
# so an active conversation stays alive while superseded snapshots age out.
_TTL_ENV = "MSHIP_RESPONSES_TTL_S"
_DEFAULT_TTL_S = 30 * 24 * 60 * 60.0  # 30 days, matching OpenAI's retention

# A background response's status once it can no longer change.
TERMINAL_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled"})

# How long a background run's HeartbeatRegistry entry survives without a refresh
# before a poller gives up on it and reports `failed`. Also the registry's own TTL
# (see get_heartbeat_registry) — one threshold, not two to keep in sync.
_STALE_ENV = "MSHIP_RESPONSES_STALE_S"
_DEFAULT_STALE_S = 30.0

# Separate, short-lived namespace for a background+stream run's replayable event log
# ("responses-stream/<identity>/<response_id>"), distinct from the response snapshot.
_STREAM_NAMESPACE = "responses-stream"
_STREAM_TTL_ENV = "MSHIP_RESPONSES_STREAM_BUFFER_TTL_S"
_DEFAULT_STREAM_TTL_S = 600.0


def ttl_seconds() -> float | None:
    """Configured conversation TTL; ``None`` (no expiry) when set to <= 0."""
    raw = os.environ.get(_TTL_ENV)
    if not raw:
        return _DEFAULT_TTL_S
    try:
        ttl = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %ss.", _TTL_ENV, raw, _DEFAULT_TTL_S)
        return _DEFAULT_TTL_S
    return ttl if ttl > 0 else None


def stale_seconds() -> float:
    """Configured heartbeat-staleness threshold for orphan detection (always positive)."""
    raw = os.environ.get(_STALE_ENV)
    if not raw:
        return _DEFAULT_STALE_S
    try:
        threshold = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %ss.", _STALE_ENV, raw, _DEFAULT_STALE_S)
        return _DEFAULT_STALE_S
    return threshold if threshold > 0 else _DEFAULT_STALE_S


class _HeartbeatStore:
    """Plain (non-actor) TTL map of key -> (req_id, deadline), factored out of
    HeartbeatRegistry so the eviction logic is unit-testable without a Ray cluster.
    ``now`` is injectable for deterministic tests. Mirrors ``_DisconnectStore``
    (``infer_config.py``) in shape."""

    def __init__(self, ttl_seconds: float, now: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._now = now
        self._entries: dict[str, tuple[str, float]] = {}

    def heartbeat(self, key: str, req_id: str) -> None:
        now = self._now()
        self._evict_expired(now)
        self._entries[key] = (req_id, now + self._ttl)

    def is_alive(self, key: str) -> bool:
        return self._live_entry(key) is not None

    def req_id(self, key: str) -> str | None:
        entry = self._live_entry(key)
        return entry[0] if entry is not None else None

    def _live_entry(self, key: str) -> tuple[str, float] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry[1] <= self._now():
            del self._entries[key]
            return None
        return entry

    def _evict_expired(self, now: float) -> None:
        for key in [k for k, (_req_id, deadline) in self._entries.items() if deadline <= now]:
            del self._entries[key]


@ray.remote(num_cpus=0)
class HeartbeatRegistry:
    """One cluster-wide actor tracking liveness per background ``/v1/responses`` run,
    keyed the same way the response snapshot is (``identity/response_id``). Separate
    from the response snapshot itself: a heartbeat refresh is then a single atomic
    actor call rather than a read-modify-write of the snapshot, which is what let a
    heartbeat tick race a terminal write (cancel/completion/staleness) and silently
    regress the response back out of its terminal status.

    Also separate from ``DisconnectRegistry``: that actor's ``is_set`` is a sticky
    one-way flag ("cancel requested"), while this one's entries decay and need
    periodic renewal ("still alive") — different enough semantics to not share one
    actor. Entries are TTL-evicted (matching ``stale_seconds()``) rather than
    explicitly cleared on terminal transition, for the same reason
    ``DisconnectRegistry`` doesn't clear on ``stop()``: an explicit clear can race a
    still-in-flight heartbeat tick and resurrect the entry right after."""

    def __init__(self, ttl_seconds: float):
        self._store = _HeartbeatStore(ttl_seconds)
        self._ttl = ttl_seconds
        self._started = time.monotonic()

    async def heartbeat(self, key: str, req_id: str) -> None:
        self._store.heartbeat(key, req_id)

    async def is_alive(self, key: str) -> bool:
        # Younger than the TTL, this registry may have replaced one that held the key.
        return self._store.is_alive(key) or time.monotonic() - self._started < self._ttl

    async def req_id(self, key: str) -> str | None:
        return self._store.req_id(key)


_heartbeat_registry = None


def get_heartbeat_registry():
    """Get-or-create the single detached, named HeartbeatRegistry shared by every
    gateway replica. Cached to keep the lookup off the hot path."""
    global _heartbeat_registry
    if _heartbeat_registry is None:
        _heartbeat_registry = HeartbeatRegistry.options(
            name="modelship_heartbeat_registry",
            get_if_exists=True,
            lifetime="detached",
            namespace="modelship",
            **head_node_options(),
        ).remote(stale_seconds())
    return _heartbeat_registry


def reset_heartbeat_registry() -> None:
    """Drop the cached handle so the next get_heartbeat_registry() re-resolves the named actor."""
    global _heartbeat_registry
    _heartbeat_registry = None


async def _call_registry(method: str, *args):
    try:
        return await getattr(get_heartbeat_registry(), method).remote(*args)
    except RayActorError:
        # The next call resolves (or creates) its replacement.
        reset_heartbeat_registry()
        raise


async def heartbeat(identity: str, response_id: str, req_id: str) -> None:
    """Refresh (or create) the liveness entry for this background run."""
    await _call_registry("heartbeat", _key(identity, response_id), req_id)


async def is_alive(identity: str, response_id: str) -> bool:
    """Whether this background run has refreshed its heartbeat within ``stale_seconds()``."""
    return await _call_registry("is_alive", _key(identity, response_id))


async def req_id_for(identity: str, response_id: str) -> str | None:
    """The request id to signal in ``DisconnectRegistry`` to cancel this run's
    in-flight inference, or ``None`` if it has no live heartbeat entry."""
    return await _call_registry("req_id", _key(identity, response_id))


def _key(identity: str, response_id: str) -> str:
    return f"{_NAMESPACE}/{identity}/{response_id}"


def read(store: StateStore, identity: str, response_id: str) -> dict | None:
    """Return the snapshot for *response_id* under *identity*, or ``None`` if absent.

    ``StateStoreUnavailableError`` propagates: a store outage must surface as a 503,
    never as a 404 that would look like a legitimately unknown id.
    """
    data = store.get(_key(identity, response_id))
    if data is None:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("response"), dict):
        logger.warning("Malformed response snapshot at %r; treating as missing.", _key(identity, response_id))
        return None
    return data


async def read_async(store: StateStore, identity: str, response_id: str) -> dict | None:
    """Async :func:`read` — same contract."""
    data = await store.get_async(_key(identity, response_id))
    if data is None:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("response"), dict):
        logger.warning("Malformed response snapshot at %r; treating as missing.", _key(identity, response_id))
        return None
    return data


async def write_async(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    input_items: list[dict],
) -> None:
    """Persist the snapshot for *response_id*.

    *response* is the full serialized ``ResponseObject`` so ``GET`` can return it
    verbatim; *input_items* is everything that went in (resolved history + this
    turn's input), so the next turn rebuilds by appending this response's output.
    """
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "input_items": input_items},
        ttl_seconds=ttl_seconds(),
    )


async def delete_async(store: StateStore, identity: str, response_id: str) -> None:
    """Drop the snapshot for *response_id*. Idempotent (per the store contract)."""
    await store.delete_async(_key(identity, response_id))


async def write_background(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    input_items: list[dict],
) -> None:
    """Persist the initial ``queued`` placeholder. Liveness (``req_id``, heartbeat)
    lives entirely in :class:`HeartbeatRegistry`, not in this snapshot — see
    :func:`heartbeat`."""
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "input_items": input_items},
        ttl_seconds=ttl_seconds(),
    )


async def write_terminal_if_not_terminal(store: StateStore, identity: str, response_id: str, *, response: dict) -> None:
    """Write *response* as the snapshot's new state, unless it's already gone or
    already terminal. Shared by every writer that could otherwise regress a terminal
    status (cancel, drain failure, staleness check)."""
    snapshot = await read_async(store, identity, response_id)
    if snapshot is None:
        return
    current_status = (snapshot.get("response") or {}).get("status")
    if current_status in TERMINAL_STATUSES:
        return
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "input_items": snapshot.get("input_items") or []},
        ttl_seconds=ttl_seconds(),
    )


def stream_buffer_ttl_seconds() -> float:
    """Configured event-buffer TTL for background+stream resume (always positive)."""
    raw = os.environ.get(_STREAM_TTL_ENV)
    if not raw:
        return _DEFAULT_STREAM_TTL_S
    try:
        ttl = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %ss.", _STREAM_TTL_ENV, raw, _DEFAULT_STREAM_TTL_S)
        return _DEFAULT_STREAM_TTL_S
    return ttl if ttl > 0 else _DEFAULT_STREAM_TTL_S


def _stream_key(identity: str, response_id: str) -> str:
    return f"{_STREAM_NAMESPACE}/{identity}/{response_id}"


async def append_stream_event(store: StateStore, identity: str, response_id: str, event: dict) -> None:
    """Durably buffer one streamed event for later replay (background+stream resume)."""
    await store.append_async(_stream_key(identity, response_id), event, ttl_seconds=stream_buffer_ttl_seconds())


async def read_stream_events_after(
    store: StateStore, identity: str, response_id: str, after_sequence: int
) -> list[dict]:
    """Buffered events after *after_sequence*, in order. Empty once discarded/expired."""
    events = await store.read_from_async(_stream_key(identity, response_id), after_sequence=after_sequence)
    return cast("list[dict]", events)


async def discard_stream_buffer(store: StateStore, identity: str, response_id: str) -> None:
    """Drop the event buffer early (run reached a terminal status) rather than waiting on its TTL."""
    await store.delete_async(_stream_key(identity, response_id))


def history_items(snapshot: dict) -> list[dict]:
    """Rebuild the conversation from a snapshot: everything that went into that turn,
    plus what it produced. This is what a continuation prepends to its own input."""
    items = snapshot.get("input_items")
    output = (snapshot.get("response") or {}).get("output")
    return [
        *(items if isinstance(items, list) else []),
        *(output if isinstance(output, list) else []),
    ]
