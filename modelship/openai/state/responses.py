"""Conversation state for ``/v1/responses``.

Each response has a record (``responses/<identity>/<response_id>``): the response as
``GET`` returns it, minus its ``output``. A turn's items, its new input and its output,
are stored once, as the entry at the turn's depth in an append-only log
(``responses-log/<identity>/<log>``) that the record names with ``log`` and ``depth``.
A continuation appends after its parent's entry; when another turn already holds that
position, it starts a log of its own whose first entry names the parent as ``base``.

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

from modelship.logging import get_logger
from modelship.openai.protocol.base import random_uuid
from modelship.state import StateStore
from modelship.utils import head_node_options

logger = get_logger("api")

# State-store namespace; one key per response: "responses/<identity>/<response_id>".
_NAMESPACE = "responses"

# How long a stored conversation lives.
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
    return max(ttl, 24 * 60 * 60.0) if ttl > 0 else None


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
    keyed the same way the response record is (``identity/response_id``). Separate
    from the response record itself: a heartbeat refresh is then a single atomic
    actor call rather than a read-modify-write of the record, which is what let a
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
        # Younger than the TTL, this registry may have restarted since the key's last heartbeat.
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
            # Restarts with the same actor id, so handles cached across a head restart keep working.
            max_restarts=-1,
            **head_node_options(),
        ).remote(stale_seconds())
    return _heartbeat_registry


async def heartbeat(identity: str, response_id: str, req_id: str) -> None:
    """Refresh (or create) the liveness entry for this background run."""
    await get_heartbeat_registry().heartbeat.remote(_key(identity, response_id), req_id)


async def is_alive(identity: str, response_id: str) -> bool:
    """Whether this background run has refreshed its heartbeat within ``stale_seconds()``."""
    return await get_heartbeat_registry().is_alive.remote(_key(identity, response_id))


async def req_id_for(identity: str, response_id: str) -> str | None:
    """The request id to signal in ``DisconnectRegistry`` to cancel this run's
    in-flight inference, or ``None`` if it has no live heartbeat entry."""
    return await get_heartbeat_registry().req_id.remote(_key(identity, response_id))


def _key(identity: str, response_id: str) -> str:
    return f"{_NAMESPACE}/{identity}/{response_id}"


def _log_key(identity: str, log: str) -> str:
    return f"responses-log/{identity}/{log}"


def _is_position(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("log"), str) and isinstance(value.get("depth"), int)


def _is_entry(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("input"), list) and isinstance(value.get("output"), list)


async def _read_record(store: StateStore, identity: str, response_id: str) -> dict | None:
    record = await store.get_async(_key(identity, response_id))
    if record is None:
        return None
    if not (
        isinstance(record, dict)
        and isinstance(record.get("response"), dict)
        and (
            _is_position(record)
            or (
                isinstance(record.get("input"), list)
                and (record.get("parent") is None or _is_position(record["parent"]))
            )
        )
    ):
        logger.warning("Malformed response record at %r; treating as missing.", _key(identity, response_id))
        return None
    return record


async def read_async(store: StateStore, identity: str, response_id: str, *, with_output: bool = True) -> dict | None:
    """The record for *response_id* (``None`` if absent), its response's ``output`` read from the log."""
    record = await _read_record(store, identity, response_id)
    if record is None or not with_output or "log" not in record:
        return record
    entries = await store.read_range_async(
        _log_key(identity, record["log"]), start=record["depth"], end=record["depth"]
    )
    if not entries or not _is_entry(entries[0]):
        return None
    return {**record, "response": {**record["response"], "output": cast(dict, entries[0])["output"]}}


async def _refresh(store: StateStore, identity: str, log: str) -> None:
    ttl = ttl_seconds()
    if ttl is not None:
        await store.expire_async(_log_key(identity, log), ttl)


async def _entries(store: StateStore, identity: str, log: str, depth: int) -> list[dict] | None:
    segments: list[list[dict]] = []
    while True:
        entries = await store.read_range_async(_log_key(identity, log), end=depth)
        if not entries or not all(_is_entry(entry) for entry in entries):
            return None
        entries = cast("list[dict]", entries)
        base = entries[0].get("base")
        if base is not None and not _is_position(base):
            return None
        if len(entries) != depth - (base["depth"] if base else 0):
            return None
        segments.append(entries)
        if base is None:
            return [entry for segment in reversed(segments) for entry in segment]
        log, depth = base["log"], base["depth"]
        await _refresh(store, identity, log)


def _items(entries: list[dict]) -> list[dict]:
    return [item for entry in entries for item in (*entry["input"], *entry["output"])]


async def history_async(store: StateStore, identity: str, record: dict) -> tuple[list[dict], dict] | None:
    await _refresh(store, identity, record["log"])
    entries = await _entries(store, identity, record["log"], record["depth"])
    if entries is None:
        return None
    return _items(entries), {"log": record["log"], "depth": record["depth"]}


async def input_items_async(store: StateStore, identity: str, record: dict) -> list[dict] | None:
    """The items that went into *record*'s turn, or ``None`` if part of them is gone."""
    if "log" in record:
        entries = await _entries(store, identity, record["log"], record["depth"])
        return None if entries is None else [*_items(entries[:-1]), *entries[-1]["input"]]
    parent = record["parent"]
    if parent is None:
        return record["input"]
    entries = await _entries(store, identity, parent["log"], parent["depth"])
    return None if entries is None else [*_items(entries), *record["input"]]


def _with_ids(items: list[dict]) -> list[dict]:
    prefixes = {
        "message": "msg",
        "function_call": "fc",
        "function_call_output": "fco",
        "reasoning": "rs",
        "mcp_call": "mcp",
        "mcp_list_tools": "mcpl",
        "mcp_approval_request": "mcpr",
        "mcp_approval_response": "mcpr",
        "compaction": "cmp",
    }
    with_ids = []
    for item in items:
        prefix = prefixes.get(item.get("type", "message"))
        with_ids.append({**item, "id": f"{prefix}_{random_uuid()}"} if prefix and not item.get("id") else item)
    return with_ids


async def _store_turn(
    store: StateStore,
    identity: str,
    response_id: str,
    response: dict,
    parent: dict | None,
    input_items: list[dict],
    *,
    if_exists: bool = False,
) -> None:
    ttl = ttl_seconds()
    key = _key(identity, response_id)
    entry = {"response_id": response_id, "input": input_items, "output": response.get("output") or []}
    record = {"response": {**response, "output": []}}
    if parent is None:
        await store.append_and_set_async(
            _log_key(identity, response_id),
            1,
            entry,
            key,
            {**record, "log": response_id, "depth": 1},
            ttl_seconds=ttl,
            if_exists=if_exists,
        )
        return
    depth = parent["depth"] + 1
    parent_log = _log_key(identity, parent["log"])
    if await store.append_and_set_async(
        parent_log,
        depth,
        entry,
        key,
        {**record, "log": parent["log"], "depth": depth},
        ttl_seconds=ttl,
        if_exists=if_exists,
    ):
        return
    taken = await store.read_range_async(parent_log, start=depth, end=depth)
    if taken and isinstance(taken[0], dict) and taken[0].get("response_id") == response_id:
        return
    await store.append_and_set_async(
        _log_key(identity, response_id),
        depth,
        {**entry, "base": parent},
        key,
        {**record, "log": response_id, "depth": depth},
        ttl_seconds=ttl,
        if_exists=if_exists,
    )


async def write_async(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    parent: dict | None,
    input_items: list[dict],
) -> None:
    """Persist a finished turn: its items as the log entry after *parent*, then its record."""
    await _store_turn(store, identity, response_id, response, parent, _with_ids(input_items))


async def delete_async(store: StateStore, identity: str, response_id: str) -> None:
    """Drop the record for *response_id*. Idempotent (per the store contract)."""
    await store.delete_async(_key(identity, response_id))


async def write_background(
    store: StateStore,
    identity: str,
    response_id: str,
    *,
    response: dict,
    parent: dict | None,
    input_items: list[dict],
) -> None:
    """Persist the initial ``queued`` placeholder. Liveness (``req_id``, heartbeat)
    lives entirely in :class:`HeartbeatRegistry`, not in this record — see
    :func:`heartbeat`."""
    await store.set_async(
        _key(identity, response_id),
        {"response": response, "parent": parent, "input": _with_ids(input_items)},
        ttl_seconds=ttl_seconds(),
    )


async def write_terminal_if_not_terminal(store: StateStore, identity: str, response_id: str, *, response: dict) -> None:
    """Write *response* as the record's new state, unless it's already gone or
    already terminal. Shared by every writer that could otherwise regress a terminal
    status (cancel, drain failure, staleness check)."""
    record = await _read_record(store, identity, response_id)
    if record is None or "log" in record or record["response"].get("status") in TERMINAL_STATUSES:
        return
    await _store_turn(store, identity, response_id, response, record["parent"], record["input"], if_exists=True)


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
    sequence_number = event.get("sequence_number")
    if isinstance(sequence_number, int):
        await store.append_async(
            _stream_key(identity, response_id), sequence_number + 1, event, ttl_seconds=stream_buffer_ttl_seconds()
        )


async def read_stream_events_after(
    store: StateStore, identity: str, response_id: str, after_sequence: int
) -> list[dict]:
    """Buffered events after *after_sequence*, in order. Empty once discarded/expired."""
    events = await store.read_range_async(_stream_key(identity, response_id), start=after_sequence + 2)
    return cast("list[dict]", events)


async def discard_stream_buffer(store: StateStore, identity: str, response_id: str) -> None:
    """Drop the event buffer early (run reached a terminal status) rather than waiting on its TTL."""
    await store.delete_async(_stream_key(identity, response_id))
