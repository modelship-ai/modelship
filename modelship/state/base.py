"""Generic durable state store.

A pluggable key→value store shared across the codebase. It stays generic: each
caller owns a domain layer over it that holds the key layout and shape — each
gateway's deploy versions (``deploy.ledger``) and the
gateway's ``/v1/responses`` conversations (``openai.state.responses``). Keys are
``/``-separated namespace paths; values are JSON/YAML-serializable (``dict`` or
``list``).

Backends differ in durability, so each caller picks the one its use needs: the
default ``memory://`` backend is cluster-scoped (shared by every process) but dies
with the cluster, while ``redis://`` survives it — required to self-heal the
effective config after cluster loss.

Sync ``get``/``set``/``delete``/``list`` are the primitive each backend must
implement; the ``*_async`` variants default to running the sync method in a thread
(non-blocking on an event loop) and a backend overrides them where a native/direct
path is better (redis: ``redis.asyncio``; memory: no thread at all).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

# JSON/YAML-serializable value. Kept deliberately narrow so every backend (file,
# Ray-actor, ConfigMap, Redis) can store it without custom encoders.
JsonValue = dict | list


def normalize_prefix(prefix: str) -> str:
    """Canonicalize a ``list()`` prefix the same way keys themselves are built (see
    each backend's key-building), so a trailing/leading/doubled slash — e.g.
    ``"responses/u1/"`` — still matches ``"responses/u1/x"`` instead of never
    matching due to a literal ``"//"`` in the comparison."""
    return "/".join(p for p in prefix.split("/") if p)


class StateStoreUnavailableError(Exception):
    """The backend is unreachable/errored — distinct from a key being absent.

    ``get``/``list`` raise this on genuine I/O failure so a caller can tell a real
    outage (surface a 503) apart from a missing key (``None`` / empty). Never raised
    for merely-corrupt stored data, which is logged and treated as missing.
    """


class StateStore(ABC):
    """Key→value store. Keys are ``/``-separated namespace paths."""

    @abstractmethod
    def get(self, key: str) -> JsonValue | None:
        """Return the value for *key*, or ``None`` if absent/expired. Raise
        ``StateStoreUnavailableError`` if the backend can't be reached."""

    @abstractmethod
    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        """Persist *value* under *key*, replacing any existing value. With
        *ttl_seconds* the entry expires after that many seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* if present (no error if absent)."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return the keys under *prefix*. Best-effort on expiry (may transiently
        include just-expired keys — ``get`` is authoritative). Raise
        ``StateStoreUnavailableError`` if the backend can't be reached."""

    # Async variants: default to offloading the sync method to a thread so a caller
    # on an event loop never blocks. Backends override where they can do better.
    async def get_async(self, key: str) -> JsonValue | None:
        return await asyncio.to_thread(self.get, key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await asyncio.to_thread(self.set, key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)

    async def list_async(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(self.list, prefix)

    # Ordered append-only log per key. Abstract with no default: a naive get+set
    # default would read-modify-write the whole log on every event.
    @abstractmethod
    async def append_async(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        """Append *event* to the log at *key* (creating it if absent). *max_len*,
        if given, bounds how many trailing entries are retained."""

    @abstractmethod
    async def read_from_async(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        """Entries at *key* with ``sequence_number`` > *after_sequence*, in order.
        Sequence numbers start at 0, so -1 means "from the start"."""
