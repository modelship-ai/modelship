"""Generic, pluggable state stores. See base.StateStore.

A store is selected by a connection URI whose **scheme** picks the backend and
whose body carries that backend's connection:

    memory://                     dict shared cluster-wide via a Ray actor (default)
    redis://host:6379/0           one JSON value per key in Redis (rediss:// = TLS)

One arg covers a full Redis connection, and a new backend is one entry in
``_BUILDERS`` with zero new flags. ``get_state_store()`` reads the configured URI
from ``MSHIP_STATE_STORE``.
"""

from __future__ import annotations

import os
import time
from urllib.parse import ParseResult, parse_qs, quote, urlparse, urlsplit, urlunsplit

from modelship.metrics import STATE_STORE_OPERATION_DURATION_SECONDS, STATE_STORE_OPERATIONS_TOTAL
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError
from modelship.state.memory import MemoryStateStore, MemoryStoreActor

__all__ = [
    "JsonValue",
    "MemoryStateStore",
    "MemoryStoreActor",
    "StateStore",
    "StateStoreUnavailableError",
    "get_state_store",
    "reject_inline_password",
    "resolve_state_store_uri",
    "state_store_env_var",
    "state_store_from_uri",
]

# Env carrying the store URI. Default is in-memory: the durable backend (redis)
# is opted into explicitly (the chart sets one for k8s).
_STATE_STORE_ENV = "MSHIP_STATE_STORE"
_DEFAULT_URI = "memory://"

# Kept out of the forwarded URI: each process reads it from its own node's env.
REDIS_PASSWORD_ENV = "MSHIP_REDIS_PASSWORD"
_REDIS_SCHEMES = ("redis", "rediss")


class _InstrumentedStateStore(StateStore):
    """Wraps any backend to record per-op latency + ok/error counts by backend, so
    a slow/failing durable store (which silently breaks self-heal) is visible."""

    def __init__(self, inner: StateStore, backend: str) -> None:
        self._inner = inner
        self._backend = backend

    @property
    def inner(self) -> StateStore:
        """The wrapped backend (the concrete store the URI selected)."""
        return self._inner

    def __getattr__(self, name: str):
        # Fires only on a miss (get/set/delete/inner are defined). Delegate any
        # backend-specific attr to the wrapped store; guard _inner so a lookup
        # before __init__ (e.g. unpickling) raises instead of recursing.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _record(self, op: str, start: float, result: str) -> None:
        # Best-effort: a metrics-agent hiccup must never mask the real op error.
        try:
            STATE_STORE_OPERATION_DURATION_SECONDS.observe(
                time.perf_counter() - start, tags={"backend": self._backend, "op": op}
            )
            STATE_STORE_OPERATIONS_TOTAL.inc(tags={"backend": self._backend, "op": op, "result": result})
        except Exception:
            pass

    def _run(self, op: str, fn):
        start = time.perf_counter()
        result = "ok"
        try:
            return fn()
        except Exception:
            result = "error"
            raise
        finally:
            self._record(op, start, result)

    async def _arun(self, op: str, coro):
        start = time.perf_counter()
        result = "ok"
        try:
            return await coro
        except Exception:
            result = "error"
            raise
        finally:
            self._record(op, start, result)

    def get(self, key: str) -> JsonValue | None:
        return self._run("get", lambda: self._inner.get(key))

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self._run("set", lambda: self._inner.set(key, value, ttl_seconds=ttl_seconds))

    def delete(self, key: str) -> None:
        self._run("delete", lambda: self._inner.delete(key))

    def list(self, prefix: str) -> list[str]:
        return self._run("list", lambda: self._inner.list(prefix))

    async def get_async(self, key: str) -> JsonValue | None:
        return await self._arun("get", self._inner.get_async(key))

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await self._arun("set", self._inner.set_async(key, value, ttl_seconds=ttl_seconds))

    async def delete_async(self, key: str) -> None:
        await self._arun("delete", self._inner.delete_async(key))

    async def list_async(self, prefix: str) -> list[str]:
        return await self._arun("list", self._inner.list_async(prefix))

    async def append_async(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        await self._arun("append", self._inner.append_async(key, event, ttl_seconds=ttl_seconds, max_len=max_len))

    async def read_from_async(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        return await self._arun("read_from", self._inner.read_from_async(key, after_sequence=after_sequence))


def _build_memory(parsed: ParseResult) -> StateStore:
    # No connection body: memory:// always names the one cluster-wide actor. Any
    # host or path (memory://foo, memory:///foo) is rejected rather than silently
    # ignored. Gate on parsed.scheme too: the bare "memory" form (no "://") is
    # parsed as an empty scheme with the word itself sitting in .path, and that
    # form must still be accepted.
    if parsed.scheme and (parsed.netloc or parsed.path):
        raise ValueError(
            f"memory:// URI takes no host/path: {parsed.geturl()!r} parses host {parsed.netloc!r} path {parsed.path!r}."
        )
    return MemoryStateStore()


def _build_redis(parsed: ParseResult) -> StateStore:
    # Hand the whole URL back to redis-py (it parses host/port/db/user/password/TLS).
    from modelship.state.redis import RedisStateStore

    return RedisStateStore(parsed.geturl())


# scheme -> builder. Add a backend here (lazy-importing its client) — no CLI change.
_BUILDERS = {
    "memory": _build_memory,
    "redis": _build_redis,
    "rediss": _build_redis,  # TLS
}


def state_store_from_uri(uri: str) -> StateStore:
    """Construct the StateStore named by *uri* (``scheme://…``)."""
    parsed = urlparse(uri)
    # A bare value like "memory" or "file" (no "://") parses with an empty scheme
    # and the word in .path — treat it as the scheme.
    scheme = parsed.scheme or parsed.path
    builder = _BUILDERS.get(scheme)
    if builder is None:
        raise ValueError(f"unknown state-store scheme {scheme!r}; known: {sorted(_BUILDERS)}")
    return _InstrumentedStateStore(builder(parsed), backend=scheme)


def _with_password(uri: str, password: str) -> str:
    """*uri* with *password* percent-encoded into its netloc; unchanged for non-redis or when
    one is present. Encoding keeps `/`, `#`, `?`, `%` from being re-read as URI syntax."""
    parts = urlsplit(uri)
    if parts.scheme not in _REDIS_SCHEMES or parts.password:
        return uri
    host = parts.netloc.rpartition("@")[2]
    user = parts.username or ""
    return urlunsplit(parts._replace(netloc=f"{user}:{quote(password, safe='')}@{host}"))


def reject_inline_password(uri: str) -> None:
    """Refuse a password written into the URI — netloc or ``?password=`` (redis-py reads
    both), directly or via an env var the URI expands to."""
    parts = urlsplit(os.path.expandvars(uri))
    if parts.password or any(parse_qs(parts.query).get("password", ())):
        raise ValueError(
            f"{_STATE_STORE_ENV} must not contain a password — runtime_env carries this URI to every "
            f"replica as plain-text cluster metadata. Drop it from the URI and set {REDIS_PASSWORD_ENV} "
            "on every node instead."
        )


def state_store_env_var() -> dict[str, str]:
    """``{MSHIP_STATE_STORE: uri}`` to forward. It carries no password: each node applies
    its own MSHIP_REDIS_PASSWORD. Empty when this process has no URI to forward."""
    uri = os.environ.get(_STATE_STORE_ENV)
    return {_STATE_STORE_ENV: uri} if uri else {}


def resolve_state_store_uri() -> str:
    """The URI this process connects with: env vars in it expanded from this node's own
    env, then MSHIP_REDIS_PASSWORD applied if the URI carries none."""
    uri = os.environ.get(_STATE_STORE_ENV) or _DEFAULT_URI
    expanded = os.path.expandvars(uri)
    if "${" in expanded:
        raise ValueError(f"{_STATE_STORE_ENV}={uri!r} references an env var that is not set on this node")
    password = os.environ.get(REDIS_PASSWORD_ENV)
    return _with_password(expanded, password) if password else expanded


def get_state_store() -> StateStore:
    """The configured default StateStore, from ``MSHIP_STATE_STORE`` (default
    ``memory://``)."""
    return state_store_from_uri(resolve_state_store_uri())
