"""Redis-backed StateStore — one JSON value per key.

Durable across cluster / head death (the value lives in Redis, not the actor), so
it's the backend for deploy versions and /v1/responses conversations in k8s, where
the same Redis also backs Ray's GCS fault tolerance. Selected by the ``redis://``
(or ``rediss://`` for TLS) URI scheme; the URL carries host/port/db/user/password,
parsed natively by ``from_url`` — so a password may be inlined (``redis://:pw@host``)
or injected by the deployment (e.g. a k8s Secret expanded into the URL). A ``namespace``
puts every key under ``modelship/state/<namespace>/``.

Two clients are held, created lazily: a sync ``redis.Redis`` for the sync methods
and a native ``redis.asyncio.Redis`` for the ``*_async`` ones — so an event-loop
caller gets true async I/O, not a thread hop. TTL uses Redis's native expiry (no
value envelope needed).
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import cast

import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError, WatchError
from redis.exceptions import TimeoutError as RedisTimeoutError

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError, normalize_prefix

logger = get_logger("startup")

_PREFIX = "modelship/state/"
# No glob characters: list() matches keys with SCAN's MATCH pattern.
_NAMESPACE = re.compile(r"[A-Za-z0-9._-]+")


def _in_namespace(key: str, prefix: str) -> bool:
    # SCAN's MATCH glob has no path-segment concept, so "prefix*" also matches a
    # sibling like "responses/u10" under prefix "responses/u1"; re-check the boundary.
    return not prefix or key == prefix or key.startswith(f"{prefix}/")


def _decode(raw: str | bytes | None) -> JsonValue | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("Corrupt JSON in redis; treating as missing.")
        return None


@contextlib.contextmanager
def _mapped(what: str):
    """Map a Redis connectivity failure to StateStoreUnavailableError (an outage, not a
    missing key). Wraps both sync calls and awaited async calls."""
    try:
        yield
    except (RedisConnectionError, RedisTimeoutError) as exc:
        raise StateStoreUnavailableError(what) from exc


class RedisStateStore(StateStore):
    def __init__(self, url: str, namespace: str | None = None) -> None:
        if namespace is not None and not _NAMESPACE.fullmatch(namespace):
            raise ValueError(f"state-store namespace {namespace!r} must be letters, digits, '.', '_' or '-'")
        self._url = url
        self._namespace = namespace
        self._prefix = f"{_PREFIX}{namespace}/" if namespace else _PREFIX
        self._sync_client: redis.Redis | None = None
        self._async_client: aioredis.Redis | None = None

    @property
    def namespace(self) -> str | None:
        return self._namespace

    def _slug(self, key: str) -> str:
        # Collapse the key path into one Redis key under the prefix, so backends stay
        # swappable by URI alone.
        return self._prefix + "/".join(p for p in key.split("/") if p)

    def _unslug(self, keys: list[str], prefix: str) -> list[str]:
        return [k[len(self._prefix) :] for k in keys if _in_namespace(k[len(self._prefix) :], prefix)]

    def _sync(self) -> redis.Redis:
        if self._sync_client is None:
            # decode_responses so get() returns str, not bytes, for json.loads.
            self._sync_client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._sync_client

    def _async(self) -> aioredis.Redis:
        if self._async_client is None:
            self._async_client = aioredis.Redis.from_url(self._url, decode_responses=True)
        return self._async_client

    @staticmethod
    def _px(ttl_seconds: float | None) -> int | None:
        # Native expiry in ms; floor at 1ms so a sub-second TTL never becomes a
        # no-expiry / rejected ``ex=0``.
        return None if ttl_seconds is None else max(1, int(ttl_seconds * 1000))

    def get(self, key: str) -> JsonValue | None:
        with _mapped(f"redis get {key!r}"):
            raw = self._sync().get(self._slug(key))
        return _decode(raw)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        with _mapped(f"redis set {key!r}"):
            self._sync().set(self._slug(key), json.dumps(value), px=self._px(ttl_seconds))

    def delete(self, key: str) -> None:
        with _mapped(f"redis delete {key!r}"):
            self._sync().delete(self._slug(key))

    def list(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        match = self._slug(prefix) + "*"
        with _mapped(f"redis scan {prefix!r}"):
            keys = list(self._sync().scan_iter(match=match))
        return self._unslug(keys, prefix)

    async def get_async(self, key: str) -> JsonValue | None:
        with _mapped(f"redis get {key!r}"):
            raw = await self._async().get(self._slug(key))
        return _decode(raw)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        with _mapped(f"redis set {key!r}"):
            await self._async().set(self._slug(key), json.dumps(value), px=self._px(ttl_seconds))

    async def delete_async(self, key: str) -> None:
        with _mapped(f"redis delete {key!r}"):
            await self._async().delete(self._slug(key))

    async def list_async(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        match = self._slug(prefix) + "*"
        keys = []
        with _mapped(f"redis scan {prefix!r}"):
            async for k in self._async().scan_iter(match=match):
                keys.append(k)
        return self._unslug(keys, prefix)

    async def append_async(
        self, key: str, position: int, value: JsonValue, *, ttl_seconds: float | None = None
    ) -> bool:
        with _mapped(f"redis xadd {key!r}"):
            try:
                await self._async().xadd(self._slug(key), {"data": json.dumps(value)}, id=f"{position}-0")
            except ResponseError as exc:
                if "equal or smaller" in str(exc):
                    return False
                raise
            if ttl_seconds is not None:
                await self._async().pexpire(self._slug(key), cast(int, self._px(ttl_seconds)))
        return True

    async def append_and_set_async(
        self,
        log_key: str,
        position: int,
        entry: JsonValue,
        key: str,
        value: JsonValue,
        *,
        ttl_seconds: float | None = None,
        if_exists: bool = False,
    ) -> bool:
        log, record, px = self._slug(log_key), self._slug(key), self._px(ttl_seconds)
        with _mapped(f"redis xadd {log_key!r} and set {key!r}"):
            async with self._async().pipeline() as pipe:
                while True:
                    try:
                        await pipe.watch(log, record)
                        if if_exists and not await pipe.exists(record):
                            return False
                        top = await pipe.xrevrange(log, count=1)
                        if top and int(cast(str, top[0][0]).split("-")[0]) >= position:
                            return False
                        pipe.multi()
                        pipe.xadd(log, {"data": json.dumps(entry)}, id=f"{position}-0")
                        if px is not None:
                            pipe.pexpire(log, px)
                        pipe.set(record, json.dumps(value), px=px)
                        await pipe.execute()
                        return True
                    except WatchError:
                        continue

    async def read_range_async(self, key: str, *, start: int = 1, end: int | None = None) -> list[JsonValue]:
        with _mapped(f"redis xrange {key!r}"):
            entries = await self._async().xrange(self._slug(key), min=str(start), max="+" if end is None else str(end))
        values = []
        for _entry_id, fields in entries or []:
            decoded = _decode(fields.get("data")) if isinstance(fields, dict) else None
            if decoded is not None:
                values.append(decoded)
        return values

    async def expire_async(self, key: str, ttl_seconds: float) -> None:
        with _mapped(f"redis pexpire {key!r}"):
            await self._async().pexpire(self._slug(key), cast(int, self._px(ttl_seconds)))
