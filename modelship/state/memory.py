"""In-memory StateStore — a dict shared cluster-wide through a detached Ray actor on the head node.

The default backend. Every process and gateway replica in the cluster shares one
``MemoryStoreActor``, so writes from one process (e.g. the deploy driver) are
visible to another (e.g. a re-run of the driver, or a gateway replica) — unlike a
plain process-local dict. It survives the actor being restarted (``max_restarts``)
but NOT the cluster dying, which is what distinguishes it from the durable
``redis://`` backend. Selected by the ``memory://`` URI scheme.
"""

from __future__ import annotations

import copy
import os
import time

import ray
from ray import exceptions as ray_exceptions

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError, normalize_prefix
from modelship.utils import head_node_options
from modelship.utils.runtime_env import MEMORY_STORE_ENV_VARS, build_env_vars

logger = get_logger("startup")

# Detached-actor identity: same namespace as the other cluster-wide coordinators
# (modelship.infer.deploy_coordinator.COORDINATOR_NAMESPACE / replica_coordinator).
# Not imported from there — modelship.state is the generic lower layer and infer
# depends on it, not the reverse.
_ACTOR_NAME = "modelship-memory-store"
_ACTOR_NAMESPACE = "modelship"

# Minimum seconds between expiry sweeps; <= 0 disables sweeping.
_SWEEP_INTERVAL_ENV = "MSHIP_STATE_SWEEP_INTERVAL_S"
_DEFAULT_SWEEP_INTERVAL_S = 300.0


def _sweep_interval_s() -> float:
    raw = os.environ.get(_SWEEP_INTERVAL_ENV)
    if not raw:
        return _DEFAULT_SWEEP_INTERVAL_S
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; falling back to %ss.", _SWEEP_INTERVAL_ENV, raw, _DEFAULT_SWEEP_INTERVAL_S
        )
        return _DEFAULT_SWEEP_INTERVAL_S


@ray.remote(num_cpus=0)
class MemoryStoreActor(StateStore):
    """Holds the dict. One actor for the whole cluster — memory:// targets
    small-traffic single-node deployments, so a single actor is the design point,
    not a stopgap. A restart returns an empty store: this fails safe for every
    caller (the replica coordinator's in-RAM registry is untouched and write-through
    repopulates it; an empty effective config makes the deploy driver's reconcile
    remove nothing rather than remove wrongly; a lost /v1/responses conversation
    surfaces as a 404 on the next previous_response_id)."""

    def __init__(self) -> None:
        # key -> (value, expires_at epoch | None). Expiry is enforced lazily on read
        # and, for keys never read again, by the sweep below.
        self._data: dict[str, tuple[JsonValue, float | None]] = {}
        self._last_sweep = time.time()
        # Read once: this actor's env is fixed for its process lifetime, so
        # re-reading it on every set() would just repeat the same parse.
        self._sweep_interval = _sweep_interval_s()

    def _maybe_sweep(self, now: float) -> None:
        """Drop expired keys, at most once per sweep interval.

        Lazy expiry only reclaims a key someone reads again. A key written once and
        never re-read — the shape of a superseded /v1/responses snapshot — would
        otherwise pin its value for the actor's lifetime. Sweeping on write rather
        than from a background task keeps this class usable as a plain object (no
        event loop, no task to cancel), which is how it is unit-tested.
        """
        interval = self._sweep_interval
        if interval <= 0 or now - self._last_sweep < interval:
            return
        self._last_sweep = now
        expired = [k for k, (_, expires_at) in self._data.items() if expires_at is not None and now >= expires_at]
        for key in expired:
            del self._data[key]
        if expired:
            logger.debug("Swept %d expired state key(s).", len(expired))

    def get(self, key: str) -> JsonValue | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._data.pop(key, None)
            return None
        # Deep-copy on the way out so callers can't mutate stored state in place.
        return copy.deepcopy(value)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        self._data[key] = (copy.deepcopy(value), expires_at)
        self._maybe_sweep(now)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def list(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        now = time.time()
        return [
            k
            for k, (_, expires_at) in list(self._data.items())
            if (not prefix or k == prefix or k.startswith(f"{prefix}/")) and (expires_at is None or now < expires_at)
        ]

    def append(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        # Mutates the actor-local list directly rather than a get+set round trip, so
        # cost per event is O(1) (amortized), not a rewrite of the whole log.
        now = time.time()
        entries, _ = self._data.get(key, ([], None))
        if not isinstance(entries, list):
            entries = []
        entries.append(copy.deepcopy(event))
        if max_len is not None and len(entries) > max_len:
            entries = entries[-max_len:]
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        self._data[key] = (entries, expires_at)
        self._maybe_sweep(now)

    def read_from(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        entry = self._data.get(key)
        if entry is None:
            return []
        entries, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._data.pop(key, None)
            return []
        if not isinstance(entries, list):
            return []
        return copy.deepcopy(
            [e for e in entries if isinstance(e, dict) and e.get("sequence_number", -1) > after_sequence]
        )

    # In-process (this runs as the actor body itself, not over RPC from within):
    # no thread needed, so skip the base's to_thread hop.
    async def get_async(self, key: str) -> JsonValue | None:
        return self.get(key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self.set(key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        self.delete(key)

    async def list_async(self, prefix: str) -> list[str]:
        return self.list(prefix)

    async def append_async(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        self.append(key, event, ttl_seconds=ttl_seconds, max_len=max_len)

    async def read_from_async(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        return self.read_from(key, after_sequence=after_sequence)


def get_or_create_memory_store_actor():
    """Return the cluster-wide memory-store actor handle, creating it on the head node if absent."""
    return MemoryStoreActor.options(
        name=_ACTOR_NAME,
        namespace=_ACTOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        max_restarts=-1,
        runtime_env={"env_vars": build_env_vars(MEMORY_STORE_ENV_VARS)},
        **head_node_options(),
    ).remote()


class MemoryStateStore(StateStore):
    """Client for the cluster-wide MemoryStoreActor. Construction is inert (no Ray
    call) so it can be built before/without a cluster, matching RedisStateStore's
    lazy-client pattern; the actor handle is resolved on first use."""

    def __init__(self) -> None:
        self._handle = None

    def _actor(self):
        if self._handle is None:
            if not ray.is_initialized():
                raise StateStoreUnavailableError(
                    "memory:// requires an initialized Ray cluster (no ray.init() has run in this process)."
                )
            self._handle = get_or_create_memory_store_actor()
        return self._handle

    def _call(self, method: str, *args, **kwargs):
        try:
            return ray.get(getattr(self._actor(), method).remote(*args, **kwargs))
        except ray_exceptions.RayActorError as exc:
            # The actor died and won't come back reachable via this stale handle
            # (max_restarts keeps the same actor id alive, but a fresh
            # lookup re-resolves it) — drop the cache and let the next call
            # re-resolve.
            self._handle = None
            raise StateStoreUnavailableError(f"memory:// store actor unreachable: {exc}") from exc

    async def _acall(self, method: str, *args, **kwargs):
        try:
            return await getattr(self._actor(), method).remote(*args, **kwargs)
        except ray_exceptions.RayActorError as exc:
            self._handle = None
            raise StateStoreUnavailableError(f"memory:// store actor unreachable: {exc}") from exc

    def get(self, key: str) -> JsonValue | None:
        return self._call("get", key)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self._call("set", key, value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> None:
        self._call("delete", key)

    def list(self, prefix: str) -> list[str]:
        return self._call("list", prefix)

    async def get_async(self, key: str) -> JsonValue | None:
        return await self._acall("get", key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await self._acall("set", key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        await self._acall("delete", key)

    async def list_async(self, prefix: str) -> list[str]:
        return await self._acall("list", prefix)

    async def append_async(
        self, key: str, event: JsonValue, *, ttl_seconds: float | None = None, max_len: int | None = None
    ) -> None:
        await self._acall("append_async", key, event, ttl_seconds=ttl_seconds, max_len=max_len)

    async def read_from_async(self, key: str, *, after_sequence: int = -1) -> list[JsonValue]:
        return await self._acall("read_from_async", key, after_sequence=after_sequence)
