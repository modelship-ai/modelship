"""A head-pinned actor granting one download lease per key; an expired lease's
leftovers are removed on its holder's node before the key frees."""

import asyncio
import time
from typing import NamedTuple

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from modelship.deploy.actor_options import build_cache_env_vars
from modelship.infer.sources import RemoteSource, remove_leftovers
from modelship.logging import configure_logging, get_logger
from modelship.utils.cache import reject_unset_cache_roots

logger = get_logger("download_leases")

LEASES_ACTOR_NAME = "modelship-download-leases"
LEASES_NAMESPACE = "modelship"

LEASE_SECONDS = 30.0
RENEW_SECONDS = 10.0
IDLE_EXIT_SECONDS = 60.0
_REAP_INTERVAL_SECONDS = 1.0


class _Lease(NamedTuple):
    holder: str
    node_id: str
    source: RemoteSource
    expires_at: float
    cleaning: bool = False


@ray.remote(num_cpus=0, max_retries=0)
def _remove_leftovers(source: RemoteSource) -> int:
    reject_unset_cache_roots()
    return remove_leftovers(source)


def remove_leftovers_on(node_id: str, source: RemoteSource) -> ray.ObjectRef:
    """Runs on `node_id` only, with a replica's cache env; fails fast if that node is gone."""
    return _remove_leftovers.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        runtime_env={"env_vars": build_cache_env_vars()},
    ).remote(source)


@ray.remote(num_cpus=0)
class DownloadLeases:
    """One lease per key. A lease unrenewed for `LEASE_SECONDS` is reaped: the key
    stays taken until its holder's leftovers are removed or its node is gone."""

    def __init__(self):
        # nothing else configures logging in this process
        configure_logging()
        self._leases: dict[str, _Lease] = {}
        now = time.monotonic()
        # holders of a crashed predecessor stop within one lease period
        self._grants_from = now + LEASE_SECONDS
        self._last_call = now
        self._closing = False
        self._cleanups: set[asyncio.Task] = set()
        self._reaper = asyncio.create_task(self._reap_forever())

    async def acquire(self, key: str, source: RemoteSource, holder: str, node_id: str) -> str | None:
        """None when granted, else what holds `key` up."""
        now = time.monotonic()
        self._last_call = now
        if self._closing:
            return "lease service shutting down"
        if now < self._grants_from:
            return "lease service starting"
        lease = self._leases.get(key)
        if lease is not None:
            return f"cleaning up after {lease.holder}" if lease.cleaning else f"held by {lease.holder}"
        self._leases[key] = _Lease(holder, node_id, source, now + LEASE_SECONDS)
        return None

    async def renew(self, key: str, holder: str) -> bool:
        self._last_call = time.monotonic()
        lease = self._leases.get(key)
        if lease is None or lease.cleaning or lease.holder != holder:
            return False
        self._leases[key] = lease._replace(expires_at=self._last_call + LEASE_SECONDS)
        return True

    async def release(self, key: str, holder: str) -> None:
        self._last_call = time.monotonic()
        lease = self._leases.get(key)
        if lease is not None and not lease.cleaning and lease.holder == holder:
            del self._leases[key]

    def _idle(self, now: float) -> bool:
        return not self._leases and now - self._last_call >= IDLE_EXIT_SECONDS

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_SECONDS)
            self._reap(time.monotonic())

    def _reap(self, now: float) -> None:
        for key, lease in list(self._leases.items()):
            if lease.cleaning or lease.expires_at > now:
                continue
            logger.warning(
                "Download lease %s expired without release (holder %s); removing its leftovers", key, lease.holder
            )
            self._leases[key] = lease._replace(cleaning=True)
            task = asyncio.create_task(self._clean(key, lease))
            self._cleanups.add(task)
            task.add_done_callback(self._cleanups.discard)
        if not self._closing and self._idle(now):
            # no await between check and flag, so no grant lands in between
            self._closing = True
            # exit_actor can leave an asyncio actor alive but unresponsive; a kill can't
            ray.kill(ray.get_runtime_context().current_actor, no_restart=True)

    async def _clean(self, key: str, lease: _Lease) -> None:
        try:
            removed = await remove_leftovers_on(lease.node_id, lease.source)
            logger.info("Removed %d leftover download file(s) for %s", removed, key)
        except Exception as e:
            # includes TaskUnschedulableError: the holder's node is gone
            logger.warning("Could not remove leftovers for %s, freeing it anyway: %r", key, e)
        finally:
            del self._leases[key]
            self._last_call = time.monotonic()


def get_or_create_leases():
    """The cluster-wide lease actor, created on the head node if absent."""
    return DownloadLeases.options(
        name=LEASES_ACTOR_NAME,
        namespace=LEASES_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        # a restart would trust an empty table; a fresh actor waits out a lease period instead
        max_restarts=0,
        resources={"node:__internal_head__": 0.001},
        # unset, a caller's placement group captures it
        scheduling_strategy="DEFAULT",
    ).remote()
