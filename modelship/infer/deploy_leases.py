"""A head-pinned actor granting one deploy lease per node, so replicas sharing a
node load their models one at a time."""

import asyncio
import contextlib
import os
import threading
import time
from typing import NamedTuple

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.logging import configure_logging, get_logger
from modelship.utils import head_node_options, random_uuid
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("deploy_leases")

LEASES_ACTOR_NAME = "modelship-deploy-leases"
LEASES_NAMESPACE = "modelship"

LEASE_SECONDS = 30.0
RENEW_SECONDS = 10.0
POLL_SECONDS = 2.0
_REAP_INTERVAL_SECONDS = 1.0
_RPC_TIMEOUT_SECONDS = 5.0
_WAIT_LOG_SECONDS = 60.0
# an unreachable actor's name resolves for ~2s after it stops answering
_MAX_LOOKUPS = 3


class DeployLeaseError(Exception):
    """The lease service could not be reached."""


class _Lease(NamedTuple):
    holder: str
    expires_at: float


@ray.remote(num_cpus=0)
class DeployLeases:
    """One lease per node id. A lease unrenewed for `LEASE_SECONDS` is freed, so a
    holder that died mid-load doesn't hold its node shut."""

    def __init__(self):
        # nothing else configures logging in this process
        configure_logging()
        self._leases: dict[str, _Lease] = {}
        # holders of a crashed predecessor stop within one lease period
        self._grants_from = time.monotonic() + LEASE_SECONDS
        self._reaper = asyncio.create_task(self._reap_forever())

    async def acquire(self, key: str, holder: str) -> str | None:
        """None when granted, else what holds `key` up."""
        now = time.monotonic()
        if now < self._grants_from:
            return "lease service starting"
        lease = self._leases.get(key)
        if lease is not None:
            return f"held by {lease.holder}"
        self._leases[key] = _Lease(holder, now + LEASE_SECONDS)
        return None

    async def renew(self, key: str, holder: str) -> bool:
        lease = self._leases.get(key)
        if lease is None or lease.holder != holder:
            return False
        self._leases[key] = lease._replace(expires_at=time.monotonic() + LEASE_SECONDS)
        return True

    async def release(self, key: str, holder: str) -> None:
        lease = self._leases.get(key)
        if lease is not None and lease.holder == holder:
            del self._leases[key]

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_SECONDS)
            self._reap(time.monotonic())

    def _reap(self, now: float) -> None:
        for key, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                logger.warning("Deploy lease %s expired without release (holder %s)", key, lease.holder)
                del self._leases[key]


def get_or_create_leases():
    """The cluster-wide deploy-lease actor, created on the head node if absent."""
    return DeployLeases.options(
        name=LEASES_ACTOR_NAME,
        namespace=LEASES_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        # a restart would trust an empty table; a fresh actor waits out a lease period instead
        max_restarts=0,
        runtime_env={"env_vars": build_env_vars(COMMON_ENV_VARS)},
        **head_node_options(),
    ).remote()


@contextlib.asynccontextmanager
async def deploy_lease(model_name: str):
    """Holds this node's deploy lease for the block."""
    key = ray.get_runtime_context().get_node_id()
    holder = f"{model_name}/{os.getpid()}/{random_uuid()[:8]}"
    leases = await _acquire(key, holder, model_name)
    stop = threading.Event()
    # a thread, not a task: the loader's constructor blocks this replica's event loop
    renewer = threading.Thread(target=_renew_until, args=(leases, key, holder, model_name, stop), daemon=True)
    renewer.start()
    try:
        yield
    finally:
        stop.set()
        # an unreleased lease expires on its own
        with contextlib.suppress(Exception):
            await asyncio.wait_for(leases.release.remote(key, holder), _RPC_TIMEOUT_SECONDS)


async def _acquire(key: str, holder: str, model_name: str):
    """Polls until granted, without limit; an unreachable lease service is retryable."""
    leases = get_or_create_leases()
    lookups = 1
    logged: tuple[str, float] | None = None
    while True:
        try:
            blocker = await leases.acquire.remote(key, holder)
        except (ActorDiedError, ActorUnavailableError) as e:
            if lookups >= _MAX_LOOKUPS:
                raise DeployLeaseError(f"deploy lease service unreachable: {e}") from e
            await asyncio.sleep(POLL_SECONDS)
            leases = get_or_create_leases()
            lookups += 1
            continue
        if blocker is None:
            return leases
        now = time.monotonic()
        if logged is None or blocker != logged[0] or now - logged[1] >= _WAIT_LOG_SECONDS:
            logger.info("%s: waiting to load on this node (%s)", model_name, blocker)
            logged = (blocker, now)
        await asyncio.sleep(POLL_SECONDS)


def _renew_until(leases, key: str, holder: str, model_name: str, stop: threading.Event) -> None:
    """Keeps the lease alive while the model loads, and gives up once it is lost.

    The load itself continues: it cannot be cancelled, and killing the replica
    would not stop whoever took the lease next — it would only repeat the load."""
    while not stop.wait(RENEW_SECONDS):
        try:
            if ray.get(leases.renew.remote(key, holder), timeout=_RPC_TIMEOUT_SECONDS):
                continue
            reason = "renewal refused"
        except Exception as e:
            reason = repr(e)
        # a release the block already sent is what refuses the renewal, not a loss
        if not stop.is_set():
            logger.warning("%s: lost the deploy lease on this node (%s); loading anyway", model_name, reason)
        return
