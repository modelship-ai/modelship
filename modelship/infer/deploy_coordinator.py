"""Cluster-wide bookkeeping for model deploys.

`DeployCoordinator` is a detached, named Ray actor on the head node, created by
the first caller and looked up by name afterwards. It holds what no single driver
or replica can:

- one deploy lease per node, held by a replica for the duration of its load, so
  loads on one node run one at a time (the holder's side is `deploy_leases.py`);
- one deploy lease per gateway, held by whatever plans, submits or deletes the
  gateway's apps; the gateway's effective config is written only for its holder;
- a per-deployment backend-death count, retiring a deployment that keeps dying;
- fatal init errors, reported by a replica and read back by the driver, which
  is how a permanently-broken model is told apart from a transient failure.
"""

import asyncio
import time
from typing import NamedTuple

import ray

from modelship.logging import configure_logging, get_logger
from modelship.state import get_state_store, state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.config_schema import parse_deployment_name
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

LEASE_SECONDS = 30.0
RENEW_SECONDS = 10.0
POLL_SECONDS = 2.0
_REAP_INTERVAL_SECONDS = 1.0
_DEATHS_PER_REPLICA = 3


class _Lease(NamedTuple):
    holder: str
    expires_at: float


def gateway_lease_key(gateway_name: str) -> str:
    return f"gateway/{gateway_name}"


@ray.remote(num_cpus=0)
class DeployCoordinator:
    """Cluster-wide deploy bookkeeping: per-node and per-gateway deploy leases, replica-death
    counts and fatal errors. A lease unrenewed for `LEASE_SECONDS` is freed, so a holder
    that died doesn't hold its node or gateway shut."""

    def __init__(self, startup_window: bool = True):
        # nothing else configures logging in this process
        configure_logging()
        self._store = get_state_store()
        self._fatal_errors: dict[str, str] = {}
        self._deaths: dict[str, int] = {}
        self._leases: dict[str, _Lease] = {}
        # holders of a crashed predecessor stop within one lease period
        self._grants_from = time.monotonic() + (LEASE_SECONDS if startup_window else 0.0)
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

    async def write_effective(self, gateway_name: str, holder: str, raw_models: list[dict]) -> bool:
        """Writes the gateway's effective config if `holder` holds the gateway's lease, renewing
        it first so it can't expire mid-write; False, writing nothing, otherwise."""
        from modelship.deploy.effective_config import write_effective

        if not await self.renew(gateway_lease_key(gateway_name), holder):
            return False
        await asyncio.to_thread(write_effective, self._store, gateway_name, raw_models)
        return True

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_SECONDS)
            self._reap(time.monotonic())

    def _reap(self, now: float) -> None:
        for key, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                logger.warning("Deploy lease %s expired without release (holder %s)", key, lease.holder)
                del self._leases[key]

    async def report_replica_death(self, deployment_name: str, replica_ceiling: int, reason: str) -> None:
        """Count one backend death against `deployment_name`, retiring it past
        `_DEATHS_PER_REPLICA * replica_ceiling`. The count is never reset by time,
        only by a redeploy — the key carries the config fingerprint. The reporting
        replica exits whatever this returns."""
        deaths = self._deaths.get(deployment_name, 0) + 1
        self._deaths[deployment_name] = deaths
        limit = _DEATHS_PER_REPLICA * max(replica_ceiling, 1)
        if deaths < limit:
            logger.warning("Replica death %d of %d for %s: %s", deaths, limit, deployment_name, reason)
            return
        logger.error("Retiring %s after %d replica death(s); last: %s", deployment_name, deaths, reason)
        self._deaths.pop(deployment_name, None)
        await self._retire(deployment_name)

    async def _retire(self, deployment_name: str) -> None:
        """Deletes the app under its gateway's lease, waiting for the lease. serve.delete blocks on
        the app's teardown, so it runs off this actor's event loop."""
        from modelship.deploy.removal import delete_apps_quietly

        parsed = parse_deployment_name(deployment_name)
        assert parsed is not None  # replicas report their own app, named by deployment_name
        key, holder = gateway_lease_key(parsed[0]), f"deploy coordinator retiring {deployment_name}"
        while await self.acquire(key, holder) is not None:
            await asyncio.sleep(POLL_SECONDS)
        renewing = asyncio.create_task(self._keep_renewed(key, holder))
        try:
            await asyncio.to_thread(delete_apps_quietly, [deployment_name])
        finally:
            renewing.cancel()
            await self.release(key, holder)

    async def _keep_renewed(self, key: str, holder: str) -> None:
        while True:
            await asyncio.sleep(RENEW_SECONDS)
            await self.renew(key, holder)

    def report_fatal_error(self, deployment_name: str, reason: str) -> None:
        self._fatal_errors[deployment_name] = reason

    def pop_fatal_error(self, deployment_name: str) -> str | None:
        return self._fatal_errors.pop(deployment_name, None)


def get_or_create_coordinator(startup_window: bool = True):
    """Return the cluster-wide deploy coordinator handle, creating it on the head node if absent.
    *startup_window* applies only when this call creates it."""
    return DeployCoordinator.options(
        name=COORDINATOR_ACTOR_NAME,
        namespace=COORDINATOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        # a restart would trust an empty lease table; a fresh actor waits out a lease period instead
        max_restarts=0,
        runtime_env={"env_vars": build_env_vars(COMMON_ENV_VARS) | state_store_env_var()},
        **head_node_options(),
    ).remote(startup_window)
