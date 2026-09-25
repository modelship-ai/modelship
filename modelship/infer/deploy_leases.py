"""Holding a deploy lease from the deploy coordinator: a node's, so replicas sharing a node
load their models one at a time, and a gateway's, so one thing at a time plans, submits or
deletes that gateway's apps."""

import asyncio
import contextlib
import os
import socket
import threading
import time
from collections.abc import Generator

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.infer.deploy_coordinator import (
    LEASE_SECONDS,
    POLL_SECONDS,
    RENEW_SECONDS,
    gateway_lease_key,
    get_or_create_coordinator,
)
from modelship.logging import get_logger
from modelship.utils import random_uuid

logger = get_logger("deploy_leases")

_RPC_TIMEOUT_SECONDS = 5.0
_WAIT_LOG_SECONDS = 60.0
# an unreachable actor's name resolves for ~2s after it stops answering
_MAX_LOOKUPS = 3


class DeployLeaseError(Exception):
    """The lease service could not be reached, or a gateway lease was lost."""


@contextlib.asynccontextmanager
async def deploy_lease(model_name: str):
    """Holds this node's deploy lease for the block."""
    key = ray.get_runtime_context().get_node_id()
    holder = f"{model_name}/{os.getpid()}/{random_uuid()[:8]}"
    leases = await _acquire(key, holder, f"{model_name}: waiting to load on this node")
    # a thread, not a task: the loader's constructor blocks this replica's event loop
    stop = _renew_in_thread(leases, key, holder, f"{model_name}: lost the deploy lease on this node; loading anyway")
    try:
        yield
    finally:
        stop.set()
        # an unreleased lease expires on its own
        with contextlib.suppress(Exception):
            await asyncio.wait_for(leases.release.remote(key, holder), _RPC_TIMEOUT_SECONDS)


class GatewayLease:
    """A held gateway deploy lease, as `gateway_lease` yields it."""

    def __init__(self, coordinator, gateway_name: str, holder: str):
        self._coordinator = coordinator
        self._gateway_name = gateway_name
        self._holder = holder

    def write_effective(self, raw_models: list[dict]) -> None:
        """Writes the gateway's effective config through the deploy coordinator, which refuses
        once this lease is lost."""
        written = self._coordinator.write_effective.remote(self._gateway_name, self._holder, raw_models)
        if not ray.get(written, timeout=_RPC_TIMEOUT_SECONDS):
            raise DeployLeaseError(
                f"lost the deploy lease of gateway {self._gateway_name!r} before writing its effective config"
            )


@contextlib.contextmanager
def gateway_lease(gateway_name: str) -> Generator[GatewayLease, None, None]:
    """Holds the gateway's deploy lease for the block, waiting while anything else holds it."""
    key = gateway_lease_key(gateway_name)
    holder = f"deploy {socket.gethostname()}/{os.getpid()}/{random_uuid()[:8]}"
    coordinator = asyncio.run(_acquire(key, holder, f"Waiting to deploy to gateway {gateway_name!r}"))
    stop = _renew_in_thread(coordinator, key, holder, f"Lost the deploy lease of gateway {gateway_name!r}")
    try:
        yield GatewayLease(coordinator, gateway_name, holder)
    finally:
        stop.set()
        # an unreleased lease expires on its own
        with contextlib.suppress(Exception):
            ray.get(coordinator.release.remote(key, holder), timeout=_RPC_TIMEOUT_SECONDS)


async def _acquire(key: str, holder: str, waiting: str):
    """Polls until granted, without limit; an unreachable lease service is retryable."""
    leases = get_or_create_coordinator()
    lookups = 1
    logged: tuple[str, float] | None = None
    while True:
        try:
            blocker = await leases.acquire.remote(key, holder)
        except (ActorDiedError, ActorUnavailableError) as e:
            if lookups >= _MAX_LOOKUPS:
                raise DeployLeaseError(f"deploy lease service unreachable: {e}") from e
            await asyncio.sleep(POLL_SECONDS)
            leases = get_or_create_coordinator()
            lookups += 1
            continue
        if blocker is None:
            return leases
        now = time.monotonic()
        if logged is None or blocker != logged[0] or now - logged[1] >= _WAIT_LOG_SECONDS:
            logger.info("%s (%s)", waiting, blocker)
            logged = (blocker, now)
        await asyncio.sleep(POLL_SECONDS)


def _renew_in_thread(leases, key: str, holder: str, lost: str) -> threading.Event:
    """Renews the lease from a daemon thread until the returned event is set."""
    stop = threading.Event()
    threading.Thread(target=_renew_until, args=(leases, key, holder, lost, stop), daemon=True).start()
    return stop


def _renew_until(leases, key: str, holder: str, lost: str, stop: threading.Event) -> None:
    """Keeps the lease alive until `stop` is set. A failed call is retried until the lease
    would have expired; a refusal ends renewal at once, logging *lost*. The holder carries on either way."""
    renewed_at = time.monotonic()
    while not stop.wait(RENEW_SECONDS):
        try:
            if ray.get(leases.renew.remote(key, holder), timeout=_RPC_TIMEOUT_SECONDS):
                renewed_at = time.monotonic()
                continue
            reason = "renewal refused"
        except Exception as e:
            if time.monotonic() - renewed_at < LEASE_SECONDS:
                continue
            reason = repr(e)
        # a release the block already sent is what refuses the renewal, not a loss
        if not stop.is_set():
            logger.warning("%s (%s)", lost, reason)
        return
