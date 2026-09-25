"""Holding a deploy lease from the deploy coordinator, so replicas sharing a node load
their models one at a time."""

import asyncio
import contextlib
import os
import threading
import time

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.infer.deploy_coordinator import LEASE_SECONDS, get_or_create_coordinator
from modelship.logging import get_logger
from modelship.utils import random_uuid

logger = get_logger("deploy_leases")

RENEW_SECONDS = 10.0
POLL_SECONDS = 2.0
_RPC_TIMEOUT_SECONDS = 5.0
_WAIT_LOG_SECONDS = 60.0
# an unreachable actor's name resolves for ~2s after it stops answering
_MAX_LOOKUPS = 3


class DeployLeaseError(Exception):
    """The lease service could not be reached."""


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
            logger.info("%s: waiting to load on this node (%s)", model_name, blocker)
            logged = (blocker, now)
        await asyncio.sleep(POLL_SECONDS)


def _renew_until(leases, key: str, holder: str, model_name: str, stop: threading.Event) -> None:
    """Keeps the lease alive while the model loads. A failed call is retried until the lease
    would have expired; a refusal ends renewal at once. The load continues either way."""
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
            logger.warning("%s: lost the deploy lease on this node (%s); loading anyway", model_name, reason)
        return
