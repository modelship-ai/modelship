"""Waiting for a node's deploy lease from the deploy coordinator, so replicas sharing a node load
their models one at a time. The deploy coordinator frees it once the replica is RUNNING or gone."""

import asyncio
import time

import ray
from ray import serve
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.infer.deploy_coordinator import POLL_SECONDS, get_or_create_coordinator
from modelship.logging import get_logger

logger = get_logger("deploy_leases")

_WAIT_LOG_SECONDS = 60.0
# an unreachable actor's name resolves for ~2s after it stops answering
_MAX_LOOKUPS = 3


class DeployLeaseError(Exception):
    """The lease service could not be reached."""


async def wait_for_deploy_lease(model_name: str) -> None:
    """Returns once this replica holds its node's deploy lease, polling without limit; an unreachable
    deploy coordinator is retried, then raises DeployLeaseError."""
    key = ray.get_runtime_context().get_node_id()
    context = serve.get_replica_context()
    replica = context.replica_id.unique_id
    holder = {
        "app": context.app_name,
        "deployment": context.deployment,
        "replica": replica,
        "label": f"{model_name}/{replica}",
    }
    waiting = f"{model_name}: waiting to load on this node"
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
            return
        now = time.monotonic()
        if logged is None or blocker != logged[0] or now - logged[1] >= _WAIT_LOG_SECONDS:
            logger.info("%s (%s)", waiting, blocker)
            logged = (blocker, now)
        await asyncio.sleep(POLL_SECONDS)
