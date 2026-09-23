"""Cluster-wide bookkeeping for model deploys.

`DeployCoordinator` is a detached, named Ray actor on the head node, created by
the first operator to deploy and looked up by name afterwards. It holds two
things no single driver or replica can:

- a per-deployment backend-death count, retiring a deployment that keeps dying;
- fatal init errors, reported by a replica and read back by the driver, which
  is how a permanently-broken model is told apart from a transient failure.

Loads are serialised elsewhere: `DeployLeases` (see `deploy_leases.py`) grants
one lease per node, held by the replica for the duration of its own load.

The durable per-gateway routing registry that gateway replicas long-poll lives on
a separate actor, `ReplicaCoordinator` (see `replica_coordinator.py`) — it shares
this module's namespace but is otherwise independent.
"""

import asyncio

import ray

from modelship.logging import get_logger
from modelship.state import state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

_DEATHS_PER_REPLICA = 3


@ray.remote(num_cpus=0)
class DeployCoordinator:
    """Cluster-wide deploy bookkeeping: replica-death counts and fatal errors."""

    def __init__(self):
        self._fatal_errors: dict[str, str] = {}
        self._deaths: dict[str, int] = {}

    async def report_replica_death(
        self,
        gateway_name: str,
        deployment_name: str,
        model_name: str,
        replica_ceiling: int,
        reason: str,
    ) -> None:
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
        await self._retire(gateway_name, deployment_name, model_name)

    async def _retire(self, gateway_name: str, deployment_name: str, model_name: str) -> None:
        """Unregister first so replicas stop routing, then delete. serve.delete
        blocks on the app's teardown, so it runs off this actor's event loop."""
        from modelship.deploy.removal import delete_apps_quietly
        from modelship.infer.replica_coordinator import get_or_create_replica_coordinator

        try:
            await get_or_create_replica_coordinator().unregister_deployment.remote(
                gateway_name, deployment_name, model_name
            )
        except Exception:
            logger.exception("Failed to unregister retired deployment %s", deployment_name)
        await asyncio.to_thread(delete_apps_quietly, [deployment_name])

    def report_fatal_error(self, deployment_name: str, reason: str) -> None:
        self._fatal_errors[deployment_name] = reason

    def pop_fatal_error(self, deployment_name: str) -> str | None:
        return self._fatal_errors.pop(deployment_name, None)


def get_or_create_coordinator():
    """Return the cluster-wide coordinator handle, creating it on the head node if absent."""
    return DeployCoordinator.options(
        name=COORDINATOR_ACTOR_NAME,
        namespace=COORDINATOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        max_restarts=-1,
        # The store URI, which _retire passes on when it recreates the replica coordinator.
        runtime_env={"env_vars": build_env_vars(COMMON_ENV_VARS) | state_store_env_var()},
        **head_node_options(),
    ).remote()
