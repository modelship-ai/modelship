"""Cluster-wide coordinator for serialising model deploys across operators.

A `mship_deploy.py` driver ("operator") cannot safely assume it is the only process
deploying models to the Ray cluster. Two operators both checking
`ray.available_resources()`, both seeing "GPU free", and both calling
`serve.run()` concurrently can trigger simultaneous VRAM loads on the same
device and OOM. This module provides a cluster-level mutex that combines
"is the lock free?" with "can this request actually be placed?" into one
atomic check, so operators never race.

Design:

- `DeployCoordinator` is a detached, named Ray actor on the head node. The
  first operator to start creates it; subsequent operators look it up by name.
- Operators reserve via `try_reserve(operator_id, probe, num_gpus, num_cpus)`.
  Granted only when the lock is unheld AND the cluster has the requested
  resources available right now.
- The operator passes a handle to a small owned actor (`OperatorProbe`) when
  reserving. The coordinator polls that handle via `__ray_ready__` to detect
  ungraceful operator death (SIGKILL, host crash, partition). Because the
  probe is owned by the operator driver, Ray tears it down when the driver
  dies — the coordinator sees `RayActorError` and force-releases the lock.
  It runs on the driver's node, so no other node's death can kill it.
- Graceful shutdown uses `release(operator_id)` from the operator's
  try/finally, cancelling the liveness watcher cleanly.

The durable per-gateway routing registry that gateway replicas long-poll lives on
a separate actor, `ReplicaCoordinator` (see `replica_coordinator.py`) — it shares
this module's namespace but is otherwise independent, so gateway watch traffic
never contends with this mutex's event loop.
"""

import asyncio
import time

import ray
from ray import exceptions as ray_exceptions
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from modelship.logging import get_logger
from modelship.metrics import (
    DEPLOY_LOCK_HELD,
    DEPLOY_RESERVATIONS_TOTAL,
    OPERATOR_FORCE_RELEASE_TOTAL,
)
from modelship.state import state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

_DEATHS_PER_REPLICA = 3

_LIVENESS_POLL_INTERVAL_S = 5.0
_LIVENESS_CALL_TIMEOUT_S = 3.0
_LIVENESS_TIMEOUT_STRIKES = 3


@ray.remote(num_cpus=0)
class OperatorProbe:
    """Empty actor whose only purpose is to be owned by the operator driver.

    Ray destroys owned actors when the owning process dies. The coordinator
    uses `__ray_ready__` on this handle as a liveness signal — if the call
    starts raising `RayActorError`, the operator is gone and the lock can be
    force-released.
    """

    def ping(self) -> str:
        return "alive"


@ray.remote(num_cpus=0)
class DeployCoordinator:
    """Cluster-wide mutex + resource-aware admission gate for model deploys."""

    def __init__(self):
        self._held_by: str | None = None
        self._held_deployment: str | None = None
        self._held_since: float = 0.0
        self._watcher_task: asyncio.Task | None = None
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

    async def try_reserve(
        self,
        operator_id: str,
        deployment_name: str,
        num_gpus: float,
        num_cpus: float,
        probe_handle,
    ) -> tuple[bool, str]:
        if self._held_by is not None:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "locked"})
            return False, f"locked_by:{self._held_by}:{self._held_deployment}"

        avail = ray.available_resources()
        eps = 1e-6
        if float(num_gpus or 0) > avail.get("GPU", 0) + eps:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "insufficient_gpu"})
            return False, "insufficient_gpu"
        if float(num_cpus or 0) > avail.get("CPU", 0) + eps:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "insufficient_cpu"})
            return False, "insufficient_cpu"

        self._held_by = operator_id
        self._held_deployment = deployment_name
        self._held_since = time.time()
        DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "granted"})
        DEPLOY_LOCK_HELD.set(1)
        self._watcher_task = asyncio.create_task(self._watch_operator_liveness(operator_id, probe_handle))
        logger.info(
            "Reserved for operator=%s deployment=%s (num_gpus=%s, num_cpus=%s)",
            operator_id,
            deployment_name,
            num_gpus,
            num_cpus,
        )
        return True, "ok"

    async def release(self, operator_id: str) -> bool:
        if self._held_by != operator_id:
            logger.warning(
                "Stale release from %s (current holder: %s) — ignoring",
                operator_id,
                self._held_by,
            )
            return False
        self._clear_hold()
        logger.info("Released by operator=%s", operator_id)
        return True

    async def status(self) -> dict:
        return {
            "held_by": self._held_by,
            "held_deployment": self._held_deployment,
            "held_for_seconds": (time.time() - self._held_since) if self._held_by else 0.0,
        }

    def _clear_hold(self):
        self._held_by = None
        self._held_deployment = None
        self._held_since = 0.0
        DEPLOY_LOCK_HELD.set(0)
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()
        self._watcher_task = None

    async def _watch_operator_liveness(self, operator_id: str, probe_handle):
        timeout_strikes = 0
        while True:
            try:
                await asyncio.sleep(_LIVENESS_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return

            if self._held_by != operator_id:
                return

            try:
                await asyncio.wait_for(
                    probe_handle.__ray_ready__.remote(),
                    timeout=_LIVENESS_CALL_TIMEOUT_S,
                )
                timeout_strikes = 0
            except ray_exceptions.RayActorError:
                logger.warning(
                    "Probe for operator=%s is gone — force-releasing lock (deployment=%s)",
                    operator_id,
                    self._held_deployment,
                )
                OPERATOR_FORCE_RELEASE_TOTAL.inc(tags={"reason": "probe_gone"})
                self._clear_hold()
                return
            except TimeoutError:
                timeout_strikes += 1
                if timeout_strikes >= _LIVENESS_TIMEOUT_STRIKES:
                    logger.warning(
                        "Probe for operator=%s unresponsive for %ds — force-releasing lock",
                        operator_id,
                        timeout_strikes * _LIVENESS_POLL_INTERVAL_S,
                    )
                    OPERATOR_FORCE_RELEASE_TOTAL.inc(tags={"reason": "unresponsive"})
                    self._clear_hold()
                    return


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


def create_operator_probe():
    """An OperatorProbe owned by this driver, on this driver's node."""
    node_id = ray.get_runtime_context().get_node_id()
    return OperatorProbe.options(
        num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
    ).remote()
