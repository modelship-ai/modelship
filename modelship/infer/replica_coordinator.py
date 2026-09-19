"""Cluster-wide routing registry shared by every gateway replica.

`ReplicaCoordinator` is a detached, named Ray actor on the head node holding the
durable mapping of model deployments each gateway owns. `mship_deploy.py` writes to
it as models are (un)deployed; every gateway replica long-polls `wait_for_change`
and reconciles its own routing table from `get_routing` — the driver never pushes
to individual replicas.

The registry is persisted through `get_state_store()` — cluster-scoped even on the
default `memory://` (backed by its own detached actor, see `modelship.state.memory`)
so a resurrected coordinator reloads live ownership instead of starting empty;
`redis://` adds survival across a full cluster loss on top of that. The
per-gateway generation counter and its wakeup `asyncio.Event` are ephemeral: on
restart the generation resets to 0, which `wait_for_change` already treats as
"changed" so replicas re-pull and reconcile from the reloaded registry.
"""

import asyncio
import contextlib

import ray

from modelship.infer.deploy_coordinator import COORDINATOR_NAMESPACE
from modelship.logging import configure_logging, get_logger
from modelship.metrics import COORDINATOR_GENERATION
from modelship.state import MemoryStateStore, get_state_store
from modelship.utils import head_node_options

logger = get_logger("replica_coordinator")

REPLICA_COORDINATOR_ACTOR_NAME = "modelship-replica-coordinator"

_STATE_KEY = "coordinator/state"

# How long a gateway replica's wait_for_change blocks before returning the
# current generation unchanged. Bounds how long a missed wake / coordinator
# restart can leave a replica un-reconciled (it re-pulls on every return).
_WATCH_TIMEOUT_S = 30.0


@ray.remote(num_cpus=0)
class ReplicaCoordinator:
    """Durable per-gateway routing registry with long-poll change notification."""

    def __init__(self):
        # Nothing else configures logging here; without it the logger falls back to Python's lastResort handler.
        configure_logging()
        # Durable ownership registry: gateway_name -> {deployment_name -> model_name}.
        # The driver writes it on (un)deploy; gateway replicas reconcile their
        # routing tables from it (see get_routing / wait_for_change), so the driver
        # never pushes to individual replicas.
        # _registry and _expected are durable (loaded below, written through on
        # every change); _generation/_change are ephemeral wakeup state. On a
        # resurrected coordinator the generation restarts at 0, which the gateway's
        # wait_for_change already treats as "changed" so replicas re-pull and
        # reconcile from the reloaded registry.
        self._store = get_state_store()
        if isinstance(getattr(self._store, "inner", self._store), MemoryStateStore):
            # The memory store is cluster-scoped (a detached actor), so it survives
            # THIS coordinator's own restart — but it dies with the cluster: a
            # coordinator resurrected on a fresh cluster reloads an empty registry,
            # and the next deploy (gen advances) re-enables removals against it —
            # dropping still-healthy models from gateway routing. Fine single-node;
            # for survival across cluster loss set MSHIP_STATE_STORE to redis://.
            logger.warning(
                "Replica coordinator is backed by a cluster-scoped (non-durable) memory state "
                "store; its routing registry survives coordinator restart but is lost if the "
                "cluster dies. Set MSHIP_STATE_STORE to redis:// to survive cluster loss."
            )
        saved = self._store.get(_STATE_KEY)
        saved = saved if isinstance(saved, dict) else {}
        registry = saved.get("registry")
        self._registry: dict[str, dict[str, str]] = registry if isinstance(registry, dict) else {}
        # Per-gateway change notification driving the gateway watch loop: a
        # monotonic generation bumped on every routing/expected change, plus an
        # asyncio.Event woken on each bump so a long-polling replica returns at
        # once. _expected is the desired model set used for gateway readiness.
        self._generation: dict[str, int] = {}
        expected = saved.get("expected")
        self._expected: dict[str, list[str]] = expected if isinstance(expected, dict) else {}
        self._change: dict[str, asyncio.Event] = {}

    # These are async so every registry / generation / Event mutation runs on the
    # actor's single event loop, serialised with wait_for_change and race-free.

    def _bump(self, gateway_name: str) -> None:
        """Advance the gateway's generation and wake any current waiters. The old
        Event is set (releasing replicas blocked on it) then replaced with a fresh
        unset Event for the next round."""
        self._generation[gateway_name] = self._generation.get(gateway_name, 0) + 1
        COORDINATOR_GENERATION.set(self._generation[gateway_name], tags={"gateway": gateway_name})
        old = self._change.get(gateway_name)
        if old is not None:
            old.set()
        self._change[gateway_name] = asyncio.Event()

    async def _persist(self) -> None:
        """Write the durable routing state through the StateStore. Async because a
        memory-backed store is an RPC to another actor — a sync ray.get here would
        block this actor's own event loop (and thus wait_for_change) on every write."""
        await self._store.set_async(_STATE_KEY, {"registry": self._registry, "expected": self._expected})

    async def register_deployment(self, gateway_name: str, deployment_name: str, model_name: str) -> None:
        """Register deployment_name for model_name, evicting any other deployment
        already registered under model_name in the same write."""
        gw = self._registry.setdefault(gateway_name, {})
        superseded = [name for name, model in gw.items() if model == model_name and name != deployment_name]
        for name in superseded:
            del gw[name]
        if superseded:
            logger.info("routing for model %s: %s -> %s", model_name, superseded, deployment_name)
        gw[deployment_name] = model_name
        await self._persist()
        self._bump(gateway_name)

    async def unregister_deployment(
        self, gateway_name: str, deployment_name: str, model_name: str | None = None
    ) -> None:
        """Drop the deployment, and the model's `_expected` entry with it once no
        other deployment serves that name. `model_name` is only needed when the
        deployment never registered — otherwise it comes from the registry."""
        gw = self._registry.get(gateway_name) or {}
        model_name = gw.pop(deployment_name, None) or model_name
        if model_name is not None and model_name not in gw.values():
            expected = self._expected.get(gateway_name)
            if expected is not None:
                self._expected[gateway_name] = [m for m in expected if m != model_name]
        if not gw:
            self._registry.pop(gateway_name, None)
        await self._persist()
        self._bump(gateway_name)

    async def set_expected(self, gateway_name: str, names: list[str]) -> None:
        """Record the desired model set for readiness; bumps so replicas adopt it."""
        self._expected[gateway_name] = list(names)
        await self._persist()
        self._bump(gateway_name)

    async def get_routing(self, gateway_name: str) -> dict:
        """Snapshot a replica pulls after a change: the app->model map, the
        expected-model set, and the current generation."""
        return {
            "models": dict(self._registry.get(gateway_name, {})),
            "expected": list(self._expected.get(gateway_name, [])),
            "generation": self._generation.get(gateway_name, 0),
        }

    async def wait_for_change(self, gateway_name: str, since_gen: int, timeout: float = _WATCH_TIMEOUT_S) -> int:
        """Long-poll for a routing change. Returns the current generation at once
        if it differs from since_gen (covers both a forward bump and a coordinator
        restart that reset it to 0); otherwise waits for the next bump up to
        timeout, then returns the current generation regardless."""
        current = self._generation.get(gateway_name, 0)
        if current != since_gen:
            return current
        event = self._change.setdefault(gateway_name, asyncio.Event())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)
        return self._generation.get(gateway_name, 0)


def get_or_create_replica_coordinator():
    """Return the cluster-wide replica-routing coordinator handle, creating it on the head node if absent."""
    return ReplicaCoordinator.options(
        name=REPLICA_COORDINATOR_ACTOR_NAME,
        namespace=COORDINATOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        max_restarts=-1,
        **head_node_options(),
    ).remote()
