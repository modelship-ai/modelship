"""Cluster-wide routing for every gateway replica.

`GatewayCoordinator` is a detached, named Ray actor on the head node. Once a second it
reads Serve's application statuses and each gateway's routing version from the deploy
coordinator, and computes every gateway's model table (`modelship.deploy.routing`).
Gateway replicas long-poll `wait_for_change`, copy their table from `get_routing`, and
report on each poll the routing version their table was built from; a deploy asks
`wait_switched` whether every replica of a gateway holds a version. Nothing is stored:
a restarted gateway coordinator recomputes everything on its first pass.
"""

import asyncio
import contextlib
import time
from typing import Any

import ray
from ray import serve

from modelship.deploy.routing import Routing, compute_routing, running_replicas
from modelship.infer.deploy_coordinator import COORDINATOR_NAMESPACE, find_coordinator
from modelship.logging import configure_logging, get_logger
from modelship.metrics import COORDINATOR_GENERATION
from modelship.state import state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.config_schema import parse_deployment_name
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("gateway_coordinator")

GATEWAY_COORDINATOR_ACTOR_NAME = "modelship-gateway-coordinator"

_PASS_INTERVAL_S = 1.0
# How long a gateway replica's wait_for_change blocks before returning the generation unchanged.
_WATCH_TIMEOUT_S = 30.0
# A replica that hasn't polled for this long no longer counts as holding its reported version.
_REPORT_TTL_S = 2 * _WATCH_TIMEOUT_S
# How long get_routing waits for the first pass before raising, so the caller retries.
_FIRST_PASS_TIMEOUT_S = 5.0
_SWITCH_POLL_S = 0.1
_RPC_TIMEOUT_S = 10.0


@ray.remote(num_cpus=0)
class GatewayCoordinator:
    """Computes each gateway's model table from Serve and its routing version, and tracks which version each
    gateway replica holds."""

    def __init__(self):
        # Nothing else configures logging here; without it the logger falls back to Python's lastResort handler.
        configure_logging()
        self._routing: dict[str, Routing] = {}
        # gateway -> {"seq": int, "apps": {model: app}}, as the deploy coordinator last gave it
        self._versions: dict[str, dict[str, Any]] = {}
        # gateway -> the routing seq its published table was computed from
        self._seqs: dict[str, int] = {}
        # Millisecond clock; at most one change per pass, so a restarted gateway coordinator never repeats a generation.
        self._first_generation = int(time.time() * 1000)
        self._generation: dict[str, int] = {}
        self._change: dict[str, asyncio.Event] = {}
        self._computed = asyncio.Event()
        # gateways whose replicas asked for a table; the rest are found through their apps
        self._watched: set[str] = set()
        # gateway -> RUNNING replicas of its own app, as of the last pass
        self._replicas: dict[str, int] = {}
        # gateway -> replica id -> (routing seq, monotonic time of the report)
        self._reports: dict[str, dict[str, tuple[int, float]]] = {}
        self._versions_unreachable = False
        self._passes = asyncio.create_task(self._compute_forever())

    async def _compute_forever(self) -> None:
        failing = False
        while True:
            try:
                await self._compute()
            except Exception:
                if not failing:
                    logger.exception("Could not compute routing; keeping the last tables")
                failing = True
            else:
                failing = False
            await asyncio.sleep(_PASS_INTERVAL_S)

    async def _compute(self) -> None:
        """One pass: the table of every gateway whose routing version is known."""
        apps = dict((await asyncio.to_thread(serve.status)).applications)
        gateways = set(self._watched)
        for name in apps:
            if (parsed := parse_deployment_name(name)) is not None:
                gateways.add(parsed[0])
        await self._fetch_versions(sorted(gateways))
        for gateway in gateways:
            if (version := self._versions.get(gateway)) is None:
                continue
            if (routing := compute_routing(gateway, version["apps"], apps)) is None:
                continue
            self._replicas[gateway] = running_replicas(apps[gateway])
            self._publish(gateway, routing, version["seq"])
        self._computed.set()

    async def _fetch_versions(self, gateways: list[str]) -> None:
        """The gateways' routing versions; on failure the last ones stay. Never creates the deploy
        coordinator, whose creator decides its lease startup window."""
        if not gateways:
            return
        try:
            coordinator: Any = await asyncio.to_thread(find_coordinator)
            if coordinator is None:
                return
            versions = await asyncio.wait_for(coordinator.routing_versions.remote(gateways), _RPC_TIMEOUT_S)
        except Exception:
            if not self._versions_unreachable:
                logger.warning("Could not read routing versions from the deploy coordinator", exc_info=True)
            self._versions_unreachable = True
            return
        self._versions_unreachable = False
        self._versions.update(versions)

    def _publish(self, gateway: str, routing: Routing, seq: int) -> None:
        previous = self._routing.get(gateway)
        self._routing[gateway] = routing
        changed = previous is None or (previous.models, previous.expected) != (routing.models, routing.expected)
        if changed or seq != self._seqs.get(gateway):
            self._bump(gateway)
        self._seqs[gateway] = seq

    def _bump(self, gateway: str) -> None:
        """Advance the gateway's generation and wake its waiting replicas."""
        self._generation[gateway] = self._generation.get(gateway, self._first_generation) + 1
        COORDINATOR_GENERATION.set(self._generation[gateway], tags={"gateway": gateway})
        if (event := self._change.pop(gateway, None)) is not None:
            event.set()

    async def get_routing(self, gateway_name: str) -> dict:
        """The gateway's model table (app -> model), expected models, generation and routing seq."""
        self._watched.add(gateway_name)
        await asyncio.wait_for(self._computed.wait(), _FIRST_PASS_TIMEOUT_S)
        routing = self._routing.get(gateway_name)
        return {
            "models": dict(routing.models) if routing else {},
            "expected": list(routing.expected) if routing else [],
            "generation": self._generation.get(gateway_name, self._first_generation),
            "routing": self._seqs.get(gateway_name),
        }

    async def wait_for_change(
        self,
        gateway_name: str,
        since_gen: int,
        timeout: float = _WATCH_TIMEOUT_S,
        *,
        replica_id: str | None = None,
        routing: int | None = None,
    ) -> int:
        """Long-poll for a change to the gateway's table: returns its generation once it
        differs from since_gen, or after timeout regardless. Before the first pass it
        reports since_gen. A replica passes its id and the routing seq its table was built from."""
        self._watched.add(gateway_name)
        if replica_id is not None and routing is not None:
            self._reports.setdefault(gateway_name, {})[replica_id] = (routing, time.monotonic())
        if not self._computed.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._computed.wait(), timeout)
            return since_gen
        current = self._generation.get(gateway_name, self._first_generation)
        if current != since_gen:
            return current
        event = self._change.setdefault(gateway_name, asyncio.Event())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)
        return self._generation.get(gateway_name, self._first_generation)

    async def wait_switched(self, gateway_name: str, routing: int, timeout: float) -> bool:
        """True once every RUNNING replica of the gateway reports holding routing seq *routing*; False after *timeout*."""
        self._watched.add(gateway_name)
        deadline = time.monotonic() + timeout
        while not self._switched(gateway_name, routing):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_SWITCH_POLL_S)
        return True

    def _switched(self, gateway_name: str, routing: int) -> bool:
        if not self._computed.is_set() or self._seqs.get(gateway_name) != routing:
            return False
        now = time.monotonic()
        reports = self._reports.get(gateway_name, {})
        for replica, (_, at) in list(reports.items()):
            if now - at > _REPORT_TTL_S:
                del reports[replica]
        holding = sum(1 for seq, _ in reports.values() if seq == routing)
        return holding >= self._replicas.get(gateway_name, 0)


def get_or_create_gateway_coordinator():
    """Return the cluster-wide gateway coordinator handle, creating it on the head node if absent."""
    return GatewayCoordinator.options(
        name=GATEWAY_COORDINATOR_ACTOR_NAME,
        namespace=COORDINATOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        max_restarts=-1,
        runtime_env={"env_vars": build_env_vars(COMMON_ENV_VARS) | state_store_env_var()},
        **head_node_options(),
    ).remote()
