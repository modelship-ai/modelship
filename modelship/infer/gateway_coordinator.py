"""Cluster-wide routing for every gateway replica.

`GatewayCoordinator` is a detached, named Ray actor on the head node. Once a second it
reads Serve's application statuses and each gateway's effective config, computes every
gateway's model table (`modelship.deploy.routing`), and deletes the apps nothing has
used for `_UNUSED_GRACE_SECONDS`. Gateway replicas long-poll `wait_for_change` and copy
their table from `get_routing`; the driver polls `get_retiring` until those deletes are
done. Nothing is stored: a restarted coordinator recomputes everything on its first pass.
"""

import asyncio
import contextlib
import time

import ray
from ray import serve

from modelship.deploy.effective_config import read_targets
from modelship.deploy.removal import delete_apps_quietly
from modelship.deploy.routing import Routing, compute_routing
from modelship.infer.deploy_coordinator import COORDINATOR_NAMESPACE
from modelship.logging import configure_logging, get_logger
from modelship.metrics import COORDINATOR_GENERATION
from modelship.state import get_state_store, state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.config_schema import parse_deployment_name
from modelship.utils.runtime_env import COMMON_ENV_VARS, build_env_vars

logger = get_logger("gateway_coordinator")

GATEWAY_COORDINATOR_ACTOR_NAME = "modelship-gateway-coordinator"

_PASS_INTERVAL_S = 1.0
# Every gateway replica re-pulls its table well within this, so an app unused this long gets no requests.
_UNUSED_GRACE_SECONDS = 10.0
# How long a gateway replica's wait_for_change blocks before returning the generation unchanged.
_WATCH_TIMEOUT_S = 30.0
# How long get_routing waits for the first pass before raising, so the caller retries.
_FIRST_PASS_TIMEOUT_S = 5.0


@ray.remote(num_cpus=0)
class GatewayCoordinator:
    """Computes each gateway's model table from Serve and the effective config, and deletes unused apps."""

    def __init__(self):
        # Nothing else configures logging here; without it the logger falls back to Python's lastResort handler.
        configure_logging()
        self._store = get_state_store()
        self._routing: dict[str, Routing] = {}
        # Millisecond clock; at most one change per pass, so a restarted coordinator never repeats a generation.
        self._first_generation = int(time.time() * 1000)
        self._generation: dict[str, int] = {}
        self._change: dict[str, asyncio.Event] = {}
        self._computed = asyncio.Event()
        # monotonic start of the last completed pass; _pass_done is set and replaced after each pass
        self._last_pass_start = float("-inf")
        self._pass_done = asyncio.Event()
        # gateways whose replicas asked for a table; the rest are found through their apps
        self._watched: set[str] = set()
        self._unreadable: set[str] = set()
        self._unused_since: dict[str, float] = {}
        self._deleting: set[str] = set()
        self._deletions: set[asyncio.Task] = set()
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
        """One pass: every gateway's table, then deletes of apps unused past the grace period."""
        started = time.monotonic()
        apps = dict((await asyncio.to_thread(serve.status)).applications)
        gateways = set(self._watched)
        for name in apps:
            if (parsed := parse_deployment_name(name)) is not None:
                gateways.add(parsed[0])
        unused: set[str] = set()
        for gateway in gateways:
            routing = compute_routing(gateway, await self._targets(gateway), apps)
            if routing is None:
                continue
            self._publish(gateway, routing)
            unused |= routing.unused
        self._delete_unused(unused)
        self._last_pass_start = started
        self._computed.set()
        self._pass_done.set()
        self._pass_done = asyncio.Event()

    async def _targets(self, gateway: str) -> dict[str, str] | None:
        """None when the effective config is missing or unreadable, which deletes nothing."""
        try:
            targets = await read_targets(self._store, gateway)
        except Exception:
            if gateway not in self._unreadable:
                logger.warning("Could not read the effective config of gateway %s", gateway, exc_info=True)
            self._unreadable.add(gateway)
            return None
        self._unreadable.discard(gateway)
        return targets

    def _publish(self, gateway: str, routing: Routing) -> None:
        previous = self._routing.get(gateway)
        self._routing[gateway] = routing
        if previous is None or (previous.models, previous.expected) != (routing.models, routing.expected):
            self._bump(gateway)

    def _bump(self, gateway: str) -> None:
        """Advance the gateway's generation and wake its waiting replicas."""
        self._generation[gateway] = self._generation.get(gateway, self._first_generation) + 1
        COORDINATOR_GENERATION.set(self._generation[gateway], tags={"gateway": gateway})
        if (event := self._change.pop(gateway, None)) is not None:
            event.set()

    def _delete_unused(self, unused: set[str]) -> None:
        """Deletes each app unused for `_UNUSED_GRACE_SECONDS`, off the event loop and one delete at a time per app."""
        now = time.monotonic()
        self._unused_since = {app: self._unused_since.get(app, now) for app in unused}
        for app, since in self._unused_since.items():
            if now - since >= _UNUSED_GRACE_SECONDS and app not in self._deleting:
                self._deleting.add(app)
                task = asyncio.create_task(self._delete(app))
                self._deletions.add(task)
                task.add_done_callback(self._deletions.discard)

    async def _delete(self, app: str) -> None:
        try:
            # serve.delete blocks until the app is gone
            await asyncio.to_thread(delete_apps_quietly, [app])
        finally:
            self._deleting.discard(app)
            # a failed delete waits out the grace period again
            self._unused_since.pop(app, None)

    async def get_routing(self, gateway_name: str) -> dict:
        """The gateway's model table (app -> model), expected models and generation."""
        self._watched.add(gateway_name)
        await asyncio.wait_for(self._computed.wait(), _FIRST_PASS_TIMEOUT_S)
        routing = self._routing.get(gateway_name)
        return {
            "models": dict(routing.models) if routing else {},
            "expected": list(routing.expected) if routing else [],
            "generation": self._generation.get(gateway_name, self._first_generation),
        }

    async def get_retiring(self, gateway_name: str) -> list[str]:
        """The gateway's apps that are unused or being deleted, as of a pass started after this call."""
        self._watched.add(gateway_name)
        called = time.monotonic()
        while self._last_pass_start <= called:
            await self._pass_done.wait()
        routing = self._routing.get(gateway_name)
        return sorted(routing.retiring) if routing else []

    async def wait_for_change(self, gateway_name: str, since_gen: int, timeout: float = _WATCH_TIMEOUT_S) -> int:
        """Long-poll for a change to the gateway's table: returns its generation once it
        differs from since_gen, or after timeout regardless. Before the first pass it
        reports since_gen."""
        self._watched.add(gateway_name)
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


def get_or_create_gateway_coordinator():
    """Return the cluster-wide replica-routing coordinator handle, creating it on the head node if absent."""
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
