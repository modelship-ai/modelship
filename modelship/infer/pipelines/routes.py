"""A pipeline replica's view of its gateway's models: the model -> app table the gateway coordinator computes,
followed the way the gateway's replicas follow it. Each poll reports the routing version the table was built from,
so a deploy's switch waits for the pipeline's replicas as for the gateway's."""

import asyncio
from typing import Any, Literal, cast

from ray import serve
from ray.serve.handle import DeploymentHandle

from modelship.logging import get_logger

logger = get_logger("pipeline")

# serving: something serves the model; starting: its gateway has it, but nothing serves it yet; unknown: its gateway
# doesn't have it
ModelState = Literal["serving", "starting", "unknown"]

# after a failed poll
_RETRY_S = 5.0
# how long a session waits for the first table
_FIRST_TABLE_S = 10.0


class ModelRoutes:
    def __init__(self, gateway: str, replica_id: str):
        self._gateway = gateway
        self._replica_id = replica_id
        # model -> (app, its handle)
        self._handles: dict[str, tuple[str, DeploymentHandle]] = {}
        self._expected: set[str] = set()
        self._generation = 0
        self._routing: int | None = None
        self._coordinator: Any = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._follow())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def ready(self) -> bool:
        """Whether the replica holds a table, waiting a while for the first one."""
        try:
            await asyncio.wait_for(self._ready.wait(), _FIRST_TABLE_S)
        except TimeoutError:
            return False
        return True

    def handle(self, model: str) -> DeploymentHandle | None:
        found = self._handles.get(model)
        return found[1] if found is not None else None

    def state(self, model: str) -> ModelState:
        if model in self._handles:
            return "serving"
        return "starting" if model in self._expected else "unknown"

    async def _follow(self) -> None:
        from modelship.deploy import gateway_coordinator

        while True:
            try:
                if self._coordinator is None:
                    self._coordinator = await asyncio.to_thread(gateway_coordinator.get_or_create_gateway_coordinator)
                coordinator = self._coordinator
                generation = await coordinator.wait_for_change.remote(
                    self._gateway, self._generation, replica_id=self._replica_id, routing=self._routing
                )
                if generation != self._generation or not self._ready.is_set():
                    self._apply(cast(dict, await coordinator.get_routing.remote(self._gateway)))
            except asyncio.CancelledError:
                return
            except Exception:
                self._coordinator = None
                logger.debug("pipeline: routing poll failed; retrying", exc_info=True)
                await asyncio.sleep(_RETRY_S)

    def _apply(self, snapshot: dict) -> None:
        """Takes *snapshot*'s table; an app that can't be reached yet stays out, and the generation stays where it
        was, so the next poll tries again."""
        handles: dict[str, tuple[str, DeploymentHandle]] = {}
        failed: list[str] = []
        for app, model in snapshot.get("models", {}).items():
            held = self._handles.get(model)
            if held is not None and held[0] == app:
                handles[model] = held
                continue
            try:
                handles[model] = (app, serve.get_app_handle(app))
            except Exception:
                logger.debug("pipeline: app %s isn't reachable yet", app, exc_info=True)
                failed.append(app)
        self._handles = handles
        self._expected = set(snapshot.get("expected", []))
        if failed:
            raise RuntimeError(f"apps not reachable yet: {failed}")
        self._generation = snapshot.get("generation", self._generation)
        self._routing = snapshot.get("routing")
        self._ready.set()
