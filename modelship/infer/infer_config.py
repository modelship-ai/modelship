"""Request-path plumbing that crosses Ray process boundaries.

The models.yaml schemas live in modelship/utils/config_schema.py, re-exported below.
"""

import asyncio
import time
from collections.abc import Callable

import ray
from fastapi import Request
from ray.exceptions import RayActorError
from starlette.datastructures import Headers, State

from modelship.logging import get_logger
from modelship.utils import head_node_options
from modelship.utils.config_schema import (
    AutoscalingConfig,
    ChatTemplateContentFormatOption,
    DiffusersConfig,
    LlamaServerConfig,
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    StableDiffusionCppConfig,
    VllmEngineConfig,
    WhispercppConfig,
    default_gpu_memory_utilization,
    resolve_gpu_memory_utilization,
)

__all__ = [
    "AutoscalingConfig",
    "ChatTemplateContentFormatOption",
    "DiffusersConfig",
    "DisconnectRegistry",
    "LlamaServerConfig",
    "ModelLoader",
    "ModelUsecase",
    "ModelshipConfig",
    "ModelshipModelConfig",
    "RawRequestProxy",
    "RequestWatcher",
    "StableDiffusionCppConfig",
    "VllmEngineConfig",
    "WhispercppConfig",
    "default_gpu_memory_utilization",
    "get_disconnect_registry",
    "reset_disconnect_registry",
    "resolve_gpu_memory_utilization",
]

_logger = get_logger("config")


# How long a recorded disconnect lingers before the registry evicts it. The
# gateway no longer clears entries on request teardown — that clear ran in the
# same finally the disconnect itself triggered, racing (and usually beating) the
# model deployment's cross-process is_disconnected() poll, so the signal was
# dropped before it was read. This TTL is now what bounds the set. It only needs
# to outlast the deployment's poll interval, not the whole generation: the entry
# is added at disconnect time, by which point the deployment is already polling.
_DISCONNECT_TTL_SECONDS = 300.0


class _DisconnectStore:
    """Plain (non-actor) TTL set of disconnected request ids, factored out of
    DisconnectRegistry so the eviction logic is unit-testable without a Ray
    cluster. ``now`` is injectable for deterministic tests."""

    def __init__(self, ttl_seconds: float, now: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._now = now
        # request_id -> monotonic deadline after which the entry is evicted.
        self._deadlines: dict[str, float] = {}

    def set(self, request_id: str) -> None:
        now = self._now()
        self._evict_expired(now)
        self._deadlines[request_id] = now + self._ttl

    def is_set(self, request_id: str) -> bool:
        deadline = self._deadlines.get(request_id)
        if deadline is None:
            return False
        if deadline <= self._now():
            del self._deadlines[request_id]
            return False
        return True

    def is_set_many(self, request_ids: list[str]) -> list[str]:
        return [request_id for request_id in request_ids if self.is_set(request_id)]

    def clear(self, request_id: str) -> None:
        self._deadlines.pop(request_id, None)

    def _evict_expired(self, now: float) -> None:
        for request_id in [rid for rid, deadline in self._deadlines.items() if deadline <= now]:
            del self._deadlines[request_id]


@ray.remote(num_cpus=0)
class DisconnectRegistry:
    """One cluster-wide actor tracking client-disconnect per request id, replacing
    the previous per-request DisconnectEvent actor. Async so concurrent polls don't
    head-of-line block on the single-threaded actor.

    Entries are TTL-evicted (``_DISCONNECT_TTL_SECONDS``) rather than cleared by the
    gateway — see ``_DISCONNECT_TTL_SECONDS`` for why."""

    def __init__(self, ttl_seconds: float = _DISCONNECT_TTL_SECONDS):
        self._store = _DisconnectStore(ttl_seconds)

    async def set(self, request_id: str) -> None:
        self._store.set(request_id)

    async def is_set(self, request_id: str) -> bool:
        return self._store.is_set(request_id)

    async def is_set_many(self, request_ids: list[str]) -> list[str]:
        return self._store.is_set_many(request_ids)

    async def clear(self, request_id: str) -> None:
        self._store.clear(request_id)


_disconnect_registry = None


def get_disconnect_registry():
    """Get-or-create the single detached, named DisconnectRegistry shared by every
    gateway replica and model deployment. Cached to keep the lookup off the hot path."""
    global _disconnect_registry
    if _disconnect_registry is None:
        _disconnect_registry = DisconnectRegistry.options(
            name="modelship_disconnect_registry",
            get_if_exists=True,
            lifetime="detached",
            namespace="modelship",
            **head_node_options(),
        ).remote()
    return _disconnect_registry


def reset_disconnect_registry() -> None:
    """Drop the cached handle so the next get_disconnect_registry() re-resolves the
    named actor. Called after a RayActorError: the detached actor died (node
    preemption, GCS restart) and the cached handle is now stale. get_if_exists makes
    every process that re-resolves converge on the same recreated actor."""
    global _disconnect_registry
    _disconnect_registry = None


class RequestWatcher:
    """Watches a FastAPI Request for client disconnect and records it in the shared
    DisconnectRegistry, keyed by request id."""

    def __init__(self, raw_request: Request, request_id: str, model: str = "", endpoint: str = ""):
        self._request = raw_request
        self._registry = get_disconnect_registry()
        self._request_id = request_id
        self._model = model
        self._endpoint = endpoint
        self._task = asyncio.create_task(self._watch())

    async def _watch(self):
        from modelship.metrics import CLIENT_DISCONNECTS_TOTAL

        while True:
            if await self._request.is_disconnected():
                CLIENT_DISCONNECTS_TOTAL.inc(tags={"model": self._model, "endpoint": self._endpoint})
                await self._record_disconnect()
                break
            await asyncio.sleep(0.1)

    async def _record_disconnect(self) -> None:
        """Record the disconnect in the shared registry, re-resolving the actor and
        retrying once if it has died — otherwise a registry blip silently loses the
        signal and the deployment runs to completion."""
        try:
            await self._registry.set.remote(self._request_id)  # type: ignore[attr-defined]
        except RayActorError:
            reset_disconnect_registry()
            self._registry = get_disconnect_registry()
            try:
                await self._registry.set.remote(self._request_id)  # type: ignore[attr-defined]
            except RayActorError:
                _logger.warning("Disconnect registry unavailable; lost disconnect for %s", self._request_id)

    def stop(self):
        """Cancel the watch task. The disconnect entry (if any) is deliberately
        left for the DisconnectRegistry to TTL-evict: clearing it here ran in the
        same teardown the disconnect triggered and raced the model deployment's
        is_disconnected() poll, dropping the signal before it was read."""
        self._task.cancel()

    @property
    def registry(self):
        return self._registry


class RawRequestProxy:
    """
    Stands in for a FastAPI Request inside model deployment actors.

    The real FastAPI Request cannot cross Ray process boundaries — it holds a live
    TCP socket and ASGI callables that are not serializable. Instead, the gateway
    extracts the serializable parts (headers as a plain dict, disconnect signal via
    the shared DisconnectRegistry actor) and passes those to the model deployment.
    RawRequestProxy reconstructs them into the interface that vllm expects:

      - raw_request.headers.get(...)     → Starlette Headers built from the dict
      - await raw_request.is_disconnected() → polls the DisconnectRegistry by id
      - raw_request.identity             → the caller's identity_key(), resolved once at
                                            the gateway (modelship/openai/auth.py)

    Any additional attributes vllm reads from raw_request in future should be added here.
    """

    def __init__(self, registry, headers: dict, request_id: str | None = None, identity: str | None = None):
        self._registry = registry
        self.headers = Headers(headers=headers)
        self.state = State()  # vllm writes per-request state here; lives in the actor process
        self.request_id = request_id
        self.identity = identity

    @property
    def is_watchable(self) -> bool:
        """Whether this proxy has a real registry + id to poll — false for internal
        requests (e.g. warmup) that have no client to disconnect."""
        return self._registry is not None and self.request_id is not None

    async def is_disconnected(self) -> bool:
        if self._registry is None:
            # No real registry (e.g. an internal warmup request) — nothing to poll.
            return False
        try:
            return await self._registry.is_set.remote(self.request_id)
        except RayActorError:
            # The shared registry actor died (node preemption, GCS restart). Disconnect
            # propagation is best-effort, so degrade to "still connected" rather than
            # failing a healthy in-flight request, and re-resolve the (recreated, via
            # get_if_exists) actor so later polls in this request reconnect.
            _logger.warning("Disconnect registry unavailable; assuming client connected")
            reset_disconnect_registry()
            self._registry = get_disconnect_registry()
            return False
