import asyncio
import inspect
from dataclasses import dataclass

import pytest
from ray import cloudpickle
from ray.serve._private.common import DeploymentHandleSource, DeploymentID, ReplicaID, RequestMetadata
from ray.serve._private.default_impl import create_init_handle_options, get_request_metadata
from ray.serve._private.handle_options import DynamicHandleOptions
from ray.serve._private.request_router.common import PendingRequest, RequestRoutingContext
from ray.serve.handle import DeploymentHandle
from ray.serve.request_router import RequestRouter

from modelship import metrics
from modelship.deploy.actor_options import build_deployment_options
from modelship.deploy.request_router import AffinityRouter
from modelship.infer.infer_config import ModelshipModelConfig

APP = DeploymentID(name="gw.m-0123456789", app_name="gw.m-0123456789")


@dataclass(eq=False)
class _Replica:
    replica_id: ReplicaID
    queue_len: int = 0
    node_id: str = "node-1"
    availability_zone: str | None = None
    max_ongoing_requests: int = 100

    async def get_queue_len(self, *, deadline_s: float) -> int:
        return self.queue_len


def _request(key: str = "") -> PendingRequest:
    return PendingRequest(
        args=[], kwargs={}, metadata=RequestMetadata(request_id="r", internal_request_id="r", session_id=key)
    )


async def _router(replicas: list[_Replica]) -> AffinityRouter:
    router = AffinityRouter(
        deployment_id=APP, handle_source=DeploymentHandleSource.UNKNOWN, use_replica_queue_len_cache=True
    )
    router.initialize_state()
    router.update_replicas(replicas)
    while any(router.replica_queue_len_cache.get(r.replica_id) is None for r in replicas):
        await asyncio.sleep(0)
    return router


def _replicas(*names: str) -> list[_Replica]:
    return [_Replica(ReplicaID(unique_id=name, deployment_id=APP)) for name in names]


async def _choose(router: AffinityRouter, request: PendingRequest) -> list[list]:
    return await router.choose_replicas(list(router.curr_replicas.values()), request)


async def _ranking(router: AffinityRouter, key: str) -> list[str]:
    groups = await _choose(router, _request(key))
    assert all(len(group) == 1 for group in groups)
    return [group[0].replica_id.unique_id for group in groups]


class TestKeyedRequests:
    @pytest.mark.asyncio
    async def test_a_key_ranks_every_replica_once(self):
        router = await _router(_replicas("a", "b", "c"))
        assert sorted(await _ranking(router, "conversation-1")) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_two_routers_rank_a_key_alike(self):
        first = await _router(_replicas("a", "b", "c"))
        second = await _router(_replicas("c", "a", "b"))
        for n in range(20):
            assert await _ranking(first, f"conversation-{n}") == await _ranking(second, f"conversation-{n}")

    @pytest.mark.asyncio
    async def test_keys_spread_over_the_replicas(self):
        router = await _router(_replicas("a", "b", "c"))
        firsts = [(await _ranking(router, f"conversation-{n}"))[0] for n in range(300)]
        assert all(firsts.count(name) > 70 for name in "abc")

    @pytest.mark.asyncio
    async def test_removing_a_replica_moves_only_its_keys(self):
        router = await _router(_replicas("a", "b", "c", "d"))
        before = {n: (await _ranking(router, f"conversation-{n}"))[0] for n in range(200)}
        router.update_replicas(_replicas("a", "b", "d"))
        after = {n: (await _ranking(router, f"conversation-{n}"))[0] for n in range(200)}

        assert all(after[n] == first for n, first in before.items() if first != "c")
        assert {after[n] for n, first in before.items() if first == "c"} == {"a", "b", "d"}

    @pytest.mark.asyncio
    async def test_a_replica_over_the_bound_goes_last(self):
        router = await _router(_replicas("a", "b", "c"))
        ranking = await _ranking(router, "conversation-1")
        router.replica_queue_len_cache.update(ReplicaID(unique_id=ranking[0], deployment_id=APP), 4)

        assert await _ranking(router, "conversation-1") == [*ranking[1:], ranking[0]]

    @pytest.mark.asyncio
    async def test_spare_requests_keep_a_quiet_clusters_first_choice(self):
        router = await _router(_replicas("a", "b", "c"))
        ranking = await _ranking(router, "conversation-1")
        router.replica_queue_len_cache.update(ReplicaID(unique_id=ranking[0], deployment_id=APP), 3)

        assert await _ranking(router, "conversation-1") == ranking

    @pytest.mark.asyncio
    async def test_a_keyed_request_backs_off_once_every_replica_is_tried(self):
        router = await _router(_replicas("a", "b"))
        request = _request("conversation-1")
        await _choose(router, request)
        assert request.routing_context.should_backoff is True


class TestRequestsWithoutAKey:
    @pytest.mark.asyncio
    async def test_get_two_different_random_replicas(self):
        router = await _router(_replicas("a", "b", "c"))
        [group] = await _choose(router, _request())
        assert len({r.replica_id for r in group}) == 2

    @pytest.mark.asyncio
    async def test_a_single_replica_is_the_only_choice(self):
        router = await _router(_replicas("a"))
        [group] = await _choose(router, _request())
        assert [r.replica_id.unique_id for r in group] == ["a"]


class _Recorder:
    def __init__(self):
        self.tags: list[dict] = []

    def inc(self, value=1.0, tags=None):
        self.tags.append(tags)


class TestRoutedCounter:
    @pytest.mark.asyncio
    async def test_counts_the_first_choice_a_spill_and_no_key(self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(metrics, "AFFINITY_ROUTED_TOTAL", recorder)
        router = await _router(_replicas("a", "b"))
        first, other = await _ranking(router, "conversation-1")

        for key, name in (("conversation-1", first), ("conversation-1", other), ("", first)):
            router.on_request_routed(_request(key), ReplicaID(unique_id=name, deployment_id=APP), None)

        assert recorder.tags == [{"model": "m", "choice": choice} for choice in ("first", "spilled", "no_key")]

    def test_the_router_serve_unpickles_counts_on_this_processes_metric(self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(metrics, "AFFINITY_ROUTED_TOTAL", recorder)
        config = ModelshipModelConfig(name="m", model="some-model", usecase="generate", loader="vllm")
        router_config = build_deployment_options(config)["request_router_config"]
        router_class = cloudpickle.loads(router_config.get_serialized_request_router_cls())
        router = router_class(deployment_id=APP, handle_source=DeploymentHandleSource.UNKNOWN)
        router.initialize_state()

        router.on_request_routed(_request(), ReplicaID(unique_id="a", deployment_id=APP), None)

        assert recorder.tags == [{"model": "m", "choice": "no_key"}]


class TestRayRouterApi:
    def test_a_handles_session_id_reaches_the_request_metadata(self):
        assert "session_id" in inspect.signature(DeploymentHandle.options).parameters
        options = DynamicHandleOptions().copy_and_update(session_id="conversation-1", stream=True)
        assert get_request_metadata(create_init_handle_options(), options).session_id == "conversation-1"

    def test_serve_backs_off_only_when_a_router_asks(self):
        assert RequestRoutingContext().should_backoff is False
        assert "routing_context.should_backoff" in inspect.getsource(RequestRouter._choose_replicas_with_backoff)

    def test_the_hooks_keep_their_shape(self):
        assert list(inspect.signature(RequestRouter.choose_replicas).parameters) == [
            "self",
            "candidate_replicas",
            "pending_request",
        ]
        assert list(inspect.signature(RequestRouter.on_request_routed).parameters) == [
            "self",
            "pending_request",
            "replica_id",
            "result",
        ]
