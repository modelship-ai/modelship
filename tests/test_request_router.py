import asyncio
import inspect
from dataclasses import dataclass

import pytest
from ray import cloudpickle
from ray.serve._private.common import DeploymentHandleSource, DeploymentID, ReplicaID, RequestMetadata
from ray.serve._private.default_impl import create_init_handle_options, get_request_metadata
from ray.serve._private.handle_options import DynamicHandleOptions
from ray.serve._private.request_router.common import PendingRequest, RequestRoutingContext
from ray.serve._private.router import AsyncioRouter
from ray.serve.handle import DeploymentHandle
from ray.serve.request_router import RequestRouter
from ray.serve.schema import LoggingConfig

from modelship import metrics
from modelship.deploy import strategy
from modelship.deploy.actor_options import PIPELINE_MAX_ONGOING_REQUESTS, build_pipeline_deployment_options
from modelship.deploy.request_router import AffinityRouter, Routing, StickyRouter
from modelship.infer.deployment_base import DeploymentBase
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.infer.pipeline_deployment import PipelineDeployment
from modelship.utils.config_schema import AssistantPipelineConfig

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

    @pytest.mark.asyncio
    async def test_an_unkeyed_request_backs_off_once_every_replica_is_tried(self):
        router = await _router(_replicas("a", "b"))
        request = _request()
        await _choose(router, request)
        assert request.routing_context.should_backoff is True


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
        router_config = ModelDeployment.func_or_class.routing(config).config()
        router_class = cloudpickle.loads(router_config.get_serialized_request_router_cls())
        router = router_class(deployment_id=APP, handle_source=DeploymentHandleSource.UNKNOWN)
        router.initialize_state()

        router.on_request_routed(_request(), ReplicaID(unique_id="a", deployment_id=APP), None)

        assert recorder.tags == [{"model": "m", "choice": "no_key"}]


PIPELINE_APP = DeploymentID(name="gw.voice-p0123456789", app_name="gw.voice-p0123456789")
PIPELINE = AssistantPipelineConfig(name="voice", type="assistant", assistant_config={"stt": "whisper"})
MODEL = ModelshipModelConfig(name="m", model="some-model", usecase="generate", loader="vllm")
STT = ModelshipModelConfig(name="parakeet", model="some-model", usecase="transcription", loader="whispercpp")


def _call(key: str = "", method: str = "realtime_event") -> PendingRequest:
    return PendingRequest(
        args=[],
        kwargs={},
        metadata=RequestMetadata(request_id="r", internal_request_id="r", session_id=key, call_method=method),
    )


def _pipeline_replicas(*names: str, node: str = "node-1") -> list[_Replica]:
    return [_Replica(ReplicaID(unique_id=name, deployment_id=PIPELINE_APP), node_id=node) for name in names]


async def _sticky(replicas: list[_Replica], routing: Routing | None = None, **locality) -> StickyRouter:
    router = StickyRouter(
        deployment_id=PIPELINE_APP,
        handle_source=DeploymentHandleSource.UNKNOWN,
        use_replica_queue_len_cache=True,
        **locality,
    )
    routing = routing or PipelineDeployment.func_or_class.routing(PIPELINE)
    router.initialize_state(**routing.config().request_router_kwargs)
    router.update_replicas(replicas)
    while any(router.replica_queue_len_cache.get(r.replica_id) is None for r in replicas):
        await asyncio.sleep(0)
    return router


async def _pick(router: StickyRouter, request: PendingRequest) -> str:
    [[replica]] = await router.choose_replicas(list(router.curr_replicas.values()), request)
    return replica.replica_id.unique_id


class TestStickyRouter:
    @pytest.mark.asyncio
    async def test_every_call_of_a_session_goes_to_the_replica_of_its_first(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        first = await _pick(router, _call("s1", "realtime"))
        router.replica_queue_len_cache.update(ReplicaID(unique_id=first, deployment_id=PIPELINE_APP), 50)
        assert {await _pick(router, _call("s1")) for _ in range(10)} == {first}

    @pytest.mark.asyncio
    async def test_a_session_waits_for_its_busy_replica(self):
        router = await _sticky(_pipeline_replicas("a", "b"))
        await _pick(router, _call("s1", "realtime"))
        request = _call("s1")
        await _pick(router, request)
        assert request.routing_context.should_backoff is True

    @pytest.mark.asyncio
    async def test_a_new_session_goes_to_a_replica_holding_the_fewest(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        picks = [await _pick(router, _call(f"s{n}", "realtime")) for n in range(6)]
        assert sorted(picks) == ["a", "a", "b", "b", "c", "c"]

    @pytest.mark.asyncio
    async def test_adding_replicas_moves_no_session(self):
        router = await _sticky(_pipeline_replicas("a", "b"))
        before = {n: await _pick(router, _call(f"s{n}", "realtime")) for n in range(4)}
        router.update_replicas(_pipeline_replicas("a", "b", "c", "d"))
        assert {n: await _pick(router, _call(f"s{n}")) for n in range(4)} == before

    @pytest.mark.asyncio
    async def test_a_session_whose_replica_is_gone_gets_another(self):
        router = await _sticky(_pipeline_replicas("a", "b"))
        first = await _pick(router, _call("s1", "realtime"))
        router.update_replicas([r for r in _pipeline_replicas("a", "b") if r.replica_id.unique_id != first])
        moved = await _pick(router, _call("s1"))
        assert moved != first
        assert await _pick(router, _call("s1")) == moved

    @pytest.mark.asyncio
    async def test_ending_a_session_unpins_it(self):
        router = await _sticky(_pipeline_replicas("a", "b"))
        first = await _pick(router, _call("s1", "realtime"))
        replica = ReplicaID(unique_id=first, deployment_id=PIPELINE_APP)
        router.on_request_routed(_call("s1"), replica, None)  # type: ignore[arg-type]
        assert "s1" in router._pins
        router.on_request_routed(_call("s1", "realtime_close"), replica, None)  # type: ignore[arg-type]
        assert "s1" not in router._pins

    @pytest.mark.asyncio
    async def test_an_unpinning_call_goes_to_its_sessions_replica(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        first = await _pick(router, _call("s1", "realtime"))
        router.replica_queue_len_cache.update(ReplicaID(unique_id=first, deployment_id=PIPELINE_APP), 50)
        assert {await _pick(router, _call("s1", "realtime_close")) for _ in range(10)} == {first}

    @pytest.mark.asyncio
    async def test_an_unpinning_call_of_an_unpinned_session_pins_nothing(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        [group] = await router.choose_replicas(list(router.curr_replicas.values()), _call("s1", "realtime_close"))
        assert len({r.replica_id for r in group}) == 2
        assert router._pins == {}

    @pytest.mark.asyncio
    async def test_a_call_to_another_method_goes_to_any_replica_and_pins_nothing(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        await _pick(router, _call("s1", "realtime"))
        [group] = await router.choose_replicas(list(router.curr_replicas.values()), _call("s1", "stt_capabilities"))
        assert len({r.replica_id for r in group}) == 2
        [other] = await router.choose_replicas(list(router.curr_replicas.values()), _call("s2", "stt_capabilities"))
        assert len(other) == 2
        assert list(router._pins) == ["s1"]

    @pytest.mark.asyncio
    async def test_a_first_call_prefers_a_replica_on_the_callers_node(self):
        replicas = [*_pipeline_replicas("far1", "far2", node="node-2"), *_pipeline_replicas("near", node="node-1")]
        router = await _sticky(replicas, self_node_id="node-1", prefer_local_node_routing=True)
        assert {await _pick(router, _call(f"s{n}", "realtime")) for n in range(4)} == {"near"}

    @pytest.mark.asyncio
    async def test_a_call_without_a_session_goes_to_any_replica(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"))
        [group] = await router.choose_replicas(list(router.curr_replicas.values()), _call())
        assert len({r.replica_id for r in group}) == 2
        assert router._pins == {}

    def test_a_pipelines_app_routes_with_it_from_the_class_serve_unpickles(self):
        assert build_pipeline_deployment_options(PIPELINE, {})["max_ongoing_requests"] == PIPELINE_MAX_ONGOING_REQUESTS
        router_config = PipelineDeployment.func_or_class.routing(PIPELINE).config()
        router_class = cloudpickle.loads(router_config.get_serialized_request_router_cls())
        router = router_class(deployment_id=PIPELINE_APP, handle_source=DeploymentHandleSource.UNKNOWN)
        router.initialize_state(**router_config.request_router_kwargs)
        assert router._pin_methods == {"realtime", "realtime_event"}
        assert router._unpin_methods == {"realtime_close"}


class TestStreamedTranscriptionRouting:
    @pytest.mark.asyncio
    async def test_a_turns_pushes_and_finish_go_to_the_replica_of_its_first_push(self):
        router = await _sticky(_pipeline_replicas("a", "b", "c"), ModelDeployment.func_or_class.routing(STT))
        first = await _pick(router, _call("s1:t1", "transcribe_push"))
        router.replica_queue_len_cache.update(ReplicaID(unique_id=first, deployment_id=PIPELINE_APP), 50)
        assert {await _pick(router, _call("s1:t1", "transcribe_push")) for _ in range(5)} == {first}
        assert await _pick(router, _call("s1:t1", "transcribe_finish")) == first

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["transcribe_finish", "transcribe_close"])
    async def test_ending_a_turn_unpins_it(self, method):
        router = await _sticky(_pipeline_replicas("a", "b"), ModelDeployment.func_or_class.routing(STT))
        first = await _pick(router, _call("s1:t1", "transcribe_push"))
        router.on_request_routed(_call("s1:t1", method), ReplicaID(unique_id=first, deployment_id=PIPELINE_APP), None)  # type: ignore[arg-type]
        assert router._pins == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["transcribe", "stt_capabilities", "transcribe_finish"])
    async def test_a_call_that_opens_no_stream_pins_nothing(self, method):
        router = await _sticky(_pipeline_replicas("a", "b", "c"), ModelDeployment.func_or_class.routing(STT))
        [group] = await router.choose_replicas(list(router.curr_replicas.values()), _call("s1", method))
        assert len({r.replica_id for r in group}) == 2
        assert router._pins == {}


class TestDeploymentRouting:
    @pytest.mark.parametrize(
        "fields",
        [
            {"model": "kokoro-en-v0_19", "usecase": "tts", "loader": "sherpa_onnx"},
            {"model": "some-model", "usecase": "generate", "loader": "llama_server"},
            {"model": "some-model", "usecase": "embed", "loader": "vllm"},
            {
                "model": "some-model",
                "usecase": "generate",
                "loader": "vllm",
                "num_gpus": 2,
                "vllm_engine_kwargs": {"tensor_parallel_size": 2},
            },
        ],
    )
    def test_a_model_routes_with_the_affinity_router(self, fields):
        config = ModelshipModelConfig(name="test-model", **fields)
        router_config = ModelDeployment.func_or_class.routing(config).config()
        assert router_config.request_router_class == "modelship.deploy.request_router.AffinityRouter"
        assert router_config.request_router_kwargs == {}

    @pytest.mark.parametrize("loader", ["whispercpp", "sherpa_onnx", "vllm"])
    def test_a_transcription_model_pins_its_streamed_transcriptions(self, loader):
        assert ModelDeployment.func_or_class.routing(STT) == ModelDeployment.func_or_class.routing(
            STT.model_copy(update={"loader": loader})
        )
        assert ModelDeployment.func_or_class.routing(STT) == Routing(
            StickyRouter, pins=("transcribe_push",), unpins=("transcribe_finish", "transcribe_close")
        )

    def test_a_pipeline_routes_with_the_sticky_router(self):
        assert PipelineDeployment.func_or_class.routing(PIPELINE) == Routing(
            StickyRouter, pins=("realtime", "realtime_event"), unpins=("realtime_close",)
        )

    @pytest.mark.parametrize(
        ("deployment", "config"), [(ModelDeployment, STT), (ModelDeployment, MODEL), (PipelineDeployment, PIPELINE)]
    )
    def test_every_method_a_routing_names_is_a_handle_method(self, deployment, config):
        routing = deployment.func_or_class.routing(config)
        for method in (*routing.pins, *routing.unpins):
            assert callable(getattr(DeploymentBase, method))

    def test_each_app_is_submitted_with_its_deployments_routing(self, monkeypatch):
        submitted = []
        monkeypatch.setattr(strategy.serve, "run_many", lambda targets, **_: submitted.extend(targets))
        strategy._run(MODEL, "gw.m-0123456789", {}, LoggingConfig())
        strategy._run_pipeline(PIPELINE, "gw.voice-p0123456789", {}, LoggingConfig())

        routers = [t.target._bound_deployment._deployment_config.request_router_config for t in submitted]
        assert routers == [
            ModelDeployment.func_or_class.routing(MODEL).config(),
            PipelineDeployment.func_or_class.routing(PIPELINE).config(),
        ]


class TestRayRouterApi:
    def test_a_calls_method_name_reaches_the_request_metadata(self):
        options = DynamicHandleOptions().copy_and_update(method_name="realtime_close", session_id="s1")
        assert get_request_metadata(create_init_handle_options(), options).call_method == "realtime_close"

    def test_a_routers_kwargs_reach_its_initialize_state(self):
        source = inspect.getsource(AsyncioRouter.request_router.fget)  # type: ignore[attr-defined]
        assert "initialize_state(**(self._request_router_kwargs))" in source

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
