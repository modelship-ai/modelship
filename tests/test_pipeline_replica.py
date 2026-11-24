"""A pipeline's replica: the relay channel a session reads from and writes to, the assistant opening and ending
sessions on it, the model table it follows from the gateway coordinator, and the calls every deployment refuses
outside its usecase."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from modelship.deploy import gateway_coordinator
from modelship.infer import base_infer
from modelship.infer.deployment_base import DeploymentBase
from modelship.infer.pipeline_deployment import load_pipeline
from modelship.infer.pipelines import routes as routes_module
from modelship.infer.pipelines.assistant import assistant_pipeline
from modelship.infer.pipelines.assistant.assistant_pipeline import AssistantPipeline, RelayChannel
from modelship.infer.pipelines.relay import RealtimeClosed, RealtimeOpened, RealtimeOpening
from modelship.infer.pipelines.routes import ModelRoutes
from modelship.utils.config_schema import AssistantPipelineConfig

CHAT = AssistantPipelineConfig(name="chat", type="assistant", llm="qwen")


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class TestRelayChannel:
    @pytest.mark.asyncio
    async def test_the_clients_frames_are_read_in_order_and_none_once_it_left(self):
        channel = RelayChannel()
        await channel.push(["a", b"b"])
        await channel.push(["c"])
        channel.leave()
        assert [await channel.receive() for _ in range(4)] == ["a", b"b", "c", None]

    @pytest.mark.asyncio
    async def test_a_hand_over_waits_while_the_queue_is_over_its_cap(self, monkeypatch):
        monkeypatch.setattr(assistant_pipeline, "MAX_QUEUED_BYTES", 5)
        channel = RelayChannel()
        push = asyncio.ensure_future(channel.push(["abc", "def"]))
        await _settle()
        assert not push.done()
        assert await channel.receive() == "abc"
        await _settle()
        assert push.done()

    @pytest.mark.asyncio
    async def test_the_sessions_frames_end_with_the_close(self):
        channel = RelayChannel()
        session = asyncio.ensure_future(asyncio.Event().wait())
        await channel.send("one")
        await channel.close(1008)
        await channel.send("after")
        assert [frame async for frame in channel.frames(session)] == ["one", RealtimeClosed(1008)]
        session.cancel()

    @pytest.mark.asyncio
    async def test_the_sessions_frames_end_once_it_ends(self):
        channel = RelayChannel()

        async def session() -> None:
            await channel.send("last")

        assert [frame async for frame in channel.frames(asyncio.ensure_future(session()))] == ["last"]

    @pytest.mark.asyncio
    async def test_a_session_that_failed_closes_the_websocket_as_an_error(self):
        channel = RelayChannel()

        async def session() -> None:
            await channel.send("last")
            raise RuntimeError("the session broke")

        task = asyncio.ensure_future(session())
        assert [frame async for frame in channel.frames(task)] == ["last", RealtimeClosed(1011)]


class _Routes:
    def __init__(self, ready: bool = True) -> None:
        self._ready = ready

    async def ready(self) -> bool:
        return self._ready


class TestAssistantPipeline:
    @pytest.mark.asyncio
    async def test_a_session_it_doesnt_hold_takes_no_frames(self):
        pipeline = AssistantPipeline(CHAT, _Routes())  # type: ignore[arg-type]
        assert await pipeline.realtime_event("nope", ["{}"]) is False

    @pytest.mark.asyncio
    async def test_a_pipeline_without_its_model_table_refuses_the_session(self, monkeypatch):
        pipeline = AssistantPipeline(CHAT, _Routes(ready=False))  # type: ignore[arg-type]
        frames = [frame async for frame in pipeline.realtime(RealtimeOpening(key="k1"))]
        assert frames[0] == RealtimeOpened()
        assert json.loads(frames[1])["error"]["code"] == "model_not_ready"
        assert frames[2:] == [RealtimeClosed(1008)]

    @pytest.mark.asyncio
    async def test_a_session_runs_on_its_channel_until_its_client_leaves(self, monkeypatch):
        served: list = []

        async def serve(channel, config, calls) -> None:
            served.append(calls)
            await channel.send(await channel.receive())
            await channel.receive()

        monkeypatch.setattr(assistant_pipeline, "serve_session", serve)
        monkeypatch.setattr(assistant_pipeline, "get_disconnect_registry", lambda: "registry")
        pipeline = AssistantPipeline(CHAT, _Routes())  # type: ignore[arg-type]
        stream = pipeline.realtime(RealtimeOpening(key="k1", headers={"x-device": "kitchen"}, identity="caller"))
        assert await anext(stream) == RealtimeOpened()
        assert await pipeline.realtime_event("k1", ["hello"]) is True
        assert await anext(stream) == "hello"
        await pipeline.realtime_close("k1")
        assert [frame async for frame in stream] == []
        assert await pipeline.realtime_event("k1", ["late"]) is False
        assert served[0]._headers == {"x-device": "kitchen"}
        assert served[0]._scan is None

    def test_an_assistant_config_loads_the_assistant(self):
        assert isinstance(load_pipeline(CHAT, MagicMock()), AssistantPipeline)


class _Coordinator:
    """The gateway coordinator: one table, at generation 5 and routing seq 2."""

    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.polls: list[tuple] = []
        self.wait_for_change = MagicMock()
        self.wait_for_change.remote = self._wait
        self.get_routing = MagicMock()
        self.get_routing.remote = AsyncMock(return_value=snapshot)

    async def _wait(self, gateway, generation, *, replica_id, routing):
        self.polls.append((gateway, generation, replica_id, routing))
        if generation == self.snapshot["generation"]:
            await asyncio.Event().wait()
        return self.snapshot["generation"]


SNAPSHOT = {"models": {"gw.qwen-0123456789": "qwen"}, "expected": ["qwen", "kokoro"], "generation": 5, "routing": 2}


class TestModelRoutes:
    @pytest.mark.asyncio
    async def test_it_follows_the_gateways_table_and_reports_what_it_holds(self, monkeypatch):
        coordinator = _Coordinator(SNAPSHOT)
        monkeypatch.setattr(gateway_coordinator, "get_or_create_gateway_coordinator", lambda: coordinator)
        handle = MagicMock()
        monkeypatch.setattr(routes_module.serve, "get_app_handle", lambda app: handle)
        routes = ModelRoutes("gw", "p1")
        routes.start()
        assert await routes.ready()
        await _settle()
        assert routes.handle("qwen") is handle
        assert [routes.state(model) for model in ("qwen", "kokoro", "nope")] == ["serving", "starting", "unknown"]
        assert coordinator.polls == [("gw", 0, "p1", None), ("gw", 5, "p1", 2)]
        await routes.close()

    def test_an_app_not_reachable_yet_stays_out_and_the_table_is_asked_for_again(self, monkeypatch):
        def get_app_handle(app):
            raise RuntimeError("not yet")

        monkeypatch.setattr(routes_module.serve, "get_app_handle", get_app_handle)
        routes = ModelRoutes("gw", "p1")
        with pytest.raises(RuntimeError, match="not reachable yet"):
            routes._apply(SNAPSHOT)
        assert routes.handle("qwen") is None
        assert routes.state("qwen") == "starting"
        assert (routes._generation, routes._routing) == (0, None)


class _Base(DeploymentBase):
    def __init__(self) -> None:
        self.config = CHAT


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_model_call_outside_the_usecase_is_refused(self):
        base = _Base()
        assert [item async for item in base.generate(MagicMock(), {}, None)] == [base_infer._NOT_SUPPORTED]
        assert [item async for item in base.speak(MagicMock(), {}, None)] == [base_infer._NOT_SUPPORTED]
        assert await base.stt_capabilities() is None

    @pytest.mark.asyncio
    async def test_a_realtime_session_outside_the_usecase_is_refused(self):
        base = _Base()
        frames = [frame async for frame in base.realtime(RealtimeOpening(key="k1"))]
        assert json.loads(frames[0])["error"]["code"] == "model_not_supported"
        assert frames[1:] == [RealtimeClosed(1008)]
        assert await base.realtime_event("k1", ["{}"]) is False
