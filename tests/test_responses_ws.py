"""Tests for the `/v1/responses` WebSocket transport.

Drives `ModelshipAPI._run_ws_turn` directly with a fake `WebSocket` (records
`send_text` calls) and a mocked `handle.respond` generator — the same style
`test_responses_api.py` uses for the HTTP route, minus the FastAPI/Starlette
plumbing (`responses_ws` itself is a thin reader-task/auth wrapper around this).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect
from ray.exceptions import RayTaskError
from starlette.websockets import WebSocketState

from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import create_error_response
from modelship.state import MemoryStoreActor
from tests.conftest import build_gateway

_ModelshipAPI = ModelshipAPI.func_or_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


class _FakeWebSocket:
    """Records every frame sent — no real socket, no ASGI, no accept()/receive()."""

    def __init__(self):
        self.sent: list[str] = []
        self.client_state = WebSocketState.CONNECTED

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    @property
    def events(self) -> list[dict]:
        return [json.loads(s) for s in self.sent]

    @property
    def types(self) -> list[str]:
        return [e["type"] for e in self.events]


@pytest.fixture
def api():
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = build_gateway("test-gateway")
        inst._watch_task = MagicMock()
        inst._state_store = _MemoryStore()
        return inst


def _wire(api, model: str, gen):
    handle = MagicMock()
    handle.respond.options.return_value.remote.return_value = gen
    api.models = {model: {f"{model}-a1b2c": handle}}
    return handle


def _frame(**overrides) -> str:
    payload = {"type": "response.create", "model": "m", "input": "hi"}
    payload.update(overrides)
    return json.dumps(payload)


async def _events(*items):
    for item in items:
        yield item


def _terminal(response_id: str, **response_overrides) -> dict:
    response = {"id": response_id, "object": "response", "status": "completed", "output": []}
    response.update(response_overrides)
    return {"type": "response.completed", "sequence_number": 1, "response": response}


class TestWsTurns:
    @pytest.mark.asyncio
    async def test_single_turn_happy_path(self, api):
        gen = _events({"type": "response.created", "sequence_number": 0}, _terminal("resp_1"))
        handle = _wire(api, "m", gen)
        ws = _FakeWebSocket()
        conn_cache: dict = {}

        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), conn_cache, MagicMock(), "req-1")

        assert ws.types == ["response.created", "response.completed"]
        assert all(s != "data: [DONE]\n\n" and s != "[DONE]" for s in ws.sent)
        # dispatched request was forced to stream=True regardless of the wire body
        dispatched = handle.respond.options.return_value.remote.call_args.args[0]
        assert dispatched.stream is True
        # store:false turn caches locally, not in the global store
        assert "resp_1" in conn_cache
        assert api._state_store.get("responses/unscoped/resp_1") is None

    @pytest.mark.asyncio
    async def test_store_not_false_persists_to_global_store_not_conn_cache(self, api):
        gen = _events(_terminal("resp_2"))
        _wire(api, "m", gen)
        ws = _FakeWebSocket()
        conn_cache: dict = {}

        await api._run_ws_turn(ws, "unscoped", {}, _frame(), conn_cache, MagicMock(), "req-1")

        assert ws.types == ["response.completed"]
        assert "resp_2" not in conn_cache
        assert api._state_store.get("responses/unscoped/resp_2")["response"]["id"] == "resp_2"

    @pytest.mark.asyncio
    async def test_sequential_turns_are_independent(self, api):
        _wire(api, "m", _events(_terminal("resp_a")))
        ws = _FakeWebSocket()
        conn_cache: dict = {}
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), conn_cache, MagicMock(), "req-1")

        _wire(api, "m", _events(_terminal("resp_b")))
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), conn_cache, MagicMock(), "req-2")

        assert ws.types == ["response.completed", "response.completed"]
        assert set(conn_cache) == {"resp_a", "resp_b"}

    @pytest.mark.asyncio
    async def test_type_field_is_stripped_before_dispatch(self, api):
        # OpenAIBaseModel is extra="allow" — an un-popped "type" would otherwise
        # silently ride the Ray hop as a stray extra field.
        handle = _wire(api, "m", _events(_terminal("resp_1")))
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")
        dispatched = handle.respond.options.return_value.remote.call_args.args[0]
        assert not dispatched.model_extra

    @pytest.mark.asyncio
    async def test_compaction_input_item_passes_through_untouched(self, api):
        # The gateway does no input interpretation — a compaction-seeded turn is
        # dispatched exactly like any other.
        seed = [{"type": "compaction", "encrypted_content": "opaque-blob"}]
        handle = _wire(api, "m", _events(_terminal("resp_1")))
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(input=seed, store=False, tools=[]), {}, MagicMock(), "req-1")
        dispatched = handle.respond.options.return_value.remote.call_args.args[0]
        assert dispatched.input == seed
        assert ws.types == ["response.completed"]

    @pytest.mark.asyncio
    async def test_mcp_tool_routes_through_mcp_loop_not_handle_respond(self, api):
        handle = _wire(api, "m", _events())  # unused if wiring is correct
        ws = _FakeWebSocket()

        async def fake_mcp_gen(*args, **kwargs):
            yield {"type": "response.created", "sequence_number": 0}
            yield _terminal("resp_mcp")

        frame = _frame(tools=[{"type": "mcp", "server_label": "s", "server_url": "http://x"}], store=False)
        with patch("modelship.openai.api.mcp_loop.run_mcp_response", side_effect=fake_mcp_gen) as run_mcp:
            await api._run_ws_turn(ws, "unscoped", {}, frame, {}, MagicMock(), "req-1")

        run_mcp.assert_called_once()
        handle.respond.options.return_value.remote.assert_not_called()
        assert ws.types == ["response.created", "response.completed"]


class TestFrameValidation:
    @pytest.mark.asyncio
    async def test_invalid_json_yields_error_frame(self, api):
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, "{not json", {}, MagicMock(), "req-1")
        assert ws.types == ["error"]
        assert ws.events[0]["status"] == 400

    @pytest.mark.asyncio
    async def test_wrong_type_field_yields_error_frame(self, api):
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, json.dumps({"type": "ping"}), {}, MagicMock(), "req-1")
        assert ws.types == ["error"]

    @pytest.mark.asyncio
    async def test_invalid_request_body_yields_error_frame(self, api):
        ws = _FakeWebSocket()
        raw = _frame(reasoning={"effort": "turbo"})
        await api._run_ws_turn(ws, "unscoped", {}, raw, {}, MagicMock(), "req-1")
        assert ws.types == ["error"]
        assert ws.events[0]["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_unknown_model_yields_error_frame(self, api, request_outcomes):
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(model="nope"), {}, MagicMock(), "req-1")
        assert ws.types == ["error"]
        assert ws.events[0]["status"] == 404
        assert request_outcomes() == [("", "create_response_ws", "error", "model_not_found")]

    @pytest.mark.asyncio
    async def test_an_expected_model_with_nothing_routed_yields_a_503_frame(self, api):
        api.expected_models = ["m"]
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(model="m"), {}, MagicMock(), "req-1")
        assert ws.events[0]["status"] == 503
        assert ws.events[0]["error"]["type"] == "api_error"


class TestPreviousResponseIdResolution:
    @pytest.mark.asyncio
    async def test_unknown_previous_response_id_sends_not_found_error_frame(self, api, request_outcomes):
        handle = _wire(api, "m", _events(_terminal("resp_1")))
        ws = _FakeWebSocket()

        await api._run_ws_turn(ws, "unscoped", {}, _frame(previous_response_id="resp_nope"), {}, MagicMock(), "req-1")

        assert ws.types == ["error"]
        assert ws.events[0]["error"]["code"] == "previous_response_not_found"
        handle.respond.options.return_value.remote.assert_not_called()
        assert request_outcomes() == [("m", "create_response_ws", "error", "http_404")]

    @pytest.mark.asyncio
    async def test_continuation_resolves_from_connection_local_cache(self, api):
        _wire(
            api, "m", _events(_terminal("resp_1", output=[{"type": "message", "role": "assistant", "content": "hey"}]))
        )
        ws = _FakeWebSocket()
        conn_cache: dict = {}
        await api._run_ws_turn(
            ws, "unscoped", {}, _frame(input="turn one", store=False), conn_cache, MagicMock(), "req-1"
        )
        assert "resp_1" in conn_cache

        handle2 = _wire(api, "m", _events(_terminal("resp_2")))
        await api._run_ws_turn(
            ws,
            "unscoped",
            {},
            _frame(input="turn two", previous_response_id="resp_1", store=False),
            conn_cache,
            MagicMock(),
            "req-2",
        )

        dispatched = handle2.respond.options.return_value.remote.call_args.args[0]
        assert dispatched.input == [
            {"type": "message", "role": "user", "content": "turn one"},
            {"type": "message", "role": "assistant", "content": "hey"},
            {"type": "message", "role": "user", "content": "turn two"},
        ]
        assert ws.types == ["response.completed", "response.completed"]
        assert "resp_2" in conn_cache

    @pytest.mark.asyncio
    async def test_reconnect_on_a_fresh_socket_misses_a_store_false_response(self, api):
        # Socket B is a different connection with its own, empty, conn_cache — a miss
        # even though resp_1 is real on socket A.
        _wire(api, "m", _events(_terminal("resp_1")))
        ws_a = _FakeWebSocket()
        conn_cache_a: dict = {}
        await api._run_ws_turn(ws_a, "unscoped", {}, _frame(store=False), conn_cache_a, MagicMock(), "req-1")
        assert "resp_1" in conn_cache_a

        handle_b = _wire(api, "m", _events(_terminal("resp_2")))
        ws_b = _FakeWebSocket()
        conn_cache_b: dict = {}
        await api._run_ws_turn(
            ws_b, "unscoped", {}, _frame(previous_response_id="resp_1"), conn_cache_b, MagicMock(), "req-2"
        )

        assert ws_b.types == ["error"]
        assert ws_b.events[0]["error"]["code"] == "previous_response_not_found"
        handle_b.respond.options.return_value.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_continuation_evicts_the_cached_previous_response_id(self, api):
        _wire(api, "m", _events(_terminal("resp_1")))
        ws = _FakeWebSocket()
        conn_cache: dict = {}
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), conn_cache, MagicMock(), "req-1")
        assert "resp_1" in conn_cache

        # Turn 2 continues resp_1 (a cache hit) but the loader fails it.
        failing_gen = _events(create_error_response("orphaned function_call_output", status_code=400))
        _wire(api, "m", failing_gen)
        await api._run_ws_turn(
            ws,
            "unscoped",
            {},
            _frame(previous_response_id="resp_1", store=False),
            conn_cache,
            MagicMock(),
            "req-2",
        )
        assert ws.types == ["response.completed", "error"]
        assert "resp_1" not in conn_cache

        # Turn 3 retries the same previous_response_id — now a clean miss, not a
        # replay of whatever broke turn 2.
        handle3 = _wire(api, "m", _events(_terminal("resp_3")))
        await api._run_ws_turn(
            ws,
            "unscoped",
            {},
            _frame(previous_response_id="resp_1", store=False),
            conn_cache,
            MagicMock(),
            "req-3",
        )
        assert ws.types[-1] == "error"
        assert ws.events[-1]["error"]["code"] == "previous_response_not_found"
        handle3.respond.options.return_value.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_store_false_continuation_gets_the_mcp_tool_list_of_an_earlier_turn(self, api):
        tool = {"type": "mcp", "server_label": "dice", "server_url": "http://dice/mcp", "require_approval": "never"}
        listing = {"type": "mcp_list_tools", "server_label": "dice", "tools": [{"name": "roll", "input_schema": {}}]}
        received = []

        def run_mcp_response(*args, mcp_listings, **kwargs):
            received.append(mcp_listings)
            output = [listing] if len(received) == 1 else []
            return _events(_terminal(f"resp_{len(received)}", output=output, tools=[tool]))

        _wire(api, "m", _events())
        ws = _FakeWebSocket()
        conn_cache: dict = {}
        with patch("modelship.openai.api.mcp_loop.run_mcp_response", side_effect=run_mcp_response):
            for n, previous in enumerate([None, "resp_1", "resp_2"]):
                frame = _frame(store=False, tools=[tool], previous_response_id=previous)
                await api._run_ws_turn(ws, "unscoped", {}, frame, conn_cache, MagicMock(), f"req-{n}")

        stored = {"dice": {"server_url": "http://dice/mcp", "allowed_tools": None, "tools": listing["tools"]}}
        assert received == [{}, stored, stored]


class TestMidStreamFailures:
    @pytest.mark.asyncio
    async def test_raytaskerror_with_value_error_cause_yields_400_error_frame(self, api, request_outcomes):
        class _FakeValidationError(ValueError):
            def __init__(self, message: str, parameter: str) -> None:
                super().__init__(message)
                self.parameter = parameter

        cause = _FakeValidationError("This model's maximum context length is 14512 tokens.", "input_tokens")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def gen():
            if False:
                yield  # pragma: no cover
            raise err

        _wire(api, "m", gen())
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert ws.types == ["error"]
        assert ws.events[0]["error"]["param"] == "input_tokens"
        assert "maximum context length" in ws.events[0]["error"]["message"]
        assert request_outcomes() == [("m", "create_response_ws", "error", "validation_error")]

    @pytest.mark.asyncio
    async def test_error_response_mid_generator_yields_error_frame_and_does_not_raise(self, api, request_outcomes):
        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            yield create_error_response("boom", status_code=500, err_type="api_error")

        _wire(api, "m", gen())
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert ws.types == ["response.created", "error"]
        assert request_outcomes() == [("m", "create_response_ws", "error", "inference_error")]

    @pytest.mark.asyncio
    async def test_unhandled_exception_yields_generic_500_error_frame(self, api, request_outcomes):
        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            raise RuntimeError("engine blew up")

        _wire(api, "m", gen())
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert ws.types == ["response.created", "error"]
        assert ws.events[-1]["status"] == 500
        assert request_outcomes() == [("m", "create_response_ws", "error", "unhandled")]

    @pytest.mark.asyncio
    async def test_a_failed_event_is_recorded_as_a_stream_error(self, api, request_outcomes):
        _wire(api, "m", _events({"type": "response.failed", "sequence_number": 0, "response": {"status": "failed"}}))
        ws = _FakeWebSocket()
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert ws.types == ["response.failed"]
        assert request_outcomes() == [("m", "create_response_ws", "error", "stream_error")]


class TestClientLeaving:
    @pytest.mark.asyncio
    async def test_a_turn_whose_client_left_is_recorded_once_as_cancelled(self, api, request_outcomes):
        ws = _FakeWebSocket()

        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            ws.client_state = WebSocketState.DISCONNECTED

        _wire(api, "m", gen())
        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert request_outcomes() == [("m", "create_response_ws", "cancelled", None)]

    @pytest.mark.asyncio
    async def test_a_turn_whose_send_fails_is_recorded_once_as_cancelled(self, api, request_outcomes):
        ws = _FakeWebSocket()
        ws.send_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
        _wire(api, "m", _events({"type": "response.created", "sequence_number": 0}, _terminal("resp_1")))

        await api._run_ws_turn(ws, "unscoped", {}, _frame(store=False), {}, MagicMock(), "req-1")

        assert request_outcomes() == [("m", "create_response_ws", "cancelled", None)]
