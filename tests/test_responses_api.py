"""Route-level tests for /v1/responses, including conversation state.

The route hands `ResponsesRequest` straight to `handle.respond` with no
chat<->Responses translation, threading the result through `_handle_response`
like `create_chat_completion`'s route. Feature-support validation (e.g.
rejecting `background`) happens inside the deployment's `create_response`,
surfacing here as a leading `ErrorResponse` from the handle.

State is the exception: the gateway resolves `previous_response_id` into
`input` before the Ray hop and stores the result on the way out. GET/DELETE
carry no model, so they can't be routed to a deployment.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from modelship.openai.api import ModelshipAPI, _error_response
from modelship.openai.protocol import ResponsesRequest, create_error_response
from modelship.openai.protocol.responses import ResponseObject, ResponseOutputMessage, ResponseOutputText, ResponseUsage
from modelship.openai.utils.responses import ResponsesApiError
from modelship.state import MemoryStoreActor, StateStoreUnavailableError
from tests.conftest import build_gateway

_ModelshipAPI = ModelshipAPI.func_or_class

# The plain class behind @ray.remote — a real store with no cluster (the actor-backed
# MemoryStateStore the gateway builds by default needs an initialized Ray).
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture
def api():
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        # See test_api.py's api fixture: stub configure_logging (via the cloudpickled
        # class's globals) so gateway instantiation doesn't leak global logging state.
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = build_gateway("test-gateway")
        # Tests set api.models directly; mark the watch loop started so the routing
        # accessors (_get_handle) don't try to reconcile from a coordinator.
        inst._watch_task = MagicMock()
        inst._state_store = _MemoryStore()
        return inst


def _raw_request():
    # headers={} resolves to the shared "unscoped" identity (auth.resolve_identity),
    # stable across requests — so two calls in one test share a conversation scope.
    raw = MagicMock()
    raw.headers = {}
    return raw


def _stored(api, response_id: str = "resp_1", identity: str = "unscoped", output=None, input_items=None):
    """Seed a stored snapshot the way a completed turn would have left it."""
    api._state_store.set(
        f"responses/{identity}/{response_id}",
        {
            "response": {"id": response_id, "object": "response", "status": "completed", "output": output or []},
            "input_items": input_items if input_items is not None else [],
        },
    )


class TestResponsesRoute:
    @pytest.mark.asyncio
    async def test_dispatches_original_request_to_respond(self, api):
        handle = MagicMock()
        remote = handle.respond.options.return_value.remote
        api.models = {"m": {"m-a1b2c": handle}}

        request = ResponsesRequest(model="m", input="hi", instructions="be terse")

        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")) as hr:
            result = await api.create_response(request, _raw_request())

        assert result == "OK"
        # No gateway-side translation: the original ResponsesRequest is handed
        # straight to the deployment, which owns chat<->Responses shaping now.
        assert remote.call_args.args[0] is request
        hr.assert_awaited_once()
        # endpoint label flows through to _handle_response for metrics.
        assert hr.call_args.args[3] == "create_response"

    @pytest.mark.asyncio
    async def test_stream_true_dispatches_and_returns_event_stream(self, api):
        # The deployment (BaseInfer.create_response) produces Responses event
        # dicts directly; the route SSE-frames them via streaming.frame_sse.
        handle = MagicMock()

        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            yield {"type": "response.completed", "sequence_number": 1, "response": {}}

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}

        request = ResponsesRequest(model="m", input="hi", stream=True)
        result = await api.create_response(request, _raw_request())

        assert result.media_type == "text/event-stream"
        assert handle.respond.options.return_value.remote.call_args.args[0] is request

        body = "".join([chunk async for chunk in result.body_iterator])
        assert "event: response.created" in body
        assert "event: response.completed" in body

    @pytest.mark.asyncio
    async def test_invalid_param_returns_400_not_500(self, api):
        # reasoning.effort validation fails in the deployment as a pydantic
        # ValidationError; that must surface as 400, not 500.
        handle = MagicMock()

        async def gen():
            yield create_error_response("bad reasoning.effort value", err_type="invalid_request_error")

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}

        request = ResponsesRequest(model="m", input="hi", reasoning={"effort": "turbo"})
        result = await api.create_response(request, _raw_request())
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_end_to_end_through_handle_response(self, api):
        # Drives the real _handle_response with a mock generator yielding an
        # already-built ResponseObject.
        handle = MagicMock()

        async def gen():
            yield ResponseObject(
                model="m",
                output=[ResponseOutputMessage(content=[ResponseOutputText(text="hello!")])],
                usage=ResponseUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            )

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}

        request = ResponsesRequest(model="m", input="hi")
        result = await api.create_response(request, _raw_request())

        body = json.loads(bytes(result.body))
        assert body["object"] == "response"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "hello!"
        assert body["usage"]["input_tokens"] == 1


class TestMcpRouting:
    """Wiring only — behavior is covered by test_mcp_loop.py. Confirms create_response
    swaps to mcp_loop.run_mcp_response for mcp tools, leaving handle.respond untouched."""

    @pytest.mark.asyncio
    async def test_mcp_tool_routes_through_mcp_loop_not_handle_respond(self, api):
        handle = MagicMock()
        api.models = {"m": {"m-a1b2c": handle}}

        async def fake_mcp_gen():
            yield ResponseObject(
                model="m", output=[ResponseOutputMessage(content=[ResponseOutputText(text="via mcp loop")])]
            )

        request = ResponsesRequest(
            model="m", input="hi", tools=[{"type": "mcp", "server_label": "s", "server_url": "http://x"}]
        )
        with patch("modelship.openai.api.mcp_loop.run_mcp_response", return_value=fake_mcp_gen()) as run_mcp:
            result = await api.create_response(request, _raw_request())

        run_mcp.assert_called_once()
        handle.respond.options.return_value.remote.assert_not_called()
        body = json.loads(bytes(result.body))
        assert body["output"][0]["content"][0]["text"] == "via mcp loop"

    @pytest.mark.asyncio
    async def test_no_mcp_tool_uses_handle_respond_as_before(self, api):
        handle = MagicMock()

        async def gen():
            yield ResponseObject(model="m", output=[])

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}

        request = ResponsesRequest(model="m", input="hi")
        with patch("modelship.openai.api.mcp_loop.run_mcp_response") as run_mcp:
            await api.create_response(request, _raw_request())

        run_mcp.assert_not_called()
        handle.respond.options.return_value.remote.assert_called_once()


def _response_gen(response_id="resp_new", text="hello!"):
    async def gen():
        yield ResponseObject(
            id=response_id,
            model="m",
            output=[ResponseOutputMessage(content=[ResponseOutputText(text=text)])],
        )

    return gen()


def _wire(api, gen):
    handle = MagicMock()
    handle.respond.options.return_value.remote.return_value = gen
    api.models = {"m": {"m-a1b2c": handle}}
    return handle


class TestPreviousResponseIdResolution:
    """History is resolved gateway-side, before the Ray hop — the loader only ever
    sees a flat `input`."""

    @pytest.mark.asyncio
    async def test_history_is_prepended_to_input_before_dispatch(self, api):
        _stored(
            api,
            "resp_1",
            output=[{"type": "message", "role": "assistant", "content": "your name is Alex"}],
            input_items=[{"type": "message", "role": "user", "content": "my name is Alex"}],
        )
        handle = _wire(api, _response_gen())

        request = ResponsesRequest(model="m", input="what is my name?", previous_response_id="resp_1")
        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")):
            await api.create_response(request, _raw_request())

        dispatched = handle.respond.options.return_value.remote.call_args.args[0]
        assert dispatched.input == [
            {"type": "message", "role": "user", "content": "my name is Alex"},
            {"type": "message", "role": "assistant", "content": "your name is Alex"},
            {"type": "message", "role": "user", "content": "what is my name?"},
        ]

    @pytest.mark.asyncio
    async def test_unknown_previous_response_id_is_404(self, api):
        _wire(api, _response_gen())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_nope")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 404
        # ResponsesApiError carries a machine-readable code; _error_response renders
        # the OpenAI envelope instead of FastAPI's default {"detail": ...} body.
        assert isinstance(exc.value, ResponsesApiError)
        assert exc.value.err.error.code == "previous_response_not_found"
        assert exc.value.err.error.param == "previous_response_id"
        body = json.loads(bytes(_error_response(exc.value.err).body))
        assert body["error"]["code"] == "previous_response_not_found"

    @pytest.mark.asyncio
    async def test_malformed_previous_response_id_is_404(self, api):
        # Never a key we could have written, so it can't resolve — 404 before any lookup.
        _wire(api, _response_gen())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="../../effective/test-gateway")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_another_identity_cannot_continue_the_conversation(self, api):
        _stored(api, "resp_1", identity="someone-else")
        _wire(api, _response_gen())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_store_outage_is_503_not_404(self, api):
        # An outage must never look like a legitimately unknown id — that would tell a
        # client to start over and silently lose the conversation.
        _stored(api, "resp_1")
        _wire(api, _response_gen())
        api._state_store = MagicMock()
        api._state_store.get_async = AsyncMock(side_effect=StateStoreUnavailableError("down"))

        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")
        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_no_previous_response_id_does_not_touch_the_store(self, api):
        api._state_store = MagicMock()
        api._state_store.get_async = AsyncMock(side_effect=AssertionError("store must not be read"))
        api._state_store.set_async = AsyncMock()
        _wire(api, _response_gen())

        request = ResponsesRequest(model="m", input="hi")
        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")):
            await api.create_response(request, _raw_request())


class TestPersistence:
    @pytest.mark.asyncio
    async def test_non_streaming_response_is_stored(self, api):
        _wire(api, _response_gen(response_id="resp_new"))
        request = ResponsesRequest(model="m", input="hi")

        await api.create_response(request, _raw_request())

        snap = api._state_store.get("responses/unscoped/resp_new")
        assert snap["response"]["id"] == "resp_new"
        assert snap["input_items"] == [{"type": "message", "role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_store_defaults_to_true_when_unset(self, api):
        # A client following OpenAI's default flow never sends `store`, then expects
        # previous_response_id to work on the next turn.
        _wire(api, _response_gen(response_id="resp_default"))
        await api.create_response(ResponsesRequest(model="m", input="hi"), _raw_request())
        assert api._state_store.get("responses/unscoped/resp_default") is not None

    @pytest.mark.asyncio
    async def test_explicit_store_false_is_not_stored(self, api):
        _wire(api, _response_gen(response_id="resp_nostore"))
        await api.create_response(ResponsesRequest(model="m", input="hi", store=False), _raw_request())
        assert api._state_store.get("responses/unscoped/resp_nostore") is None

    @pytest.mark.asyncio
    async def test_stored_snapshot_accumulates_across_turns(self, api):
        _stored(
            api,
            "resp_1",
            output=[{"type": "message", "role": "assistant", "content": "hi back"}],
            input_items=[{"type": "message", "role": "user", "content": "hi"}],
        )
        _wire(api, _response_gen(response_id="resp_2"))

        request = ResponsesRequest(model="m", input="again", previous_response_id="resp_1")
        await api.create_response(request, _raw_request())

        # Turn 2's snapshot embeds turn 1's, for O(1) reads.
        snap = api._state_store.get("responses/unscoped/resp_2")
        assert snap["input_items"] == [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "hi back"},
            {"type": "message", "role": "user", "content": "again"},
        ]

    @pytest.mark.asyncio
    async def test_non_streaming_store_failure_is_503(self, api):
        _wire(api, _response_gen())
        api._state_store = MagicMock()
        api._state_store.set_async = AsyncMock(side_effect=StateStoreUnavailableError("down"))

        result = await api.create_response(ResponsesRequest(model="m", input="hi"), _raw_request())
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_streaming_response_is_stored(self, api):
        completed = {
            "type": "response.completed",
            "sequence_number": 3,
            "response": {"id": "resp_stream", "object": "response", "status": "completed", "output": []},
        }

        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            yield completed

        _wire(api, gen())
        result = await api.create_response(ResponsesRequest(model="m", input="hi", stream=True), _raw_request())
        body = "".join([chunk async for chunk in result.body_iterator])

        assert "event: response.completed" in body
        assert api._state_store.get("responses/unscoped/resp_stream")["response"]["id"] == "resp_stream"

    @pytest.mark.asyncio
    async def test_streaming_store_failure_reports_response_failed(self, api):
        # The 200 and response.created are already sent, so this can't be a 503. Telling
        # the client it completed would hand back an id that 404s on the next turn.
        completed = {
            "type": "response.completed",
            "sequence_number": 3,
            "response": {"id": "resp_stream", "object": "response", "status": "completed", "output": []},
        }

        async def gen():
            yield {"type": "response.created", "sequence_number": 0}
            yield completed

        _wire(api, gen())
        api._state_store = MagicMock()
        api._state_store.set_async = AsyncMock(side_effect=StateStoreUnavailableError("down"))

        result = await api.create_response(ResponsesRequest(model="m", input="hi", stream=True), _raw_request())
        body = "".join([chunk async for chunk in result.body_iterator])

        assert "event: response.failed" in body
        assert "event: response.completed" not in body
        # [DONE] still follows a store-write failure (streaming.frame_sse), so split
        # on the frame boundary rather than assuming failed is the last event.
        failed_frame = next(part for part in body.split("\n\n") if part.startswith("event: response.failed"))
        failed = json.loads(failed_frame.split("data: ", 1)[1])
        assert failed["response"]["status"] == "failed"
        assert failed["sequence_number"] == 3  # replaces the terminal event, not appended after it


class TestGetDeleteInputItems:
    @pytest.mark.asyncio
    async def test_get_returns_stored_response_verbatim(self, api):
        _stored(api, "resp_1", output=[{"type": "message", "role": "assistant", "content": "hi"}])
        result = await api.get_response("resp_1", _raw_request())
        body = json.loads(bytes(result.body))
        assert body["id"] == "resp_1"
        assert body["output"] == [{"type": "message", "role": "assistant", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_get_unknown_is_404(self, api):
        with pytest.raises(HTTPException) as exc:
            await api.get_response("resp_nope", _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_another_identity_is_404(self, api):
        _stored(api, "resp_1", identity="someone-else")
        with pytest.raises(HTTPException) as exc:
            await api.get_response("resp_1", _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_store_outage_is_503(self, api):
        api._state_store = MagicMock()
        api._state_store.get_async = AsyncMock(side_effect=StateStoreUnavailableError("down"))
        with pytest.raises(HTTPException) as exc:
            await api.get_response("resp_1", _raw_request())
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_delete_removes_and_reports(self, api):
        _stored(api, "resp_1")
        result = await api.delete_response("resp_1", _raw_request())
        body = json.loads(bytes(result.body))
        assert body == {"id": "resp_1", "object": "response", "deleted": True}
        assert api._state_store.get("responses/unscoped/resp_1") is None

    @pytest.mark.asyncio
    async def test_delete_unknown_is_404(self, api):
        with pytest.raises(HTTPException) as exc:
            await api.delete_response("resp_nope", _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_another_identity_is_404_and_leaves_it_intact(self, api):
        _stored(api, "resp_1", identity="someone-else")
        with pytest.raises(HTTPException) as exc:
            await api.delete_response("resp_1", _raw_request())
        assert exc.value.status_code == 404
        assert api._state_store.get("responses/someone-else/resp_1") is not None

    @pytest.mark.asyncio
    async def test_input_items_lists_stored_input(self, api):
        _stored(api, "resp_1", input_items=[{"id": "msg_1", "type": "message", "role": "user", "content": "hi"}])
        result = await api.get_response_input_items("resp_1", _raw_request())
        body = json.loads(bytes(result.body))
        assert body["object"] == "list"
        assert body["data"][0]["content"] == "hi"
        assert body["first_id"] == "msg_1"
        assert body["last_id"] == "msg_1"
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_input_items_empty(self, api):
        _stored(api, "resp_1", input_items=[])
        result = await api.get_response_input_items("resp_1", _raw_request())
        body = json.loads(bytes(result.body))
        assert body["data"] == []
        assert body["first_id"] is None

    @pytest.mark.asyncio
    async def test_input_items_unknown_is_404(self, api):
        with pytest.raises(HTTPException) as exc:
            await api.get_response_input_items("resp_nope", _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.parametrize("bad_id", ["../../effective/test-gateway", "a/b", "resp 1", "", "x" * 129])
    @pytest.mark.asyncio
    async def test_malformed_id_is_404(self, api, bad_id):
        # response_id becomes a state-key segment; enforce our own id shape rather than
        # letting arbitrary text through to the backend.
        with pytest.raises(HTTPException) as exc:
            await api.get_response(bad_id, _raw_request())
        assert exc.value.status_code == 404
