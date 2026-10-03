"""Route-level tests for `background:true` on /v1/responses. The drain task runs
detached, so most tests call `_drain_background_tasks(api)` after triggering the
route before asserting on the stored snapshot."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import ResponsesRequest
from modelship.openai.state import responses as responses_state
from modelship.openai.state.responses import _HeartbeatStore
from modelship.openai.utils.responses import _StreamBuffer
from modelship.state import MemoryStoreActor, StateStoreUnavailableError
from tests.conftest import build_gateway

_ModelshipAPI = ModelshipAPI.func_or_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def _fake_disconnect_registry():
    """`utils.responses` imports `get_disconnect_registry` directly, so the
    conftest-level patch on `infer_config`'s copy doesn't reach it — fake it here."""
    registry = MagicMock()
    registry.set.remote = AsyncMock()
    registry.is_set.remote = AsyncMock(return_value=False)
    with patch("modelship.openai.utils.responses.get_disconnect_registry", return_value=registry):
        yield registry


class _FakeHeartbeatRegistry:
    """Stand-in for the HeartbeatRegistry actor handle, backed by a real
    _HeartbeatStore; `.remote()` mimics Ray actor-method dispatch."""

    def __init__(self):
        self._store = _HeartbeatStore(ttl_seconds=3600.0)
        self.heartbeat = self._Method(self._heartbeat)
        self.is_alive = self._Method(self._is_alive)
        self.req_id = self._Method(self._req_id)

    async def _heartbeat(self, key, req_id):
        self._store.heartbeat(key, req_id)

    async def _is_alive(self, key):
        return self._store.is_alive(key)

    async def _req_id(self, key):
        return self._store.req_id(key)

    def expire(self, identity: str, response_id: str) -> None:
        """Force this key's entry stale, simulating a dead worker's heartbeat aging
        out, without waiting out a real TTL."""
        key = f"responses/{identity}/{response_id}"
        req_id = self._store.req_id(key)
        if req_id is not None:
            self._store._entries[key] = (req_id, -1.0)

    class _Method:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)


@pytest.fixture(autouse=True)
def heartbeat_registry():
    reg = _FakeHeartbeatRegistry()
    with patch("modelship.openai.state.responses.get_heartbeat_registry", return_value=reg):
        yield reg


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


def _raw_request():
    raw = MagicMock()
    raw.headers = {}
    return raw


def _stored(api, response_id="resp_1", identity="unscoped", status="completed", output=None):
    api._state_store.set(
        f"responses/{identity}/{response_id}",
        {
            "response": {"id": response_id, "object": "response", "status": status, "output": output or []},
            "input_items": [],
        },
    )


async def _stored_background(
    api, response_id="resp_1", identity="unscoped", status="in_progress", req_id="req-1", output=None
):
    response = {"id": response_id, "object": "response", "status": status, "background": True, "output": output or []}
    value = {"response": response, "input_items": []}
    api._state_store.set(f"responses/{identity}/{response_id}", value)
    if status in ("queued", "in_progress"):
        await responses_state.heartbeat(identity, response_id, req_id)


def _wire(api, side_effect):
    """Wire `handle.respond.options(...).remote(...)` to `side_effect`."""
    handle = MagicMock()
    handle.respond.options.return_value.remote.side_effect = side_effect
    api.models = {"m": {"m-a1b2c": handle}}
    return handle


def _background_gen_factory(text="hello background!", status="completed"):
    """A fake `handle.respond` remote-call side effect: reads the injected
    `response_id` (positional arg 5) so the terminal event's id matches the placeholder's."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            yield {
                "type": f"response.{status}",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": status,
                    "background": True,
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    ],
                },
            }

        return gen()

    return _make_gen


def _completes_after_callback_factory(callback, text="hello background!"):
    """Like `_background_gen_factory`, but runs `callback()` between the `created`
    and `completed` events, racing it against the completion write."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            await callback(response_id)
            yield {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "background": True,
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    ],
                },
            }

        return gen()

    return _make_gen


def _hanging_gen_factory(started: asyncio.Event, release: asyncio.Event):
    """A fake remote-call side effect that yields one event, signals `started`, then
    blocks on `release`, simulating a run still in flight."""

    def _make_gen(*args, **kwargs):
        response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

        async def gen():
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            started.set()
            await release.wait()

        return gen()

    return _make_gen


async def _drain_background_tasks(api):
    tasks = list(api._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


class TestBackgroundCreate:
    @pytest.mark.asyncio
    async def test_returns_queued_immediately(self, api):
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", background=True)

        result = await api.create_response(request, _raw_request())

        body = json.loads(bytes(result.body))
        assert body["status"] == "queued"
        assert body["id"].startswith("resp_")
        assert body["background"] is True
        assert "_mship" not in body

        # The drain task hasn't run yet (nothing has awaited it): an immediate GET
        # sees the same queued placeholder.
        get_result = await api.get_response(body["id"], _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "queued"

        await _drain_background_tasks(api)

    @pytest.mark.asyncio
    async def test_drains_to_completed(self, api):
        _wire(api, _background_gen_factory(text="hello background!"))
        request = ResponsesRequest(model="m", input="hi", background=True)

        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        body = json.loads(bytes(get_result.body))
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "hello background!"
        assert "_mship" not in body


class TestMcpBackgroundWiring:
    """Wiring only — behavior covered by test_mcp_loop.py. Confirms run_background
    routes through mcp_loop.run_mcp_response and persists an mcp_approval_request."""

    @pytest.mark.asyncio
    async def test_background_mcp_routes_through_loop_and_persists_approval_request(self, api):
        handle = MagicMock()
        api.models = {"m": {"m-a1b2c": handle}}

        async def fake_mcp_gen(*args, **kwargs):
            response_id = kwargs.get("response_id")
            yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
            yield {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "background": True,
                    "output": [
                        {
                            "type": "mcp_approval_request",
                            "id": "mcpr_1",
                            "name": "roll",
                            "arguments": "{}",
                            "server_label": "dice",
                        }
                    ],
                },
            }

        request = ResponsesRequest(
            model="m",
            input="roll some dice",
            background=True,
            tools=[{"type": "mcp", "server_label": "dice", "server_url": "http://x"}],
        )
        with patch("modelship.openai.mcp.loop.run_mcp_response", side_effect=fake_mcp_gen) as run_mcp:
            result = await api.create_response(request, _raw_request())
            response_id = json.loads(bytes(result.body))["id"]
            await _drain_background_tasks(api)

        run_mcp.assert_called_once()
        assert run_mcp.call_args.kwargs["response_id"] == response_id
        handle.respond.options.return_value.remote.assert_not_called()

        get_result = await api.get_response(response_id, _raw_request())
        body = json.loads(bytes(get_result.body))
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "mcp_approval_request"
        assert body["output"][0]["name"] == "roll"


class TestGuards:
    @pytest.mark.asyncio
    async def test_background_and_store_false_is_400(self, api):
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", background=True, store=False)

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400
        assert api._background_tasks == set()

    @pytest.mark.asyncio
    async def test_previous_response_id_in_progress_is_400(self, api):
        await _stored_background(api, "resp_running", status="in_progress")
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_running")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400
        assert exc.value.err.error.param == "previous_response_id"

    @pytest.mark.asyncio
    async def test_previous_response_id_queued_is_400(self, api):
        await _stored_background(api, "resp_queued", status="queued")
        _wire(api, _background_gen_factory())
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_queued")

        with pytest.raises(HTTPException) as exc:
            await api.create_response(request, _raw_request())
        assert exc.value.status_code == 400


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_in_progress_marks_cancelled_and_survives_abnormal_end(self, api):
        started = asyncio.Event()
        release = asyncio.Event()
        _wire(api, _hanging_gen_factory(started, release))

        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]
        await started.wait()

        cancel_result = await api.cancel_response(response_id, _raw_request())
        body = json.loads(bytes(cancel_result.body))
        assert body["status"] == "cancelled"

        # Let the "generation" end abnormally (no terminal event) — the drain task's
        # fallback-failure write must not regress an already-cancelled status.
        release.set()
        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_wins_race_against_a_genuine_completion(self, api):
        # A cancel that lands mid-flight must not be clobbered by the generation's
        # own later, unconditional completion write.
        async def _cancel_mid_stream(response_id):
            await api.cancel_response(response_id, _raw_request())

        _wire(api, _completes_after_callback_factory(_cancel_mid_stream))
        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        get_result = await api.get_response(response_id, _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_on_terminal_background_is_idempotent(self, api):
        await _stored_background(api, "resp_done", status="completed")

        result = await api.cancel_response("resp_done", _raw_request())

        assert json.loads(bytes(result.body))["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancel_on_non_background_response_is_400(self, api):
        _stored(api, "resp_chat", status="completed")

        with pytest.raises(HTTPException) as exc:
            await api.cancel_response("resp_chat", _raw_request())
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_unknown_is_404(self, api):
        with pytest.raises(HTTPException) as exc:
            await api.cancel_response("resp_nope", _raw_request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_store_outage_writing_cancelled_snapshot_is_503(self, api):
        # A store outage writing the cancelled snapshot must surface as 503, not a
        # raw StateStoreUnavailableError.
        await _stored_background(api, "resp_1", status="in_progress")

        with (
            patch.object(api._state_store, "set_async", AsyncMock(side_effect=StateStoreUnavailableError("down"))),
            pytest.raises(HTTPException) as exc,
        ):
            await api.cancel_response("resp_1", _raw_request())
        assert exc.value.status_code == 503


class TestReplicaShutdown:
    @pytest.mark.asyncio
    async def test_a_stopping_replica_waits_for_its_background_runs(self, api):
        started = asyncio.Event()
        release = asyncio.Event()

        async def hold(_response_id):
            started.set()
            await release.wait()

        _wire(api, _completes_after_callback_factory(hold))
        result = await api.create_response(ResponsesRequest(model="m", input="hi", background=True), _raw_request())
        response_id = json.loads(bytes(result.body))["id"]
        await started.wait()

        # The class @serve.ingress wraps; the wrapper's own __del__ also stops the ASGI lifespan.
        stopping = asyncio.create_task(_ModelshipAPI.__bases__[0].__del__(api))
        _, pending = await asyncio.wait({stopping}, timeout=0.1)
        assert stopping in pending

        release.set()
        await stopping
        get_result = await api.get_response(response_id, _raw_request())
        assert json.loads(bytes(get_result.body))["status"] == "completed"


class TestDeleteImpliesCancel:
    @pytest.mark.asyncio
    async def test_delete_in_flight_signals_cancel_and_removes_snapshot(self, api):
        started = asyncio.Event()
        release = asyncio.Event()
        _wire(api, _hanging_gen_factory(started, release))

        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]
        await started.wait()

        delete_result = await api.delete_response(response_id, _raw_request())
        assert json.loads(bytes(delete_result.body))["deleted"] is True
        assert api._state_store.get(f"responses/unscoped/{response_id}") is None

        # The drain task finishes only after the delete; it must not resurrect what
        # was just removed.
        release.set()
        await _drain_background_tasks(api)

        assert api._state_store.get(f"responses/unscoped/{response_id}") is None

    @pytest.mark.asyncio
    async def test_delete_wins_race_against_a_genuine_completion(self, api):
        # A delete that lands mid-flight must not be resurrected by that generation's
        # own later, unconditional-looking completion write.
        async def _delete_mid_stream(response_id):
            await api.delete_response(response_id, _raw_request())

        _wire(api, _completes_after_callback_factory(_delete_mid_stream))
        request = ResponsesRequest(model="m", input="hi", background=True)
        result = await api.create_response(request, _raw_request())
        response_id = json.loads(bytes(result.body))["id"]

        await _drain_background_tasks(api)

        assert api._state_store.get(f"responses/unscoped/{response_id}") is None


class TestOrphanDetection:
    @pytest.mark.asyncio
    async def test_stale_heartbeat_reports_failed_on_get(self, api, heartbeat_registry):
        await _stored_background(api, "resp_stale", status="in_progress")
        heartbeat_registry.expire("unscoped", "resp_stale")

        result = await api.get_response("resp_stale", _raw_request())

        body = json.loads(bytes(result.body))
        assert body["status"] == "failed"
        assert "_mship" not in body

    @pytest.mark.asyncio
    async def test_fresh_heartbeat_stays_in_progress(self, api):
        await _stored_background(api, "resp_fresh", status="in_progress")

        result = await api.get_response("resp_fresh", _raw_request())

        assert json.loads(bytes(result.body))["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_store_outage_writing_failed_snapshot_is_503(self, api, heartbeat_registry):
        # A store outage in reconcile_staleness's snapshot write must surface as
        # the documented 503, not a raw StateStoreUnavailableError.
        await _stored_background(api, "resp_stale", status="in_progress")
        heartbeat_registry.expire("unscoped", "resp_stale")

        with (
            patch.object(api._state_store, "set_async", AsyncMock(side_effect=StateStoreUnavailableError("down"))),
            pytest.raises(HTTPException) as exc,
        ):
            await api.get_response("resp_stale", _raw_request())
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_store_outage_on_post_write_readback_is_503(self, api, heartbeat_registry):
        # reconcile_staleness's final read-back after a successful write must also
        # surface a store outage as 503.
        await _stored_background(api, "resp_stale", status="in_progress")
        heartbeat_registry.expire("unscoped", "resp_stale")

        real_get_async = api._state_store.get_async
        calls = 0

        async def flaky_get_async(key):
            # 1st call: get_response's own load_snapshot. 2nd: write_terminal_if_not_terminal's
            # internal pre-write read. 3rd: reconcile_staleness's final read-back — fail there.
            nonlocal calls
            calls += 1
            if calls > 2:
                raise StateStoreUnavailableError("down")
            return await real_get_async(key)

        with (
            patch.object(api._state_store, "get_async", flaky_get_async),
            pytest.raises(HTTPException) as exc,
        ):
            await api.get_response("resp_stale", _raw_request())
        assert exc.value.status_code == 503


def _parse_sse(body: str) -> list[dict]:
    events = []
    for frame in body.split("\n\n"):
        lines = [line for line in frame.split("\n") if line]
        if not lines:
            continue
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        if data_line is None:
            continue
        data = data_line[len("data: ") :]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


async def _consume_body(result) -> str:
    chunks = await asyncio.wait_for(_collect(result), timeout=5.0)
    return "".join(chunks)


async def _collect(result):
    return [chunk async for chunk in result.body_iterator]


class TestBackgroundStream:
    @pytest.mark.asyncio
    async def test_background_and_stream_streams_live_and_completes(self, api):
        # A real gap between events so the tailer's poll observes `response.created`
        # while buffered, rather than racing a drain task that finishes first.
        def _slow_gen_factory(*args, **kwargs):
            response_id = args[5] if len(args) > 5 else kwargs.get("response_id")

            async def gen():
                yield {"type": "response.created", "sequence_number": 0, "response": {"id": response_id}}
                await asyncio.sleep(0.4)
                yield {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "background": True,
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "hello background!"}],
                            }
                        ],
                    },
                }

            return gen()

        _wire(api, _slow_gen_factory)
        request = ResponsesRequest(model="m", input="hi", background=True, stream=True)

        result = await api.create_response(request, _raw_request())
        assert result.media_type == "text/event-stream"

        body = await _consume_body(result)
        events = _parse_sse(body)

        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["status"] == "completed"
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == "hello background!"

        await _drain_background_tasks(api)

    @pytest.mark.asyncio
    async def test_get_stream_resumes_from_buffered_events(self, api):
        response_id = "resp_1"
        await _stored_background(api, response_id, status="completed")
        key = f"responses-stream/unscoped/{response_id}"
        await api._state_store.append_async(key, {"type": "response.created", "sequence_number": 0, "response": {}})
        await api._state_store.append_async(
            key, {"type": "response.output_text.delta", "sequence_number": 1, "delta": "hi"}
        )
        await api._state_store.append_async(
            key,
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {"id": response_id, "object": "response", "status": "completed"},
            },
        )

        result = await api.get_response(response_id, _raw_request(), stream=True, starting_after=0)
        body = await _consume_body(result)
        events = _parse_sse(body)

        # starting_after=0 excludes the seq-0 event that was already seen before disconnect.
        assert [e["sequence_number"] for e in events] == [1, 2]
        assert events[-1]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_get_stream_on_terminal_response_synthesizes_terminal_event(self, api):
        # A fresh stream (no starting_after) always gets a response.created first,
        # even when nothing was ever buffered for it.
        await _stored_background(api, "resp_done", status="completed")

        result = await api.get_response("resp_done", _raw_request(), stream=True)
        body = await _consume_body(result)
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["response.created", "response.completed"]
        assert events[-1]["response"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_synthesized_created_for_fresh_tail_has_in_progress_envelope(self, api):
        # The synthesized response.created for a fresh tail must carry
        # status=in_progress, output=[] — not the terminal response's fields.
        response = {
            "id": "resp_done",
            "object": "response",
            "status": "completed",
            "background": True,
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        api._state_store.set("responses/unscoped/resp_done", {"response": response, "input_items": []})

        result = await api.get_response("resp_done", _raw_request(), stream=True)
        body = await _consume_body(result)
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["response.created", "response.completed"]
        created = events[0]["response"]
        assert created["status"] == "in_progress"
        assert created["output"] == []
        assert created["usage"] is None
        # The terminal event, in contrast, must still carry the real final state.
        assert events[1]["response"]["status"] == "completed"
        assert events[1]["response"]["output"] == response["output"]

    @pytest.mark.asyncio
    async def test_get_stream_resume_from_a_real_cursor_skips_the_synthesized_created(self, api):
        # A genuine resume (a real starting_after, meaning the client already saw
        # response.created on its original connection) must not get a second one.
        await _stored_background(api, "resp_done", status="completed")

        result = await api.get_response("resp_done", _raw_request(), stream=True, starting_after=0)
        body = await _consume_body(result)
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["response.completed"]

    @pytest.mark.asyncio
    async def test_get_stream_ends_cleanly_on_buffer_read_outage(self, api):
        await _stored_background(api, "resp_1", status="in_progress")
        with patch.object(
            api._state_store, "read_from_async", AsyncMock(side_effect=StateStoreUnavailableError("down"))
        ):
            result = await api.get_response("resp_1", _raw_request(), stream=True)
            body = await _consume_body(result)

        # Ends the stream quietly rather than raising mid-response (headers already sent).
        assert _parse_sse(body) == []


class TestStreamBufferOrdering:
    @pytest.mark.asyncio
    async def test_appends_land_in_event_order_even_if_an_earlier_one_is_slower(self):
        landed = []

        async def fake_append(store, identity, response_id, event):
            await asyncio.sleep(0.05 if event["sequence_number"] == 0 else 0)
            landed.append(event["sequence_number"])

        with (
            patch("modelship.openai.utils.responses.responses_state.append_stream_event", fake_append),
            patch("modelship.openai.utils.responses.responses_state.discard_stream_buffer", AsyncMock()),
        ):
            buf = _StreamBuffer(MagicMock(), "u1", "resp_1")
            buf.append({"sequence_number": 0})
            buf.append({"sequence_number": 1})
            await buf.flush_and_discard()

        assert landed == [0, 1]

    @pytest.mark.asyncio
    async def test_flush_and_discard_waits_for_the_pending_append_first(self):
        order = []

        async def fake_append(store, identity, response_id, event):
            await asyncio.sleep(0.05)
            order.append("append")

        async def fake_discard(store, identity, response_id):
            order.append("discard")

        with (
            patch("modelship.openai.utils.responses.responses_state.append_stream_event", fake_append),
            patch("modelship.openai.utils.responses.responses_state.discard_stream_buffer", fake_discard),
        ):
            buf = _StreamBuffer(MagicMock(), "u1", "resp_1")
            buf.append({"sequence_number": 0})
            await buf.flush_and_discard()

        assert order == ["append", "discard"]
