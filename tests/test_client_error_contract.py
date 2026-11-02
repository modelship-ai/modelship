import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from ray.exceptions import ActorUnavailableError, RayTaskError

from modelship.infer.model_deployment import ModelDeployment
from modelship.openai.api import ModelshipAPI, app
from modelship.openai.protocol import create_error_response, encode_error_sse
from modelship.state import MemoryStoreActor
from tests.conftest import build_gateway
from tests.test_responses_background import _FakeHeartbeatRegistry

MARKER = "secret-internal-detail-7f3a"
REQUEST_ID = re.compile(r"request id: [0-9a-f]+\)")

_ModelDeployment = ModelDeployment.func_or_class
_ModelshipAPI = ModelshipAPI.func_or_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_AUDIO = {"data": {"model": "m"}, "files": {"file": ("a.wav", b"RIFF", "audio/wav")}}
_IMAGE = {"image": ("a.png", b"PNG", "image/png")}
MODEL_ROUTES = {
    ("POST", "/v1/chat/completions"): {"json": _CHAT},
    ("POST", "/v1/responses"): {"json": {"model": "m", "input": "hi"}},
    ("POST", "/v1/responses/compact"): {"json": {"model": "m", "input": "hi"}},
    ("POST", "/v1/embeddings"): {"json": {"model": "m", "input": "hi"}},
    ("POST", "/v1/audio/transcriptions"): _AUDIO,
    ("POST", "/v1/audio/translations"): _AUDIO,
    ("POST", "/v1/audio/speech"): {"json": {"model": "m", "input": "hi", "voice": "alloy"}},
    ("POST", "/v1/images/generations"): {"json": {"model": "m", "prompt": "a cat"}},
    ("POST", "/v1/images/edits"): {"data": {"model": "m", "prompt": "a cat"}, "files": _IMAGE},
    ("POST", "/v1/images/variations"): {"data": {"model": "m"}, "files": _IMAGE},
}
OTHER_ROUTES = {
    ("GET", "/health"),
    ("GET", "/readyz"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{model}"),
    ("GET", "/v1/responses/{response_id}"),
    ("POST", "/v1/responses/{response_id}/cancel"),
    ("DELETE", "/v1/responses/{response_id}"),
    ("GET", "/v1/responses/{response_id}/input_items"),
}
_ROUTE_IDS = [path for _, path in MODEL_ROUTES]


class _FakeInfer:
    result = None

    def __getattr__(self, name):
        async def create(*_args, **_kwargs):
            return self.result()

        return create


class _FakeHandle:
    def __init__(self, deployment):
        self._deployment = deployment
        self.crash: Exception | None = None

    def __getattr__(self, name):
        return _FakeMethod(self, getattr(self._deployment, name))


class _FakeMethod:
    def __init__(self, handle, method):
        self._handle, self._method = handle, method

    def options(self, **_):
        return self

    def remote(self, *args):
        return _relay(self._handle.crash, self._method(*args))


async def _relay(crash, items):
    if crash is not None:
        raise crash
    try:
        async for item in items:
            yield item
    except Exception as exc:
        raise RayTaskError("method", f"Traceback: {exc!r}", exc).as_instanceof_cause() from exc


@pytest.fixture
def gateway():
    infer = _FakeInfer()
    deployment = _ModelDeployment.__new__(_ModelDeployment)
    deployment.config = MagicMock()
    deployment.config.name = "m"
    deployment.infer = infer
    handle = _FakeHandle(deployment)
    registry = MagicMock()
    registry.set.remote = AsyncMock()
    registry.is_set.remote = AsyncMock(return_value=False)
    with (
        patch("modelship.openai.api.serve.get_replica_context") as ctx,
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
        patch("modelship.openai.api.get_disconnect_registry", return_value=registry),
        patch("modelship.openai.utils.responses.get_disconnect_registry", return_value=registry),
        patch("modelship.openai.state.responses.get_heartbeat_registry", return_value=_FakeHeartbeatRegistry()),
    ):
        ctx.return_value.app_name = "test-gateway"
        api = build_gateway("test-gateway")
        api._watch_task = MagicMock()
        api._state_store = _MemoryStore()
        api.models = {"m": {"m-a1b2c": handle}}
        api.expected_models = ["m"]
        ctx.return_value.servable_object = api
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, infer=infer, handle=handle)


def _reply(status):
    return lambda: create_error_response(MARKER, status_code=status)


def _raise(exc_type):
    def result():
        raise exc_type(MARKER)

    return result


def _stream(*items):
    async def gen():
        for item in items:
            yield item

    return gen


def _stream_then_raise(first):
    async def gen():
        yield first
        raise RuntimeError(MARKER)

    return gen


_CREATED = {
    "type": "response.created",
    "sequence_number": 0,
    "response": {"id": "resp_1", "object": "response", "status": "in_progress", "output": []},
}


def _failed(**error):
    response = {"id": "resp_1", "object": "response", "status": "failed", "output": [], "error": error}
    return {"type": "response.failed", "sequence_number": 1, "response": response}


SERVER_FAILURES = {
    "5xx reply": _reply(502),
    "raises": _raise(RuntimeError),
    "raises TypeError": _raise(TypeError),
    "raises OverflowError": _raise(OverflowError),
}
CLIENT_FAILURES = {"4xx reply": _reply(400), "raises ValueError": _raise(ValueError)}
CHAT_STREAM_SERVER_FAILURES = {
    "error chunk": _stream("data: {}\n\n", encode_error_sse(create_error_response(MARKER, status_code=502))),
    "raises mid-stream": _stream_then_raise("data: {}\n\n"),
}
RESPONSES_STREAM_SERVER_FAILURES = {
    "failed without code": _stream(_CREATED, _failed(message=MARKER)),
    "failed with server_error": _stream(_CREATED, _failed(code="server_error", message=MARKER)),
    "raises mid-stream": _stream_then_raise(_CREATED),
}


def _assert_hidden(text, caplog):
    assert MARKER not in text
    assert REQUEST_ID.search(text), text
    assert MARKER in caplog.text


def _ws_frames(client):
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "m", "input": "hi"})
        frames = []
        while not frames or frames[-1]["type"] not in ("error", "response.failed", "response.completed"):
            frames.append(ws.receive_json())
        return frames


def test_every_route_is_in_the_contract():
    routes = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes |= {(method, route.path) for method in route.methods}
        elif isinstance(route, APIWebSocketRoute):
            routes.add(("WS", route.path))

    assert routes == set(MODEL_ROUTES) | OTHER_ROUTES | {("WS", "/v1/responses")}


@pytest.mark.parametrize("result", SERVER_FAILURES.values(), ids=SERVER_FAILURES.keys())
@pytest.mark.parametrize("route", MODEL_ROUTES, ids=_ROUTE_IDS)
def test_a_server_failure_reaches_the_client_without_its_text(gateway, route, result, caplog):
    gateway.infer.result = result

    resp = gateway.client.request(*route, **MODEL_ROUTES[route])

    assert resp.status_code >= 500
    assert resp.json()["error"]["type"] == "api_error"
    _assert_hidden(resp.text, caplog)


@pytest.mark.parametrize("route", MODEL_ROUTES, ids=_ROUTE_IDS)
def test_a_dead_model_reaches_the_client_without_its_text(gateway, route, caplog):
    gateway.handle.crash = ActorUnavailableError(MARKER, None)

    resp = gateway.client.request(*route, **MODEL_ROUTES[route])

    assert resp.status_code == 500
    _assert_hidden(resp.text, caplog)


@pytest.mark.parametrize("result", CLIENT_FAILURES.values(), ids=CLIENT_FAILURES.keys())
@pytest.mark.parametrize("route", MODEL_ROUTES, ids=_ROUTE_IDS)
def test_a_client_failure_keeps_its_text(gateway, route, result):
    gateway.infer.result = result

    resp = gateway.client.request(*route, **MODEL_ROUTES[route])

    assert resp.status_code == 400
    assert MARKER in resp.text


@pytest.mark.parametrize("result", CHAT_STREAM_SERVER_FAILURES.values(), ids=CHAT_STREAM_SERVER_FAILURES.keys())
def test_a_chat_stream_failure_reaches_the_client_without_its_text(gateway, result, caplog):
    gateway.infer.result = result

    resp = gateway.client.post("/v1/chat/completions", json={**_CHAT, "stream": True})

    _assert_hidden(resp.text, caplog)


def test_a_chat_stream_client_error_keeps_its_text(gateway):
    gateway.infer.result = _stream("data: {}\n\n", encode_error_sse(create_error_response(MARKER)))

    resp = gateway.client.post("/v1/chat/completions", json={**_CHAT, "stream": True})

    assert MARKER in resp.text


@pytest.mark.parametrize(
    "result", RESPONSES_STREAM_SERVER_FAILURES.values(), ids=RESPONSES_STREAM_SERVER_FAILURES.keys()
)
def test_a_responses_stream_failure_reaches_the_client_without_its_text(gateway, result, caplog):
    gateway.infer.result = result

    resp = gateway.client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})

    _assert_hidden(resp.text, caplog)


def test_a_responses_stream_client_error_keeps_its_text(gateway):
    gateway.infer.result = _stream(_CREATED, _failed(code="invalid_prompt", message=MARKER))

    resp = gateway.client.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})

    assert MARKER in resp.text


@pytest.mark.parametrize(
    "result",
    [*SERVER_FAILURES.values(), *RESPONSES_STREAM_SERVER_FAILURES.values()],
    ids=[*SERVER_FAILURES.keys(), *RESPONSES_STREAM_SERVER_FAILURES.keys()],
)
def test_a_websocket_failure_reaches_the_client_without_its_text(gateway, result, caplog):
    gateway.infer.result = result

    frames = _ws_frames(gateway.client)

    _assert_hidden(str(frames), caplog)


def test_a_websocket_client_error_keeps_its_text(gateway):
    gateway.infer.result = _stream(_CREATED, _failed(code="invalid_prompt", message=MARKER))

    frames = _ws_frames(gateway.client)

    assert MARKER in str(frames)


@pytest.mark.parametrize(
    "result",
    [_stream(_CREATED, _failed(message=MARKER)), _stream_then_raise(_CREATED)],
    ids=["failed without code", "raises mid-stream"],
)
def test_a_background_failure_is_stored_without_its_text(gateway, result, caplog):
    gateway.infer.result = result

    started = gateway.client.post("/v1/responses", json={"model": "m", "input": "hi", "background": True})
    for _ in range(100):
        record = gateway.client.get(f"/v1/responses/{started.json()['id']}")
        if record.json()["status"] == "failed":
            break
        time.sleep(0.05)

    assert record.json()["status"] == "failed"
    assert MARKER not in record.text
    assert MARKER in caplog.text
