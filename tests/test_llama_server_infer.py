"""Tests for the llama_server loader: subprocess lifecycle (fake executable,
no real llama-server binary in CI) and HTTP request/response projection
(httpx mocked, no real socket)."""

from __future__ import annotations

import asyncio
import contextlib
import stat
import sys
import textwrap
from typing import ClassVar
from unittest.mock import patch

import httpx
import pytest

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    RawRequestProxy,
)
from modelship.infer.llama_server.llama_server_infer import LlamaServerInfer
from modelship.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    ResponsesRequest,
)

# ---------------------------------------------------------------------------
# Fake llama-server executables (plain scripts; no real binary needed in CI)
# ---------------------------------------------------------------------------

_FAKE_HEALTHY_SERVER = textwrap.dedent(
    """
    import http.server
    import socketserver
    import sys

    def _port():
        for i, a in enumerate(sys.argv):
            if a == "--port":
                return int(sys.argv[i + 1])
        raise SystemExit("no --port passed")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/props":
                import json

                body = json.dumps({"total_slots": 1, "default_generation_settings": {"n_ctx": 2048}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", _port()), Handler) as httpd:
        httpd.serve_forever()
    """
)

_FAKE_CRASHING_SERVER = "import sys\nsys.exit(1)\n"


def _write_fake_executable(tmp_path, source: str, name: str = "fake-llama-server") -> str:
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n{source}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _make_config(num_gpus: float = 0, **llama_server_kwargs) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_server,
        num_gpus=num_gpus,
        llama_server_config=LlamaServerConfig(**llama_server_kwargs),
    )
    cfg._resolved_path = "/fake/model.gguf"
    return cfg


class TestSubprocessLifecycle:
    @pytest.mark.asyncio
    async def test_start_waits_for_health_then_shutdown_terminates(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config())
        await infer.start()
        try:
            assert infer._proc is not None
            assert infer._proc.poll() is None
            assert infer._client is not None
        finally:
            infer.shutdown()

        assert infer._proc is None

    @pytest.mark.asyncio
    async def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
        infer = LlamaServerInfer(_make_config())
        with pytest.raises(ValueError, match="MSHIP_LLAMA_SERVER_BIN"):
            await infer.start()

    @pytest.mark.asyncio
    async def test_missing_resolved_path_raises(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)
        config = _make_config()
        config._resolved_path = None
        infer = LlamaServerInfer(config)
        with pytest.raises(ValueError, match="resolved model path"):
            await infer.start()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("num_gpus", "config_kwargs", "expected_ngl"),
        [
            (0, {}, "0"),  # no reserved GPU: never offload, even with a GPU build
            (0, {"n_gpu_layers": 24}, "0"),
            (1, {}, "-1"),  # llama-server auto-fit default
            (1, {"n_gpu_layers": 24}, "24"),
            # Any negative n_gpu_layers hits llama-server's own auto-fit code path
            # (`params.n_gpu_layers < 0`); this loader passes the configured int through verbatim.
            (2, {"n_gpu_layers": -2}, "-2"),
        ],
    )
    async def test_ngl_follows_num_gpus(self, tmp_path, monkeypatch, num_gpus, config_kwargs, expected_ngl):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config(num_gpus=num_gpus, **config_kwargs))
        await infer.start()
        try:
            args = list(infer._proc.args)
            assert args[args.index("-ngl") + 1] == expected_ngl
        finally:
            infer.shutdown()

    @pytest.mark.asyncio
    async def test_threads_flag_only_appears_when_set(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer_unset = LlamaServerInfer(_make_config())
        await infer_unset.start()
        try:
            assert "--threads" not in list(infer_unset._proc.args)
        finally:
            infer_unset.shutdown()

        infer_set = LlamaServerInfer(_make_config(threads=8))
        await infer_set.start()
        try:
            args = list(infer_set._proc.args)
            assert args[args.index("--threads") + 1] == "8"
        finally:
            infer_set.shutdown()

    @pytest.mark.asyncio
    async def test_cache_flags_at_defaults(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config())
        await infer.start()
        try:
            args = list(infer._proc.args)
            assert "--cache-reuse" not in args
            assert "--context-shift" not in args
            assert args[args.index("--cache-ram") + 1] == "0"
        finally:
            infer.shutdown()

    @pytest.mark.asyncio
    async def test_cache_flags_appear_when_set(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config(cache_reuse=256, context_shift=True, cache_ram_mib=4096))
        await infer.start()
        try:
            args = list(infer._proc.args)
            assert args[args.index("--cache-reuse") + 1] == "256"
            assert "--context-shift" in args
            assert args[args.index("--cache-ram") + 1] == "4096"
        finally:
            infer.shutdown()

    @pytest.mark.asyncio
    async def test_cache_ram_no_limit_is_forwarded(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config(cache_ram_mib=-1))
        await infer.start()
        try:
            args = list(infer._proc.args)
            assert args[args.index("--cache-ram") + 1] == "-1"
        finally:
            infer.shutdown()

    @pytest.mark.asyncio
    async def test_immediate_crash_retries_then_raises(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_CRASHING_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)
        infer = LlamaServerInfer(_make_config())
        with (
            patch("modelship.infer.llama_server.llama_server_infer._LAUNCH_RETRY_LIMIT", 2),
            pytest.raises(RuntimeError, match="failed to start after"),
        ):
            await infer.start()

    @pytest.mark.asyncio
    async def test_shutdown_is_non_blocking_and_kills_on_timeout(self, tmp_path, monkeypatch):
        import subprocess
        import threading
        import time

        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        infer = LlamaServerInfer(_make_config())
        await infer.start()

        proc = infer._proc
        assert proc is not None

        original_kill = proc.kill

        wait_called = threading.Event()
        kill_called = threading.Event()

        def mocked_wait(timeout=None):
            wait_called.set()
            raise subprocess.TimeoutExpired(cmd=proc.args, timeout=timeout)

        def mocked_kill():
            with contextlib.suppress(Exception):
                original_kill()
            kill_called.set()

        proc.wait = mocked_wait
        proc.kill = mocked_kill

        start_time = time.monotonic()
        infer.shutdown()
        end_time = time.monotonic()

        assert end_time - start_time < 0.5
        assert infer._proc is None

        assert wait_called.wait(timeout=2.0)
        assert kill_called.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Request/response projection over a mocked httpx transport
# ---------------------------------------------------------------------------


def _infer_with_client(handler) -> LlamaServerInfer:
    infer = LlamaServerInfer(_make_config())
    infer._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    return infer


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[{"role": "user", "content": "hi"}], **kwargs)


class _DisconnectingRawRequest:
    """A RawRequestProxy stand-in whose is_disconnected() flips True after
    `disconnect_after` seconds; self-registers by id for a patched _poll_disconnected_ids."""

    _by_id: ClassVar[dict[str, _DisconnectingRawRequest]] = {}

    def __init__(self, disconnect_after: float):
        self._disconnect_after = disconnect_after
        self._start = asyncio.get_event_loop().time()
        self.request_id = "req-1"
        self.is_watchable = True
        _DisconnectingRawRequest._by_id[self.request_id] = self

    async def is_disconnected(self) -> bool:
        return asyncio.get_event_loop().time() - self._start >= self._disconnect_after


async def _poll_disconnected_via_fakes(request_ids: list[str]) -> list[str]:
    """Test-only BaseInfer._poll_disconnected_ids replacement: consults
    _DisconnectingRawRequest's self-registry instead of a real Ray actor."""
    return [rid for rid in request_ids if await _DisconnectingRawRequest._by_id[rid].is_disconnected()]


class TestNonStreamingProjection:
    @pytest.mark.asyncio
    async def test_strips_extension_fields_and_maps_reasoning_and_tools(self):
        raw = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 123,
            "model": "some-gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "thinking...",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "timings": {"predicted_ms": 42.0},  # llama.cpp extension — must not leak
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json=raw)

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(), RawRequestProxy(None, {}))

        assert not isinstance(result, ErrorResponse)
        assert "timings" not in result.model_dump()
        choice = result.choices[0]
        assert choice.message.reasoning == "thinking..."
        assert choice.message.tool_calls[0].function.name == "get_weather"
        assert choice.finish_reason == "tool_calls"
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 14

    @pytest.mark.asyncio
    async def test_forwards_normalized_messages_and_model_name(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {}},
            )

        infer = _infer_with_client(handler)
        await infer.create_chat_completion(
            _request(tools=[{"type": "function", "function": {"name": "f"}}]), RawRequestProxy(None, {})
        )

        assert captured["payload"]["model"] == "test-model"
        assert captured["payload"]["stream"] is False
        assert captured["payload"]["tools"][0]["function"]["name"] == "f"

    @pytest.mark.asyncio
    async def test_tool_choice_none_drops_the_tools(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {}},
            )

        infer = _infer_with_client(handler)
        await infer.create_chat_completion(
            _request(tools=[{"type": "function", "function": {"name": "f"}}], tool_choice="none"),
            RawRequestProxy(None, {}),
        )

        assert "tools" not in captured["payload"]
        assert captured["payload"]["tool_choice"] == "none"

    @pytest.mark.asyncio
    async def test_drops_logprobs_fields_not_yet_supported(self):
        # ChatCompletionRequest defaults top_logprobs to 0 (not None), so a naive
        # exclude_none dump forwards it; llama-server rejects that unless logprobs=true.
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {}},
            )

        infer = _infer_with_client(handler)
        await infer.create_chat_completion(_request(), RawRequestProxy(None, {}))

        assert "logprobs" not in captured["payload"]
        assert "top_logprobs" not in captured["payload"]

    @pytest.mark.asyncio
    async def test_rejects_unsupported_image_content(self):
        infer = _infer_with_client(lambda request: httpx.Response(500))
        request = _request()
        request.messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        result = await infer.create_chat_completion(request, RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_http_error_becomes_error_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(), RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)
        assert "bad request" in result.error.message

    @pytest.mark.asyncio
    async def test_disconnect_aborts_inflight_request_without_waiting_for_response(self):
        # A client disconnect cancels the in-flight httpx call to llama-server
        # (freeing its GPU/CPU slot) rather than waiting for a response.
        handler_finished = False

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal handler_finished
            await asyncio.sleep(10)
            handler_finished = True
            return httpx.Response(200, json={"choices": [], "usage": {}})

        infer = _infer_with_client(handler)
        with patch.object(BaseInfer, "_poll_disconnected_ids", staticmethod(_poll_disconnected_via_fakes)):
            result = await infer.create_chat_completion(_request(), _DisconnectingRawRequest(disconnect_after=0.05))

        assert isinstance(result, ErrorResponse)
        assert "disconnect" in result.error.message.lower()
        assert handler_finished is False

    @pytest.mark.asyncio
    async def test_no_client_falls_back_to_not_supported(self):
        infer = _infer_with_client(lambda request: httpx.Response(200))
        infer._client = None
        result = await infer.create_chat_completion(_request(), RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)


class TestStreamingProjection:
    @pytest.mark.asyncio
    async def test_streams_deltas_tool_calls_and_final_usage(self):
        sse_body = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body.encode(), headers={"content-type": "text/event-stream"})

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(stream=True), RawRequestProxy(None, {}))

        chunks = [chunk async for chunk in result]
        assert chunks[-1] == "data: [DONE]\n\n"

        import json

        content = ""
        tool_call_seen = False
        usage_seen = None
        for raw in chunks[:-1]:
            payload = json.loads(raw[len("data: ") :])
            for choice in payload["choices"]:
                delta = choice["delta"]
                if delta.get("content"):
                    content += delta["content"]
                if delta.get("tool_calls"):
                    tool_call_seen = True
                    assert delta["tool_calls"][0]["function"]["name"] == "get_weather"
            if payload.get("usage"):
                usage_seen = payload["usage"]

        assert content == "Hello"
        assert tool_call_seen
        assert usage_seen is not None
        assert usage_seen["prompt_tokens"] == 5
        assert usage_seen["completion_tokens"] == 2
        assert usage_seen["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_stream_http_error_yields_error_chunk(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "boom"}})

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(stream=True), RawRequestProxy(None, {}))
        chunks = [chunk async for chunk in result]
        assert any("boom" in c for c in chunks)
        # An SSE stream always terminates with [DONE], even on an in-band error,
        # so clients waiting for it (rather than connection close) don't hang.
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_transport_error_yields_error_chunk(self):
        # Simulates a dropped connection / crashed subprocess mid-stream: httpx
        # raises TransportError rather than handing back a Response at all.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("connection reset")

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(stream=True), RawRequestProxy(None, {}))
        chunks = [chunk async for chunk in result]
        assert any("connection reset" in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stage_b4_embedding_support(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        model_config = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.embed,
            loader=ModelLoader.llama_server,
        )
        model_config._resolved_path = "/fake/model.gguf"

        infer = LlamaServerInfer(model_config)
        await infer.start()
        try:
            args = list(infer._proc.args)
            assert "--embedding" in args
        finally:
            infer.shutdown()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/embeddings"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3],
                        }
                    ],
                    "model": "test-model",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 0,
                        "total_tokens": 5,
                    },
                    "timings": "secret",  # extension field to be stripped
                },
            )

        infer_client = LlamaServerInfer(model_config)
        infer_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

        req = EmbeddingRequest(model="test-model", input="hello")
        res = await infer_client.create_embedding(req, RawRequestProxy(None, {}))
        assert isinstance(res, EmbeddingResponse)
        assert res.data[0].embedding == [0.1, 0.2, 0.3]
        assert "timings" not in res.model_dump()

    @pytest.mark.asyncio
    async def test_stage_b4_vision_support(self, tmp_path, monkeypatch):
        binary = _write_fake_executable(tmp_path, _FAKE_HEALTHY_SERVER)
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", binary)

        # mmproj is resolved once on the driver, same as the primary model path —
        # the actor receives an already-resolved path.
        model_config = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            llama_server_config=LlamaServerConfig(mmproj="/resolved/mmproj.gguf"),
        )
        model_config._resolved_path = "/fake/model.gguf"

        infer = LlamaServerInfer(model_config)

        await infer.start()
        try:
            args = list(infer._proc.args)
            assert "--mmproj" in args
            assert "/resolved/mmproj.gguf" in args
        finally:
            infer.shutdown()

        # Gateway rejection of image content is skipped when mmproj is configured.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "I see an image"}}]},
            )

        infer_client = LlamaServerInfer(model_config)
        infer_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

        req = ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    ],
                }
            ],
        )
        res = await infer_client.create_chat_completion(req, RawRequestProxy(None, {}))
        assert not isinstance(res, ErrorResponse)
        assert res.choices[0].message.content == "I see an image"

    @pytest.mark.asyncio
    async def test_stage_b4_concurrency_coupling(self):
        model_config = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            llama_server_config=LlamaServerConfig(parallel=4),
        )
        from modelship.deploy.actor_options import build_deployment_options

        opts = build_deployment_options(model_config)
        assert opts["max_ongoing_requests"] == 4

        model_config.max_ongoing_requests = 10
        opts = build_deployment_options(model_config)
        assert opts["max_ongoing_requests"] == 10

    @pytest.mark.asyncio
    async def test_stage_b4_logprobs_support(self):
        def handler(request: httpx.Request) -> httpx.Response:
            import json

            payload = json.loads(request.read())
            assert payload.get("logprobs") is True
            assert payload.get("top_logprobs") == 2
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "hello"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "hello",
                                        "logprob": -0.1,
                                        "top_logprobs": [
                                            {"token": "hello", "logprob": -0.1},
                                            {"token": "hi", "logprob": -1.5},
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                },
            )

        model_config = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
        )
        infer_client = LlamaServerInfer(model_config)
        infer_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=2,
        )
        res = await infer_client.create_chat_completion(req, RawRequestProxy(None, {}))
        assert not isinstance(res, ErrorResponse)
        assert res.choices[0].logprobs is not None
        assert res.choices[0].logprobs.content[0].token == "hello"
        assert res.choices[0].logprobs.content[0].top_logprobs[1].token == "hi"

    @pytest.mark.asyncio
    async def test_stage_b4_inline_json_errors_handled(self):
        def chat_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"error": {"message": "inline chat error detail"}},
            )

        model_config = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
        )
        infer_client = LlamaServerInfer(model_config)
        infer_client._client = httpx.AsyncClient(transport=httpx.MockTransport(chat_handler), base_url="http://test")

        res = await infer_client.create_chat_completion(_request(), RawRequestProxy(None, {}))
        assert isinstance(res, ErrorResponse)
        assert res.error.message == "inline chat error detail"

        def embed_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"error": "inline embedding error detail"},
            )

        model_config_embed = ModelshipModelConfig(
            name="test-model",
            model="org/test-model",
            usecase=ModelUsecase.embed,
            loader=ModelLoader.llama_server,
        )
        infer_client_embed = LlamaServerInfer(model_config_embed)
        infer_client_embed._client = httpx.AsyncClient(
            transport=httpx.MockTransport(embed_handler), base_url="http://test"
        )

        req = EmbeddingRequest(model="test-model", input="hello")
        res_embed = await infer_client_embed.create_embedding(req, RawRequestProxy(None, {}))
        assert isinstance(res_embed, ErrorResponse)
        assert res_embed.error.message == "inline embedding error detail"

    @pytest.mark.asyncio
    async def test_stage_b4_mid_stream_json_errors_handled(self):
        sse_body = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            'data: {"error": {"message": "inline mid-stream error detail"}}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body.encode(), headers={"content-type": "text/event-stream"})

        infer = _infer_with_client(handler)
        result = await infer.create_chat_completion(_request(stream=True), RawRequestProxy(None, {}))

        chunks = [chunk async for chunk in result]
        # First 2 are normal choices, then the error chunk, then [DONE]
        assert len(chunks) == 4
        assert "inline mid-stream error detail" in chunks[2]


def _responses_request(**kwargs) -> ResponsesRequest:
    return ResponsesRequest(model="test-model", input="hi", **kwargs)


class TestResponsesProjection:
    """LlamaServerInfer.create_response shapes items directly from
    llama-server's own parsed reasoning/tool_calls, same as the chat projection tests do."""

    @pytest.mark.asyncio
    async def test_non_stream_maps_reasoning_and_tools_to_responses_items(self):
        raw = {
            "id": "chatcmpl-xyz",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "thinking...",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "timings": {"predicted_ms": 42.0},  # llama.cpp extension — must not leak
        }

        infer = _infer_with_client(lambda request: httpx.Response(200, json=raw))
        result = await infer.create_response(_responses_request(), RawRequestProxy(None, {}))

        assert not isinstance(result, ErrorResponse)
        assert result.object == "response"
        assert [item.type for item in result.output] == ["reasoning", "function_call"]
        assert result.output[0].summary[0].text == "thinking..."
        assert result.output[1].name == "get_weather"
        assert result.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_background_is_not_special_cased_at_the_loader(self):
        # background:true is handled at the gateway; the loader sees only an echoed flag.
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"choices": [], "usage": {}})

        infer = _infer_with_client(handler)
        result = await infer.create_response(_responses_request(background=True), RawRequestProxy(None, {}))

        assert not isinstance(result, ErrorResponse)
        assert called is True
        assert result.background is True

    @pytest.mark.asyncio
    async def test_no_client_falls_back_to_not_supported(self):
        infer = _infer_with_client(lambda request: httpx.Response(200))
        infer._client = None
        result = await infer.create_response(_responses_request(), RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_stream_events_sequence_and_forces_stream_true_on_wire(self):
        captured = {}
        sse_body = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, content=sse_body.encode(), headers={"content-type": "text/event-stream"})

        infer = _infer_with_client(handler)
        result = await infer.create_response(_responses_request(stream=True), RawRequestProxy(None, {}))
        events = [chunk async for chunk in result]
        types = [e["type"] for e in events]

        # responses_request_to_chat hardcodes stream=False; create_response flips it
        # back to True before the wire, or llama-server answers with JSON, not SSE.
        assert captured["payload"]["stream"] is True
        assert "response.created" in types
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_stream_mid_stream_error_emits_failed_event(self):
        sse_body = (
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            'data: {"error": {"message": "inline mid-stream error detail"}}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body.encode(), headers={"content-type": "text/event-stream"})

        infer = _infer_with_client(handler)
        result = await infer.create_response(_responses_request(stream=True), RawRequestProxy(None, {}))
        events = [chunk async for chunk in result]
        types = [e["type"] for e in events]

        assert types[-1] == "response.failed"
        assert events[-1]["response"]["error"]["message"] == "inline mid-stream error detail"
        assert "response.completed" not in types

    @pytest.mark.asyncio
    async def test_stream_disconnect_ends_without_hanging(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10)
            return httpx.Response(200, content=b"data: [DONE]\n\n")  # pragma: no cover - never reached

        infer = _infer_with_client(handler)
        with patch.object(BaseInfer, "_poll_disconnected_ids", staticmethod(_poll_disconnected_via_fakes)):
            result = await infer.create_response(
                _responses_request(stream=True), _DisconnectingRawRequest(disconnect_after=0.05)
            )

            # translator.start()'s two events are synchronous and yield immediately;
            # the disconnect bites once the generator blocks on the (never-answered) HTTP call.
            events = [chunk async for chunk in result]
        types = [e["type"] for e in events]
        assert "response.created" in types
        assert "response.completed" not in types
        assert "response.failed" not in types
