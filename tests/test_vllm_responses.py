"""Unit tests for VllmInfer.create_response's own wiring: gating, request
translation error mapping, and pre-generation-vs-mid-stream error handling.

`engine_ops` is mocked out entirely; full DTO-shaping correctness is covered
by test_vllm_engine_ops.py instead.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.exceptions import VLLMValidationError as VllmValidationError

from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    ResponsesRequest,
)


class _FakeRawRequest:
    """Never disconnects — is_watchable=False keeps it out of BaseInfer's
    disconnect pump, so no real DisconnectRegistry actor is needed."""

    request_id = "req-1"
    is_watchable = False
    identity = None

    async def is_disconnected(self) -> bool:
        return False


def _make_infer() -> VllmInfer:
    """Build a VllmInfer with __init__/start bypassed; only the fields
    create_response reads are populated (mirrors test_vllm_vision.py)."""
    infer = VllmInfer.__new__(VllmInfer)
    infer._caps = VllmCapabilities(supports_image=False, supports_audio=False)
    infer.model_config = MagicMock()
    infer.model_config.name = "m"
    infer.openai_serving_render = MagicMock()
    infer._tokenizer = MagicMock()
    infer.engine = MagicMock()
    infer._enable_auto_tools = True
    return infer


@pytest.mark.asyncio
async def test_no_render_pipeline_falls_back_to_not_supported():
    infer = VllmInfer.__new__(VllmInfer)  # no openai_serving_render attribute at all
    request = ResponsesRequest(model="m", input="hi")

    result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 404


@pytest.mark.asyncio
async def test_background_is_not_rejected_before_engine_ops_touched():
    # background:true is handled at the gateway; here it just reaches engine_ops like any request.
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", background=True)

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.render_and_params = AsyncMock(side_effect=VllmValidationError("stop here"))
        result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    mock_ops.render_and_params.assert_called_once()


@pytest.mark.asyncio
async def test_invalid_reasoning_effort_returns_400():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", reasoning={"effort": "turbo"})

    result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400


@pytest.mark.asyncio
async def test_stream_prevalidation_error_returns_plain_error_not_generator():
    """A VllmValidationError from render_and_params must short-circuit to a plain
    ErrorResponse before any generator is created, not a broken/empty stream."""
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(side_effect=VllmValidationError("too long", parameter="messages"))
        result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400


@pytest.mark.asyncio
async def test_stream_success_produces_native_responses_events():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="hi!"))]
        )
        yield ChatCompletionStreamResponse(
            model="m",
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
        )

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        result = await infer.create_response(request, raw_request=_FakeRawRequest())
        events = [chunk async for chunk in result]

    # Plain event dicts — SSE framing and [DONE] are added at the gateway edge.
    types = [e["type"] for e in events]
    assert "response.created" in types
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"


@pytest.mark.asyncio
async def test_a_response_stream_is_timed_to_its_first_token():
    infer = _make_infer()
    raw_request = _FakeRawRequest()

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="hi!"))]
        )

    with (
        patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops,
        patch.object(VllmInfer, "_observe_first_token", side_effect=lambda chunks, _raw: chunks) as observe,
    ):
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        result = await infer.create_response(ResponsesRequest(model="m", input="hi", stream=True), raw_request)
        [event async for event in result]

    assert observe.call_args.args[1] is raw_request


@pytest.mark.asyncio
async def test_a_chat_stream_is_timed_to_its_first_token():
    infer = _make_infer()
    raw_request = _FakeRawRequest()

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="hi!"))]
        )

    with (
        patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops,
        patch.object(VllmInfer, "_observe_first_token", side_effect=lambda chunks, _raw: chunks) as observe,
    ):
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
        [chunk async for chunk in infer._create_chat_completion_stream(request, MagicMock(), raw_request)]

    assert observe.call_args.args[1] is raw_request


@pytest.mark.asyncio
async def test_stream_mid_stream_exception_emits_failed_event():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="partial"))]
        )
        raise RuntimeError("engine blew up")

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        result = await infer.create_response(request, raw_request=_FakeRawRequest())
        events = [chunk async for chunk in result]

    types = [e["type"] for e in events]
    assert types[-1] == "response.failed"
    assert "response.completed" not in types
