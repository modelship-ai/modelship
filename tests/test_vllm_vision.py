"""Tests for vLLM chat-completion handling of vision (image_url) message parts.

Verifies the protocol hop (``ChatCompletionRequest`` -> ``model_dump()`` ->
``VllmChatCompletionRequest``) preserves image parts, plus the wrapper-level
400-rejection gating in :class:`VllmInfer.create_chat_completion`.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as VllmChatCompletionRequest

from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import ChatCompletionRequest, ErrorResponse

# ---------------------------------------------------------------------------
# Protocol hop — exercise the real vLLM ChatCompletionRequest validator
# ---------------------------------------------------------------------------


def test_image_url_part_survives_protocol_hop_into_vllm_request():
    """image_url parts must round-trip into ``VllmChatCompletionRequest``
    unchanged — the shape vLLM's multimodal preprocessing reads."""
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    )

    vllm_request = VllmChatCompletionRequest(**request.model_dump())

    parts = vllm_request.messages[0]["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "https://example.com/cat.png"
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "describe"


def test_data_uri_image_survives_protocol_hop():
    """Base64 data URIs are the common alternative to remote URLs; vLLM must
    accept them on the same code path."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            }
        ],
    )

    vllm_request = VllmChatCompletionRequest(**request.model_dump())

    parts = vllm_request.messages[0]["content"]
    assert parts[0]["image_url"]["url"] == data_uri


# ---------------------------------------------------------------------------
# Wrapper gating — exercise VllmInfer.create_chat_completion's 400 path
# ---------------------------------------------------------------------------


def _make_infer(*, supports_image: bool, monkeypatch: pytest.MonkeyPatch) -> VllmInfer:
    """Build a VllmInfer with __init__/start bypassed; only the fields the
    chat-completion path reads are populated.

    `_prepare_chat` renders the request via `engine_ops.render_and_params` before
    streaming, so that call is stubbed to avoid a real render against the mock.
    """
    infer = VllmInfer.__new__(VllmInfer)
    infer._caps = VllmCapabilities(supports_image=supports_image)  # type: ignore[attr-defined]
    infer.model_config = MagicMock()
    infer.openai_serving_render = MagicMock()
    infer._tokenizer = MagicMock()
    infer._prompt_endings = []
    infer._create_chat_completion_stream = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "modelship.infer.vllm.engine_ops.render_and_params",
        AsyncMock(return_value=(MagicMock(), MagicMock())),
    )
    return infer


@pytest.mark.asyncio
async def test_image_part_rejected_on_text_only_model_with_400(monkeypatch: pytest.MonkeyPatch):
    """A text-only model receiving image_url returns a 400 BadRequest with
    our error envelope — same shape llama.cpp produces."""
    infer = _make_infer(supports_image=False, monkeypatch=monkeypatch)
    request = ChatCompletionRequest(
        model="llm",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    )

    result = await infer.create_chat_completion(request, raw_request=MagicMock(identity=None))

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400
    assert "image" in result.error.message.lower()
    # The reject must happen before we hand off to vLLM.
    infer._create_chat_completion_stream.assert_not_called()


@pytest.mark.asyncio
async def test_text_only_request_reaches_streaming_path_on_vlm(monkeypatch: pytest.MonkeyPatch):
    """A plain text request on a VLM is not blocked by the gating layer."""
    infer = _make_infer(supports_image=True, monkeypatch=monkeypatch)
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    await infer.create_chat_completion(request, raw_request=MagicMock(identity=None))

    infer._create_chat_completion_stream.assert_called_once()
