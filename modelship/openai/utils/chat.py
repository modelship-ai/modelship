"""Validation and normalization helpers for OpenAI chat-completion messages.

Responses-specific shaping (parsed chat output -> Responses items) and gateway-side
Responses helpers live in the sibling :mod:`.responses` module instead.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ToolCall,
    UsageInfo,
)

logger = get_logger("openai.utils.chat")


class UnsupportedContentError(ValueError):
    """A chat-completion message contains a part the model cannot accept.

    Subclasses ``ValueError`` so :func:`modelship.openai.protocol.create_error_response`
    maps it to a 400 BadRequestError automatically.
    """


# Part types that collapse to plain text (mirrors vllm's chat_utils text aliases).
_TEXT_TYPES = frozenset({"text", "input_text", "output_text", "refusal", "thinking"})

_IMAGE_TYPES = frozenset({"image_url", "input_image"})
_AUDIO_TYPES = frozenset({"input_audio", "audio_url"})


def _tool_name_by_call_id(messages: list[dict]) -> dict[str, str]:
    """Map tool_call_id -> function name from assistant tool_calls.

    Strict chat templates (FunctionGemma, Mistral) require ``role: tool``
    messages to carry ``name``, but the OpenAI API makes it optional and clients
    like Home Assistant omit it. Recover it from the assistant turn that issued
    the matching call.
    """
    mapping: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            call_id = call.get("id")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(call_id, str) and isinstance(name, str):
                mapping[call_id] = name
    return mapping


def normalize_chat_messages(
    messages: list[dict],
    *,
    supports_image: bool = False,
    supports_audio: bool = False,
) -> list[dict]:
    """Validate and normalize OpenAI chat-completion messages.

    Aimed at backends that consume the OpenAI message shape directly (e.g. llama.cpp),
    where the underlying chat template is strict about input structure.

    Behavior:

    - String / ``None`` content passes through unchanged.
    - List content is validated part-by-part:
        - Plain strings inside the list are accepted as text.
        - Text-like parts (``text`` / ``input_text`` / ``output_text`` /
          ``refusal`` / ``thinking``) are normalized to ``{"type": "text", "text": ...}``.
        - Image parts (``image_url`` / ``input_image``) require ``supports_image``;
          otherwise :class:`UnsupportedContentError` is raised. Both are normalized
          to ``{"type": "image_url", "image_url": {"url": ...}}``.
        - Audio parts (``input_audio`` / ``audio_url``) require ``supports_audio``;
          otherwise :class:`UnsupportedContentError` is raised.
        - Parts with empty / missing content are dropped with a warning
          (matches vllm's tolerant behavior).
        - Unknown part types raise :class:`UnsupportedContentError`.
        - Malformed parts (wrong types for required fields) raise :class:`UnsupportedContentError`.
    - List content that ends up text-only is collapsed to a single string
      joined with ``"\\n"`` so loaders whose Jinja templates only accept string
      content keep working.
    - ``role: tool`` messages missing ``name`` get it backfilled from the
      matching assistant ``tool_calls`` entry — strict templates require it.
    """
    tool_names = _tool_name_by_call_id(messages)
    normalized: list[dict] = []
    for idx, msg in enumerate(messages):
        out = dict(msg)
        if out.get("role") == "tool" and not out.get("name"):
            call_id = out.get("tool_call_id")
            name = tool_names.get(call_id) if isinstance(call_id, str) else None
            if name:
                out["name"] = name
        content = msg.get("content")
        if not isinstance(content, list):
            normalized.append(out)
            continue

        validated: list[dict] = []
        all_text = True
        for part in content:
            v = _validate_part(part, idx, supports_image=supports_image, supports_audio=supports_audio)
            if v is None:
                continue
            validated.append(v)
            if v.get("type") != "text":
                all_text = False

        if all_text:
            out["content"] = "\n".join(p["text"] for p in validated)
        else:
            out["content"] = validated
        normalized.append(out)
    return normalized


def _validate_part(
    part: Any,
    msg_idx: int,
    *,
    supports_image: bool,
    supports_audio: bool,
) -> dict | None:
    if isinstance(part, str):
        return {"type": "text", "text": part}

    if not isinstance(part, dict):
        raise UnsupportedContentError(
            f"messages[{msg_idx}].content: each part must be an object or string, got {type(part).__name__}"
        )

    ptype = part.get("type")
    if ptype is None:
        raise UnsupportedContentError(f"messages[{msg_idx}].content: part is missing required 'type' field")

    if ptype in _TEXT_TYPES:
        text = part.get("text") or part.get(ptype)
        if text is None:
            logger.warning("messages[%d].content: skipping empty %r part", msg_idx, ptype)
            return None
        if not isinstance(text, str):
            raise UnsupportedContentError(f"messages[{msg_idx}].content: {ptype!r} part must carry string content")
        return {"type": "text", "text": text}

    if ptype in _IMAGE_TYPES:
        if not supports_image:
            raise UnsupportedContentError(f"messages[{msg_idx}].content: this model does not support image input")
        # Both types carry the URL under `image_url`: bare string for input_image, {"url": ...} for image_url.
        img = part.get("image_url")
        if img is None:
            logger.warning("messages[%d].content: skipping empty %r part", msg_idx, ptype)
            return None
        if isinstance(img, str):
            url, detail = img, part.get("detail")
        elif isinstance(img, dict):
            url, detail = img.get("url"), img.get("detail")
        else:
            raise UnsupportedContentError(
                f"messages[{msg_idx}].content: {ptype!r} must be a URL string or an object with a 'url' field"
            )
        if not isinstance(url, str) or not url:
            raise UnsupportedContentError(
                f"messages[{msg_idx}].content: {ptype!r}.url must be a non-empty string (http(s) URL or data: URI)"
            )
        if ptype == "image_url" and isinstance(img, dict):
            return part
        image_url: dict[str, Any] = {"url": url}
        if isinstance(detail, str):
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    if ptype in _AUDIO_TYPES:
        if not supports_audio:
            raise UnsupportedContentError(f"messages[{msg_idx}].content: this model does not support audio input")
        if ptype == "audio_url":
            audio = part.get("audio_url")
            if audio is None:
                logger.warning("messages[%d].content: skipping empty 'audio_url' part", msg_idx)
                return None
            url = audio if isinstance(audio, str) else (audio.get("url") if isinstance(audio, dict) else None)
            if not isinstance(url, str) or not url:
                raise UnsupportedContentError(
                    f"messages[{msg_idx}].content: 'audio_url.url' must be a non-empty string"
                )
            return part
        audio = part.get("input_audio")
        if audio is None:
            logger.warning("messages[%d].content: skipping empty 'input_audio' part", msg_idx)
            return None
        if (
            not isinstance(audio, dict)
            or not isinstance(audio.get("data"), str)
            or not isinstance(audio.get("format"), str)
        ):
            raise UnsupportedContentError(
                f"messages[{msg_idx}].content: 'input_audio' must be an object with string 'data' and 'format' fields"
            )
        return part

    raise UnsupportedContentError(f"messages[{msg_idx}].content: unsupported content part type {ptype!r}")


async def truncate_messages_to_fit(
    messages: list[dict],
    budget: int,
    count_tokens: Callable[[list[dict]], Awaitable[int]],
) -> list[dict]:
    """Drop oldest messages until ``count_tokens`` of the kept list fits ``budget``,
    always keeping leading system/developer messages, the final message, and each
    tool result's preceding call. Binary-searches the drop point, assuming
    ``count_tokens`` never rises as messages are dropped."""
    if budget <= 0 or await count_tokens(messages) <= budget:
        return messages
    head = 0
    while head < len(messages) and messages[head].get("role") in ("system", "developer"):
        head += 1
    leading, rest = messages[:head], messages[head:]
    if len(rest) <= 1:
        return messages
    max_drop = len(rest) - 1
    while max_drop > 0 and rest[max_drop].get("role") == "tool":
        max_drop -= 1
    starts = [i for i in range(1, max_drop + 1) if rest[i].get("role") != "tool"]
    lo, hi, drop = 0, len(starts) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if await count_tokens(leading + rest[starts[mid] :]) <= budget:
            drop, hi = starts[mid], mid - 1
        else:
            lo = mid + 1
    if drop is None:
        return leading + rest[max_drop:]
    return leading + rest[drop:]


@dataclass(frozen=True)
class ParsedChatOutput:
    """Aggregate result of parsing a model's full chat-completion text.

    This is a loader-agnostic 3-field DTO representing the parsed output.
    """

    content: str | None
    reasoning: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def build_from_parsed(
    *,
    request_id: str,
    model_name: str,
    choices: list[ParsedChatOutput],
    usage: UsageInfo,
    finish_reasons: list[str | None] | str | None = None,
    created: int | None = None,
    logprobs: list[Any] | None = None,
) -> ChatCompletionResponse:
    """Build a ChatCompletionResponse from parsed choice DTOs.

    Allows multi-choice responses from day one.
    """
    if created is None:
        created = int(time.time())

    response_choices = []
    for idx, parsed in enumerate(choices):
        # Determine finish reason for this choice
        if isinstance(finish_reasons, list) and idx < len(finish_reasons):
            fr = finish_reasons[idx]
        elif isinstance(finish_reasons, str):
            fr = finish_reasons
        else:
            fr = "tool_calls" if parsed.has_tool_calls else "stop"

        choice_logprobs = None
        if logprobs is not None and idx < len(logprobs):
            choice_logprobs = logprobs[idx]

        response_choices.append(
            ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=parsed.content,
                    reasoning=parsed.reasoning,
                    tool_calls=parsed.tool_calls,
                ),
                logprobs=choice_logprobs,
                finish_reason=fr,
            )
        )

    return ChatCompletionResponse(
        id=request_id,
        model=model_name,
        choices=response_choices,
        usage=usage,
        created=created,
    )


def encode_chat_sse_chunk(chunk: ChatCompletionStreamResponse) -> str:
    """Encode one chat-completion stream chunk as an SSE `data:` line."""
    return f"data: {chunk.model_dump_json()}\n\n"
