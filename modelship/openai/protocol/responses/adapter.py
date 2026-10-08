"""Translation between the Responses API and chat completions: request-side shaping
(``responses_request_to_chat``) plus the response-envelope helpers every loader's
native ``create_response`` and the streaming translator share.

Never import from the top-level ``modelship.openai.protocol`` package here — it
imports this module, so that would be a cycle.
"""

from __future__ import annotations

from typing import Any, Literal

from cryptography.fernet import InvalidToken

from modelship.logging import get_logger
from modelship.openai import compaction_crypto
from modelship.openai.protocol.chat import ChatCompletionRequest, StreamOptions
from modelship.openai.protocol.responses.schemas import (
    McpApprovalRequestItem,
    McpCallItem,
    McpListToolsItem,
    ResponseFunctionToolCall,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputTokensDetails,
    ResponseReasoningItem,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.protocol.usage import UsageInfo

logger = get_logger("openai.protocol.responses.adapter")


class UnsupportedResponsesFeatureError(ValueError):
    """Unsupported Responses feature; subclasses ``ValueError`` so ``create_error_response`` maps it to a 400."""


# ---------------------------------------------------------------------------
# Request: Responses → chat
# ---------------------------------------------------------------------------


def responses_request_to_chat(request: ResponsesRequest) -> ChatCompletionRequest:
    """Translate a ``ResponsesRequest`` into a ``ChatCompletionRequest``.

    ``previous_response_id`` is already resolved into ``input`` by the gateway before
    the Ray hop; it survives here only to be echoed back. Raises
    :class:`UnsupportedResponsesFeatureError` for features the adapter can't fulfill.
    ``background`` is handled entirely at the gateway; here it's just an echoed flag.
    """
    messages = messages_from_input(request.input, request.instructions)

    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": bool(request.stream),
    }
    if request.stream:
        # Needed for the streaming translator to get a usage-bearing final chunk; only
        # legal when stream=True, so this must stay conditional.
        kwargs["stream_options"] = StreamOptions(include_usage=True)
    if request.max_output_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = request.parallel_tool_calls
    if request.user is not None:
        kwargs["user"] = request.user

    tools = _tools_to_chat(request.tools)
    if tools is not None and request.tool_choice != "none":
        kwargs["tools"] = tools
    tool_choice = _tool_choice_to_chat(request.tool_choice)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response_format = _response_format_from_text(request.text)
    if response_format is not None:
        kwargs["response_format"] = response_format

    effort = (request.reasoning or {}).get("effort")
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    return ChatCompletionRequest(**kwargs)


def messages_from_input(input_: str | list[dict[str, Any]], instructions: str | None) -> list[dict[str, Any]]:
    """Build chat messages from Responses ``input``, merging every system-level item
    (``instructions`` plus any ``system``- or ``developer``-role item found anywhere in
    ``input``, at any nesting depth) into a single leading system message. Some chat
    templates (e.g. Qwen3.5) hard-reject a second system-level message or one not in
    slot 0; clients like Codex CLI send ``instructions``, a separate ``system``-role
    item, *and* a ``developer``-role item (OpenAI's system-equivalent role for newer
    models), so a naive per-item translation trips that template restriction."""
    system_parts: list[str] = [instructions] if instructions else []
    messages = _messages_from_items(input_, system_parts)
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return messages


def _messages_from_items(input_: str | list[dict[str, Any]], system_parts: list[str]) -> list[dict[str, Any]]:
    if isinstance(input_, str):
        return [{"role": "user", "content": input_}]

    messages: list[dict[str, Any]] = []
    # call_ids seen from a function_call earlier in this list; an unmatched
    # function_call_output is orphaned. Reset per call, not shared across recursion.
    seen_call_ids: set[str] = set()
    for item in input_:
        itype = item.get("type")
        if itype == "function_call":
            # call_id and name identify the call; both required or downstream tool-call handling breaks.
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if not call_id or not name:
                raise UnsupportedResponsesFeatureError(
                    "function_call input items require both 'call_id' (or 'id') and 'name'."
                )
            seen_call_ids.add(call_id)
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif itype == "mcp_call":
            # Mirrors the function_call branch: an assistant tool_calls message plus its
            # tool-result message, synthesized as a pair since mcp_call has no call_id.
            call_id = item.get("id") or ""
            name = item.get("name")
            if not call_id or not name:
                raise UnsupportedResponsesFeatureError("mcp_call input items require both 'id' and 'name'.")
            output, error = item.get("output"), item.get("error")
            if output is None and error is None:
                raise UnsupportedResponsesFeatureError("mcp_call input items require one of 'output' or 'error'.")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
            content = _text_of(error) if error is not None else _text_of(output)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        elif itype in ("mcp_list_tools", "mcp_approval_request", "mcp_approval_response"):
            # Not prompt material: mcp_list_tools is discovery bookkeeping, and approval
            # request/response are consumed by the MCP loop before the loader ever sees them.
            continue
        elif itype == "function_call_output":
            # call_id must reference a function_call seen earlier — an orphaned tool result should 400, not silently pass through.
            call_id = item.get("call_id")
            if not call_id:
                raise UnsupportedResponsesFeatureError("function_call_output input items require 'call_id'.")
            content = _text_of(item.get("output"))
            if call_id not in seen_call_ids:
                raise UnsupportedResponsesFeatureError(
                    f"function_call_output references unknown call_id {call_id!r} (no matching function_call)."
                )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        elif itype == "reasoning":
            # Don't replay raw chain-of-thought back into the prompt.
            continue
        elif itype == "compaction":
            # Round-trip half of /v1/responses/compact: decrypt the opaque blob back
            # into the items it was built from. Nesting is rejected below rather than
            # trusted, since the crypto layer can't tell a forged blob from a real one.
            encrypted_content = item.get("encrypted_content")
            if not encrypted_content:
                raise UnsupportedResponsesFeatureError("compaction input items require 'encrypted_content'.")
            try:
                decoded_items = compaction_crypto.decrypt_items(encrypted_content)
            except InvalidToken:
                raise UnsupportedResponsesFeatureError("compaction item could not be decoded.") from None
            if any(isinstance(d, dict) and d.get("type") == "compaction" for d in decoded_items):
                raise UnsupportedResponsesFeatureError("compaction items cannot nest another compaction item.")
            messages.extend(_messages_from_items(decoded_items, system_parts))
        elif itype == "message" or "role" in item:
            role = item.get("role", "user")
            if role in ("system", "developer"):
                text = _text_of(item.get("content"))
                if text:
                    system_parts.append(text)
                continue
            messages.append({"role": role, "content": _content_to_chat(item.get("content"))})
        else:
            raise UnsupportedResponsesFeatureError(f"unsupported input item type {itype!r}.")

    return messages


def _text_of(content: Any) -> str | None:
    """Reduce a Responses content value to plain text (tool-result content, always text)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "".join(parts) if parts else None
    raise UnsupportedResponsesFeatureError(f"unsupported content shape {content!r}.")


_CONTENT_TEXT_TYPES = frozenset({"input_text", "output_text", "text"})
_CONTENT_IMAGE_TYPES = frozenset({"input_image", "image_url"})


def _content_to_chat(content: Any) -> str | list[dict[str, Any]] | None:
    """Translate a Responses message ``content`` value into chat message content.

    Unlike ``_text_of``, this preserves images: ``input_image`` becomes chat's
    ``image_url`` shape. Unknown part types are rejected rather than dropped.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise UnsupportedResponsesFeatureError(f"unsupported content shape {content!r}.")

    parts: list[dict[str, Any]] = []
    for p in content:
        if not isinstance(p, dict):
            raise UnsupportedResponsesFeatureError(f"unsupported content part {p!r}.")
        ptype = p.get("type")
        if ptype in _CONTENT_TEXT_TYPES:
            parts.append({"type": "text", "text": p.get("text")})
        elif ptype in _CONTENT_IMAGE_TYPES:
            # Both part types carry the URL/data-URI under `image_url` (string or {"url": ...}).
            url = p.get("image_url")
            parts.append({"type": "image_url", "image_url": {"url": url} if isinstance(url, str) else url})
        else:
            raise UnsupportedResponsesFeatureError(f"unsupported content part type {ptype!r}.")

    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] or "" for part in parts)
    return parts


def _tools_to_chat(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype == "mcp":
            # The gateway expands mcp tools into plain function tools before the Ray hop;
            # reaching here means a loader was called directly, bypassing that expansion.
            raise UnsupportedResponsesFeatureError(
                "mcp tools are handled at the gateway and must not reach the loader directly."
            )
        if ttype != "function":
            # Hosted tool types (e.g. OpenAI's own web_search, or Codex CLI's
            # 'namespace') have no client-defined equivalent a self-hosted backend can
            # fulfill. Drop them rather than failing the whole request over a
            # capability the client didn't ask the model to actually use — llama.cpp's
            # server took the same approach for the same Codex CLI incompatibility
            # (ggml-org/llama.cpp#23041).
            logger.warning("dropping unsupported Responses tool type %r", ttype)
            continue
        # Responses flattens the function fields onto the tool; chat nests them.
        fn = {k: tool[k] for k in ("name", "description", "parameters", "strict") if k in tool}
        out.append({"type": "function", "function": fn})
    return out or None


def _tool_choice_to_chat(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    raise UnsupportedResponsesFeatureError(f"tool_choice type {tool_choice.get('type')!r} is not supported.")


def _response_format_from_text(text: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate Responses ``text.format`` into chat ``response_format``."""
    if not text:
        return None
    fmt = text.get("format")
    if not fmt:
        return None
    if not isinstance(fmt, dict):
        raise UnsupportedResponsesFeatureError(
            f"text.format must be an object with a 'type' field, got {type(fmt).__name__}."
        )
    ftype = fmt.get("type")
    if ftype == "json_schema":
        # Responses flattens name/schema/strict; chat nests them under json_schema.
        nested = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        return {"type": "json_schema", "json_schema": nested}
    if ftype in ("json_object", "text"):
        return {"type": ftype}
    raise UnsupportedResponsesFeatureError(f"text.format type {ftype!r} is not supported.")


# ---------------------------------------------------------------------------
# Response envelope: shared by every loader's native create_response and by
# the streaming translator.
# ---------------------------------------------------------------------------


def build_response_object(
    request: ResponsesRequest,
    *,
    status: str,
    output: list[Any],
    usage: ResponseUsage | None,
    incomplete: dict[str, Any] | None,
    model: str | None = None,
    response_id: str | None = None,
    created_at: int | None = None,
    completed_at: int | None = None,
    error: Any | None = None,
    background: bool = False,
) -> ResponseObject:
    """Build a ``ResponseObject``, echoing the request settings OpenAI returns. Shared
    by the non-streaming adapter and the streaming translator for an identical envelope
    shape. ``response_id``/``created_at`` let streaming keep one stable id across events."""
    kwargs: dict[str, Any] = {
        "model": model or request.model or "",
        "status": status,
        "output": output,
        "usage": usage,
        "incomplete_details": incomplete,
        "background": background,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        # Effective values, not a bare echo: OpenAI's own defaults when unset.
        "temperature": request.temperature if request.temperature is not None else 1.0,
        "top_p": request.top_p if request.top_p is not None else 1.0,
        "tools": _echo_tools(request.tools),
        "tool_choice": request.tool_choice if request.tool_choice is not None else "auto",
        "parallel_tool_calls": request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        "text": request.text if request.text else {"format": {"type": "text"}},
        "reasoning": request.reasoning,
        "metadata": request.metadata or {},
        # OpenAI stores by default; only an explicit `store: false` opts out.
        "store": request.store is not False,
        "previous_response_id": request.previous_response_id,
    }
    if response_id is not None:
        kwargs["id"] = response_id
    if created_at is not None:
        kwargs["created_at"] = created_at
    if completed_at is not None:
        kwargs["completed_at"] = completed_at
    if error is not None:
        kwargs["error"] = error
    return ResponseObject(**kwargs)


def _echo_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Backfill echoed ``tools[]`` so every ``FunctionTool`` carries all five spec keys
    (the request may omit optional ones, but the echo requires them present, nullable).
    ``mcp`` tools have their credentials scrubbed (OpenAI redacts the same way) since
    this echo is both returned to the client and written into the state-store snapshot."""
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype == "mcp":
            scrubbed = {**tool, "headers": None}
            scrubbed.pop("authorization", None)
            out.append(scrubbed)
            continue
        if ttype != "function":
            out.append(tool)
            continue
        out.append(
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                "strict": tool.get("strict"),
            }
        )
    return out


def _usage_from_chat(usage: UsageInfo) -> ResponseUsage:
    """Remap chat usage to Responses usage; a direct field copy since both use the same
    ``cached_tokens``/``reasoning_tokens`` sub-field names. Loaders reporting no details
    leave them at the zero default."""
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return ResponseUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens,
        input_tokens_details=ResponseInputTokensDetails(
            cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0
        ),
        output_tokens_details=ResponseOutputTokensDetails(
            reasoning_tokens=(completion_details.reasoning_tokens or 0) if completion_details else 0
        ),
    )


_OUTPUT_ITEM_CLASSES: dict[str, type[ResponseOutputItem]] = {
    "reasoning": ResponseReasoningItem,
    "message": ResponseOutputMessage,
    "function_call": ResponseFunctionToolCall,
    "mcp_list_tools": McpListToolsItem,
    "mcp_call": McpCallItem,
    "mcp_approval_request": McpApprovalRequestItem,
}


def parse_output_item(d: dict[str, Any]) -> ResponseOutputItem:
    """Rebuild a typed output item from a dict, dispatching explicitly on ``type``.
    ``ResponseOutputItem`` is a plain (non-discriminated) union, so letting pydantic
    smart-match a dict against it risks mis-coercion between the similarly-shaped
    item classes."""
    cls = _OUTPUT_ITEM_CLASSES.get(d.get("type", ""))
    if cls is None:
        raise UnsupportedResponsesFeatureError(f"unsupported output item type {d.get('type')!r}.")
    return cls.model_validate(d)


def _status_for(
    finish_reason: str | None,
) -> tuple[Literal["completed", "incomplete"], dict[str, Any] | None]:
    # chat finish_reason -> Responses status + incomplete_details.reason.
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None
