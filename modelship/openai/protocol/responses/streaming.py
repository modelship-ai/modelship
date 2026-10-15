"""Streaming translation: typed chat chunks → Responses event stream.

``vllm``/``llama_server`` feed :class:`ResponsesStreamTranslator` directly with
typed ``ChatCompletionStreamResponse`` chunks (the same DTO their chat streaming
path produces) — no SSE text round-trip. The Responses API wants a *semantic*
event protocol instead: named events (``response.created``,
``response.output_text.delta``, …) each carrying a monotonically increasing
``sequence_number`` plus the relevant output index, wrapping each output item
in explicit added/done brackets.

:class:`ResponsesStreamTranslator` consumes the parsed chat chunks and brackets
them. Design choices that matter:

- **Three flat channels, no nesting.** Each chat ``DeltaMessage`` carries
  independent ``content`` / ``reasoning`` / ``tool_calls`` fields; the upstream
  scanner already resolved any marker nesting. Responses ``output`` is likewise
  a flat list of items. So we map channels → items 1:1 (one message item, one
  reasoning item, one ``function_call`` per tool index).
- **Open on first delta, close at ``finish``.** No ordering is assumed between
  channels (a model may interleave answer/tool/answer text). An item's bracket
  is opened the first time its channel produces a delta and stays open until the
  stream ends, so late text on an already-seen channel always has a home. The
  only cost is that an earlier item's ``output_item.done`` arrives at the end
  rather than the instant the next item starts — spec-valid, and SDK accumulators
  key off ``item_id`` so they reconstruct identically.

Imports stay within the package's leaf submodules (``schemas`` + ``adapter``),
never the top-level ``modelship.openai.protocol`` package, to avoid an import
cycle (that package imports this one).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any

from modelship.openai.protocol.base import random_uuid
from modelship.openai.protocol.chat import ChatCompletionStreamResponse, DeltaToolCall
from modelship.openai.protocol.error import ErrorResponse
from modelship.openai.protocol.responses.adapter import (
    _status_for,
    _usage_from_chat,
    build_response_object,
)
from modelship.openai.protocol.responses.schemas import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummary,
    ResponsesRequest,
)

# Terminal events carrying a complete response object. Both are continuable, so both
# are worth persisting; ``response.failed`` deliberately is not.
TERMINAL_EVENT_TYPES = ("response.completed", "response.incomplete")


def store_failure_event(terminal_payload: dict[str, Any], message: str) -> dict[str, Any]:
    """Rewrite a terminal completed/incomplete event as ``response.failed``.

    The gateway persists a stored response *before* forwarding its terminal event, so
    a store failure can still change what the client is told. Generation did succeed,
    but the response is uncontinuable — reporting completion would hand back an id
    that 404s on the next turn. Reuses the terminal event's sequence number because it
    replaces that event rather than following it.
    """
    response = {
        **(terminal_payload.get("response") or {}),
        "status": "failed",
        "error": {"message": message},
    }
    return {
        "type": "response.failed",
        "sequence_number": terminal_payload.get("sequence_number", 0),
        "response": response,
    }


def _frame_payload(frame: Any) -> dict[str, Any]:
    """Best-effort decode of an SSE frame's ``data:`` JSON. ``{}`` for anything that
    isn't one — a non-str chunk, the ``[DONE]`` terminator, malformed JSON."""
    if not isinstance(frame, str):
        return {}
    for line in frame.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[len("data: ") :])
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def failure_event_from_frames(created_frame: Any, last_frame: Any, message: str, *, model: str = "") -> dict[str, Any]:
    """Terminal ``response.failed`` for a stream that died outside the loader, where no
    translator survives to call :meth:`ResponsesStreamTranslator.fail`. Rebuilds the
    envelope from the frames already sent: ``response.created`` carries id/created_at/
    model, the last frame the sequence number this one must follow. Output is dropped —
    the frames carry it as deltas, not items."""
    response = dict(_frame_payload(created_frame).get("response") or {})
    response.update(status="failed", error={"message": message}, output=[])
    response.setdefault("object", "response")
    response.setdefault("id", f"resp_{random_uuid()}")
    response.setdefault("created_at", int(time.time()))
    response.setdefault("model", model)
    seq = _frame_payload(last_frame).get("sequence_number")
    return {
        "type": "response.failed",
        "sequence_number": seq + 1 if isinstance(seq, int) else 0,
        "response": response,
    }


def encode_sse_event(event: dict[str, Any]) -> str:
    """Wire-format one Responses event dict as an SSE frame (named event line + JSON data line)."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


async def frame_sse(gen: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
    """HTTP transport edge: SSE-frame each event dict in *gen*, passing anything else
    (a non-streaming ``ResponseObject``, a pre-generation ``ErrorResponse``) through
    unchanged. Appends ``data: [DONE]\\n\\n`` once the stream ends, but only if
    something was actually framed.
    """
    framed_any = False
    async for item in gen:
        if isinstance(item, dict):
            yield encode_sse_event(item)
            framed_any = True
        else:
            yield item
    if framed_any:
        yield "data: [DONE]\n\n"


def error_ws_frame(err: ErrorResponse) -> str:
    """WS transport edge: render an ``ErrorResponse`` as a ``webSocketErrorEventSchema``
    text frame. Falls back to the error ``type`` for ``code`` since the WS schema requires it non-null.
    """
    error = err.error.model_dump(mode="json", exclude_none=True)
    if not error.get("code"):
        error["code"] = err.error.type
    return json.dumps({"type": "error", "status": err._http_status, "error": error})


class ResponsesStreamTranslator:
    """Stateful translator from chat stream chunks to Responses events.

    One instance per request. Emit :meth:`start` first, feed every parsed chunk
    to :meth:`process`, then emit :meth:`finish` once the chat stream ends.

    Transport-neutral: every method yields plain event dicts, not pre-framed SSE
    strings — framing happens at the gateway edge (see :func:`frame_sse`).
    """

    def __init__(self, request: ResponsesRequest, *, response_id: str | None = None):
        self.request = request
        # Injected for a background run to keep the same id across placeholder/stream/record.
        self.response_id = response_id or f"resp_{random_uuid()}"
        self.created_at: int | None = None  # pinned after the first envelope
        self.model = request.model or ""
        self._seq = 0
        self._next_oi = 0

        self.usage = None
        self.finish_reason: str | None = None

        # One reasoning item (mapped to a single summary part).
        self._reasoning: ResponseReasoningItem | None = None
        self._reasoning_oi = -1
        self._reasoning_text = ""

        # One assistant message item (single output_text part).
        self._message: ResponseOutputMessage | None = None
        self._message_oi = -1
        self._message_text = ""

        # Tool calls, keyed by their chat-stream delta index.
        self._tools: dict[int, ResponseFunctionToolCall] = {}
        self._tool_args: dict[int, str] = {}
        self._tool_oi: dict[int, int] = {}

    # -- low-level helpers --------------------------------------------------

    def _event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = {"type": event_type, "sequence_number": self._seq, **payload}
        self._seq += 1
        return out

    def _take_oi(self) -> int:
        oi = self._next_oi
        self._next_oi += 1
        return oi

    def _envelope(
        self,
        event_type: str,
        status: str,
        output: list[Any],
        usage,
        incomplete,
        error: Any | None = None,
        completed_at: int | None = None,
    ) -> dict[str, Any]:
        response = build_response_object(
            self.request,
            status=status,
            output=output,
            usage=usage,
            incomplete=incomplete,
            model=self.model,
            response_id=self.response_id,
            created_at=self.created_at,
            completed_at=completed_at,
            error=error,
            background=bool(self.request.background),
        )
        # Pin created_at after the first build so every envelope is identical.
        self.created_at = response.created_at
        return self._event(event_type, {"response": response.model_dump(mode="json")})

    def _close_all(self) -> Iterator[dict[str, Any]]:
        # Close every open item, in output-index (first-seen) order.
        yield from self._close_reasoning()
        yield from self._close_message()
        yield from self._close_tools()

    def _collect_output(self) -> list[Any]:
        output: list[Any] = []
        if self._reasoning is not None:
            output.append(self._reasoning)
        if self._message is not None:
            output.append(self._message)
        for idx in sorted(self._tools):
            output.append(self._tools[idx])
        return output

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> Iterator[dict[str, Any]]:
        yield self._envelope("response.created", "in_progress", [], None, None)
        yield self._envelope("response.in_progress", "in_progress", [], None, None)

    def process(self, chunk: ChatCompletionStreamResponse) -> Iterator[dict[str, Any]]:
        if chunk.model:
            self.model = chunk.model
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices:
            delta = choice.delta
            if delta.reasoning:
                yield from self._on_reasoning(delta.reasoning)
            if delta.content:
                yield from self._on_content(delta.content)
            for tc in delta.tool_calls or []:
                yield from self._on_tool_call(tc)
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason

    def finish(self) -> Iterator[dict[str, Any]]:
        yield from self._close_all()
        output = self._collect_output()

        status, incomplete = _status_for(self.finish_reason)
        usage = _usage_from_chat(self.usage) if self.usage is not None else None
        terminal = "response.incomplete" if status == "incomplete" else "response.completed"
        completed_at = int(time.time()) if status == "completed" else None
        yield self._envelope(terminal, status, output, usage, incomplete, completed_at=completed_at)

    def fail(self, message: str) -> Iterator[dict[str, Any]]:
        """Terminal ``response.failed`` event for a mid-stream error (a loader
        exception, not a normal completion). Still closes any already-open item
        brackets so a client sees whatever partial content was generated."""
        yield from self._close_all()
        output = self._collect_output()
        yield self._envelope("response.failed", "failed", output, None, None, error={"message": message})

    # -- reasoning channel --------------------------------------------------

    def _on_reasoning(self, text: str) -> Iterator[dict[str, Any]]:
        if self._reasoning is None:
            self._reasoning = ResponseReasoningItem(summary=[])
            self._reasoning_oi = self._take_oi()
            yield self._event(
                "response.output_item.added",
                {"output_index": self._reasoning_oi, "item": self._reasoning.model_dump(mode="json")},
            )
            yield self._event(
                "response.reasoning_summary_part.added",
                {
                    "item_id": self._reasoning.id,
                    "output_index": self._reasoning_oi,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                },
            )
        self._reasoning_text += text
        yield self._event(
            "response.reasoning_summary_text.delta",
            {
                "item_id": self._reasoning.id,
                "output_index": self._reasoning_oi,
                "summary_index": 0,
                "delta": text,
            },
        )

    def _close_reasoning(self) -> Iterator[dict[str, Any]]:
        if self._reasoning is None:
            return
        item = self._reasoning
        yield self._event(
            "response.reasoning_summary_text.done",
            {"item_id": item.id, "output_index": self._reasoning_oi, "summary_index": 0, "text": self._reasoning_text},
        )
        yield self._event(
            "response.reasoning_summary_part.done",
            {
                "item_id": item.id,
                "output_index": self._reasoning_oi,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": self._reasoning_text},
            },
        )
        item.summary = [ResponseReasoningSummary(text=self._reasoning_text)]
        yield self._event(
            "response.output_item.done",
            {"output_index": self._reasoning_oi, "item": item.model_dump(mode="json")},
        )

    # -- message (content) channel -----------------------------------------

    def _on_content(self, text: str) -> Iterator[dict[str, Any]]:
        if self._message is None:
            self._message = ResponseOutputMessage(status="in_progress", content=[])
            self._message_oi = self._take_oi()
            yield self._event(
                "response.output_item.added",
                {"output_index": self._message_oi, "item": self._message.model_dump(mode="json")},
            )
            yield self._event(
                "response.content_part.added",
                {
                    "item_id": self._message.id,
                    "output_index": self._message_oi,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        self._message_text += text
        yield self._event(
            "response.output_text.delta",
            {
                "item_id": self._message.id,
                "output_index": self._message_oi,
                "content_index": 0,
                "delta": text,
            },
        )

    def _close_message(self) -> Iterator[dict[str, Any]]:
        if self._message is None:
            return
        item = self._message
        yield self._event(
            "response.output_text.done",
            {"item_id": item.id, "output_index": self._message_oi, "content_index": 0, "text": self._message_text},
        )
        yield self._event(
            "response.content_part.done",
            {
                "item_id": item.id,
                "output_index": self._message_oi,
                "content_index": 0,
                "part": {"type": "output_text", "text": self._message_text, "annotations": []},
            },
        )
        item.content = [ResponseOutputText(text=self._message_text)]
        item.status = "completed"
        yield self._event(
            "response.output_item.done",
            {"output_index": self._message_oi, "item": item.model_dump(mode="json")},
        )

    # -- tool-call channel --------------------------------------------------

    def _on_tool_call(self, tc: DeltaToolCall) -> Iterator[dict[str, Any]]:
        idx = tc.index
        if idx not in self._tools:
            call_id = tc.id or f"call_{random_uuid()}"
            name = tc.function.name if tc.function else None
            item = ResponseFunctionToolCall(call_id=call_id, name=name or "", arguments="", status="in_progress")
            self._tools[idx] = item
            self._tool_args[idx] = ""
            self._tool_oi[idx] = self._take_oi()
            yield self._event(
                "response.output_item.added",
                {"output_index": self._tool_oi[idx], "item": item.model_dump(mode="json")},
            )
        else:
            item = self._tools[idx]
            # Some loaders send the function name in a later chunk than the id.
            if tc.function and tc.function.name and not item.name:
                item.name = tc.function.name

        if tc.function and tc.function.arguments:
            self._tool_args[idx] += tc.function.arguments
            yield self._event(
                "response.function_call_arguments.delta",
                {"item_id": item.id, "output_index": self._tool_oi[idx], "delta": tc.function.arguments},
            )

    def _close_tools(self) -> Iterator[dict[str, Any]]:
        for idx in sorted(self._tools):
            item = self._tools[idx]
            args = self._tool_args[idx]
            oi = self._tool_oi[idx]
            item.arguments = args
            item.status = "completed"
            yield self._event(
                "response.function_call_arguments.done",
                {"item_id": item.id, "output_index": oi, "arguments": args},
            )
            yield self._event(
                "response.output_item.done",
                {"output_index": oi, "item": item.model_dump(mode="json")},
            )
