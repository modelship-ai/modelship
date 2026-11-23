"""One realtime response: the pipeline's LLM answers through the gateway's Responses path, and the client gets its
answer as Realtime events. Reasoning reaches neither the client nor the conversation."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ray.exceptions import RayActorError, RayTaskError

from modelship.logging import get_logger
from modelship.metrics import RequestTracker
from modelship.openai.protocol import ErrorResponse
from modelship.openai.protocol.error import validation_error_from_cause
from modelship.openai.protocol.realtime import (
    RealtimeError,
    RealtimeResponse,
    ResponseStatus,
    StatusDetails,
    error_event,
    response_event,
    server_event,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.openai.realtime.conversation import Conversation

logger = get_logger("realtime")

Send = Callable[[dict], Awaitable[None]]

_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})
_PART_EVENTS = frozenset(
    {
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
    }
)


@dataclass
class LlmCall:
    """A response's call to its LLM."""

    handle: Any
    request: ResponsesRequest
    headers: dict[str, str]
    # the disconnect registry; None without one
    registry: Any
    request_id: str
    identity: str
    session_id: str


def _usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Responses usage in the Realtime shape: all text."""
    if usage is None:
        return None
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "total_tokens": usage["total_tokens"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "input_token_details": {
            "text_tokens": usage["input_tokens"],
            "audio_tokens": 0,
            "cached_tokens": cached,
            "cached_tokens_details": {"text_tokens": cached, "audio_tokens": 0},
        },
        "output_token_details": {"text_tokens": usage["output_tokens"], "audio_tokens": 0},
    }


class ActiveResponse:
    """A response in progress, sending its events through *send*. Its output joins *conversation*, unless that is
    None."""

    def __init__(self, response: RealtimeResponse, conversation: Conversation | None, call: LlmCall, send: Send):
        self.response = response
        self._conversation = conversation
        self._call = call
        self._send = send
        # its message items by id; reasoning and tool calls have none
        self._items: dict[str, dict[str, Any]] = {}
        self._terminal: dict[str, Any] | None = None
        self._started = False
        self._cancelled = False
        self._finishing = False
        self.task = asyncio.ensure_future(self._run())

    def done(self) -> bool:
        return self.task.done()

    def cancel(self) -> None:
        """Ends the response early, with a `response.done` of status cancelled."""
        if self._cancelled or self._finishing:
            return
        self._cancelled = True
        # a task cancelled before its first step never runs `_run`, so it would send nothing
        if self._started:
            self.task.cancel()

    async def abandon(self) -> None:
        """Stops the response and sends nothing more: the client has left."""
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    async def _run(self) -> None:
        self._started = True
        tracker = RequestTracker(self._call.request.model or "", "realtime_response")
        try:
            await self._send(response_event("response.created", self.response))
            if self._cancelled:
                raise asyncio.CancelledError
            failure = await self._generate()
        except asyncio.CancelledError:
            tracker.finish("cancelled")
            if not self._cancelled:
                raise
            self._finishing = True
            await self._end("cancelled", StatusDetails(type="cancelled", reason="client_cancelled"))
            return
        except RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, BaseException) else None
            if isinstance(cause, ValueError):
                error = validation_error_from_cause(cause).error
                failure = (RealtimeError(message=error.message, type=error.type, code=error.code), "validation_error")
            else:
                logger.exception("Realtime response %s failed", self.response.id)
                failure = (_server_error(), "unhandled")
        except Exception:
            logger.exception("Realtime response %s failed", self.response.id)
            failure = (_server_error(), "unhandled")
        self._finishing = True
        terminal = self._terminal
        if failure is None and terminal is None:
            failure = (_server_error(), "stream_error")
        if failure is None and terminal is not None and terminal.get("status") == "failed":
            error = terminal.get("error") or {}
            code = error.get("code") or "server_error"
            kind = "server_error" if code == "server_error" else "invalid_request_error"
            failure = (RealtimeError(message=error.get("message") or "", type=kind, code=code), "stream_error")
        if failure is not None:
            error, error_type = failure
            tracker.finish("error", error_type)
            await self._send(error_event(error))
            await self._end("failed", StatusDetails(type="failed", error={"type": error.type, "code": error.code}))
            return
        assert terminal is not None
        self.response.usage = _usage(terminal.get("usage"))
        tracker.finish("ok")
        if terminal.get("status") == "incomplete":
            reason = (terminal.get("incomplete_details") or {}).get("reason")
            known = reason if reason in ("max_output_tokens", "content_filter") else None
            await self._end("incomplete", StatusDetails(type="incomplete", reason=known))
            return
        await self._end("completed", None)

    async def _generate(self) -> tuple[RealtimeError, str] | None:
        """Streams the LLM's answer to the client. The error a loader answered with instead, or None."""
        call = self._call
        stream = call.handle.respond.options(stream=True, session_id=call.session_id).remote(
            call.request, call.headers, call.registry, call.request_id, call.identity
        )
        failure = None
        exhausted = False
        try:
            async for event in stream:
                if isinstance(event, ErrorResponse):
                    error = event.error
                    failure = (
                        RealtimeError(message=error.message, type=error.type, code=error.code),
                        "inference_error",
                    )
                    continue
                await self._on_event(event)
            exhausted = True
        finally:
            if not exhausted:
                stream.cancel()
                await self._stop_generation()
        return failure

    async def _stop_generation(self) -> None:
        if self._call.registry is None:
            return
        try:
            await self._call.registry.set.remote(self._call.request_id)
        except RayActorError:
            logger.warning("Disconnect registry unavailable; lost the cancel of %s", self._call.request_id)

    async def _on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in _TERMINAL:
            self._terminal = event["response"]
            return
        if kind == "response.output_item.added":
            if event["item"].get("type") == "message":
                await self._add_item(event["item"]["id"])
            return
        item_id = event["item"]["id"] if kind == "response.output_item.done" else event.get("item_id")
        item = self._items.get(item_id or "")
        if item is None:
            return
        where = {"response_id": self.response.id, "item_id": item["id"], "output_index": self._index(item)}
        if kind in _PART_EVENTS:
            where["content_index"] = event["content_index"]
        if kind == "response.content_part.added":
            item["content"].append({"type": "output_text", "text": ""})
            await self._send(server_event(kind, **where, part={"type": "text", "text": ""}))
        elif kind == "response.output_text.delta":
            item["content"][-1]["text"] += event["delta"]
            await self._send(server_event(kind, **where, delta=event["delta"]))
        elif kind == "response.output_text.done":
            await self._send(server_event(kind, **where, text=event["text"]))
        elif kind == "response.content_part.done":
            await self._send(server_event(kind, **where, part={"type": "text", "text": event["part"]["text"]}))
        elif kind == "response.output_item.done":
            item["status"] = event["item"]["status"]
            await self._item_done(item)

    def _index(self, item: dict[str, Any]) -> int:
        return next(i for i, other in enumerate(self.response.output) if other is item)

    async def _add_item(self, item_id: str) -> None:
        item = {
            "id": item_id,
            "type": "message",
            "object": "realtime.item",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        self._items[item_id] = item
        self.response.output.append(item)
        await self._send(
            server_event(
                "response.output_item.added", response_id=self.response.id, output_index=self._index(item), item=item
            )
        )
        if self._conversation is not None:
            before = self._conversation.insert(item)
            await self._send(server_event("conversation.item.added", previous_item_id=before, item=item))

    async def _item_done(self, item: dict[str, Any]) -> None:
        await self._send(
            server_event(
                "response.output_item.done", response_id=self.response.id, output_index=self._index(item), item=item
            )
        )
        if self._conversation is not None and self._conversation.get(item["id"]) is not None:
            before = self._conversation.before(item["id"])
            await self._send(server_event("conversation.item.done", previous_item_id=before, item=item))

    async def _end(self, status: ResponseStatus, details: StatusDetails | None) -> None:
        for item in self.response.output:
            if item["status"] == "in_progress":
                item["status"] = "incomplete"
                await self._item_done(item)
        self.response.status = status
        self.response.status_details = details
        await self._send(response_event("response.done", self.response))


def _server_error() -> RealtimeError:
    return RealtimeError(message="Internal error during generation.", type="server_error", code="server_error")
