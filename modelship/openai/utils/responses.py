"""Utility helpers for ``/v1/responses``: loader-side output shaping plus
gateway-side conversation-state plumbing (history, persistence, records)."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException
from pydantic import ValidationError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

from modelship.infer.infer_config import get_disconnect_registry
from modelship.logging import get_logger
from modelship.metrics import RequestTracker
from modelship.openai import compaction_crypto
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    ResponseObject,
    ResponsesRequest,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.base import random_uuid
from modelship.openai.protocol.responses.adapter import (
    _status_for,
    _usage_from_chat,
    build_response_object,
    messages_from_input,
)
from modelship.openai.protocol.responses.schemas import (
    CompactionItem,
    CompactResource,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummary,
)
from modelship.openai.protocol.responses.streaming import TERMINAL_EVENT_TYPES, store_failure_event
from modelship.openai.state import responses as responses_state
from modelship.openai.utils.chat import ParsedChatOutput
from modelship.state import StateStore, StateStoreUnavailableError

logger = get_logger("openai.utils.responses")

# Shape of the response ids we mint (`resp_<uuid>`). Ids arriving from a client are
# checked against it before becoming a state-store key segment, so a malformed id is
# a clean 404 rather than a lookup for something we could never have written.
RESPONSE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# --- Shaping: parsed chat output -> Responses items (loader-side) ---


def build_responses_items_from_parsed(parsed: ParsedChatOutput) -> list[ResponseOutputItem]:
    """Shape one parsed choice into Responses ``output[]`` items: reasoning, then
    message, then one ``function_call`` per tool call."""
    output: list[ResponseOutputItem] = []
    if parsed.reasoning:
        output.append(ResponseReasoningItem(summary=[ResponseReasoningSummary(text=parsed.reasoning)]))
    if parsed.content:
        output.append(ResponseOutputMessage(content=[ResponseOutputText(text=parsed.content)]))
    for call in parsed.tool_calls:
        output.append(
            ResponseFunctionToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
        )
    return output


def build_response_from_parsed(
    parsed: ParsedChatOutput,
    request: ResponsesRequest,
    *,
    usage: UsageInfo,
    finish_reason: str | None,
    model: str,
) -> ResponseObject:
    """Build a non-streaming ``ResponseObject`` from one loader's parsed chat output.
    Shared status/usage/envelope assembly for every loader's non-streaming `create_response`."""
    status, incomplete = _status_for(finish_reason)
    return build_response_object(
        request,
        status=status,
        output=build_responses_items_from_parsed(parsed),
        usage=_usage_from_chat(usage),
        incomplete=incomplete,
        model=model,
        completed_at=int(time.time()) if status == "completed" else None,
        background=bool(request.background),
    )


# Structured rather than a bare "summarize this" so a continuation loses nothing essential.
_COMPACTION_SYSTEM_PROMPT = (
    "Summarize this conversation so it can be continued from a fresh context window "
    "with nothing essential lost. Structure the summary with these sections, in order: "
    "(1) the user's explicit requests and intent, verbatim where it matters; "
    "(2) key facts, decisions, names, and identifiers established so far; "
    "(3) specific file paths, code, or command output already produced, in enough "
    "detail to avoid re-deriving them; "
    "(4) errors encountered and how they were fixed, including any correction the "
    "user gave about how to approach the task; "
    "(5) work explicitly still pending; "
    "(6) exactly what was being worked on immediately before this summary, so the "
    "next turn picks up from there rather than guessing."
)


def build_summarization_request(model: str, items: list[Any], instructions: str | None = None) -> ChatCompletionRequest:
    """The internal chat request ``/v1/responses/compact`` issues to summarize *items*.
    *instructions*, if given, is inserted as an extra system message."""
    messages = messages_from_input(items, None)
    messages.insert(0, {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT})
    if instructions:
        messages.insert(1, {"role": "system", "content": instructions})
    return ChatCompletionRequest(model=model, messages=messages, stream=False, reasoning_effort="none")


def build_compaction(*, summary_items: list[Any], usage: UsageInfo) -> CompactResource:
    """Build a ``CompactResource`` from a ``/v1/responses/compact`` summarization call.
    ``id``/``created_at`` are freshly minted; a compaction result is never persisted."""
    encrypted_content = compaction_crypto.encrypt_items(summary_items)
    return CompactResource(output=[CompactionItem(encrypted_content=encrypted_content)], usage=_usage_from_chat(usage))


def responses_validation_error(exc: ValidationError) -> ErrorResponse:
    """400 for a pydantic ``ValidationError`` surfaced by ``responses_request_to_chat``,
    same shape as every other rejection."""
    return create_error_response(message=str(exc), err_type="invalid_request_error")


# --- Gateway-side conversation-state plumbing ---


class ResponsesApiError(HTTPException):
    """An ``HTTPException`` that also carries a full OpenAI-shaped ``ErrorResponse``,
    so one raise renders correctly for both the HTTP route and the WS turn-runner."""

    def __init__(self, err: ErrorResponse):
        self.err = err
        super().__init__(status_code=err._http_status, detail=err.error.message)


def _not_found_error(response_id: str, *, previous: bool = False) -> ResponsesApiError:
    if previous:
        # "previous_response_not_found" is OpenAI's actual code for this failure.
        return ResponsesApiError(
            create_error_response(
                f"Previous response with id '{response_id}' not found.",
                err_type="invalid_request_error",
                status_code=HTTPStatus.NOT_FOUND,
                param="previous_response_id",
                code="previous_response_not_found",
            )
        )
    return ResponsesApiError(
        create_error_response(
            f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
        )
    )


def _store_unavailable_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "Conversation state store is unavailable; retry shortly.",
            err_type="api_error",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    )


def _previous_in_progress_error(response_id: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            f"Previous response with id '{response_id}' is still in progress.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param="previous_response_id",
        )
    )


def background_store_false_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "background:true requires store:true; a background response can't be polled if it isn't stored.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param="background",
        )
    )


def _not_background_error(response_id: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            f"Response with id '{response_id}' is not a background response.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    )


def _mark_failed(response: dict, message: str) -> dict:
    return {**response, "status": "failed", "error": {"message": message}}


def as_input_items(input_: str | list[Any]) -> list[Any]:
    """Normalize a Responses ``input`` to item form, so stored history and this turn's
    input concatenate."""
    if isinstance(input_, str):
        return [{"type": "message", "role": "user", "content": input_}]
    return list(input_)


async def resolve_previous(
    store: StateStore, identity: str, previous_response_id: str
) -> tuple[list[Any], dict, dict[str, dict]]:
    """History stored up to *previous_response_id*, and its log position. 404 if unknown, 503 if the store is down."""
    if not RESPONSE_ID_RE.match(previous_response_id):
        raise _not_found_error(previous_response_id, previous=True)
    try:
        record = await responses_state.read_async(store, identity, previous_response_id, with_output=False)
        if record is None:
            raise _not_found_error(previous_response_id, previous=True)
        if record["response"].get("status") in ("queued", "in_progress"):
            # No output yet to continue from.
            raise _previous_in_progress_error(previous_response_id)
        resolved = await responses_state.history_async(store, identity, record)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable resolving previous_response_id=%s", previous_response_id)
        raise _store_unavailable_error() from None
    if resolved is None:
        raise _not_found_error(previous_response_id, previous=True)
    return resolved


async def persist_response(
    gen,
    store: StateStore,
    *,
    identity: str,
    parent: dict | None = None,
    input_items: list[Any] | None = None,
    conditional: bool = False,
):
    """Tee `respond`'s output, storing the turn as it passes. `conditional=True`
    (background mode) routes the terminal write through `write_terminal_if_not_terminal`
    so a concurrent cancel/delete always wins over a genuine completion."""
    async for item in gen:
        if isinstance(item, ResponseObject):
            try:
                if conditional:
                    await responses_state.write_terminal_if_not_terminal(
                        store, identity, item.id, response=item.model_dump(mode="json")
                    )
                else:
                    await responses_state.write_async(
                        store,
                        identity,
                        item.id,
                        response=item.model_dump(mode="json"),
                        parent=parent,
                        input_items=input_items or [],
                    )
            except StateStoreUnavailableError:
                logger.exception("State store unavailable persisting response %s", item.id)
                yield create_error_response(
                    "Conversation state store is unavailable; the response was generated but not stored.",
                    err_type="api_error",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            yield item
            continue

        if not isinstance(item, dict) or item.get("type") not in TERMINAL_EVENT_TYPES:
            yield item
            continue

        response = item.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(response, dict) or not response_id:
            logger.warning("Terminal Responses event has no usable response id; not storing.")
            yield item
            continue

        try:
            if conditional:
                await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=response)
            else:
                await responses_state.write_async(
                    store, identity, response_id, response=response, parent=parent, input_items=input_items or []
                )
        except StateStoreUnavailableError:
            logger.exception("State store unavailable persisting streamed response %s", response.get("id"))
            yield store_failure_event(
                item, "Conversation state store is unavailable; the response was generated but not stored."
            )
            return
        yield item


async def load_record(store: StateStore, identity: str, response_id: str) -> dict:
    """The stored record for *response_id*, scoped to the caller's identity."""
    if not RESPONSE_ID_RE.match(response_id):
        raise _not_found_error(response_id)
    try:
        record = await responses_state.read_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reading response %s", response_id)
        raise _store_unavailable_error() from None
    if record is None:
        raise _not_found_error(response_id)
    return record


async def load_input_items(store: StateStore, identity: str, response_id: str) -> list[dict]:
    """The items that went into *response_id*'s turn, scoped to the caller's identity."""
    if not RESPONSE_ID_RE.match(response_id):
        raise _not_found_error(response_id)
    try:
        record = await responses_state.read_async(store, identity, response_id, with_output=False)
        items = None if record is None else await responses_state.input_items_async(store, identity, record)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reading input items of %s", response_id)
        raise _store_unavailable_error() from None
    if items is None:
        raise _not_found_error(response_id)
    return items


async def delete_record(store: StateStore, identity: str, response_id: str) -> None:
    """Delete the record for *response_id*, and its stream buffer if it had one
    (a no-op if it didn't). Caller must confirm existence first via :func:`load_record`."""
    try:
        await responses_state.delete_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable deleting response %s", response_id)
        raise _store_unavailable_error() from None
    with contextlib.suppress(StateStoreUnavailableError):
        await responses_state.discard_stream_buffer(store, identity, response_id)


# --- Background mode (background:true) ---

# Drain task's heartbeat cadence; state/responses.py's staleness threshold is a multiple of this.
_HEARTBEAT_INTERVAL_S = 5.0

# How often a tailer (live background+stream, or a GET resume) polls the event buffer
# for new entries.
_TAIL_POLL_INTERVAL_S = 0.25

_TERMINAL_EVENT_OR_FAILED = (*TERMINAL_EVENT_TYPES, "response.failed")

# Held so a fire-and-forget buffer-append task isn't GC'd mid-write.
_pending_buffer_appends: set[asyncio.Task] = set()


class _StreamBuffer:
    """Appends events to the replay log one at a time, in event order, without
    blocking the caller on each individual write. `flush_and_discard` waits for
    every scheduled append before dropping the buffer, so a straggler can't land
    after it's gone."""

    def __init__(self, store: StateStore, identity: str, response_id: str) -> None:
        self._store = store
        self._identity = identity
        self._response_id = response_id
        self._pending: asyncio.Task | None = None

    def append(self, event: dict) -> None:
        self._pending = asyncio.ensure_future(self._chained_append(self._pending, event))
        _pending_buffer_appends.add(self._pending)
        self._pending.add_done_callback(_pending_buffer_appends.discard)

    async def _chained_append(self, prev: asyncio.Task | None, event: dict) -> None:
        if prev is not None:
            await prev
        try:
            await responses_state.append_stream_event(self._store, self._identity, self._response_id, event)
        except Exception:
            logger.warning("Failed to buffer event for response %s", self._response_id, exc_info=True)

    async def flush_and_discard(self) -> None:
        if self._pending is not None:
            await self._pending
            self._pending = None
        with contextlib.suppress(StateStoreUnavailableError):
            await responses_state.discard_stream_buffer(self._store, self._identity, self._response_id)


async def buffer_stream_events(gen, buffer: _StreamBuffer):
    """Tee every event dict flowing through *gen* into *buffer*, flushing and
    discarding it once a terminal event passes through."""
    async for item in gen:
        if isinstance(item, dict):
            buffer.append(item)
            if item.get("type") in _TERMINAL_EVENT_OR_FAILED:
                await buffer.flush_and_discard()
        yield item


def _terminal_event_from_response(response: dict, after_sequence: int) -> dict | None:
    """Synthesize the terminal event a tailer missed, from the response's own final
    state. No event exists for `cancelled`; tailing simply ends there."""
    event_type = {
        "completed": "response.completed",
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }.get(response.get("status") or "")
    if event_type is None:
        return None
    return {"type": event_type, "sequence_number": after_sequence + 1, "response": response}


async def tail_background_events(store: StateStore, identity: str, response_id: str, *, after_sequence: int = -1):
    """Replay buffered events after *after_sequence*, then poll until terminal.
    Ends quietly (no error) if the response is deleted mid-tail."""
    while True:
        try:
            events = await responses_state.read_stream_events_after(store, identity, response_id, after_sequence)
        except StateStoreUnavailableError:
            logger.exception("State store unavailable tailing response %s", response_id)
            return
        if events:
            for event in events:
                after_sequence = event.get("sequence_number", after_sequence)
                yield event
            if events[-1].get("type") in _TERMINAL_EVENT_OR_FAILED:
                return
            continue

        try:
            record = await responses_state.read_async(store, identity, response_id)
        except StateStoreUnavailableError:
            logger.exception("State store unavailable tailing response %s", response_id)
            return
        if record is None:
            return
        record = await reconcile_staleness(store, identity, response_id, record)
        response = record["response"]
        if response.get("status") in responses_state.TERMINAL_STATUSES:
            if after_sequence < 0:
                # Nothing has been seen yet (a fresh tail, not a resume from a real
                # cursor). The buffer can be fully written-and-discarded before this
                # tailer's first poll — buffer_stream_events discards it the instant
                # the terminal event passes through, and a tiny/fast generation can
                # finish quicker than _TAIL_POLL_INTERVAL_S — so without this, a
                # fresh connection would see only the terminal event and never
                # response.created. Its envelope must look like the opening one a live
                # connection would have seen — status in_progress, no output/usage yet —
                # not the terminal response dict, which would otherwise leak a completed
                # status and full output into what's supposed to be the opening event.
                created_response = {
                    **response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                    "incomplete_details": None,
                    "error": None,
                    "completed_at": None,
                }
                yield {"type": "response.created", "sequence_number": 0, "response": created_response}
                after_sequence = 0
            terminal = _terminal_event_from_response(response, after_sequence)
            if terminal is not None:
                yield terminal
            return

        await asyncio.sleep(_TAIL_POLL_INTERVAL_S)


async def start_background(
    store: StateStore,
    identity: str,
    request: ResponsesRequest,
    req_id: str,
    parent: dict | None,
    input_items: list[Any],
) -> dict:
    """Mint a response id, persist the initial `queued` placeholder, and return
    the serialized `ResponseObject` to send back to the client immediately."""
    response_id = f"resp_{random_uuid()}"
    response = build_response_object(
        request,
        status="queued",
        output=[],
        usage=None,
        incomplete=None,
        response_id=response_id,
        created_at=int(time.time()),
        background=True,
    )
    response_dict = response.model_dump(mode="json")
    try:
        await responses_state.write_background(
            store, identity, response_id, response=response_dict, parent=parent, input_items=input_items
        )
    except StateStoreUnavailableError:
        logger.exception("State store unavailable creating background response %s", response_id)
        raise _store_unavailable_error() from None
    # Best-effort: a failure here just means the first heartbeat tick (up to
    # _HEARTBEAT_INTERVAL_S later) re-seeds it — the placeholder record above is
    # already the durable source of truth, so this shouldn't fail the whole request.
    with contextlib.suppress(Exception):
        await responses_state.heartbeat(identity, response_id, req_id)
    return response_dict


async def run_background(
    handle: DeploymentHandle,
    request: ResponsesRequest,
    headers: dict[str, str],
    req_id: str,
    identity: str,
    *,
    store: StateStore,
    response_id: str,
    tracker: RequestTracker,
    stream_buffer: bool = False,
    mcp_listings: dict[str, dict] | None = None,
) -> None:
    """Detached-task body for `background:true`: drives the streaming Responses
    generator to completion, persisting every terminal transition. `stream_buffer=True`
    also tees every event into the durable replay log a poller reads from."""
    request.stream = True
    buffer = _StreamBuffer(store, identity, response_id) if stream_buffer else None
    try:
        try:
            registry = get_disconnect_registry()
            # Local import: modelship.openai.mcp.{spec,egress} import ResponsesApiError from this
            # module, so importing mcp.loop at module scope here would be a cycle.
            from modelship.openai.mcp import loop as mcp_loop

            if mcp_loop.wants_mcp(request):
                response_gen = mcp_loop.run_mcp_response(
                    handle,
                    request,
                    headers,
                    registry,
                    req_id,
                    identity,
                    response_id=response_id,
                    mcp_listings=mcp_listings,
                )
            else:
                response_gen = cast(
                    "DeploymentResponseGenerator[Any]",
                    handle.respond.options(stream=True).remote(
                        request, headers, registry, req_id, identity, response_id
                    ),
                )
        except Exception:
            logger.exception("background response %s: dispatch failed", response_id)
            tracker.finish("error", "unhandled")
            await _write_failed(store, identity, response_id, "Generation failed to start.")
            return
        piped = persist_response(response_gen, store, identity=identity, conditional=True)
        if buffer is not None:
            piped = buffer_stream_events(piped, buffer)

        drain_task = asyncio.ensure_future(_drain_background_stream(piped, response_id))
        heartbeat_task = asyncio.ensure_future(_heartbeat_loop(identity, response_id, req_id))
        try:
            done, _pending = await asyncio.wait({drain_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
            if drain_task not in done:
                # Heartbeat found this run cancelled: stop draining, no further write.
                tracker.finish("cancelled")
                drain_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain_task
                if isinstance(response_gen, DeploymentResponseGenerator):
                    response_gen.cancel()  # best-effort abort; registry signal above is the primary path
                else:
                    # mcp_loop.run_mcp_response: a plain async generator, not a Ray one;
                    # aclose() throws GeneratorExit into it, which cancels its own
                    # in-flight inner Ray generator (see loop.py's _events finally block).
                    with contextlib.suppress(Exception):
                        await response_gen.aclose()
                return
            saw_ok_terminal, failed_response = drain_task.result()
        except Exception:
            logger.exception("background response %s: drain task raised unexpectedly", response_id)
            saw_ok_terminal, failed_response = False, None
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

        if failed_response is not None:
            tracker.finish("error", "stream_error")
            # persist_response doesn't persist response.failed events; the poller needs it stored.
            await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed_response)
        elif not saw_ok_terminal:
            with contextlib.suppress(Exception):
                if await registry.is_set.remote(req_id):
                    tracker.finish("cancelled")
            tracker.finish("error", "stream_error")
            # Stream ended with no terminal event (cancel race or actor crash) — don't leave a poller hanging.
            await _write_failed(store, identity, response_id, "Generation ended unexpectedly without a terminal event.")
        else:
            tracker.finish("ok")
    finally:
        # Idempotent: the terminal path already flushed via buffer_stream_events; this
        # is what covers the abnormal exits above, which never reach a terminal event.
        if buffer is not None:
            await buffer.flush_and_discard()


async def _write_failed(store: StateStore, identity: str, response_id: str, message: str) -> None:
    record = await responses_state.read_async(store, identity, response_id)
    if record is not None:
        failed = _mark_failed(record.get("response") or {}, message)
        await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed)


async def _drain_background_stream(piped, response_id: str) -> tuple[bool, dict | None]:
    """Consume *piped* (the `persist_response`-wrapped event stream) to completion.
    Returns `(saw_ok_terminal, failed_response)` — `failed_response` carries a
    loader-side `response.failed` payload, which `persist_response` doesn't persist itself."""
    saw_ok_terminal = False
    failed_response: dict | None = None
    try:
        async for item in piped:
            if not isinstance(item, dict):
                continue  # persist_response's own terminal-write-failed signal; nothing new to persist
            event_type = item.get("type")
            if event_type in TERMINAL_EVENT_TYPES:
                saw_ok_terminal = True
            elif event_type == "response.failed":
                failed_response = item.get("response")
    except Exception:
        logger.exception("background response %s: event stream raised mid-drain", response_id)
    return saw_ok_terminal, failed_response


async def _heartbeat_loop(identity: str, response_id: str, req_id: str) -> None:
    """Refresh this run's HeartbeatRegistry entry every `_HEARTBEAT_INTERVAL_S`, and
    stop as soon as `DisconnectRegistry` shows `req_id` cancelled — the same signal
    `cancel_background`/`signal_background_cancel_if_in_progress` already set, so
    cancellation is detected without ever touching the response record."""
    registry = get_disconnect_registry()
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        with contextlib.suppress(Exception):
            await responses_state.heartbeat(identity, response_id, req_id)
        with contextlib.suppress(Exception):
            if await registry.is_set.remote(req_id):
                return


async def cancel_background(store: StateStore, identity: str, response_id: str) -> dict:
    """Cancel an in-flight background response: signal `DisconnectRegistry` with the
    run's request id, then mark the record `cancelled`. Idempotent on an already-
    terminal response; 400s if it was never a background run."""
    record = await load_record(store, identity, response_id)
    response = record["response"]
    if not response.get("background"):
        raise _not_background_error(response_id)
    if response.get("status") in responses_state.TERMINAL_STATUSES:
        return response

    req_id = None
    with contextlib.suppress(Exception):
        req_id = await responses_state.req_id_for(identity, response_id)
    if req_id:
        registry = get_disconnect_registry()
        with contextlib.suppress(Exception):
            await registry.set.remote(req_id)

    cancelled = {**response, "status": "cancelled", "completed_at": int(time.time())}
    try:
        await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=cancelled)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable cancelling response %s", response_id)
        raise _store_unavailable_error() from None
    with contextlib.suppress(StateStoreUnavailableError):
        await responses_state.discard_stream_buffer(store, identity, response_id)
    updated = await load_record(store, identity, response_id)
    return updated["response"]


async def signal_background_cancel_if_in_progress(identity: str, response_id: str, record: dict) -> None:
    """`DELETE /{id}` on an in-flight background run implies cancel: signal the
    registry (same mechanism `cancel_background` uses) before the caller deletes the
    record, so the drain task's heartbeat loop tears the run down."""
    response = record.get("response") or {}
    if not response.get("background") or response.get("status") in responses_state.TERMINAL_STATUSES:
        return
    req_id = None
    with contextlib.suppress(Exception):
        req_id = await responses_state.req_id_for(identity, response_id)
    if req_id:
        registry = get_disconnect_registry()
        with contextlib.suppress(Exception):
            await registry.set.remote(req_id)


async def reconcile_staleness(store: StateStore, identity: str, response_id: str, record: dict) -> dict:
    """`GET`-time orphan detection: if *record* is a non-terminal background run
    with a stale heartbeat, transition it to `failed` and persist. No-op otherwise."""
    response = record.get("response") or {}
    if response.get("status") not in ("queued", "in_progress"):
        return record
    try:
        alive = await responses_state.is_alive(identity, response_id)
    except Exception:
        # Registry unreachable: ambiguous, not evidence of staleness — don't risk
        # marking a genuinely healthy run failed over a transient hiccup.
        logger.warning("HeartbeatRegistry unavailable reconciling staleness for response %s", response_id)
        return record
    if alive:
        return record
    failed = _mark_failed(response, "Background response orphaned: its worker stopped reporting progress.")
    try:
        await responses_state.write_terminal_if_not_terminal(store, identity, response_id, response=failed)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reconciling staleness for response %s", response_id)
        raise _store_unavailable_error() from None
    with contextlib.suppress(StateStoreUnavailableError):
        await responses_state.discard_stream_buffer(store, identity, response_id)
    try:
        updated = await responses_state.read_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reconciling staleness for response %s", response_id)
        raise _store_unavailable_error() from None
    return updated if updated is not None else record
