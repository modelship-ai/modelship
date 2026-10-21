"""Server-side MCP tool execution for ``/v1/responses``.

Discovers a request's ``mcp`` tool servers, drives the tool-call loop against the
loader, and stitches the resulting N inner Responses-stream turns into one logical
outer response: one ``response.created``, one monotonic ``sequence_number`` series,
one ``output_index`` series, one terminal event. The loader never sees an ``mcp``
tool — it is expanded into plain ``function`` tools before each inner Ray hop, so
this is loader-agnostic and runs entirely at the gateway.

Public surface: :func:`wants_mcp` (gate) and :func:`run_mcp_response`, a drop-in
generator source for ``handle.respond.options(stream=True).remote(...)`` at every
gateway call site (HTTP, background, WebSocket) — it yields exactly what that call
yields today: event dicts when ``request.stream`` is truthy, a single
``ResponseObject`` when falsy, ``ErrorResponse`` passthrough on failure.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, cast

from pydantic import ValidationError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

from modelship.openai.mcp.client import McpCallError, McpConnections
from modelship.openai.mcp.egress import validate_server_url
from modelship.openai.mcp.spec import (
    McpToolSpec,
    check_collisions,
    filter_tools,
    requires_approval,
    split_mcp_tools,
)
from modelship.openai.protocol import ErrorResponse, create_error_response
from modelship.openai.protocol.base import random_uuid
from modelship.openai.protocol.responses.adapter import build_response_object, parse_output_item
from modelship.openai.protocol.responses.schemas import (
    McpApprovalRequestItem,
    McpCallItem,
    McpListToolsItem,
    McpListToolsTool,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputTokensDetails,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.utils import affinity
from modelship.openai.utils.responses import ResponsesApiError, as_input_items

# Loader turns per response; matches LocalAI's agent.max_iterations default.
DEFAULT_TURN_CAP = 10

_ToolIndex = dict[str, tuple[McpToolSpec, McpListToolsTool]]


def wants_mcp(request: ResponsesRequest) -> bool:
    """Whether *request* declares at least one ``mcp`` tool."""
    return any(isinstance(t, dict) and t.get("type") == "mcp" for t in (request.tools or []))


def _mcp_error(message: str) -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            message, err_type="invalid_request_error", status_code=HTTPStatus.BAD_REQUEST, param="tools"
        )
    )


def _sum_usage(a: ResponseUsage | None, b: ResponseUsage | None) -> ResponseUsage | None:
    if a is None:
        return b
    if b is None:
        return a
    a_in = a.input_tokens_details.cached_tokens if a.input_tokens_details else 0
    b_in = b.input_tokens_details.cached_tokens if b.input_tokens_details else 0
    a_out = a.output_tokens_details.reasoning_tokens if a.output_tokens_details else 0
    b_out = b.output_tokens_details.reasoning_tokens if b.output_tokens_details else 0
    return ResponseUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        input_tokens_details=ResponseInputTokensDetails(cached_tokens=a_in + b_in),
        output_tokens_details=ResponseOutputTokensDetails(reasoning_tokens=a_out + b_out),
    )


def _usage_from_dict(usage: dict[str, Any] | None) -> ResponseUsage | None:
    return None if usage is None else ResponseUsage.model_validate(usage)


def _tools_as_functions(tool_index: _ToolIndex) -> list[dict[str, Any]]:
    return [
        {"type": "function", "name": name, "description": tool.description, "parameters": tool.input_schema}
        for name, (_spec, tool) in tool_index.items()
    ]


@dataclass
class _BufferedCall:
    """Per-inner-tool-call state, keyed by the loader's own item id (``fc_...``)
    until the tool name resolves and the call can be classified."""

    inner_item_id: str
    classification: str | None = None  # None | "client" | "mcp" | "mcp_approval"
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    abs_oi: int | None = None
    outer_id: str | None = None  # minted mcp_<uuid>, only for "mcp"/"mcp_approval"
    name: str = ""
    arguments: str = ""
    spec: McpToolSpec | None = None
    tool: McpListToolsTool | None = None


class Stitcher:
    """Owns the outer envelope's sequence/offset counters across N inner loader
    turns, and rewrites ``function_call`` events into ``mcp_call`` events for tool
    calls that match a discovered MCP tool. One instance per outer response."""

    def __init__(self, tool_index: _ToolIndex):
        self._seq = 0
        self._offset = 0
        self._tool_index = tool_index
        self.usage_total: ResponseUsage | None = None
        # Set fresh by run_turn(); read by the caller once that generator is exhausted.
        self.pending_mcp_calls: list[_BufferedCall] = []
        self.had_client_function_call = False
        self.turn_output: list[tuple[int, dict[str, Any]]] = []
        self.inner_error: ErrorResponse | None = None
        self.inner_failed_response: dict[str, Any] | None = None

    def next_seq(self) -> int:
        n = self._seq
        self._seq += 1
        return n

    def claim_slot(self) -> int:
        """Reserve the next outer output slot for an item the loop owns directly
        (not absorbed from an inner turn): discovery items, approval-resume items,
        approval-request items."""
        oi = self._offset
        self._offset += 1
        return oi

    def stamped(self, event_type: str, output_index: int | None = None, **payload: Any) -> dict[str, Any]:
        """Build a fresh event with an absolute (already-offset) output_index."""
        out: dict[str, Any] = {"type": event_type, "sequence_number": self.next_seq(), **payload}
        if output_index is not None:
            out["output_index"] = output_index
        return out

    def _stamp(self, event: dict[str, Any]) -> dict[str, Any]:
        """Forward an inner-turn event verbatim: renumber sequence_number, offset
        its turn-relative output_index (if any) by the outer offset."""
        out = {**event, "sequence_number": self.next_seq()}
        if "output_index" in out:
            out["output_index"] = out["output_index"] + self._offset
        return out

    def _classify(self, name: str) -> tuple[str, McpToolSpec | None, McpListToolsTool | None]:
        entry = self._tool_index.get(name)
        if entry is None:
            return "client", None, None
        spec, tool = entry
        return ("mcp_approval" if requires_approval(spec, tool) else "mcp"), spec, tool

    def _flush(self, buf: _BufferedCall) -> Iterator[dict[str, Any]]:
        """Classify *buf* (name now known) and emit its buffered events transformed
        for that classification. ``mcp_approval`` suppresses all streaming output —
        nothing is emitted for it until the post-turn approval-request pair."""
        classification, spec, tool = self._classify(buf.name)
        buf.classification = classification
        buf.spec, buf.tool = spec, tool

        if classification == "client":
            for raw in buf.raw_events:
                yield self._stamp(raw)
            buf.raw_events = []
            return

        if classification == "mcp_approval":
            for raw in buf.raw_events:
                if raw["type"] == "response.output_item.added":
                    buf.abs_oi = raw["output_index"] + self._offset
                    break
            buf.raw_events = []
            return

        assert spec is not None
        buf.outer_id = f"mcp_{random_uuid()}"
        for raw in buf.raw_events:
            rtype = raw["type"]
            if rtype == "response.output_item.added":
                buf.abs_oi = raw["output_index"] + self._offset
                item = McpCallItem(
                    id=buf.outer_id, name=buf.name, arguments="", server_label=spec.server_label, status="in_progress"
                )
                yield self.stamped("response.output_item.added", buf.abs_oi, item=item.model_dump(mode="json"))
            elif rtype == "response.function_call_arguments.delta":
                yield self.stamped(
                    "response.mcp_call_arguments.delta", buf.abs_oi, item_id=buf.outer_id, delta=raw.get("delta", "")
                )
            elif rtype == "response.function_call_arguments.done":
                buf.arguments = raw.get("arguments", "")
                yield self.stamped(
                    "response.mcp_call_arguments.done", buf.abs_oi, item_id=buf.outer_id, arguments=buf.arguments
                )
        buf.raw_events = []

    async def run_turn(self, inner_gen: AsyncIterator[Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Consume one inner turn's event stream, yielding renumbered/rewritten
        events for immediate emission. Resets and populates, as attributes read by
        the caller once this generator is exhausted: ``pending_mcp_calls``,
        ``had_client_function_call``, ``turn_output`` (this turn's final items as
        ``(abs_output_index, item)`` pairs, ready for the caller to place directly
        into the outer accumulator by index — mcp/approval slots are excluded here
        since the caller fills those in by ``buf.abs_oi`` after execution),
        ``inner_error``, ``inner_failed_response``."""
        buffers: dict[str, _BufferedCall] = {}
        self.pending_mcp_calls = []
        self.had_client_function_call = False
        self.turn_output = []
        self.inner_error = None
        self.inner_failed_response = None

        async for event in inner_gen:
            if isinstance(event, ErrorResponse):
                self.inner_error = event
                return
            if not isinstance(event, dict):
                continue
            etype = event.get("type")

            if etype in ("response.created", "response.in_progress"):
                continue

            if etype == "response.failed":
                self.inner_failed_response = event.get("response") or {}
                return

            if etype in ("response.completed", "response.incomplete"):
                response = event.get("response") or {}
                self.usage_total = _sum_usage(self.usage_total, _usage_from_dict(response.get("usage")))
                output = response.get("output") or []
                base = self._offset
                for i, item in enumerate(output):
                    if item.get("type") != "function_call":
                        self.turn_output.append((base + i, item))
                        continue
                    buf = buffers.get(item.get("id") or "")
                    if buf is None or buf.classification == "client":
                        self.turn_output.append((base + i, item))
                        self.had_client_function_call = True
                self._offset += len(output)
                continue

            item: dict[str, Any] = event.get("item") or {}
            is_tool_item = item.get("type") == "function_call"

            if etype == "response.output_item.added" and is_tool_item:
                inner_id = item["id"]
                buf = _BufferedCall(inner_item_id=inner_id)
                buffers[inner_id] = buf
                buf.raw_events.append(event)
                name = item.get("name") or ""
                if name:
                    buf.name = name
                    for out_event in self._flush(buf):
                        yield out_event
                continue

            if etype == "response.function_call_arguments.delta":
                buf = buffers.get(event.get("item_id") or "")
                if buf is None:
                    continue
                if buf.classification is None:
                    buf.raw_events.append(event)
                elif buf.classification == "client":
                    yield self._stamp(event)
                elif buf.classification == "mcp":
                    yield self.stamped(
                        "response.mcp_call_arguments.delta",
                        buf.abs_oi,
                        item_id=buf.outer_id,
                        delta=event.get("delta", ""),
                    )
                continue

            if etype == "response.function_call_arguments.done":
                buf = buffers.get(event.get("item_id") or "")
                if buf is None:
                    continue
                if buf.classification is None:
                    buf.raw_events.append(event)
                elif buf.classification == "client":
                    yield self._stamp(event)
                elif buf.classification == "mcp":
                    buf.arguments = event.get("arguments", "")
                    yield self.stamped(
                        "response.mcp_call_arguments.done", buf.abs_oi, item_id=buf.outer_id, arguments=buf.arguments
                    )
                else:  # mcp_approval, arguments not final yet — captured at output_item.done below
                    buf.arguments = event.get("arguments", "")
                continue

            if etype == "response.output_item.done" and is_tool_item:
                inner_id = item["id"]
                buf = buffers[inner_id]
                buf.name = item.get("name") or buf.name
                if buf.classification is None:
                    for out_event in self._flush(buf):
                        yield out_event
                buf.arguments = item.get("arguments", buf.arguments)
                if buf.classification == "client":
                    yield self._stamp(event)
                else:
                    self.pending_mcp_calls.append(buf)
                continue

            # Non-tool events (reasoning/message channels): stream through with
            # only renumbering.
            yield self._stamp(event)


async def _execute_mcp_call(
    stitcher: Stitcher,
    *,
    outer_id: str,
    abs_oi: int,
    name: str,
    arguments: str,
    spec: McpToolSpec,
    connections: McpConnections,
    approval_request_id: str | None = None,
) -> tuple[list[dict[str, Any]], McpCallItem]:
    events: list[dict[str, Any]] = [stitcher.stamped("response.mcp_call.in_progress", abs_oi, item_id=outer_id)]
    try:
        output = await connections.call_tool(spec, name, arguments)
        item = McpCallItem(
            id=outer_id,
            name=name,
            arguments=arguments,
            server_label=spec.server_label,
            approval_request_id=approval_request_id,
            output=output,
            status="completed",
        )
        events.append(stitcher.stamped("response.mcp_call.completed", abs_oi, item_id=outer_id))
    except McpCallError as exc:
        item = McpCallItem(
            id=outer_id,
            name=name,
            arguments=arguments,
            server_label=spec.server_label,
            approval_request_id=approval_request_id,
            error=str(exc),
            status="failed",
        )
        events.append(stitcher.stamped("response.mcp_call.failed", abs_oi, item_id=outer_id))
    events.append(stitcher.stamped("response.output_item.done", abs_oi, item=item.model_dump(mode="json")))
    return events, item


async def run_mcp_response(
    handle: DeploymentHandle,
    request: ResponsesRequest,
    headers: dict[str, str],
    registry: Any,
    req_id: str,
    identity: str,
    *,
    response_id: str | None = None,
    mcp_listings: dict[str, dict] | None = None,
) -> AsyncGenerator[Any, None]:
    is_stream = bool(request.stream)
    connections = McpConnections()
    gen = _events(handle, request, headers, registry, req_id, identity, response_id, connections, mcp_listings or {})
    try:
        if is_stream:
            async for event in gen:
                yield event
            return

        terminal_response: dict[str, Any] | None = None
        async for event in gen:
            if isinstance(event, ErrorResponse):
                yield event
                return
            if isinstance(event, dict) and event.get("type") in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                terminal_response = event.get("response")
        if terminal_response is not None:
            yield ResponseObject.model_validate(terminal_response)
    finally:
        # Explicitly close rather than relying on GC, so _events()'s own finally
        # (which cancels the in-flight inner Ray generator) runs deterministically.
        with contextlib.suppress(Exception):
            await gen.aclose()
        connections.close()


def _stored_tools(spec: McpToolSpec, listing: dict | None) -> list[McpListToolsTool] | None:
    if listing is None or (listing["server_url"], listing["allowed_tools"]) != (spec.server_url, spec.allowed_tools):
        return None
    try:
        tools = [McpListToolsTool.model_validate(tool) for tool in listing["tools"]]
    except ValidationError:
        return None
    return filter_tools(spec, tools)


async def _events(
    handle: DeploymentHandle,
    request: ResponsesRequest,
    headers: dict[str, str],
    registry: Any,
    req_id: str,
    identity: str,
    response_id: str | None,
    connections: McpConnections,
    mcp_listings: dict[str, dict],
) -> AsyncGenerator[dict[str, Any] | ErrorResponse, None]:
    response_id = response_id or f"resp_{random_uuid()}"
    created_at = int(time.time())
    accumulator: list[dict[str, Any]] = []

    try:
        specs, other_tools = split_mcp_tools(request.tools)
        for spec in specs:
            validate_server_url(spec.server_url)
        if isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "mcp":
            raise _mcp_error("tool_choice forcing a specific mcp tool is not supported.")
        original_input_items = as_input_items(request.input)
        if request.previous_response_id is None and any(
            isinstance(item, dict) and item.get("type") == "mcp_approval_response" for item in original_input_items
        ):
            # An mcp_approval_request only ever exists as output from a prior, stored
            # response — without previous_response_id there is no genuine one to
            # resume, so a same-request approval_request/approval_response pair can
            # only be self-supplied, bypassing the require_approval pause entirely.
            raise ResponsesApiError(
                create_error_response(
                    "mcp_approval_response requires previous_response_id: an approval can only be "
                    "resumed from a prior response's own mcp_approval_request, not supplied directly "
                    "on a fresh request.",
                    err_type="invalid_request_error",
                    status_code=HTTPStatus.BAD_REQUEST,
                    param="input",
                )
            )
    except ResponsesApiError as exc:
        yield exc.err
        return

    tool_index: _ToolIndex = {}
    stitcher = Stitcher(tool_index)

    def _failed_event(message: str) -> dict[str, Any]:
        response = build_response_object(
            request,
            status="failed",
            output=[parse_output_item(d) for d in accumulator],
            usage=stitcher.usage_total,
            incomplete=None,
            response_id=response_id,
            created_at=created_at,
            completed_at=int(time.time()),
            error={"message": message},
            background=bool(request.background),
        )
        return stitcher.stamped("response.failed", response=response.model_dump(mode="json"))

    created_response = build_response_object(
        request,
        status="in_progress",
        output=[],
        usage=None,
        incomplete=None,
        response_id=response_id,
        created_at=created_at,
        background=bool(request.background),
    )
    yield stitcher.stamped("response.created", response=created_response.model_dump(mode="json"))
    yield stitcher.stamped("response.in_progress", response=created_response.model_dump(mode="json"))

    specs_by_label = {s.server_label: s for s in specs}

    # --- Discovery -------------------------------------------------------------
    # Runs before approval resume below: the resumed approval's tool name must be
    # checked against this turn's actual discovered+filtered tool list, not trusted
    # from the client-echoed mcp_approval_request item — otherwise a client could
    # forge/edit that item's name to execute a tool that allowed_tools would have
    # filtered out, or that doesn't exist on the server at all.
    discovered: dict[str, list[McpListToolsTool]] = {}
    for spec in specs:
        tools = _stored_tools(spec, mcp_listings.get(spec.server_label))
        if tools is None:
            oi = stitcher.claim_slot()
            list_item_id = f"mcpl_{random_uuid()}"
            placeholder = McpListToolsItem(id=list_item_id, server_label=spec.server_label, tools=[])
            yield stitcher.stamped("response.output_item.added", oi, item=placeholder.model_dump(mode="json"))
            yield stitcher.stamped("response.mcp_list_tools.in_progress", oi, item_id=list_item_id)
            try:
                tools = await connections.list_tools(spec)
            except McpCallError as exc:
                failed_item = McpListToolsItem(
                    id=list_item_id, server_label=spec.server_label, tools=[], error=str(exc)
                )
                yield stitcher.stamped("response.mcp_list_tools.failed", oi, item_id=list_item_id)
                yield stitcher.stamped("response.output_item.done", oi, item=failed_item.model_dump(mode="json"))
                accumulator.append(failed_item.model_dump(mode="json"))
                yield _failed_event(f"mcp server {spec.server_label!r} discovery failed: {exc}")
                return

            tools = filter_tools(spec, tools)
            ok_item = McpListToolsItem(id=list_item_id, server_label=spec.server_label, tools=tools)
            yield stitcher.stamped("response.mcp_list_tools.completed", oi, item_id=list_item_id)
            yield stitcher.stamped("response.output_item.done", oi, item=ok_item.model_dump(mode="json"))
            accumulator.append(ok_item.model_dump(mode="json"))
        discovered[spec.server_label] = tools
        for tool in tools:
            tool_index[tool.name] = (spec, tool)

    try:
        check_collisions(specs, discovered, other_tools)
    except ResponsesApiError as exc:
        yield _failed_event(exc.err.error.message)
        return

    # --- Approval resume ---------------------------------------------------
    approval_requests_by_id = {
        item.get("id"): item
        for item in original_input_items
        if isinstance(item, dict) and item.get("type") == "mcp_approval_request"
    }
    for item in original_input_items:
        if not isinstance(item, dict) or item.get("type") != "mcp_approval_response":
            continue
        approval_request_id = item.get("approval_request_id")
        req_item = approval_requests_by_id.get(approval_request_id)
        if req_item is None:
            yield create_error_response(
                f"mcp_approval_response references unknown approval_request_id {approval_request_id!r}.",
                err_type="invalid_request_error",
            )
            return

        name = req_item.get("name", "")
        server_label = req_item.get("server_label", "")
        arguments = req_item.get("arguments", "")
        spec = specs_by_label.get(server_label)
        # The approval-request item is client-echoed input, not server-verified
        # state — re-check the name against this turn's real discovery/filter
        # result rather than trusting it, or a forged/edited name would execute.
        tool_known = spec is not None and any(t.name == name for t in discovered.get(server_label, []))
        outer_id = f"mcp_{random_uuid()}"
        oi = stitcher.claim_slot()

        if item.get("approve") and tool_known:
            assert spec is not None
            placeholder = McpCallItem(
                id=outer_id,
                name=name,
                arguments=arguments,
                server_label=server_label,
                approval_request_id=approval_request_id,
                status="in_progress",
            )
            yield stitcher.stamped("response.output_item.added", oi, item=placeholder.model_dump(mode="json"))
            # Arguments are already fully known (they came from the original
            # mcp_approval_request item, not a live stream) — emit the delta/done
            # pair as a single chunk so the event shape still matches a normal
            # (non-resumed) mcp_call: added -> arguments.delta -> arguments.done -> ...
            yield stitcher.stamped("response.mcp_call_arguments.delta", oi, item_id=outer_id, delta=arguments)
            yield stitcher.stamped("response.mcp_call_arguments.done", oi, item_id=outer_id, arguments=arguments)
            events, call_item = await _execute_mcp_call(
                stitcher,
                outer_id=outer_id,
                abs_oi=oi,
                name=name,
                arguments=arguments,
                spec=spec,
                connections=connections,
                approval_request_id=approval_request_id,
            )
            for e in events:
                yield e
            accumulator.append(call_item.model_dump(mode="json"))
        else:
            if spec is None:
                message = f"mcp server {server_label!r} is not configured on this request."
            elif not tool_known:
                message = f"tool {name!r} is not available on mcp server {server_label!r}."
            else:
                message = "Tool call was rejected."
                reason = item.get("reason")
                if reason:
                    message = f"{message} {reason}"
            call_item = McpCallItem(
                id=outer_id,
                name=name,
                arguments=arguments,
                server_label=server_label,
                approval_request_id=approval_request_id,
                error=message,
                status="failed",
            )
            yield stitcher.stamped("response.output_item.added", oi, item=call_item.model_dump(mode="json"))
            yield stitcher.stamped("response.mcp_call.in_progress", oi, item_id=outer_id)
            yield stitcher.stamped("response.mcp_call.failed", oi, item_id=outer_id)
            yield stitcher.stamped("response.output_item.done", oi, item=call_item.model_dump(mode="json"))
            accumulator.append(call_item.model_dump(mode="json"))

    # --- Turn loop -------------------------------------------------------------
    max_tool_calls = request.max_tool_calls
    executed_call_count = 0
    final_status = "completed"
    incomplete_reason: str | None = None
    active_inner_gen: DeploymentResponseGenerator[Any] | None = None
    session_id = affinity.responses_key(request, identity)

    try:
        for turn_index in range(DEFAULT_TURN_CAP):
            # A forced tool_choice is a one-shot nudge, not a standing constraint —
            # reapplying it every turn would force tool calls forever.
            tool_choice = request.tool_choice if turn_index == 0 else "auto"
            inner_request = request.model_copy(
                update={
                    "input": [*original_input_items, *accumulator],
                    "tools": [*other_tools, *_tools_as_functions(tool_index)],
                    "tool_choice": tool_choice,
                    "stream": True,
                    "previous_response_id": None,
                    "background": False,
                    "max_tool_calls": None,
                }
            )
            inner_gen = cast(
                "DeploymentResponseGenerator[Any]",
                handle.respond.options(stream=True, session_id=session_id).remote(
                    inner_request, headers, registry, req_id, identity
                ),
            )
            active_inner_gen = inner_gen
            async for event in stitcher.run_turn(inner_gen):
                yield event
            active_inner_gen = None

            if stitcher.inner_error is not None:
                yield stitcher.inner_error
                return
            if stitcher.inner_failed_response is not None:
                yield stitcher.stamped(
                    "response.failed", response={**stitcher.inner_failed_response, "id": response_id}
                )
                return

            # Items land in the outer accumulator at their absolute output_index
            # rather than being appended in whatever order the two groups (plain
            # turn_output items vs. executed mcp calls) happen to be processed in
            # here — a turn that interleaves them (e.g. message, mcp_call, message)
            # would otherwise land in the wrong order in the final response.output[].
            accumulator.extend({} for _ in range(len(stitcher.turn_output) + len(stitcher.pending_mcp_calls)))
            for abs_index, item in stitcher.turn_output:
                accumulator[abs_index] = item

            any_approval_this_turn = False
            for buf in stitcher.pending_mcp_calls:
                if buf.classification == "mcp_approval":
                    assert buf.spec is not None and buf.abs_oi is not None
                    item = McpApprovalRequestItem(
                        name=buf.name, arguments=buf.arguments, server_label=buf.spec.server_label
                    )
                    yield stitcher.stamped("response.output_item.added", buf.abs_oi, item=item.model_dump(mode="json"))
                    yield stitcher.stamped("response.output_item.done", buf.abs_oi, item=item.model_dump(mode="json"))
                    accumulator[buf.abs_oi] = item.model_dump(mode="json")
                    any_approval_this_turn = True
                else:
                    assert buf.spec is not None and buf.outer_id is not None and buf.abs_oi is not None
                    events, item = await _execute_mcp_call(
                        stitcher,
                        outer_id=buf.outer_id,
                        abs_oi=buf.abs_oi,
                        name=buf.name,
                        arguments=buf.arguments,
                        spec=buf.spec,
                        connections=connections,
                    )
                    for e in events:
                        yield e
                    accumulator[buf.abs_oi] = item.model_dump(mode="json")
                    executed_call_count += 1

            if any_approval_this_turn or stitcher.had_client_function_call:
                final_status = "completed"
                break
            if not stitcher.pending_mcp_calls:
                final_status = "completed"
                break
            if max_tool_calls is not None and executed_call_count >= max_tool_calls:
                final_status, incomplete_reason = "incomplete", "max_tool_calls"
                break
        else:
            final_status, incomplete_reason = "incomplete", "max_tool_calls"
    finally:
        if active_inner_gen is not None:
            with contextlib.suppress(Exception):
                active_inner_gen.cancel()

    incomplete = {"reason": incomplete_reason} if incomplete_reason else None
    response = build_response_object(
        request,
        status=final_status,
        output=[parse_output_item(d) for d in accumulator],
        usage=stitcher.usage_total,
        incomplete=incomplete,
        response_id=response_id,
        created_at=created_at,
        completed_at=int(time.time()),
        background=bool(request.background),
    )
    terminal_type = "response.completed" if final_status == "completed" else "response.incomplete"
    yield stitcher.stamped(terminal_type, response=response.model_dump(mode="json"))


__all__ = ["DEFAULT_TURN_CAP", "Stitcher", "run_mcp_response", "wants_mcp"]
