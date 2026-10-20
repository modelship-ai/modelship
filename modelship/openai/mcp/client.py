"""This module is the only place that imports ``mcp``."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx2
from mcp_types import INVALID_REQUEST

from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from modelship.openai.mcp.spec import McpToolSpec
from modelship.openai.protocol.responses.schemas import McpListToolsTool

LIST_TOOLS_TIMEOUT_S = 30
CALL_TOOL_TIMEOUT_S = 60

_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MiB

_tasks: set[asyncio.Task] = set()


class McpCallError(Exception):
    """Raised for any MCP-side failure: network, protocol, or output-too-large.
    The loop records this on the ``mcp_call`` item's ``error`` field rather than
    letting it escape — a broken tool server doesn't take down the response."""


def _headers_for(spec: McpToolSpec) -> dict[str, str]:
    headers = dict(spec.headers or {})
    if spec.authorization and not any(k.lower() == "authorization" for k in headers):
        headers["Authorization"] = f"Bearer {spec.authorization}"
    return headers


def _session_expired(exc: MCPError) -> bool:
    return exc.code == INVALID_REQUEST and exc.message == "Session terminated"


@dataclass
class _Request:
    operation: Callable[[Client], Awaitable[Any]]
    timeout_s: float
    future: asyncio.Future[Any]


def _fail(request: _Request, exc: BaseException) -> None:
    if not request.future.done():
        request.future.set_exception(exc)


class _Connection:
    def __init__(self, spec: McpToolSpec) -> None:
        self._spec = spec
        self._requests: asyncio.Queue[_Request | None] = asyncio.Queue()
        self._current: _Request | None = None
        self.closed = False
        self._task = asyncio.create_task(self._serve())
        _tasks.add(self._task)
        self._task.add_done_callback(_tasks.discard)

    async def run(self, operation: Callable[[Client], Awaitable[Any]], timeout_s: float) -> Any:
        request = _Request(operation, timeout_s, asyncio.get_running_loop().create_future())
        self._requests.put_nowait(request)
        return await request.future

    def close(self) -> None:
        self.closed = True
        if self._current is not None:
            self._task.cancel()
        else:
            self._requests.put_nowait(None)

    async def _serve(self) -> None:
        try:
            async with (
                httpx2.AsyncClient(headers=_headers_for(self._spec), timeout=CALL_TOOL_TIMEOUT_S) as http_client,
                contextlib.AsyncExitStack() as stack,
            ):
                client: Client | None = None
                while (request := await self._requests.get()) is not None:
                    self._current = request
                    try:
                        async with asyncio.timeout(request.timeout_s):
                            if client is None:
                                transport = streamable_http_client(self._spec.server_url, http_client=http_client)
                                client = await stack.enter_async_context(Client(transport, mode="auto"))
                            result = await request.operation(client)
                    except MCPError as exc:
                        self.closed = self.closed or _session_expired(exc)
                        _fail(request, exc)
                        if self.closed:
                            return
                    except TimeoutError as exc:
                        self.closed = True
                        _fail(request, exc)
                        return
                    else:
                        if not request.future.done():
                            request.future.set_result(result)
                    self._current = None
        except Exception as exc:
            self.closed = True
            if self._current is not None:
                _fail(self._current, exc)
        finally:
            self.closed = True
            if self._current is not None and not self._current.future.done():
                self._current.future.cancel()
            while not self._requests.empty():
                if (queued := self._requests.get_nowait()) is not None:
                    _fail(queued, ConnectionError("the MCP connection closed"))


class McpConnections:
    def __init__(self) -> None:
        self._connections: dict[str, _Connection] = {}

    async def list_tools(self, spec: McpToolSpec) -> list[McpListToolsTool]:
        try:
            result = await self._run(spec, lambda client: client.list_tools(), LIST_TOOLS_TIMEOUT_S)
        except TimeoutError as exc:
            raise McpCallError(
                f"server {spec.server_label!r}: tools/list timed out after {LIST_TOOLS_TIMEOUT_S}s."
            ) from exc
        except MCPError as exc:
            raise McpCallError(f"server {spec.server_label!r}: tools/list failed: {exc.message}") from exc
        except Exception as exc:
            raise McpCallError(f"server {spec.server_label!r}: tools/list failed: {exc}") from exc

        return [
            McpListToolsTool(
                name=tool.name,
                input_schema=tool.input_schema,
                annotations=tool.annotations.model_dump(exclude_none=True) if tool.annotations else None,
                description=tool.description,
            )
            for tool in result.tools
        ]

    async def call_tool(self, spec: McpToolSpec, name: str, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            raise McpCallError(f"tool {name!r} on server {spec.server_label!r}: invalid arguments JSON: {exc}") from exc

        try:
            result = await self._run(spec, lambda client: client.call_tool(name, arguments), CALL_TOOL_TIMEOUT_S)
        except TimeoutError as exc:
            raise McpCallError(
                f"tool {name!r} on server {spec.server_label!r} timed out after {CALL_TOOL_TIMEOUT_S}s."
            ) from exc
        except MCPError as exc:
            raise McpCallError(f"tool {name!r} on server {spec.server_label!r} failed: {exc.message}") from exc
        except Exception as exc:
            raise McpCallError(f"tool {name!r} on server {spec.server_label!r} failed: {exc}") from exc

        output = _flatten_content(result.content)
        if result.is_error:
            raise McpCallError(f"tool {name!r} on server {spec.server_label!r} returned an error: {output}")
        if len(output.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise McpCallError(
                f"tool {name!r} on server {spec.server_label!r} output exceeds {_MAX_OUTPUT_BYTES} bytes."
            )
        return output

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()

    async def _run(self, spec: McpToolSpec, operation: Callable[[Client], Awaitable[Any]], timeout_s: float) -> Any:
        try:
            return await self._connection(spec).run(operation, timeout_s)
        except MCPError as exc:
            if not _session_expired(exc):
                raise
        return await self._connection(spec).run(operation, timeout_s)

    def _connection(self, spec: McpToolSpec) -> _Connection:
        connection = self._connections.get(spec.server_label)
        if connection is None or connection.closed:
            connection = self._connections[spec.server_label] = _Connection(spec)
        return connection


def _flatten_content(content: list) -> str:
    parts: list[str] = []
    for part in content:
        if getattr(part, "type", None) == "text":
            parts.append(part.text)
        else:
            parts.append(part.model_dump_json(exclude_none=True))
    return "".join(parts)


__all__ = ["CALL_TOOL_TIMEOUT_S", "LIST_TOOLS_TIMEOUT_S", "McpCallError", "McpConnections"]
