"""End-to-end test for the MCP loop against a real MCP server (an in-process
`MCPServer` served over streamable HTTP via uvicorn), unlike test_mcp_loop.py.
Only the loader side is scripted.

No external network or GPU/Ray, so this runs in the default (non-integration)
suite.
"""

import asyncio
import contextlib
import json
import threading

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from modelship.openai.mcp import client as mcp_client
from modelship.openai.mcp import loop as mcp_loop
from modelship.openai.protocol.responses.schemas import ResponsesRequest

_PORT = 8933


def _recording(app, seen: list[str], expire_first_session_call: bool, stall_first_call: bool):
    async def wrapped(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        messages = [await receive()]
        while messages[-1].get("more_body"):
            messages.append(await receive())
        body = b"".join(message.get("body", b"") for message in messages)
        method = json.loads(body).get("method") if body else scope["method"]
        seen.append(method)
        if stall_first_call and method == "tools/call" and seen.count("tools/call") == 1:
            await asyncio.sleep(2)
        has_session = b"mcp-session-id" in dict(scope["headers"])
        if expire_first_session_call and (
            method == "server/discover" or (method == "tools/call" and has_session and "expired" not in seen)
        ):
            if method == "tools/call":
                seen.append("expired")
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return

        async def replay():
            return messages.pop(0) if messages else await receive()

        await app(scope, replay, send)

    return wrapped


@contextlib.contextmanager
def serve_dice(port: int, seen: list[str], *, expire_first_session_call: bool = False, stall_first_call: bool = False):
    server = MCPServer("dice")

    @server.tool()
    def roll(num_dice: int) -> str:
        """Roll num_dice six-sided dice."""
        return f"total: {num_dice * 3}"

    app = _recording(server.streamable_http_app(), seen, expire_first_session_call, stall_first_call)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)

    def run():
        asyncio.run(uv_server.serve())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv_server.started:
            break
        threading.Event().wait(0.05)
    assert uv_server.started, "test MCP server failed to start"
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        uv_server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def _dice():
    seen: list[str] = []
    with serve_dice(_PORT, seen) as url:
        yield url, seen


@pytest.fixture
def dice_server(_dice):
    return _dice[0]


@pytest.fixture
def dice_requests(_dice):
    _dice[1].clear()
    return _dice[1]


@pytest.fixture
def expiring_dice():
    seen: list[str] = []
    with serve_dice(8935, seen, expire_first_session_call=True) as url:
        yield url, seen


@pytest.fixture
def stalling_dice():
    seen: list[str] = []
    with serve_dice(8936, seen, stall_first_call=True) as url:
        yield url, seen


def _created(inner_id: str):
    return {"type": "response.created", "sequence_number": 0, "response": {"id": inner_id, "output": []}}


def _in_progress(inner_id: str):
    return {"type": "response.in_progress", "sequence_number": 1, "response": {"id": inner_id, "output": []}}


def _roll_call_turn(*, oi: int = 0):
    fc_item = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "roll",
        "arguments": "",
        "status": "in_progress",
    }
    completed_item = {**fc_item, "arguments": '{"num_dice": 2}', "status": "completed"}
    return [
        _created("i1"),
        _in_progress("i1"),
        {"type": "response.output_item.added", "sequence_number": 2, "output_index": oi, "item": fc_item},
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "item_id": "fc_1",
            "output_index": oi,
            "delta": '{"num_dice": 2}',
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 4,
            "item_id": "fc_1",
            "output_index": oi,
            "arguments": '{"num_dice": 2}',
        },
        {"type": "response.output_item.done", "sequence_number": 5, "output_index": oi, "item": completed_item},
        {
            "type": "response.completed",
            "sequence_number": 6,
            "response": {
                "id": "i1",
                "output": [completed_item],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            },
        },
    ]


def _final_answer_turn():
    msg_item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "The total is 6.", "annotations": []}],
    }
    return [
        _created("i2"),
        _in_progress("i2"),
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": {**msg_item, "status": "in_progress"},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "The total is 6.",
        },
        {"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": msg_item},
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": {
                "id": "i2",
                "output": [msg_item],
                "usage": {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
            },
        },
    ]


class FakeGen:
    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item

    def cancel(self):
        pass


class FakeHandle:
    """Stands in for the Ray DeploymentHandle: the loader side stays scripted
    (no real model in this test), only the MCP server side is real."""

    def __init__(self, turns: list[list]):
        self._turns = turns
        self.calls = 0
        self.requests: list[ResponsesRequest] = []

    @property
    def respond(self):
        return self

    def options(self, stream=True):
        return self

    def remote(self, *args, **kwargs):
        items = self._turns[self.calls]
        self.calls += 1
        self.requests.append(args[0])
        return FakeGen(items)


class TestRealMcpServer:
    @pytest.mark.asyncio
    async def test_discovery_and_tool_call_against_real_server(self, dice_server):
        handle = FakeHandle([_roll_call_turn(), _final_answer_turn()])
        request = ResponsesRequest(
            model="m",
            input="roll 2 dice",
            stream=True,
            tools=[{"type": "mcp", "server_label": "dice", "server_url": dice_server, "require_approval": "never"}],
        )

        events = [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped")]

        seqs = [e["sequence_number"] for e in events]
        assert seqs == list(range(len(seqs)))
        assert sum(1 for e in events if e["type"] == "response.created") == 1

        terminals = [e for e in events if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_call", "message"]

        list_tools_item = output[0]
        assert list_tools_item["tools"][0]["name"] == "roll"

        mcp_call_item = output[1]
        assert mcp_call_item["status"] == "completed"
        # Real tool execution: the dice server actually ran and returned this.
        assert mcp_call_item["output"] == "total: 6"

    @pytest.mark.asyncio
    async def test_discovery_failure_against_unreachable_server(self):
        handle = FakeHandle([[]])
        request = ResponsesRequest(
            model="m",
            input="roll 2 dice",
            stream=True,
            tools=[{"type": "mcp", "server_label": "dice", "server_url": "http://127.0.0.1:1/mcp"}],
        )

        events = [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped")]

        failed = [e for e in events if e["type"] == "response.failed"]
        assert len(failed) == 1
        assert "dice" in failed[0]["response"]["error"]["message"]
        assert handle.calls == 0


def _request(server_url: str) -> ResponsesRequest:
    tool = {"type": "mcp", "server_label": "dice", "server_url": server_url, "require_approval": "never"}
    return ResponsesRequest(model="m", input="roll 2 dice", stream=True, tools=[tool])


def _stored(server_url: str, overrides: dict | None = None) -> dict[str, dict]:
    tools = [{"name": "roll", "input_schema": {"type": "object"}, "description": "Roll num_dice six-sided dice."}]
    return {"dice": {"server_url": server_url, "allowed_tools": None, "tools": tools, **(overrides or {})}}


async def _output(handle: FakeHandle, request: ResponsesRequest, **kwargs) -> list[dict]:
    events = [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped", **kwargs)]
    return next(e for e in events if e["type"] == "response.completed")["response"]["output"]


class TestConnectionsAndStoredListings:
    @pytest.mark.asyncio
    async def test_one_connection_serves_discovery_and_every_call(self, dice_server, dice_requests):
        handle = FakeHandle([_roll_call_turn(), _roll_call_turn(), _final_answer_turn()])

        output = await _output(handle, _request(dice_server))

        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_call", "mcp_call", "message"]
        assert dice_requests == ["server/discover", "tools/list", "tools/call", "tools/call"]

    @pytest.mark.asyncio
    async def test_a_stored_listing_is_reused_without_contacting_the_server(self, dice_server, dice_requests):
        handle = FakeHandle([_final_answer_turn()])

        output = await _output(handle, _request(dice_server), mcp_listings=_stored(dice_server))

        assert [o["type"] for o in output] == ["message"]
        assert [tool["name"] for tool in handle.requests[0].tools] == ["roll"]
        assert dice_requests == []

    @pytest.mark.asyncio
    async def test_a_reused_listing_still_executes_calls(self, dice_server, dice_requests):
        handle = FakeHandle([_roll_call_turn(), _final_answer_turn()])

        output = await _output(handle, _request(dice_server), mcp_listings=_stored(dice_server))

        assert [o["type"] for o in output] == ["mcp_call", "message"]
        assert output[0]["output"] == "total: 6"
        assert sorted(dice_requests) == ["server/discover", "tools/call", "tools/list"]

    @pytest.mark.parametrize("changed", [{"server_url": "http://127.0.0.1:1/mcp"}, {"allowed_tools": ["roll"]}])
    @pytest.mark.asyncio
    async def test_a_listing_stored_under_other_settings_is_fetched_again(self, dice_server, dice_requests, changed):
        handle = FakeHandle([_final_answer_turn()])

        output = await _output(handle, _request(dice_server), mcp_listings=_stored(dice_server, changed))

        assert [o["type"] for o in output] == ["mcp_list_tools", "message"]
        assert dice_requests == ["server/discover", "tools/list"]

    @pytest.mark.asyncio
    async def test_an_expired_session_is_retried_on_a_new_connection(self, expiring_dice):
        server_url, seen = expiring_dice
        handle = FakeHandle([_roll_call_turn(), _final_answer_turn()])

        output = await _output(handle, _request(server_url))

        assert output[1]["status"] == "completed"
        assert output[1]["output"] == "total: 6"
        assert seen.count("expired") == 1
        assert seen.count("initialize") == 2

    @pytest.mark.asyncio
    async def test_a_timed_out_call_leaves_the_next_call_a_new_connection(self, stalling_dice, monkeypatch):
        server_url, seen = stalling_dice
        monkeypatch.setattr(mcp_client, "CALL_TOOL_TIMEOUT_S", 0.5)
        handle = FakeHandle([_roll_call_turn(), _roll_call_turn(), _final_answer_turn()])

        output = await _output(handle, _request(server_url))

        assert [o["status"] for o in output[1:3]] == ["failed", "completed"]
        assert "timed out" in output[1]["error"]
        assert output[2]["output"] == "total: 6"
        assert seen.count("server/discover") == 2
