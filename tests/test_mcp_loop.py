"""Tests for modelship.openai.mcp.loop: the MCP orchestrator + stream stitcher.

Inner loader turns are scripted as plain async generators of event dicts (the same
shape ResponsesStreamTranslator produces), fed through a fake Ray handle — no real
Ray cluster, no real MCP network calls.
"""

from unittest.mock import patch

import pytest

from modelship.openai.mcp import loop as mcp_loop
from modelship.openai.mcp.client import McpCallError
from modelship.openai.protocol import ErrorResponse, create_error_response
from modelship.openai.protocol.responses.schemas import McpListToolsTool, ResponseObject, ResponsesRequest
from modelship.openai.utils import affinity


def _created(inner_id: str):
    return {"type": "response.created", "sequence_number": 0, "response": {"id": inner_id, "output": []}}


def _in_progress(inner_id: str):
    return {"type": "response.in_progress", "sequence_number": 1, "response": {"id": inner_id, "output": []}}


def _tool_call_events(*, call_id: str, fc_id: str, name: str, arguments: str, oi: int, start_seq: int):
    seq = start_seq
    events = []
    events.append(
        {
            "type": "response.output_item.added",
            "sequence_number": seq,
            "output_index": oi,
            "item": {
                "id": fc_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress",
            },
        }
    )
    seq += 1
    events.append(
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": seq,
            "item_id": fc_id,
            "output_index": oi,
            "delta": arguments,
        }
    )
    seq += 1
    events.append(
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": seq,
            "item_id": fc_id,
            "output_index": oi,
            "arguments": arguments,
        }
    )
    seq += 1
    completed_item = {
        "id": fc_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }
    events.append(
        {"type": "response.output_item.done", "sequence_number": seq, "output_index": oi, "item": completed_item}
    )
    return events, completed_item, seq + 1


def _message_events(*, oi: int, text: str, start_seq: int):
    seq = start_seq
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": seq,
            "output_index": oi,
            "item": {"id": "msg_1", "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        }
    ]
    seq += 1
    events.append(
        {
            "type": "response.output_text.delta",
            "sequence_number": seq,
            "item_id": "msg_1",
            "output_index": oi,
            "content_index": 0,
            "delta": text,
        }
    )
    seq += 1
    msg_item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    events.append({"type": "response.output_item.done", "sequence_number": seq, "output_index": oi, "item": msg_item})
    return events, msg_item, seq + 1


def _terminal(inner_id: str, output: list, *, seq: int, usage=None, status="completed"):
    etype = "response.completed" if status == "completed" else "response.incomplete"
    return {
        "type": etype,
        "sequence_number": seq,
        "response": {
            "id": inner_id,
            "output": output,
            "usage": usage or {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }


class FakeGen:
    def __init__(self, items):
        self._items = items
        self.cancelled = False

    async def __aiter__(self):
        for item in self._items:
            yield item

    def cancel(self):
        self.cancelled = True


class FakeHandle:
    """Stands in for the Ray DeploymentHandle: .respond.options(stream=True).remote(...)
    returns the next scripted turn (a list of events) each time it's called."""

    def __init__(self, turns: list[list]):
        self._turns = turns
        self._calls = 0
        self.generators: list[FakeGen] = []
        self.remote_calls: list[tuple[tuple, dict]] = []
        self.session_ids: list[str] = []

    @property
    def respond(self):
        return self

    def options(self, stream=True, session_id=""):
        self.session_ids.append(session_id)
        return self

    def remote(self, *args, **kwargs):
        self.remote_calls.append((args, kwargs))
        items = self._turns[self._calls]
        self._calls += 1
        gen = FakeGen(items)
        self.generators.append(gen)
        return gen


def _req(**overrides) -> ResponsesRequest:
    payload = {
        "model": "m",
        "input": "hi",
        "stream": True,
        "tools": [
            {"type": "mcp", "server_label": "dice", "server_url": "http://fake/mcp", "require_approval": "never"}
        ],
    }
    payload.update(overrides)
    return ResponsesRequest(**payload)


async def _run(handle, request, **kwargs) -> list:
    return [e async for e in mcp_loop.run_mcp_response(handle, request, {}, None, "req_1", "unscoped", **kwargs)]


@pytest.fixture(autouse=True)
def _mcp_tools():
    async def fake_list(_self, _spec):
        return [McpListToolsTool(name="roll", input_schema={"type": "object"}, description="Roll dice.")]

    with patch("modelship.openai.mcp.loop.McpConnections.list_tools", fake_list):
        yield


class TestSingleTurnNoToolCalls:
    @pytest.mark.asyncio
    async def test_no_tool_calls_completes_after_discovery(self):
        events, msg_item, seq = _message_events(oi=0, text="hi there", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        result = await _run(handle, _req())

        seqs = [e["sequence_number"] for e in result]
        assert seqs == list(range(len(seqs)))
        assert sum(1 for e in result if e["type"] == "response.created") == 1
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        output_types = [o["type"] for o in terminals[0]["response"]["output"]]
        assert output_types == ["mcp_list_tools", "message"]
        assert terminals[0]["response"]["status"] == "completed"


class TestMcpCallExecutesAndContinues:
    @pytest.mark.asyncio
    async def test_mcp_call_auto_executes_and_turn_continues(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments='{"n":2}', oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]

        msg_events, msg_item, seq2 = _message_events(oi=0, text="You rolled a 9.", start_seq=2)
        turn2 = [_created("i2"), _in_progress("i2"), *msg_events, _terminal("i2", [msg_item], seq=seq2)]

        handle = FakeHandle([turn1, turn2])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            result = await _run(handle, _req())

        seqs = [e["sequence_number"] for e in result]
        assert seqs == list(range(len(seqs)))
        assert sum(1 for e in result if e["type"] == "response.created") == 1
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_call", "message"]
        assert output[1]["output"] == "9"
        assert output[1]["status"] == "completed"
        assert terminals[0]["response"]["usage"]["total_tokens"] == 4

        # mcp_call streaming shape: added -> arguments.delta -> arguments.done -> in_progress -> completed -> output_item.done
        mcp_related = [e["type"] for e in result if "mcp_call" in e["type"]]
        assert mcp_related == [
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
        ]

    @pytest.mark.asyncio
    async def test_every_round_routes_with_the_requests_key(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments='{"n":2}', oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        msg_events, msg_item, seq2 = _message_events(oi=0, text="You rolled a 9.", start_seq=2)
        turn2 = [_created("i2"), _in_progress("i2"), *msg_events, _terminal("i2", [msg_item], seq=seq2)]
        handle = FakeHandle([turn1, turn2])
        request = _req()
        key = affinity.responses_key(request, "unscoped")

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            await _run(handle, request)

        assert handle.session_ids == [key, key]

    @pytest.mark.asyncio
    async def test_tool_call_sandwiched_between_messages_keeps_output_order(self):
        # Output order must match the turn's own interleaving, not tool-calls-last.
        msg1_events, msg1_item, seq = _message_events(oi=0, text="Rolling...", start_seq=2)
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments='{"n":2}', oi=1, start_seq=seq
        )
        msg2_events, msg2_item, seq = _message_events(oi=2, text="Got it.", start_seq=seq)
        turn1 = [
            _created("i1"),
            _in_progress("i1"),
            *msg1_events,
            *tc_events,
            *msg2_events,
            _terminal("i1", [msg1_item, tc_item, msg2_item], seq=seq),
        ]

        msg3_events, msg3_item, seq2 = _message_events(oi=0, text="You rolled a 9.", start_seq=2)
        turn2 = [_created("i2"), _in_progress("i2"), *msg3_events, _terminal("i2", [msg3_item], seq=seq2)]

        handle = FakeHandle([turn1, turn2])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            result = await _run(handle, _req())

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "message", "mcp_call", "message", "message"]
        assert output[1]["content"][0]["text"] == "Rolling..."
        assert output[2]["output"] == "9"
        assert output[3]["content"][0]["text"] == "Got it."

    @pytest.mark.asyncio
    async def test_forced_tool_choice_only_applies_to_first_turn(self):
        # tool_choice forcing only applies to the first turn — reapplying it every
        # turn would force endless tool calls.
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments="{}", oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        msg_events, msg_item, seq2 = _message_events(oi=0, text="Done.", start_seq=2)
        turn2 = [_created("i2"), _in_progress("i2"), *msg_events, _terminal("i2", [msg_item], seq=seq2)]
        handle = FakeHandle([turn1, turn2])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            await _run(handle, _req(tool_choice="required"))

        first_inner_request = handle.remote_calls[0][0][0]
        second_inner_request = handle.remote_calls[1][0][0]
        assert first_inner_request.tool_choice == "required"
        assert second_inner_request.tool_choice == "auto"

    @pytest.mark.asyncio
    async def test_failed_mcp_call_records_error_and_continues(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments="{}", oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        msg_events, msg_item, seq2 = _message_events(oi=0, text="Sorry, that failed.", start_seq=2)
        turn2 = [_created("i2"), _in_progress("i2"), *msg_events, _terminal("i2", [msg_item], seq=seq2)]
        handle = FakeHandle([turn1, turn2])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", side_effect=McpCallError("boom")):
            result = await _run(handle, _req())

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        mcp_call_item = next(o for o in output if o["type"] == "mcp_call")
        assert mcp_call_item["status"] == "failed"
        assert "boom" in mcp_call_item["error"]
        failed_events = [e["type"] for e in result if e["type"] == "response.mcp_call.failed"]
        assert failed_events == ["response.mcp_call.failed"]


class TestApprovalRequired:
    @pytest.mark.asyncio
    async def test_approval_required_terminates_completed_without_executing(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments="{}", oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        handle = FakeHandle([turn1])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool") as mock_call:
            result = await _run(
                handle, _req(tools=[{"type": "mcp", "server_label": "dice", "server_url": "http://fake/mcp"}])
            )

        mock_call.assert_not_called()
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        assert terminals[0]["response"]["status"] == "completed"
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_approval_request"]
        assert output[1]["name"] == "roll"
        # No mcp_call streaming events at all for a suppressed approval-bound call.
        assert not any("mcp_call" in e["type"] for e in result if "type" in e)
        # Approval-request output_item events must carry the correct offset output_index.
        approval_events = [e for e in result if e.get("item", {}).get("type") == "mcp_approval_request"]
        assert len(approval_events) == 2
        assert all(e["output_index"] == 1 for e in approval_events)

    @pytest.mark.asyncio
    async def test_approval_resume_approved_executes_call(self):
        # Turn after resume: model just answers, no more tool calls.
        msg_events, msg_item, seq = _message_events(oi=0, text="You rolled a 9.", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *msg_events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        request = _req(
            previous_response_id="resp_prior",
            input=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_1",
                    "name": "roll",
                    "arguments": '{"n":2}',
                    "server_label": "dice",
                },
                {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True},
            ],
        )
        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            result = await _run(handle, request)

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "mcp_call", "message"]
        assert output[1]["output"] == "9"
        assert output[1]["approval_request_id"] == "mcpr_1"
        assert output[1]["arguments"] == '{"n":2}'

        # The resumed call's event shape matches a normal mcp_call even though the
        # arguments are already known (from mcp_approval_request, not a live stream).
        mcp_related = [e["type"] for e in result if "mcp_call" in e["type"]]
        assert mcp_related == [
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
        ]
        delta_event = next(e for e in result if e["type"] == "response.mcp_call_arguments.delta")
        done_event = next(e for e in result if e["type"] == "response.mcp_call_arguments.done")
        assert delta_event["delta"] == '{"n":2}'
        assert done_event["arguments"] == '{"n":2}'

    @pytest.mark.asyncio
    async def test_approval_resume_rejected_records_failed_call_without_executing(self):
        msg_events, msg_item, seq = _message_events(oi=0, text="okay, skipping.", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *msg_events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        request = _req(
            previous_response_id="resp_prior",
            input=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_1",
                    "name": "roll",
                    "arguments": "{}",
                    "server_label": "dice",
                },
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": "mcpr_1",
                    "approve": False,
                    "reason": "no thanks",
                },
            ],
        )
        with patch("modelship.openai.mcp.loop.McpConnections.call_tool") as mock_call:
            result = await _run(handle, request)

        mock_call.assert_not_called()
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        rejected = next(o for o in output if o["type"] == "mcp_call")
        assert rejected["status"] == "failed"
        assert "rejected" in rejected["error"]
        assert "no thanks" in rejected["error"]
        # A rejected approval-resume call emits the same in_progress/failed streaming
        # pair as _execute_mcp_call, not just a terminal output item.
        assert [e["type"] for e in result if e["type"].startswith("response.mcp_call.")] == [
            "response.mcp_call.in_progress",
            "response.mcp_call.failed",
        ]

    @pytest.mark.asyncio
    async def test_approval_resume_rejects_forged_tool_name_not_in_discovery(self):
        # mcp_approval_request is client-echoed input, not server-verified — a
        # forged tool name must not execute.
        msg_events, msg_item, seq = _message_events(oi=0, text="okay.", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *msg_events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        request = _req(
            previous_response_id="resp_prior",
            input=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_1",
                    "name": "delete_everything",
                    "arguments": "{}",
                    "server_label": "dice",
                },
                {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True},
            ],
        )
        with patch("modelship.openai.mcp.loop.McpConnections.call_tool") as mock_call:
            result = await _run(handle, request)

        mock_call.assert_not_called()
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        rejected = next(o for o in output if o["type"] == "mcp_call")
        assert rejected["status"] == "failed"
        assert "not available" in rejected["error"]

    @pytest.mark.asyncio
    async def test_approval_resume_rejects_tool_filtered_out_by_allowed_tools(self):
        # A tool filtered out by this turn's allowed_tools policy must not execute
        # via approval-resume, even if it exists on the server.
        msg_events, msg_item, seq = _message_events(oi=0, text="okay.", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *msg_events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        request = _req(
            previous_response_id="resp_prior",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "dice",
                    "server_url": "http://fake/mcp",
                    "require_approval": "never",
                    "allowed_tools": ["roll"],
                }
            ],
            input=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_1",
                    "name": "peek_secrets",
                    "arguments": "{}",
                    "server_label": "dice",
                },
                {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True},
            ],
        )

        async def fake_list_two_tools(_self, _spec):
            return [
                McpListToolsTool(name="roll", input_schema={"type": "object"}, description="Roll dice."),
                McpListToolsTool(name="peek_secrets", input_schema={"type": "object"}, description="Leak secrets."),
            ]

        with (
            patch("modelship.openai.mcp.loop.McpConnections.list_tools", fake_list_two_tools),
            patch("modelship.openai.mcp.loop.McpConnections.call_tool") as mock_call,
        ):
            result = await _run(handle, request)

        mock_call.assert_not_called()
        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        output = terminals[0]["response"]["output"]
        assert "peek_secrets" not in [t["name"] for t in output[0]["tools"]]
        rejected = next(o for o in output if o["type"] == "mcp_call")
        assert rejected["status"] == "failed"
        assert "not available" in rejected["error"]

    @pytest.mark.asyncio
    async def test_approval_resume_unknown_request_id_yields_error(self):
        handle = FakeHandle([[]])
        request = _req(
            previous_response_id="resp_prior",
            input=[{"type": "mcp_approval_response", "approval_request_id": "nope", "approve": True}],
        )

        result = await _run(handle, request)

        assert isinstance(result[-1], ErrorResponse)
        assert handle._calls == 0

    @pytest.mark.asyncio
    async def test_approval_response_without_previous_response_id_rejected(self):
        # mcp_approval_request only ever exists as output from a prior response;
        # without previous_response_id there's no genuine one to resume from, so a
        # client can't self-supply both items to bypass approval.
        handle = FakeHandle([[]])
        request = _req(
            previous_response_id=None,
            input=[
                {
                    "type": "mcp_approval_request",
                    "id": "mcpr_1",
                    "name": "roll",
                    "arguments": "{}",
                    "server_label": "dice",
                },
                {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True},
            ],
        )

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool") as mock_call:
            result = await _run(handle, request)

        mock_call.assert_not_called()
        assert len(result) == 1
        assert isinstance(result[0], ErrorResponse)
        assert "previous_response_id" in result[0].error.message
        assert handle._calls == 0


class TestClientFunctionCallMixedWithMcp:
    @pytest.mark.asyncio
    async def test_client_tool_call_terminates_turn_and_is_handed_back(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="client_tool", arguments="{}", oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        handle = FakeHandle([turn1])

        request = _req(
            tools=[
                {"type": "mcp", "server_label": "dice", "server_url": "http://fake/mcp", "require_approval": "never"},
                {"type": "function", "name": "client_tool", "parameters": {}},
            ]
        )
        result = await _run(handle, request)

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert terminals[0]["response"]["status"] == "completed"
        output = terminals[0]["response"]["output"]
        assert [o["type"] for o in output] == ["mcp_list_tools", "function_call"]
        assert output[1]["name"] == "client_tool"
        # Client function_call events forwarded unchanged (not rewritten to mcp_call_*).
        added = next(
            e for e in result if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"
        )
        assert added["item"]["name"] == "client_tool"


class TestCaps:
    @pytest.mark.asyncio
    async def test_turn_cap_hit_terminates_incomplete(self):
        turns = []
        for i in range(mcp_loop.DEFAULT_TURN_CAP):
            tc_events, tc_item, seq = _tool_call_events(
                call_id=f"call_{i}", fc_id=f"fc_{i}", name="roll", arguments="{}", oi=0, start_seq=2
            )
            turns.append([_created(f"i{i}"), _in_progress(f"i{i}"), *tc_events, _terminal(f"i{i}", [tc_item], seq=seq)])
        handle = FakeHandle(turns)

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            result = await _run(handle, _req())

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert len(terminals) == 1
        assert terminals[0]["type"] == "response.incomplete"
        assert terminals[0]["response"]["incomplete_details"]["reason"] == "max_tool_calls"
        assert handle._calls == mcp_loop.DEFAULT_TURN_CAP

    @pytest.mark.asyncio
    async def test_max_tool_calls_caps_before_turn_limit(self):
        tc_events, tc_item, seq = _tool_call_events(
            call_id="call_1", fc_id="fc_1", name="roll", arguments="{}", oi=0, start_seq=2
        )
        turn1 = [_created("i1"), _in_progress("i1"), *tc_events, _terminal("i1", [tc_item], seq=seq)]
        handle = FakeHandle([turn1])

        with patch("modelship.openai.mcp.loop.McpConnections.call_tool", return_value="9"):
            result = await _run(handle, _req(max_tool_calls=1))

        terminals = [e for e in result if e["type"] in ("response.completed", "response.incomplete")]
        assert terminals[0]["type"] == "response.incomplete"
        assert terminals[0]["response"]["incomplete_details"]["reason"] == "max_tool_calls"
        assert handle._calls == 1


class TestNonStreaming:
    @pytest.mark.asyncio
    async def test_non_streaming_yields_single_response_object(self):
        events, msg_item, seq = _message_events(oi=0, text="hi there", start_seq=2)
        turn1 = [_created("i1"), _in_progress("i1"), *events, _terminal("i1", [msg_item], seq=seq)]
        handle = FakeHandle([turn1])

        result = await _run(handle, _req(stream=False))

        assert len(result) == 1
        assert isinstance(result[0], ResponseObject)
        assert result[0].status == "completed"
        assert [o.type for o in result[0].output] == ["mcp_list_tools", "message"]


class TestValidationErrors:
    @pytest.mark.asyncio
    async def test_missing_server_url_yields_error_response_before_any_event(self):
        handle = FakeHandle([[]])
        result = await _run(handle, _req(tools=[{"type": "mcp", "server_label": "s"}]))
        assert len(result) == 1
        assert isinstance(result[0], ErrorResponse)
        assert handle._calls == 0

    @pytest.mark.asyncio
    async def test_tool_choice_forcing_mcp_rejected(self):
        handle = FakeHandle([[]])
        result = await _run(handle, _req(tool_choice={"type": "mcp", "server_label": "dice"}))
        assert len(result) == 1
        assert isinstance(result[0], ErrorResponse)

    @pytest.mark.parametrize(
        "bad",
        [
            {"allowed_tools": "roll"},
            {"allowed_tools": {"tool_names": "roll"}},
            {"require_approval": "sometimes"},
            {"require_approval": {"never": "roll"}},
        ],
    )
    @pytest.mark.asyncio
    async def test_malformed_policy_shape_errors_before_any_event(self, bad):
        handle = FakeHandle([[]])
        tool = {"type": "mcp", "server_label": "dice", "server_url": "http://fake/mcp", **bad}
        result = await _run(handle, _req(tools=[tool]))
        assert len(result) == 1
        assert isinstance(result[0], ErrorResponse)
        assert handle._calls == 0


class TestDiscoveryFailure:
    @pytest.mark.asyncio
    async def test_discovery_failure_yields_response_failed(self):
        handle = FakeHandle([[]])
        with patch("modelship.openai.mcp.loop.McpConnections.list_tools", side_effect=McpCallError("unreachable")):
            result = await _run(handle, _req())

        failed = [e for e in result if e["type"] == "response.failed"]
        assert len(failed) == 1
        assert "unreachable" in failed[0]["response"]["error"]["message"]
        assert handle._calls == 0

    @pytest.mark.asyncio
    async def test_discovery_failure_non_streaming_yields_a_failed_response_object_not_nothing(self):
        # The non-streaming branch must capture response.failed like
        # completed/incomplete, or the generator yields nothing and _await_first
        # surfaces a generic 500.
        handle = FakeHandle([[]])
        with patch("modelship.openai.mcp.loop.McpConnections.list_tools", side_effect=McpCallError("unreachable")):
            result = await _run(handle, _req(stream=False))

        assert len(result) == 1
        assert isinstance(result[0], ResponseObject)
        assert result[0].status == "failed"
        assert "unreachable" in result[0].error["message"]

    @pytest.mark.asyncio
    async def test_discovery_failure_closes_the_connections(self):
        with (
            patch("modelship.openai.mcp.loop.McpConnections.list_tools", side_effect=McpCallError("unreachable")),
            patch("modelship.openai.mcp.loop.McpConnections.close") as close,
        ):
            await _run(FakeHandle([[]]), _req())

        close.assert_called_once_with()


class TestInnerFailurePassthrough:
    @pytest.mark.asyncio
    async def test_inner_error_response_passthrough(self):
        handle = FakeHandle([[create_error_response("bad request")]])
        result = await _run(handle, _req())
        assert isinstance(result[-1], ErrorResponse)
        # No terminal envelope follows an inner error: the loop stops right there.
        assert not any(
            isinstance(e, dict) and e["type"] in ("response.completed", "response.incomplete") for e in result
        )

    @pytest.mark.asyncio
    async def test_inner_response_failed_rewrites_id_to_outer(self):
        handle = FakeHandle(
            [
                [
                    _created("i1"),
                    _in_progress("i1"),
                    {
                        "type": "response.failed",
                        "sequence_number": 2,
                        "response": {"id": "i1", "status": "failed", "error": {"message": "loader exploded"}},
                    },
                ]
            ]
        )
        result = await _run(handle, _req())
        failed = [e for e in result if e["type"] == "response.failed"]
        assert len(failed) == 1
        assert failed[0]["response"]["id"] != "i1"
        assert failed[0]["response"]["error"]["message"] == "loader exploded"
