"""Tests for modelship.openai.utils.responses's parsed-output shaping."""

import pytest

from modelship.openai.protocol import FunctionCall, ResponsesRequest, ToolCall, UsageInfo
from modelship.openai.utils.chat import ParsedChatOutput
from modelship.openai.utils.responses import build_response_from_parsed, build_responses_items_from_parsed


def test_build_responses_items_from_parsed_orders_reasoning_message_tools():
    parsed = ParsedChatOutput(
        content="Hello!",
        reasoning="Thinking...",
        tool_calls=[ToolCall(id="call_1", type="function", function=FunctionCall(name="get_weather", arguments="{}"))],
    )

    output = build_responses_items_from_parsed(parsed)

    assert [item.type for item in output] == ["reasoning", "message", "function_call"]
    assert output[0].summary[0].text == "Thinking..."
    assert output[1].content[0].text == "Hello!"
    assert output[2].call_id == "call_1"
    assert output[2].name == "get_weather"
    assert output[2].arguments == "{}"


def test_build_responses_items_from_parsed_skips_empty_fields():
    parsed = ParsedChatOutput(content=None, reasoning=None, tool_calls=[])

    assert build_responses_items_from_parsed(parsed) == []


def _usage() -> UsageInfo:
    return UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)


def test_build_response_from_parsed_sets_completed_at_when_completed():
    request = ResponsesRequest(model="m", input="hi")
    parsed = ParsedChatOutput(content="hi", reasoning=None, tool_calls=[])

    resp = build_response_from_parsed(parsed, request, usage=_usage(), finish_reason="stop", model="m")

    assert resp.status == "completed"
    assert isinstance(resp.completed_at, int)


def test_build_response_from_parsed_leaves_completed_at_none_when_incomplete():
    request = ResponsesRequest(model="m", input="hi")
    parsed = ParsedChatOutput(content="hi", reasoning=None, tool_calls=[])

    resp = build_response_from_parsed(parsed, request, usage=_usage(), finish_reason="length", model="m")

    assert resp.status == "incomplete"
    assert resp.completed_at is None


_CALL = ToolCall(id="call_1", type="function", function=FunctionCall(name="f", arguments='{"a":'))


@pytest.mark.parametrize(
    ("parsed", "finish_reason", "expected"),
    [
        (
            ParsedChatOutput(content="hi", reasoning="th", tool_calls=[_CALL]),
            "length",
            ["completed", "completed", "incomplete"],
        ),
        (ParsedChatOutput(content=None, reasoning="th", tool_calls=[]), "length", ["incomplete"]),
        (
            ParsedChatOutput(content="hi", reasoning="th", tool_calls=[_CALL]),
            "stop",
            ["completed", "completed", "completed"],
        ),
    ],
)
def test_only_the_last_item_of_a_cut_off_response_is_incomplete(parsed, finish_reason, expected):
    request = ResponsesRequest(model="m", input="hi")

    resp = build_response_from_parsed(parsed, request, usage=_usage(), finish_reason=finish_reason, model="m")

    assert [item.status for item in resp.output] == expected
