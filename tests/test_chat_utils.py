"""Tests for modelship.openai.utils.chat."""

import pytest

from modelship.openai.protocol import FunctionCall, ToolCall, UsageInfo
from modelship.openai.utils.chat import (
    ParsedChatOutput,
    build_from_parsed,
    normalize_chat_messages,
    truncate_messages_to_fit,
)


async def _char_count(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


@pytest.mark.asyncio
async def test_truncate_under_budget_returns_unchanged():
    messages = [{"role": "user", "content": "x" * 10}]
    out = await truncate_messages_to_fit(messages, 100, _char_count)
    assert out is messages


@pytest.mark.asyncio
async def test_truncate_zero_budget_returns_unchanged():
    messages = [{"role": "user", "content": "x" * 10}]
    out = await truncate_messages_to_fit(messages, 0, _char_count)
    assert out is messages


@pytest.mark.asyncio
async def test_truncate_drops_oldest_keeping_system_and_final():
    messages = [
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "a" * 20},
        {"role": "assistant", "content": "b" * 20},
        {"role": "user", "content": "c" * 5},
    ]
    out = await truncate_messages_to_fit(messages, 40, _char_count)
    assert out == [messages[0], messages[2], messages[3]]


@pytest.mark.asyncio
async def test_truncate_keeps_all_leading_system_messages():
    messages = [
        {"role": "system", "content": "A" * 5},
        {"role": "developer", "content": "B" * 5},
        {"role": "user", "content": "a" * 20},
        {"role": "user", "content": "c" * 5},
    ]
    out = await truncate_messages_to_fit(messages, 20, _char_count)
    assert out == [messages[0], messages[1], messages[3]]


@pytest.mark.asyncio
async def test_truncate_falls_back_to_system_plus_final():
    messages = [
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "a" * 20},
        {"role": "assistant", "content": "b" * 20},
        {"role": "user", "content": "c" * 5},
    ]
    out = await truncate_messages_to_fit(messages, 16, _char_count)
    assert out == [messages[0], messages[3]]


@pytest.mark.asyncio
async def test_truncate_never_orphans_a_tool_result():
    messages = [
        {"role": "system", "content": "S" * 5},
        {"role": "assistant", "content": "a" * 10},
        {"role": "tool", "tool_call_id": "x", "content": "t" * 10},
        {"role": "user", "content": "u" * 5},
    ]
    out = await truncate_messages_to_fit(messages, 22, _char_count)
    assert out == [messages[0], messages[3]]
    assert messages[2] not in out


@pytest.mark.asyncio
async def test_truncate_all_system_returns_unchanged():
    messages = [{"role": "system", "content": "x" * 50}]
    out = await truncate_messages_to_fit(messages, 10, _char_count)
    assert out is messages


@pytest.mark.asyncio
async def test_truncate_keeps_trailing_tool_result_with_its_call():
    messages = [
        {"role": "system", "content": "S" * 5},
        {"role": "user", "content": "u" * 30},
        {"role": "assistant", "content": "a" * 10},
        {"role": "tool", "tool_call_id": "x", "content": "t" * 10},
    ]
    out = await truncate_messages_to_fit(messages, 30, _char_count)
    assert out == [messages[0], messages[2], messages[3]]


@pytest.mark.asyncio
async def test_truncate_trailing_tool_pair_preserved_when_overflowing():
    messages = [
        {"role": "system", "content": "S" * 5},
        {"role": "user", "content": "u" * 30},
        {"role": "assistant", "content": "a" * 40},
        {"role": "tool", "tool_call_id": "x", "content": "t" * 40},
    ]
    out = await truncate_messages_to_fit(messages, 30, _char_count)
    assert out == [messages[0], messages[2], messages[3]]
    assert out[1].get("role") != "tool"


def test_tool_message_name_backfilled_from_assistant_call():
    messages = [
        {"role": "user", "content": "turn off the lights"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "HassTurnOff"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "content": '{"success": true}'},
    ]

    out = normalize_chat_messages(messages)

    tool_msg = out[-1]
    assert tool_msg["name"] == "HassTurnOff"
    # caller's dicts are not mutated
    assert "name" not in messages[-1]


def test_existing_tool_message_name_preserved():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "function": {"name": "HassTurnOff"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "name": "explicit", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert out[-1]["name"] == "explicit"


def test_orphan_tool_message_left_unchanged():
    messages = [
        {"role": "tool", "tool_call_id": "missing", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert "name" not in out[-1]


def test_malformed_tool_calls_do_not_raise():
    messages = [
        {
            "role": "assistant",
            # tool_calls / function in unexpected shapes must not blow up
            "tool_calls": [
                "not-a-dict",
                {"id": "x", "function": "not-a-dict"},
                {"id": ["unhashable"], "function": {"name": "Skipped"}},
                {"id": "y", "function": {"name": "Real"}},
            ],
        },
        {"role": "assistant", "tool_calls": "not-a-list"},
        {"role": "tool", "tool_call_id": "y", "content": "ok"},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert out[-2]["name"] == "Real"
    # the call whose function was malformed yields no name
    assert "name" not in out[-1]


def test_non_string_tool_call_id_does_not_raise():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "function": {"name": "HassTurnOff"}}],
        },
        # unhashable tool_call_id must not raise on the mapping lookup
        {"role": "tool", "tool_call_id": ["abc"], "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert "name" not in out[-1]


def test_text_part_collapse_still_works():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ],
        },
    ]

    out = normalize_chat_messages(messages)

    assert out[0]["content"] == "hello\nworld"


def test_build_from_parsed_multi_choice_and_dto():
    choices = [
        ParsedChatOutput(
            content="Hello from choice 0",
            reasoning="Thinking 0...",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="get_weather", arguments="{}"),
                )
            ],
        ),
        ParsedChatOutput(
            content="Hello from choice 1",
            reasoning="Thinking 1...",
            tool_calls=[],
        ),
    ]
    usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    res1 = build_from_parsed(
        request_id="test_req_1",
        model_name="test_model",
        choices=choices,
        usage=usage,
        finish_reasons="length",
        created=12345,
    )

    assert res1.id == "test_req_1"
    assert res1.model == "test_model"
    assert len(res1.choices) == 2
    assert res1.choices[0].index == 0
    assert res1.choices[0].message.content == "Hello from choice 0"
    assert res1.choices[0].message.reasoning == "Thinking 0..."
    assert len(res1.choices[0].message.tool_calls) == 1
    assert res1.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert res1.choices[0].finish_reason == "length"

    assert res1.choices[1].index == 1
    assert res1.choices[1].message.content == "Hello from choice 1"
    assert res1.choices[1].message.reasoning == "Thinking 1..."
    assert len(res1.choices[1].message.tool_calls) == 0
    assert res1.choices[1].finish_reason == "length"

    res2 = build_from_parsed(
        request_id="test_req_2",
        model_name="test_model",
        choices=choices,
        usage=usage,
        created=12345,
    )
    assert res2.choices[0].finish_reason == "tool_calls"
    assert res2.choices[1].finish_reason == "stop"

    res3 = build_from_parsed(
        request_id="test_req_3",
        model_name="test_model",
        choices=choices,
        usage=usage,
        finish_reasons=["length", "stop"],
        created=12345,
    )
    assert res3.choices[0].finish_reason == "length"
    assert res3.choices[1].finish_reason == "stop"
