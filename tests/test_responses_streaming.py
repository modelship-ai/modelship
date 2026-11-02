"""Tests for the streaming Responses <-> chat-completions translator.

The translator consumes typed chat stream chunks every loader emits and emits
the Responses event protocol as plain dicts (transport-neutral — framing happens
at the gateway edge, not here). These tests drive it directly with synthesized
chat chunks (no Ray, no loader) and assert the event sequence/shape.
"""

import json

import pytest

from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ResponsesRequest,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.responses.streaming import ResponsesStreamTranslator, error_ws_frame, frame_sse


def _req(**overrides) -> ResponsesRequest:
    payload = {"model": "m", "input": "hi"}
    payload.update(overrides)
    return ResponsesRequest(**payload)


def _chunk(delta: DeltaMessage | None = None, *, finish_reason=None, usage=None) -> ChatCompletionStreamResponse:
    choices = []
    if delta is not None or finish_reason is not None:
        choices = [
            ChatCompletionResponseStreamChoice(index=0, delta=delta or DeltaMessage(), finish_reason=finish_reason)
        ]
    return ChatCompletionStreamResponse(model="m", choices=choices, usage=usage)


def _events(translator: ResponsesStreamTranslator, chunks) -> list[dict]:
    """Run start -> process(chunks) -> finish and return the event dicts."""
    events: list[dict] = []
    events.extend(translator.start())
    for c in chunks:
        events.extend(translator.process(c))
    events.extend(translator.finish())
    return events


def _events_then_fail(translator: ResponsesStreamTranslator, chunks, message: str) -> list[dict]:
    """Run start -> process(chunks) -> fail(message) and return the event dicts."""
    events: list[dict] = []
    events.extend(translator.start())
    for c in chunks:
        events.extend(translator.process(c))
    events.extend(translator.fail(message, "server_error"))
    return events


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


class TestTextStream:
    def test_envelope_and_text_event_sequence(self):
        translator = ResponsesStreamTranslator(_req())
        chunks = [
            _chunk(DeltaMessage(role="assistant")),
            _chunk(DeltaMessage(content="Hel")),
            _chunk(DeltaMessage(content="lo")),
            _chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
        ]
        events = _events(translator, chunks)
        assert _types(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

    def test_text_deltas_carry_pieces_and_done_carries_full(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [_chunk(DeltaMessage(content="Hel")), _chunk(DeltaMessage(content="lo"), finish_reason="stop")],
        )
        deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
        assert deltas == ["Hel", "lo"]
        done = next(e for e in events if e["type"] == "response.output_text.done")
        assert done["text"] == "Hello"

    def test_sequence_numbers_are_monotonic(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        seqs = [e["sequence_number"] for e in events]
        assert seqs == list(range(len(events)))

    def test_completed_response_carries_full_text_and_usage(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(content="hello world")),
                _chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
            ],
        )
        completed = events[-1]
        assert completed["type"] == "response.completed"
        resp = completed["response"]
        assert resp["status"] == "completed"
        assert resp["output"][0]["type"] == "message"
        assert resp["output"][0]["content"][0]["text"] == "hello world"
        assert resp["usage"]["input_tokens"] == 4
        assert resp["usage"]["output_tokens"] == 6

    def test_completed_response_has_completed_at(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        assert isinstance(events[-1]["response"]["completed_at"], int)

    def test_created_and_in_progress_envelopes_have_no_completed_at(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        assert events[0]["response"]["completed_at"] is None
        assert events[1]["response"]["completed_at"] is None

    def test_item_id_is_stable_across_events(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        msg_events = [e for e in events if e.get("item_id", "").startswith("msg_")]
        ids = {e["item_id"] for e in msg_events}
        assert len(ids) == 1
        # and the same id appears on the final output item
        item = next(e for e in events if e["type"] == "response.output_item.done")["item"]
        assert item["id"] == next(iter(ids))


class TestReasoningStream:
    def test_reasoning_emits_summary_events_before_message(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(reasoning="be")),
                _chunk(DeltaMessage(reasoning="cause")),
                _chunk(DeltaMessage(content="answer"), finish_reason="stop"),
            ],
        )
        types = _types(events)
        assert "response.reasoning_summary_text.delta" in types
        # reasoning item opens before the message item
        assert types.index("response.reasoning_summary_part.added") < types.index("response.content_part.added")
        done = next(e for e in events if e["type"] == "response.reasoning_summary_text.done")
        assert done["text"] == "because"

    def test_completed_has_reasoning_then_message(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [_chunk(DeltaMessage(reasoning="think")), _chunk(DeltaMessage(content="ans"), finish_reason="stop")],
        )
        out = events[-1]["response"]["output"]
        assert out[0]["type"] == "reasoning"
        assert out[0]["summary"][0]["text"] == "think"
        assert out[1]["type"] == "message"


class TestToolCallStream:
    def test_tool_call_args_stream_and_finalize(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(
                    DeltaMessage(
                        tool_calls=[DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name="get_weather"))]
                    )
                ),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='{"city":'))])
                ),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='"NYC"}'))])
                ),
                _chunk(finish_reason="tool_calls"),
            ],
        )
        types = _types(events)
        assert "response.function_call_arguments.delta" in types
        arg_deltas = [e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"]
        assert "".join(arg_deltas) == '{"city":"NYC"}'
        done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
        assert done["arguments"] == '{"city":"NYC"}'
        # completed output carries the function_call item with name + call_id
        fc = next(o for o in events[-1]["response"]["output"] if o["type"] == "function_call")
        assert fc["name"] == "get_weather"
        assert fc["call_id"] == "call_1"
        assert fc["arguments"] == '{"city":"NYC"}'
        # tool_calls finish reason is still a completed response
        assert events[-1]["type"] == "response.completed"

    def test_interleaved_content_after_tool_call_is_merged_into_one_message(self):
        # answer -> tool_call -> more answer: the trailing text appends to the
        # already-open message item (not dropped, not a second message).
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(content="before ")),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name="f"))])
                ),
                _chunk(DeltaMessage(content="after"), finish_reason="tool_calls"),
            ],
        )
        out = events[-1]["response"]["output"]
        messages = [o for o in out if o["type"] == "message"]
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "before after"


class TestTerminalStatus:
    def test_length_finish_reason_is_incomplete(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="length")])
        assert events[-1]["type"] == "response.incomplete"
        assert events[-1]["response"]["status"] == "incomplete"
        assert events[-1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
        # incomplete isn't "done" — completed_at stays unset.
        assert events[-1]["response"]["completed_at"] is None

    def test_content_filter_finish_reason_is_incomplete(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="content_filter")])
        assert events[-1]["type"] == "response.incomplete"
        assert events[-1]["response"]["incomplete_details"] == {"reason": "content_filter"}


class TestFailedStream:
    def test_fail_before_any_output_emits_bare_failed_event(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events_then_fail(translator, [], "boom")
        assert _types(events) == ["response.created", "response.in_progress", "response.failed"]
        failed = events[-1]["response"]
        assert failed["status"] == "failed"
        assert failed["error"] == {"code": "server_error", "message": "boom"}
        assert failed["output"] == []

    def test_fail_mid_message_closes_the_open_item_with_partial_text(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events_then_fail(translator, [_chunk(DeltaMessage(content="partial"))], "boom")
        assert _types(events)[-4:] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.failed",
        ]
        failed = events[-1]["response"]
        assert failed["status"] == "failed"
        assert failed["output"][0]["type"] == "message"
        assert failed["output"][0]["content"][0]["text"] == "partial"

    def test_fail_mid_tool_call_closes_it_with_partial_arguments(self):
        translator = ResponsesStreamTranslator(_req())
        tc = DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name="get_weather", arguments='{"lo'))
        events = _events_then_fail(translator, [_chunk(DeltaMessage(tool_calls=[tc]))], "boom")
        failed = events[-1]["response"]
        assert failed["output"][0]["type"] == "function_call"
        assert failed["output"][0]["arguments"] == '{"lo'
        assert failed["output"][0]["status"] == "completed"


class TestFrameSse:
    """Wire-format pin for the HTTP transport edge: dict -> SSE frame, `[DONE]`
    appended only once something was actually framed."""

    @staticmethod
    async def _drain(items: list) -> list[str]:
        async def gen():
            for item in items:
                yield item

        return [chunk async for chunk in frame_sse(gen())]

    @pytest.mark.asyncio
    async def test_event_dict_framed_exactly_like_the_old_sse_builder(self):
        event = {"type": "response.created", "sequence_number": 0}
        out = await self._drain([event])
        assert out == [f"event: response.created\ndata: {json.dumps(event)}\n\n", "data: [DONE]\n\n"]

    @pytest.mark.asyncio
    async def test_done_appended_after_a_framed_stream(self):
        events = [
            {"type": "response.created", "sequence_number": 0},
            {"type": "response.completed", "sequence_number": 1, "response": {}},
        ]
        out = await self._drain(events)
        assert out[-1] == "data: [DONE]\n\n"
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_non_dict_passthrough_gets_no_done_sentinel(self):
        # A single-shot reply, not a stream — nothing was framed, so no [DONE].
        out = await self._drain(["not a dict"])
        assert out == ["not a dict"]


class TestErrorWsFrame:
    def test_valid_websocket_error_event_shape(self):
        err = create_error_response("bad request", err_type="invalid_request_error", code="previous_response_not_found")
        frame = json.loads(error_ws_frame(err))
        assert frame == {
            "type": "error",
            "status": 400,
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "code": "previous_response_not_found",
            },
        }

    def test_missing_code_falls_back_to_error_type(self):
        err = create_error_response("boom", err_type="api_error", status_code=500)
        frame = json.loads(error_ws_frame(err))
        assert frame["error"]["code"] == "api_error"
