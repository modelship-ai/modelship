"""`/v1/realtime` sessions in the gateway: opening on a pipeline, session.update, the conversation's items, and
responses from a fake LLM that streams real Responses events."""

import asyncio

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    PromptTokenUsageInfo,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.realtime import RealtimeResponse
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.openai.protocol.responses.streaming import ResponsesStreamTranslator
from modelship.openai.realtime import serve_session
from modelship.openai.realtime.response import ActiveResponse

ASSISTANT = {"name": "assistant", "stt": "whisper", "llm": "qwen", "session_defaults": {"instructions": "Be brief."}}
SCRIBE = {"name": "scribe", "stt": "whisper"}


def _chunk(delta: DeltaMessage | None = None, *, finish_reason=None, usage=None) -> ChatCompletionStreamResponse:
    choices = []
    if delta is not None or finish_reason is not None:
        choices = [
            ChatCompletionResponseStreamChoice(index=0, delta=delta or DeltaMessage(), finish_reason=finish_reason)
        ]
    return ChatCompletionStreamResponse(model="qwen", choices=choices, usage=usage)


USAGE = UsageInfo(
    prompt_tokens=7, completion_tokens=2, total_tokens=9, prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=3)
)


def _answer(*text: str, reasoning: str | None = None, finish: str | None = "stop", fail: bool = False):
    """A script: the Responses events a loader streams for an answer of *text*. With *finish* None the answer
    never ends, until the call is cancelled."""

    def script(request: ResponsesRequest) -> list:
        translator = ResponsesStreamTranslator(request)
        events = [*translator.start()]
        if reasoning:
            events += translator.process(_chunk(DeltaMessage(reasoning=reasoning)))
        for piece in text:
            events += translator.process(_chunk(DeltaMessage(content=piece)))
        if fail:
            return events + list(translator.fail("engine died", "server_error"))
        if finish is not None:
            events += translator.process(_chunk(finish_reason=finish, usage=USAGE))
            events += translator.finish()
        return events

    script.endless = finish is None and not fail  # type: ignore[attr-defined]
    return script


class _Llm:
    """The model "qwen", answering each call with the next script and recording the calls."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.requests: list[ResponsesRequest] = []
        self.sessions: list[str] = []
        self.started = 0
        # per answer: whether it was closed before its end
        self.closed_early: list[bool] = []

    def respond(self, model: str, request: ResponsesRequest, session_id: str):
        if model != "qwen":
            return None
        self.requests.append(request)
        self.sessions.append(session_id)
        script = self.scripts.pop(0)
        events = script(request) if callable(script) else script
        return self._answer(events, getattr(script, "endless", False))

    async def _answer(self, events: list, endless: bool):
        self.started += 1
        finished = False
        try:
            for event in events:
                yield event
            if endless:
                await asyncio.Event().wait()
            finished = True
        finally:
            self.closed_early.append(not finished)


def _client(pipelines: dict, known=("whisper", "qwen"), serving=("whisper", "qwen"), llm: _Llm | None = None):
    app = FastAPI()

    def respond(model: str, request: ResponsesRequest, session_id: str):
        return llm.respond(model, request, session_id) if llm is not None else None

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        await serve_session(websocket, pipelines, list(known), set(serving), respond)

    return TestClient(app)


def _route(raw: dict, ready: bool = True) -> dict:
    return {raw["name"]: {"config": raw, "app": "gw.x-p0123456789", "ready": ready}}


def _refused(client: TestClient, query: str) -> dict:
    with client.websocket_connect(f"/v1/realtime{query}") as ws:
        event = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1008
    assert event["type"] == "error"
    return event["error"]


def _open(ws) -> dict:
    """The session, once its opening events are read."""
    session = ws.receive_json()["session"]
    if session["type"] == "realtime":
        assert ws.receive_json()["type"] == "conversation.created"
    return session


def _until(ws, kind: str) -> list[dict]:
    """Events up to and including the first of *kind*."""
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["type"] == kind:
            return events


def _say(ws, text: str, role: str = "user", id: str | None = None, previous_item_id: str | None = None) -> dict:
    part = "output_text" if role == "assistant" else "input_text"
    item = {"type": "message", "role": role, "content": [{"type": part, "text": text}]}
    if id is not None:
        item["id"] = id
    event = {"type": "conversation.item.create", "item": item}
    if previous_item_id is not None:
        event["previous_item_id"] = previous_item_id
    ws.send_json(event)
    added, done = ws.receive_json(), ws.receive_json()
    assert (added["type"], done["type"]) == ("conversation.item.added", "conversation.item.done")
    return added


def _text(item: dict) -> str:
    return "".join(part["text"] for part in item["content"])


class TestOpening:
    def test_a_known_pipeline_opens_with_its_defaults(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            event = ws.receive_json()
            conversation = ws.receive_json()
        assert event["type"] == "session.created"
        assert event["event_id"].startswith("event_")
        session = event["session"]
        assert session["type"] == "realtime"
        assert session["object"] == "realtime.session"
        assert session["model"] == "assistant"
        assert session["instructions"] == "Be brief."
        assert session["output_modalities"] == ["text"]
        assert session["id"].startswith("sess_")
        assert conversation["type"] == "conversation.created"
        assert conversation["conversation"]["id"].startswith("conv_")

    def test_a_pipeline_without_an_llm_opens_a_transcription_session(self):
        with _client(_route(SCRIBE)).websocket_connect("/v1/realtime?model=scribe") as ws:
            session = ws.receive_json()["session"]
        assert (session["type"], session["object"]) == ("transcription", "realtime.transcription_session")

    def test_without_a_model_it_is_refused(self):
        assert _refused(_client(_route(ASSISTANT)), "")["code"] == "missing_model"

    def test_an_unknown_pipeline_is_refused(self):
        error = _refused(_client(_route(ASSISTANT)), "?model=nope")
        assert error["code"] == "model_not_found"
        assert "No pipeline 'nope'" in error["message"]

    def test_a_pipeline_whose_model_the_gateway_lacks_is_refused_naming_it(self):
        error = _refused(_client(_route(ASSISTANT), known=("whisper",)), "?model=assistant")
        assert error["code"] == "model_not_found"
        assert "uses model 'qwen', which this gateway doesn't have" in error["message"]

    def test_a_pipeline_whose_model_is_not_serving_yet_is_refused(self):
        error = _refused(_client(_route(ASSISTANT), serving=("whisper",)), "?model=assistant")
        assert error["code"] == "model_not_ready"
        assert "'qwen', which isn't ready yet" in error["message"]

    def test_a_pipeline_whose_audio_app_is_not_ready_is_refused(self):
        error = _refused(_client(_route(ASSISTANT, ready=False)), "?model=assistant")
        assert error == {**error, "code": "model_not_ready", "message": "Pipeline 'assistant' isn't ready yet."}


class TestSessionUpdate:
    def test_session_update_answers_with_the_updated_session(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            update = {"type": "realtime", "instructions": "Speak French.", "reasoning": {"effort": "low"}}
            ws.send_json({"type": "session.update", "session": update})
            event = ws.receive_json()
        assert event["type"] == "session.updated"
        assert event["session"]["instructions"] == "Speak French."
        assert event["session"]["reasoning"] == {"effort": "low"}

    def test_session_fields_not_used_yet_are_ignored(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "marin"}}}})
            event = ws.receive_json()
        assert event["type"] == "session.updated"
        assert "audio" not in event["session"]

    def test_the_session_type_cannot_change(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "event_id": "e1", "session": {"type": "transcription"}})
            error = ws.receive_json()["error"]
        assert (error["param"], error["event_id"]) == ("session.type", "e1")

    def test_an_invalid_field_names_its_param(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"output_modalities": ["video"]}})
            error = ws.receive_json()["error"]
        assert error["code"] == "invalid_value"
        assert error["param"].startswith("session.output_modalities")

    def test_audio_output_is_not_supported_yet(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"output_modalities": ["audio"]}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("unsupported_value", "session.output_modalities")

    def test_max_output_tokens_below_one_is_refused(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"max_output_tokens": 0}})
            error = ws.receive_json()["error"]
            ws.send_json({"type": "session.update", "session": {"max_output_tokens": "inf"}})
            updated = ws.receive_json()
        assert error["param"].startswith("session.max_output_tokens")
        assert updated["session"]["max_output_tokens"] == "inf"


class TestEvents:
    def test_audio_events_are_not_supported_yet_and_the_session_stays_open(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "input_audio_buffer.append", "event_id": "e2", "audio": ""})
            error = ws.receive_json()["error"]
            ws.send_json({"type": "session.update", "session": {}})
            still_open = ws.receive_json()
        assert (error["code"], error["event_id"]) == ("unsupported_event", "e2")
        assert error["message"] == "input_audio_buffer.append is not supported yet."
        assert still_open["type"] == "session.updated"

    @pytest.mark.parametrize("kind", ["conversation.item.create", "response.create", "response.cancel"])
    def test_a_transcription_session_takes_no_conversation_or_response(self, kind):
        with _client(_route(SCRIBE)).websocket_connect("/v1/realtime?model=scribe") as ws:
            _open(ws)
            ws.send_json({"type": kind})
            error = ws.receive_json()["error"]
        assert error["code"] == "invalid_event"

    @pytest.mark.parametrize("frame", ["not json", "[1, 2]", '{"no": "type"}'])
    def test_a_malformed_event_is_an_error(self, frame):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_text(frame)
            event = ws.receive_json()
        assert event["type"] == "error"

    def test_a_binary_frame_is_an_error(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_bytes(b"\x00\x01")
            event = ws.receive_json()
        assert event["error"]["message"] == "Send events as JSON text frames."


class TestConversation:
    def test_items_are_added_in_order_and_retrieved(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            first = _say(ws, "Hi")
            second = _say(ws, "Be nice.", role="system")
            ws.send_json({"type": "conversation.item.retrieve", "item_id": first["item"]["id"]})
            retrieved = ws.receive_json()
        assert first["previous_item_id"] is None
        assert second["previous_item_id"] == first["item"]["id"]
        assert first["item"] == {
            **first["item"],
            "type": "message",
            "object": "realtime.item",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hi"}],
        }
        assert first["item"]["id"].startswith("item_")
        assert retrieved == {**retrieved, "type": "conversation.item.retrieved", "item": first["item"]}

    def test_an_item_goes_after_its_previous_item_or_first_at_root(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            _say(ws, "one", id="a")
            _say(ws, "three", id="c")
            middle = _say(ws, "two", id="b", previous_item_id="a")
            first = _say(ws, "zero", id="z", previous_item_id="root")
        assert (middle["previous_item_id"], first["previous_item_id"]) == ("a", None)

    def test_a_deleted_item_is_gone(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            _say(ws, "Hi", id="a")
            ws.send_json({"type": "conversation.item.delete", "item_id": "a"})
            deleted = ws.receive_json()
            ws.send_json({"type": "conversation.item.retrieve", "item_id": "a"})
            error = ws.receive_json()["error"]
        assert deleted == {**deleted, "type": "conversation.item.deleted", "item_id": "a"}
        assert (error["code"], error["param"]) == ("item_not_found", "item_id")

    @pytest.mark.parametrize(
        ("item", "code", "param"),
        [
            ({"type": "function_call", "name": "f"}, "unsupported_value", "item.type"),
            (
                {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": ""}]},
                "unsupported_value",
                "item.content[0]",
            ),
            (
                {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "x"}]},
                "invalid_value",
                "item.content[0]",
            ),
            ({"type": "message", "role": "tool", "content": []}, "invalid_value", "item.role"),
            ({"type": "image"}, "invalid_value", "item.type"),
            ("hi", "invalid_value", "item"),
        ],
    )
    def test_an_item_it_cannot_hold_is_refused(self, item, code, param):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "conversation.item.create", "item": item})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == (code, param)

    def test_an_unknown_previous_item_or_a_taken_id_is_refused(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            _say(ws, "Hi", id="a")
            item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}
            ws.send_json({"type": "conversation.item.create", "item": item, "previous_item_id": "nope"})
            unknown = ws.receive_json()["error"]
            ws.send_json({"type": "conversation.item.create", "item": {**item, "id": "a"}})
            taken = ws.receive_json()["error"]
        assert (unknown["code"], unknown["param"]) == ("item_not_found", "previous_item_id")
        assert (taken["code"], taken["param"]) == ("invalid_value", "item.id")


class TestResponses:
    def _session(self, llm: _Llm, pipeline: dict = ASSISTANT):
        return _client(_route(pipeline), llm=llm).websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def test_a_response_answers_the_conversation_and_joins_it(self):
        llm = _Llm(_answer("Hello", " there"))
        with self._session(llm) as ws:
            session = _open(ws)
            user = _say(ws, "Hi")["item"]
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        assert [event["type"] for event in events] == [
            "response.created",
            "response.output_item.added",
            "conversation.item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        created, done = events[0]["response"], events[-1]["response"]
        assert (created["status"], created["object"], created["id"]) == ("in_progress", "realtime.response", done["id"])
        assert done["status"] == "completed"
        assert done["output_modalities"] == ["text"]
        assert done["conversation_id"].startswith("conv_")
        (answer,) = done["output"]
        assert answer == {**answer, "role": "assistant", "status": "completed", "object": "realtime.item"}
        assert answer["content"] == [{"type": "output_text", "text": "Hello there"}]
        assert events[2]["previous_item_id"] == user["id"]
        assert events[4] == {**events[4], "response_id": done["id"], "item_id": answer["id"], "delta": "Hello"}
        assert events[7]["part"] == {"type": "text", "text": "Hello there"}
        assert done["usage"]["input_tokens"] == 7
        assert done["usage"]["output_tokens"] == 2
        assert done["usage"]["input_token_details"]["cached_tokens"] == 3
        (request,) = llm.requests
        assert request.input == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
        assert (request.model, request.instructions, request.store, request.stream) == (
            "qwen",
            "Be brief.",
            False,
            True,
        )
        assert request.max_output_tokens is None
        assert llm.sessions == [session["id"]]
        assert llm.closed_early == [False]

    def test_the_next_response_sees_the_last_answer(self):
        llm = _Llm(_answer("Four."), _answer("Eight."))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "2+2?")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
            _say(ws, "Double it.")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert [(item["role"], _text(item)) for item in llm.requests[1].input] == [
            ("user", "2+2?"),
            ("assistant", "Four."),
            ("user", "Double it."),
        ]

    def test_reasoning_reaches_neither_the_client_nor_the_conversation(self):
        llm = _Llm(_answer("Yes.", reasoning="Let me think."), _answer("Ok."))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "Sure?")
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert not [event for event in events if "reasoning" in event["type"]]
        assert [event["output_index"] for event in events if "output_index" in event] == [0] * 6
        assert [item["role"] for item in llm.requests[1].input] == ["user", "assistant"]

    def test_a_response_takes_its_own_settings_over_the_sessions(self):
        llm = _Llm(_answer("A."), _answer("B."))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json(
                {"type": "session.update", "session": {"reasoning": {"effort": "low"}, "max_output_tokens": 9}}
            )
            ws.receive_json()
            _say(ws, "Hi")
            ws.send_json({"type": "response.create"})
            first = _until(ws, "response.done")[-1]["response"]
            overrides = {"instructions": "Rhyme.", "max_output_tokens": "inf", "reasoning": {"effort": "high"}}
            ws.send_json({"type": "response.create", "response": overrides})
            second = _until(ws, "response.done")[-1]["response"]
        assert [(r.instructions, r.max_output_tokens, r.reasoning) for r in llm.requests] == [
            ("Be brief.", 9, {"effort": "low"}),
            ("Rhyme.", None, {"effort": "high"}),
        ]
        assert (first["max_output_tokens"], second["max_output_tokens"]) == (9, "inf")

    def test_an_out_of_band_response_stays_out_of_the_conversation(self):
        llm = _Llm(_answer("Kitchen."), _answer("Fine."))
        with self._session(llm) as ws:
            _open(ws)
            first = _say(ws, "Where am I?")["item"]
            _say(ws, "Hi")
            params = {
                "conversation": "none",
                "metadata": {"topic": "where"},
                "input": [
                    {"type": "item_reference", "id": first["id"]},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "One word."}]},
                ],
            }
            ws.send_json({"type": "response.create", "response": params})
            events = _until(ws, "response.done")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert not [event for event in events if event["type"].startswith("conversation.")]
        done = events[-1]["response"]
        assert (done["conversation_id"], done["metadata"]) == (None, {"topic": "where"})
        assert [_text(item) for item in llm.requests[0].input] == ["Where am I?", "One word."]
        assert [_text(item) for item in llm.requests[1].input] == ["Where am I?", "Hi"]

    def test_an_unknown_reference_or_a_bad_input_item_is_refused(self):
        with self._session(_Llm()) as ws:
            _open(ws)
            ws.send_json({"type": "response.create", "response": {"input": [{"type": "item_reference", "id": "x"}]}})
            unknown = ws.receive_json()["error"]
            ws.send_json({"type": "response.create", "response": {"input": [{"type": "message", "role": "dog"}]}})
            bad = ws.receive_json()["error"]
        assert (unknown["code"], unknown["param"]) == ("item_not_found", "response.input[0].id")
        assert (bad["code"], bad["param"]) == ("invalid_value", "response.input[0].role")

    def test_one_response_at_a_time(self):
        llm = _Llm(_answer("Thinking", finish=None))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "Hi")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.output_text.delta")
            ws.send_json({"type": "response.create", "event_id": "again"})
            error = ws.receive_json()["error"]
        assert (error["code"], error["event_id"]) == ("conversation_already_has_active_response", "again")

    def test_a_cancelled_response_ends_with_what_it_said(self):
        llm = _Llm(_answer("Once upon", finish=None), _answer("Ok."))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "Tell me a story.")
            ws.send_json({"type": "response.create"})
            created = _until(ws, "response.output_text.delta")[0]["response"]
            ws.send_json({"type": "response.cancel", "response_id": created["id"]})
            events = _until(ws, "response.done")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert [event["type"] for event in events] == [
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        assert events[0]["item"]["status"] == "incomplete"
        done = events[-1]["response"]
        assert (done["status"], done["status_details"]["reason"]) == ("cancelled", "client_cancelled")
        assert llm.closed_early[0]
        assert [(item["role"], _text(item)) for item in llm.requests[1].input][-1] == ("assistant", "Once upon")

    def test_cancelling_with_nothing_in_progress_is_an_error(self):
        llm = _Llm(_answer("Hi."))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "response.cancel"})
            idle = ws.receive_json()["error"]
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
            ws.send_json({"type": "response.cancel"})
            finished = ws.receive_json()["error"]
        assert idle["code"] == finished["code"] == "response_cancel_not_active"

    def test_leaving_mid_response_cancels_the_call(self):
        llm = _Llm(_answer("Once upon", finish=None))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "Tell me a story.")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.output_text.delta")
        assert llm.closed_early == [True]

    def test_a_loader_error_fails_the_response(self):
        llm = _Llm([create_error_response("The prompt is too long.", err_type="invalid_request_error")])
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        assert [event["type"] for event in events] == ["response.created", "error", "response.done"]
        assert events[1]["error"]["message"] == "The prompt is too long."
        done = events[-1]["response"]
        assert (done["status"], done["status_details"]["type"]) == ("failed", "failed")
        assert done["status_details"]["error"]["type"] == "invalid_request_error"

    def test_a_failure_mid_answer_fails_the_response_and_keeps_what_it_said(self):
        llm = _Llm(_answer("Half", fail=True))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        assert [event["type"] for event in events][-2:] == ["error", "response.done"]
        assert events[-2]["error"] == {**events[-2]["error"], "message": "engine died", "code": "server_error"}
        done = events[-1]["response"]
        assert done["status"] == "failed"
        assert _text(done["output"][0]) == "Half"

    def test_an_answer_cut_at_max_output_tokens_is_incomplete(self):
        llm = _Llm(_answer("Long", finish="length"))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "response.create", "response": {"max_output_tokens": 1}})
            done = _until(ws, "response.done")[-1]["response"]
        assert (done["status"], done["status_details"]["reason"]) == ("incomplete", "max_output_tokens")
        assert done["output"][0]["status"] == "incomplete"

    def test_audio_output_is_not_supported_yet(self):
        with self._session(_Llm()) as ws:
            _open(ws)
            ws.send_json({"type": "response.create", "response": {"output_modalities": ["audio"]}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("unsupported_value", "response.output_modalities")

    def test_an_llm_that_stopped_serving_refuses_the_response(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "response.create"})
            error = ws.receive_json()["error"]
        assert error == {**error, "code": "model_not_ready", "message": "Pipeline model 'qwen' isn't ready."}


class TestActiveResponse:
    @pytest.mark.asyncio
    async def test_a_cancel_before_its_task_starts_still_opens_and_ends_it(self):
        llm = _Llm(_answer("Hi."))
        sent: list[dict] = []

        async def send(event: dict) -> None:
            sent.append(event)

        answer = llm.respond("qwen", ResponsesRequest(model="qwen", input=[]), "s1")
        assert answer is not None
        response = RealtimeResponse(id="resp_1", output_modalities=["text"], max_output_tokens="inf")
        active = ActiveResponse(response, "qwen", answer, None, send)
        active.cancel()
        await active.task
        assert [event["type"] for event in sent] == ["response.created", "response.done"]
        assert sent[-1]["response"]["status"] == "cancelled"
        assert llm.started == 0
