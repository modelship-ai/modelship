"""`/v1/realtime` sessions in the gateway: opening on a pipeline, session.update, the conversation's items, audio
turns a fake speech-to-text model transcribes, and responses from a fake LLM that streams real Responses events."""

import asyncio
import base64
import io
import warnings
import wave

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    PromptTokenUsageInfo,
    SttCapabilities,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionUsageAudio,
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


WHISPER = SttCapabilities(streaming=False, reports_language=True)


class _Stt:
    """The speech-to-text model "whisper": answers each turn with the next of *transcripts* (an ErrorResponse fails
    it), detecting *language*, and records what it was sent."""

    def __init__(self, *transcripts, language: str = "de", delay: float = 0.0):
        self.transcripts = list(transcripts)
        self.language = language
        self.delay = delay
        self.requests: list[TranscriptionRequest] = []
        self.audio: list[bytes] = []

    async def transcribe(self, audio: bytes, request: TranscriptionRequest):
        self.requests.append(request)
        self.audio.append(audio)
        await asyncio.sleep(self.delay)
        found = self.transcripts.pop(0)
        if not isinstance(found, str):
            return found
        if request.response_format == "verbose_json":
            return TranscriptionResponseVerbose(language=self.language, duration=1.0, text=found)
        return TranscriptionResponse(text=found, usage=TranscriptionUsageAudio(seconds=1.0))


class _Calls:
    """`PipelineCalls` over a fake LLM, "qwen", and a speech-to-text model, "whisper", answering *stt*."""

    def __init__(self, llm: _Llm | None, stt: SttCapabilities | Exception | None, transcriber: _Stt | None = None):
        self.llm = llm
        self.stt = stt
        self.transcriber = transcriber
        self.asked: list[str] = []

    async def transcribe(self, model: str, audio: bytes, request: TranscriptionRequest, session_id: str):
        assert model == "whisper" and self.transcriber is not None
        return await self.transcriber.transcribe(audio, request)

    def respond(self, model: str, request: ResponsesRequest, session_id: str):
        return self.llm.respond(model, request, session_id) if self.llm is not None else None

    async def stt_capabilities(self, model: str) -> SttCapabilities | None:
        self.asked.append(model)
        if isinstance(self.stt, Exception):
            raise self.stt
        return self.stt


def _client(
    pipelines: dict,
    known=("whisper", "qwen"),
    serving=("whisper", "qwen"),
    llm: _Llm | None = None,
    stt: SttCapabilities | Exception | None = WHISPER,
    sessions: list | None = None,
    transcriber: _Stt | None = None,
) -> TestClient:
    """A gateway app serving *pipelines*; each session's `_Calls` go into *sessions*."""
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        calls = _Calls(llm, stt, transcriber)
        if sessions is not None:
            sessions.append(calls)
        await serve_session(websocket, pipelines, list(known), set(serving), calls)

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


class TestSttCapabilities:
    def test_a_session_asks_its_stt_model_once_and_keeps_the_answer(self):
        sessions: list[_Calls] = []
        llm = _Llm(_answer("Hi."))
        with _client(_route(ASSISTANT), llm=llm, sessions=sessions).websocket_connect(
            "/v1/realtime?model=assistant"
        ) as ws:
            _open(ws)
            _say(ws, "Hi")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert sessions[0].asked == ["whisper"]

    def test_a_pipeline_without_stt_asks_nothing(self):
        sessions: list[_Calls] = []
        chat = {"name": "chat", "llm": "qwen"}
        with _client(_route(chat), sessions=sessions).websocket_connect("/v1/realtime?model=chat") as ws:
            _open(ws)
        assert sessions[0].asked == []

    def test_an_stt_model_that_does_not_transcribe_refuses_the_session(self):
        error = _refused(_client(_route(ASSISTANT), stt=None), "?model=assistant")
        assert error["code"] == "model_not_supported"
        assert "uses model 'whisper' as stt, which doesn't transcribe" in error["message"]

    def test_an_stt_model_that_fails_to_answer_refuses_the_session(self):
        error = _refused(_client(_route(ASSISTANT), stt=RuntimeError("replica died")), "?model=assistant")
        assert error == {
            **error,
            "code": "model_not_ready",
            "message": "Pipeline 'assistant' couldn't ask model 'whisper' what it can do.",
        }


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
        assert "output" not in event["session"]["audio"]

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
    def test_audio_output_events_are_not_supported_yet_and_the_session_stays_open(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "output_audio_buffer.clear", "event_id": "e2"})
            error = ws.receive_json()["error"]
            ws.send_json({"type": "session.update", "session": {}})
            still_open = ws.receive_json()
        assert (error["code"], error["event_id"]) == ("unsupported_event", "e2")
        assert error["message"] == "output_audio_buffer.clear is not supported yet."
        assert still_open["type"] == "session.updated"

    @pytest.mark.parametrize(
        "event",
        [
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            },
            {"type": "response.create"},
            {"type": "response.cancel"},
        ],
    )
    def test_a_transcription_session_takes_no_conversation_or_response(self, event):
        with _client(_route(SCRIBE)).websocket_connect("/v1/realtime?model=scribe") as ws:
            _open(ws)
            ws.send_json(event)
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
            ({"type": "function_call", "name": "f"}, "invalid_value", "item.arguments"),
            ({"type": "function_call_output", "output": "on"}, "invalid_value", "item.call_id"),
            ({"type": "mcp_call", "name": "f"}, "invalid_value", "item"),
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
            ({"type": "message", "role": "tool", "content": []}, "invalid_value", "item"),
            ({"type": "image"}, "invalid_value", "item"),
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
        assert (bad["code"], bad["param"]) == ("invalid_value", "response.input[0]")

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

    @pytest.mark.parametrize(
        ("event", "param"),
        [
            ({"type": "response.create", "response": {"output_modalities": []}}, "response.output_modalities"),
            ({"type": "session.update", "session": {"output_modalities": []}}, "session.output_modalities"),
        ],
    )
    def test_no_output_modality_is_refused(self, event, param):
        with self._session(_Llm()) as ws:
            _open(ws)
            ws.send_json(event)
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", param)

    def test_an_llm_that_stopped_serving_fails_the_response(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        assert [event["type"] for event in events] == ["response.created", "error", "response.done"]
        error = events[1]["error"]
        assert error == {**error, "code": "model_not_ready", "message": "Pipeline model 'qwen' isn't ready."}
        assert events[-1]["response"]["status"] == "failed"


LIGHTS = {
    "type": "function",
    "name": "turn_on",
    "description": "Turns a room's lights on.",
    "parameters": {"type": "object", "properties": {"room": {"type": "string"}}},
}


def _tool_call(name: str, *arguments: str):
    """A script: the Responses events a loader streams for one call of *name* with *arguments* in pieces."""

    def script(request: ResponsesRequest) -> list:
        translator = ResponsesStreamTranslator(request)
        events = [*translator.start()]
        first = DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name=name, arguments=""))
        events += translator.process(_chunk(DeltaMessage(tool_calls=[first])))
        for piece in arguments:
            more = DeltaToolCall(index=0, function=DeltaFunctionCall(arguments=piece))
            events += translator.process(_chunk(DeltaMessage(tool_calls=[more])))
        events += translator.process(_chunk(finish_reason="tool_calls", usage=USAGE))
        return events + list(translator.finish())

    return script


class TestTools:
    def _session(self, llm: _Llm):
        return _client(_route(ASSISTANT), llm=llm).websocket_connect("/v1/realtime?model=assistant")

    def test_a_function_call_streams_its_arguments_and_joins_the_conversation(self):
        llm = _Llm(_tool_call("turn_on", '{"room":', ' "kitchen"}'))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"tools": [LIGHTS]}})
            ws.receive_json()
            _say(ws, "Kitchen lights on.")
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        assert [event["type"] for event in events] == [
            "response.created",
            "response.output_item.added",
            "conversation.item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        assert [event["delta"] for event in events[3:5]] == ['{"room":', ' "kitchen"}']
        assert events[5] == {**events[5], "call_id": "call_1", "arguments": '{"room": "kitchen"}'}
        (call,) = events[-1]["response"]["output"]
        assert call == {
            **call,
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "turn_on",
            "arguments": '{"room": "kitchen"}',
        }
        (request,) = llm.requests
        assert request.tools == [LIGHTS]

    def test_the_function_output_goes_back_to_the_model_with_its_call(self):
        llm = _Llm(_tool_call("turn_on", '{"room": "kitchen"}'), _answer("Done."))
        with self._session(llm) as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"tools": [LIGHTS]}})
            ws.receive_json()
            _say(ws, "Kitchen lights on.")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
            output = {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'}
            ws.send_json({"type": "conversation.item.create", "item": output})
            added = ws.receive_json()
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert added["item"] == {**added["item"], "object": "realtime.item", **output}
        assert llm.requests[1].input[1:] == [
            {"type": "function_call", "call_id": "call_1", "name": "turn_on", "arguments": '{"room": "kitchen"}'},
            output,
        ]

    def test_a_response_takes_its_own_tools_over_the_sessions(self):
        llm = _Llm(_answer("A."), _answer("B."))
        with self._session(llm) as ws:
            _open(ws)
            session = {"tools": [LIGHTS], "tool_choice": "required", "parallel_tool_calls": False}
            ws.send_json({"type": "session.update", "session": session})
            updated = ws.receive_json()["session"]
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
            named = {"type": "function", "name": "turn_on"}
            ws.send_json({"type": "response.create", "response": {"tools": [], "tool_choice": named}})
            _until(ws, "response.done")
        assert (updated["tools"], updated["tool_choice"]) == ([LIGHTS], "required")
        first, second = llm.requests
        assert (first.tools, first.tool_choice, first.parallel_tool_calls) == ([LIGHTS], "required", False)
        assert (second.tools, second.tool_choice) == (None, named)

    def test_only_function_tools_are_taken(self):
        mcp = {"type": "mcp", "server_label": "home", "server_url": "http://ha.local/mcp"}
        with self._session(_Llm()) as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"tools": [mcp]}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.tools[0].type")


def _pcm(seconds: float, rate: int = 24000) -> bytes:
    return b"\x01\x00" * int(seconds * rate)


def _append(ws, audio: bytes) -> None:
    ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()})


def _commit(ws) -> list[dict]:
    """A commit's events, up to its transcript's item.done."""
    ws.send_json({"type": "input_audio_buffer.commit"})
    return _until(ws, "conversation.item.done")


def _wav(audio: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(audio)) as wav:
        return wav.getframerate(), wav.getnframes()


class TestAudioInput:
    def _session(self, stt: _Stt, llm: _Llm | None = None, pipeline: dict = ASSISTANT, caps=WHISPER):
        client = _client(_route(pipeline), llm=llm, stt=caps, transcriber=stt)
        return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def test_a_committed_turn_is_transcribed_into_the_conversation(self):
        stt = _Stt("Hallo Welt.")
        with self._session(stt) as ws:
            session = _open(ws)
            _append(ws, _pcm(0.5))
            _append(ws, _pcm(0.5))
            events = _commit(ws)
        assert session["audio"]["input"] == {**session["audio"]["input"], "turn_detection": None}
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert [event["type"] for event in events] == [
            "input_audio_buffer.committed",
            "conversation.item.added",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.done",
        ]
        item_id = events[0]["item_id"]
        assert events[1]["item"]["content"] == [{"type": "input_audio", "transcript": None}]
        assert events[2] == {**events[2], "item_id": item_id, "content_index": 0, "transcript": "Hallo Welt."}
        assert events[2]["usage"] == {"type": "duration", "seconds": 1.0}
        assert events[3]["item"]["content"] == [{"type": "input_audio", "transcript": "Hallo Welt."}]
        assert _wav(stt.audio[0]) == (24000, 24000)

    def test_the_first_turn_detects_the_language_and_later_turns_reuse_it(self):
        stt = _Stt("Hallo.", "Wie geht's?")
        with self._session(stt) as ws:
            _open(ws)
            _append(ws, _pcm(0.2))
            _commit(ws)
            _append(ws, _pcm(0.2))
            _commit(ws)
        assert [(r.response_format, r.language) for r in stt.requests] == [("verbose_json", None), ("json", "de")]

    def test_a_model_that_does_not_report_its_language_is_never_asked(self):
        stt = _Stt("Hi.", "There.")
        with self._session(stt, caps=SttCapabilities(streaming=False, reports_language=False)) as ws:
            _open(ws)
            for _ in range(2):
                _append(ws, _pcm(0.2))
                _commit(ws)
        assert [(r.response_format, r.language) for r in stt.requests] == [("json", None), ("json", None)]

    def test_the_clients_language_and_prompt_reach_the_model(self):
        stt = _Stt("Bonjour.")
        with self._session(stt) as ws:
            _open(ws)
            transcription = {"model": "whisper-1", "language": "fr", "prompt": "Maison"}
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"transcription": transcription}}}})
            ws.receive_json()
            _append(ws, _pcm(0.2))
            _commit(ws)
        (request,) = stt.requests
        assert (request.model, request.language, request.prompt, request.response_format) == (
            "whisper",
            "fr",
            "Maison",
            "json",
        )

    def test_a_response_waits_for_the_transcript_and_answers_it(self):
        stt = _Stt("Turn on the lights.", delay=0.05)
        llm = _Llm(_answer("Done."))
        with self._session(stt, llm=llm) as ws:
            _open(ws)
            _append(ws, _pcm(0.2))
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        (request,) = llm.requests
        assert request.input == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Turn on the lights."}]}
        ]

    def test_a_failed_turn_is_reported_and_left_out_of_the_answer(self):
        stt = _Stt(create_error_response("Audio too noisy.", err_type="invalid_request_error"))
        llm = _Llm(_answer("Pardon?"))
        with self._session(stt, llm=llm) as ws:
            _open(ws)
            _say(ws, "Hi")
            _append(ws, _pcm(0.2))
            events = _commit(ws)
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        failed = events[2]
        assert failed["type"] == "conversation.item.input_audio_transcription.failed"
        assert failed["error"] == {**failed["error"], "message": "Audio too noisy.", "type": "invalid_request_error"}
        assert [_text(item) for item in llm.requests[0].input] == ["Hi"]

    def test_g711_is_decoded_to_8_khz_pcm(self):
        stt = _Stt("Hi.")
        with self._session(stt) as ws:
            _open(ws)
            ws.send_json(
                {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcmu"}}}}}
            )
            ws.receive_json()
            _append(ws, bytes(range(256)) * 8)
            _commit(ws)
        assert _wav(stt.audio[0]) == (8000, 2048)

    def test_audio_at_another_pcm_rate_keeps_it(self):
        stt = _Stt("Hi.")
        with self._session(stt) as ws:
            _open(ws)
            pcm16k = {"type": "audio/pcm", "rate": 16000}
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"format": pcm16k}}}})
            ws.receive_json()
            _append(ws, _pcm(0.5, 16000))
            _commit(ws)
        assert _wav(stt.audio[0]) == (16000, 8000)

    def test_a_commit_under_100_ms_is_refused_and_clear_empties_the_buffer(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            _append(ws, _pcm(0.05))
            ws.send_json({"type": "input_audio_buffer.commit"})
            short = ws.receive_json()["error"]
            _append(ws, _pcm(0.5))
            ws.send_json({"type": "input_audio_buffer.clear"})
            cleared = ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.commit"})
            empty = ws.receive_json()["error"]
        assert short["code"] == empty["code"] == "input_audio_buffer_commit_empty"
        assert cleared["type"] == "input_audio_buffer.cleared"

    def _g711(self, ws) -> None:
        # one byte per sample at 8 kHz keeps long turns small
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcmu"}}}}})
        ws.receive_json()

    def test_the_buffer_holds_at_most_120_seconds(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            self._g711(ws)
            _append(ws, b"\xff" * 8000 * 121)
            error = ws.receive_json()["error"]
        assert error["code"] == "input_audio_buffer_full"

    def test_the_turns_waiting_for_transcription_hold_at_most_120_seconds(self):
        stt = _Stt("One.", "Two.", delay=0.5)
        with self._session(stt) as ws:
            _open(ws)
            self._g711(ws)
            _append(ws, b"\xff" * 8000 * 70)
            ws.send_json({"type": "input_audio_buffer.commit"})
            _until(ws, "conversation.item.added")
            _append(ws, b"\xff" * 8000 * 70)
            ws.send_json({"type": "input_audio_buffer.commit"})
            error = ws.receive_json()["error"]
            _until(ws, "conversation.item.done")
            ws.send_json({"type": "input_audio_buffer.commit"})
            later = _until(ws, "conversation.item.done")
        assert error["code"] == "input_audio_transcription_busy"
        assert later[0]["type"] == "input_audio_buffer.committed"

    def test_a_pcm_rate_outside_8_to_48_khz_is_refused(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            fast = {"type": "audio/pcm", "rate": 96000}
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"format": fast}}}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.audio.input.format.rate")

    def test_the_format_cannot_change_with_audio_in_the_buffer(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            _append(ws, _pcm(0.2))
            ws.send_json(
                {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcma"}}}}}
            )
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.audio.input.format")

    def test_turn_detection_is_not_supported_yet(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            vad = {"turn_detection": {"type": "server_vad"}}
            ws.send_json({"type": "session.update", "session": {"audio": {"input": vad}}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("unsupported_value", "session.audio.input.turn_detection")

    def test_audio_inside_an_item_is_not_supported_yet(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            item = {"type": "message", "role": "user", "content": [{"type": "input_audio", "transcript": "hi"}]}
            ws.send_json({"type": "conversation.item.create", "item": item})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("unsupported_value", "item.content[0]")

    @pytest.mark.parametrize("audio", ["not base64!", "é"])
    def test_audio_that_is_not_base64_is_refused(self, audio):
        with self._session(_Stt()) as ws:
            _open(ws)
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "audio")

    def test_a_pipeline_without_stt_takes_no_audio(self):
        with self._session(_Stt(), pipeline={"name": "chat", "llm": "qwen"}) as ws:
            _open(ws)
            _append(ws, _pcm(0.2))
            error = ws.receive_json()["error"]
        assert error["code"] == "invalid_event"

    def test_a_transcription_session_transcribes_its_turns(self):
        stt = _Stt("Note one.", "Note two.")
        with self._session(stt, pipeline=SCRIBE) as ws:
            _open(ws)
            transcripts = []
            for _ in range(2):
                _append(ws, _pcm(0.2))
                transcripts.append(_commit(ws)[2]["transcript"])
        assert transcripts == ["Note one.", "Note two."]


def test_g711_tables_match_audioop():
    from modelship.openai.realtime.audio import to_pcm16

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        audioop = pytest.importorskip("audioop")
    codes = bytes(range(256))
    assert to_pcm16(codes, "audio/pcmu") == audioop.ulaw2lin(codes, 2)
    assert to_pcm16(codes, "audio/pcma") == audioop.alaw2lin(codes, 2)


class TestActiveResponse:
    @pytest.mark.asyncio
    async def test_a_cancel_before_its_task_starts_still_opens_and_ends_it(self):
        llm = _Llm(_answer("Hi."))
        sent: list[dict] = []

        async def send(event: dict) -> None:
            sent.append(event)

        async def start():
            return llm.respond("qwen", ResponsesRequest(model="qwen", input=[]), "s1")

        response = RealtimeResponse(id="resp_1", output_modalities=["text"], max_output_tokens="inf")
        active = ActiveResponse(response, "qwen", start, None, send)
        active.cancel()
        await active.task
        assert [event["type"] for event in sent] == ["response.created", "response.done"]
        assert sent[-1]["response"]["status"] == "cancelled"
        assert llm.requests == []
