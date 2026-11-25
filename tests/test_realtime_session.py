"""`/v1/realtime` sessions, through the gateway's relay to an assistant pipeline held in-process: opening on a
pipeline, session.update, the conversation's items, audio turns a fake speech-to-text model transcribes, responses
from a fake LLM that streams real Responses events, and the spoken answers a fake text-to-speech model says."""

import asyncio
import base64
import io
import time
import warnings
import wave

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from modelship.infer.pipelines.assistant.assistant_pipeline import AssistantPipeline
from modelship.infer.pipelines.assistant.response import ActiveResponse
from modelship.infer.pipelines.relay import RealtimeOpening
from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    PromptTokenUsageInfo,
    RawSpeechResponse,
    SpeechRequest,
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
from modelship.openai.realtime_relay import serve_relay
from modelship.utils.config_schema import AssistantPipelineConfig, parse_pipeline
from tests.test_realtime_turns import FakeVad

ASSISTANT = {"name": "assistant", "stt": "whisper", "llm": "qwen", "session_defaults": {"instructions": "Be brief."}}
SCRIBE = {"name": "scribe", "stt": "whisper"}
SPEAKER = {"name": "speaker", "stt": "whisper", "llm": "qwen", "tts": "kokoro"}
# ASSISTANT with the client committing each turn
MANUAL = {
    **ASSISTANT,
    "session_defaults": {"instructions": "Be brief.", "audio": {"input": {"turn_detection": None}}},
}
MODELS = ("whisper", "qwen", "kokoro")


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


TTS_RATE = 24000


def _spoken(text: str) -> bytes:
    """The fake tts model's WAV for *text*: 10 ms of a tone per character, at 24 kHz."""
    count = TTS_RATE // 100 * len(text)
    tone = (np.sin(2 * np.pi * 440 * np.arange(count) / TTS_RATE) * 8000).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TTS_RATE)
        wav.writeframes(tone.tobytes())
    return out.getvalue()


class _Tts:
    """The text-to-speech model "kokoro": speaks each text as `_spoken`, recording the requests. The first call takes
    *delay* seconds, or never ends with *hold*; the call *fail* calls in answers with an error."""

    def __init__(self, delay: float = 0.0, hold: bool = False, fail: int | None = None):
        self.delay = delay
        self.hold = hold
        self.fail = fail
        self.requests: list[SpeechRequest] = []
        self.sessions: list[str] = []
        self.cancelled = 0

    async def speak(self, request: SpeechRequest, session_id: str):
        self.requests.append(request)
        self.sessions.append(session_id)
        first = len(self.requests) == 1
        try:
            if first and self.hold:
                await asyncio.Event().wait()
            if first:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail is not None and len(self.requests) > self.fail:
            return create_error_response("The voice broke.", err_type="server_error")
        return RawSpeechResponse(audio=_spoken(request.input))


class _Calls:
    """`PipelineCalls` over a fake LLM, "qwen", a speech-to-text model, "whisper", answering *stt*, a
    text-to-speech model, "kokoro", and speech detection with *vad*; the gateway has the models *known*,
    and *serving* serve."""

    def __init__(
        self,
        llm: _Llm | None,
        stt: SttCapabilities | Exception | None,
        transcriber: _Stt | None = None,
        speaker: _Tts | None = None,
        vad: FakeVad | None = None,
        known=MODELS,
        serving=MODELS,
    ):
        self.llm = llm
        self.stt = stt
        self.transcriber = transcriber
        self.speaker = speaker
        self.vad = vad or FakeVad()
        self.known = known
        self.serving = serving
        self.asked: list[str] = []

    def model_state(self, model: str):
        if model not in self.known:
            return "unknown"
        return "serving" if model in self.serving else "starting"

    async def transcribe(self, model: str, audio: bytes, request: TranscriptionRequest, session_id: str):
        assert model == "whisper" and self.transcriber is not None
        return await self.transcriber.transcribe(audio, request)

    async def speak(self, model: str, request: SpeechRequest, session_id: str):
        if model != "kokoro" or self.speaker is None:
            return None
        return await self.speaker.speak(request, session_id)

    def respond(self, model: str, request: ResponsesRequest, session_id: str):
        return self.llm.respond(model, request, session_id) if self.llm is not None else None

    async def stt_capabilities(self, model: str) -> SttCapabilities | None:
        self.asked.append(model)
        if isinstance(self.stt, Exception):
            raise self.stt
        return self.stt


def _client(
    pipelines: dict,
    known=MODELS,
    serving=MODELS,
    llm: _Llm | None = None,
    stt: SttCapabilities | Exception | None = WHISPER,
    sessions: list | None = None,
    transcriber: _Stt | None = None,
    speaker: _Tts | None = None,
    vad: FakeVad | None = None,
) -> TestClient:
    """A gateway app relaying sessions on *pipelines* to the pipeline, held in-process; each session's `_Calls` go
    into *sessions*."""
    app = FastAPI()

    def link(name: str) -> _Link:
        calls = _Calls(llm, stt, transcriber, speaker, vad, known, serving)
        if sessions is not None:
            sessions.append(calls)
        config = parse_pipeline(pipelines[name]["config"])
        assert isinstance(config, AssistantPipelineConfig)
        return _Link(_Pipeline(config, calls))

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        await serve_relay(websocket, pipelines, link, {}, "")

    return TestClient(app)


class _Routes:
    async def ready(self) -> bool:
        return True


class _Pipeline(AssistantPipeline):
    """The assistant pipeline, calling *calls* and finding speech with their *vad*."""

    def __init__(self, config: AssistantPipelineConfig, calls: _Calls):
        super().__init__(config, _Routes())  # type: ignore[arg-type]
        self.calls = calls

    def _calls(self, opening: RealtimeOpening):
        return self.calls

    def _finder(self):
        return self.calls.vad.open


class _Link:
    """The gateway's link to *pipeline*: a direct call where Ray would be."""

    def __init__(self, pipeline: AssistantPipeline):
        self.pipeline = pipeline
        self.stream = None

    def open(self, opening: RealtimeOpening):
        self.stream = self.pipeline.realtime(opening)
        return self.stream

    async def send(self, key: str, frames: list) -> bool:
        return await self.pipeline.realtime_event(key, frames)

    async def leave(self, key: str) -> None:
        await self.pipeline.realtime_close(key)

    async def stop(self) -> None:
        if self.stream is not None:
            await self.stream.aclose()


def _route(raw: dict, ready: bool = True) -> dict:
    """The routing entry of an assistant *raw* describes flat: its name, and the rest its assistant_config."""
    own = {key: value for key, value in raw.items() if key != "name"}
    config = {"name": raw["name"], "type": "assistant", "assistant_config": own}
    return {raw["name"]: {"config": config, "app": "gw.x-p0123456789", "ready": ready}}


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
        error = _refused(_client(_route(ASSISTANT)), "")
        assert (error["code"], error["param"]) == ("missing_required_parameter", "model")

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
        assert (error["type"], error["code"]) == ("server_error", None)
        assert "'qwen', which isn't ready yet" in error["message"]

    def test_a_pipeline_whose_audio_app_is_not_ready_is_refused(self):
        error = _refused(_client(_route(ASSISTANT, ready=False)), "?model=assistant")
        assert error == {
            **error,
            "type": "server_error",
            "code": None,
            "message": "Pipeline 'assistant' isn't ready yet.",
        }


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
        assert (error["code"], error["param"]) == ("invalid_value", "model")
        assert "uses model 'whisper' as stt, which doesn't transcribe" in error["message"]

    def test_an_stt_model_that_fails_to_answer_refuses_the_session(self):
        error = _refused(_client(_route(ASSISTANT), stt=RuntimeError("replica died")), "?model=assistant")
        assert error == {
            **error,
            "type": "server_error",
            "code": None,
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
            ws.send_json({"type": "session.update", "session": {"tracing": "auto", "audio": {"noise": "near"}}})
            event = ws.receive_json()
        assert event["type"] == "session.updated"
        assert "tracing" not in event["session"]
        assert "noise" not in event["session"]["audio"]

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

    def test_audio_output_needs_a_tts_model(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"output_modalities": ["audio"]}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.output_modalities")
        assert error["message"] == "Pipeline 'assistant' has no tts model to speak with; ask for text output."

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
    def test_an_event_type_it_does_not_take_is_an_invalid_value_and_the_session_stays_open(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_json({"type": "output_audio_buffer.clear", "event_id": "e2"})
            error = ws.receive_json()["error"]
            ws.send_json({"type": "session.update", "session": {}})
            still_open = ws.receive_json()
        assert (error["type"], error["code"], error["param"], error["event_id"]) == (
            "invalid_request_error",
            "invalid_value",
            "type",
            "e2",
        )
        assert error["message"].startswith(
            "Invalid value: 'output_audio_buffer.clear'. Supported values are: 'session.update', "
        )
        assert "'input_audio_buffer.append'" in error["message"]
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
            {"type": "conversation.item.truncate", "item_id": "item_1", "content_index": 0, "audio_end_ms": 0},
        ],
    )
    def test_a_transcription_session_takes_no_conversation_or_response(self, event):
        with _client(_route(SCRIBE)).websocket_connect("/v1/realtime?model=scribe") as ws:
            _open(ws)
            ws.send_json(event)
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "type")
        assert f"'{event['type']}'" not in error["message"].split("Supported values are:")[1]

    @pytest.mark.parametrize(
        ("frame", "code", "param"),
        [
            ("not json", None, None),
            ("[1, 2]", None, None),
            ('{"no": "type"}', "missing_required_parameter", "type"),
            ('{"type": "input_audio_buffer.append"}', "missing_required_parameter", "audio"),
        ],
    )
    def test_a_malformed_event_is_an_error(self, frame, code, param):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            _open(ws)
            ws.send_text(frame)
            error = ws.receive_json()["error"]
        assert (error["type"], error["code"], error["param"]) == ("invalid_request_error", code, param)

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
            ({"type": "function_call", "name": "f"}, "missing_required_parameter", "item.arguments"),
            ({"type": "function_call_output", "output": "on"}, "missing_required_parameter", "item.call_id"),
            ({"type": "mcp_call", "name": "f"}, "invalid_value", "item"),
            (
                {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": ""}]},
                "invalid_value",
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

    def test_audio_output_needs_a_tts_model(self):
        with self._session(_Llm()) as ws:
            _open(ws)
            ws.send_json({"type": "response.create", "response": {"output_modalities": ["audio"]}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "response.output_modalities")

    @pytest.mark.parametrize(
        ("event", "param"),
        [
            ({"type": "response.create", "response": {"output_modalities": []}}, "response.output_modalities"),
            ({"type": "session.update", "session": {"output_modalities": []}}, "session.output_modalities"),
            (
                {"type": "response.create", "response": {"output_modalities": ["text", "audio"]}},
                "response.output_modalities",
            ),
            (
                {"type": "session.update", "session": {"output_modalities": ["audio", "text"]}},
                "session.output_modalities",
            ),
        ],
    )
    def test_anything_but_one_output_modality_is_refused(self, event, param):
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
        assert error == {
            **error,
            "type": "server_error",
            "code": None,
            "message": "Pipeline model 'qwen' isn't ready.",
        }
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
    def _session(self, stt: _Stt, llm: _Llm | None = None, pipeline: dict = MANUAL, caps=WHISPER):
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

    def test_the_buffer_refuses_audio_past_its_cap(self, monkeypatch):
        monkeypatch.setattr("modelship.infer.pipelines.assistant.audio.MAX_SECONDS", 2.0)
        with self._session(_Stt()) as ws:
            _open(ws)
            self._g711(ws)
            _append(ws, b"\xff" * 8000 * 3)
            error = ws.receive_json()["error"]
        assert (error["type"], error["code"], error["param"]) == ("invalid_request_error", None, "audio")

    def test_a_commit_is_taken_while_turns_wait_for_transcription(self):
        stt = _Stt("One.", "Two.", delay=0.5)
        with self._session(stt) as ws:
            _open(ws)
            self._g711(ws)
            _append(ws, b"\xff" * 8000)
            ws.send_json({"type": "input_audio_buffer.commit"})
            _until(ws, "conversation.item.added")
            _append(ws, b"\xff" * 8000)
            ws.send_json({"type": "input_audio_buffer.commit"})
            second = ws.receive_json()
            done = [_until(ws, "conversation.item.done")[-1] for _ in range(2)]
        assert second["type"] == "input_audio_buffer.committed"
        assert [event["item"]["content"][0]["transcript"] for event in done] == ["One.", "Two."]

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

    @pytest.mark.parametrize(
        ("turn_detection", "code", "param"),
        [
            ({"type": "semantic_vad"}, "invalid_value", "session.audio.input.turn_detection"),
            (
                {"type": "server_vad", "idle_timeout_ms": 5000},
                "invalid_value",
                "session.audio.input.turn_detection.idle_timeout_ms",
            ),
            ({"type": "server_vad", "threshold": 2}, "invalid_value", "session.audio.input.turn_detection.threshold"),
        ],
    )
    def test_turn_detection_it_cannot_do_is_refused(self, turn_detection, code, param):
        with self._session(_Stt()) as ws:
            _open(ws)
            audio = {"input": {"turn_detection": turn_detection}}
            ws.send_json({"type": "session.update", "session": {"audio": audio}})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == (code, param)

    def test_server_vad_needs_an_stt_model(self):
        with self._session(_Stt(), pipeline={"name": "chat", "llm": "qwen"}) as ws:
            session = _open(ws)
            audio = {"input": {"turn_detection": {"type": "server_vad"}}}
            ws.send_json({"type": "session.update", "session": {"audio": audio}})
            error = ws.receive_json()["error"]
        assert session["audio"]["input"]["turn_detection"] is None
        assert (error["code"], error["param"]) == ("invalid_value", "session.audio.input.turn_detection")

    def test_audio_inside_an_item_is_not_supported_yet(self):
        with self._session(_Stt()) as ws:
            _open(ws)
            item = {"type": "message", "role": "user", "content": [{"type": "input_audio", "transcript": "hi"}]}
            ws.send_json({"type": "conversation.item.create", "item": item})
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "item.content[0]")

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
            ws.send_json({"type": "input_audio_buffer.commit"})
            # malformed too: the type is still what it reports
            ws.send_json({"type": "input_audio_buffer.append", "event_id": "e3"})
            append = ws.receive_json()["error"]
            commit = ws.receive_json()["error"]
            malformed = ws.receive_json()["error"]
            ws.send_json({"type": "input_audio_buffer.clear"})
            cleared = ws.receive_json()
        assert (append["code"], append["param"]) == (commit["code"], commit["param"]) == ("invalid_value", "type")
        assert (malformed["code"], malformed["param"], malformed["event_id"]) == ("invalid_value", "type", "e3")
        assert append["message"].startswith("Invalid value: 'input_audio_buffer.append'.")
        assert "'input_audio_buffer.append'" not in append["message"].split("Supported values are:")[1]
        assert cleared["type"] == "input_audio_buffer.cleared"

    def test_a_transcription_session_transcribes_its_turns(self):
        stt = _Stt("Note one.", "Note two.")
        with self._session(stt, pipeline=SCRIBE) as ws:
            _open(ws)
            transcripts = []
            for _ in range(2):
                _append(ws, _pcm(0.2))
                transcripts.append(_commit(ws)[2]["transcript"])
        assert transcripts == ["Note one.", "Note two."]


KITCHEN = ("The kitchen lights are on now. ", "The oven is preheating as well.")


def _audio(events: list[dict]) -> bytes:
    return b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")


def _types(events: list[dict]) -> list[str]:
    """The event types, each run of deltas of one type as one."""
    found: list[str] = []
    for event in events:
        if not (found and found[-1] == event["type"] and event["type"].endswith(".delta")):
            found.append(event["type"])
    return found


class TestAudioOutput:
    def _session(self, llm: _Llm, tts: _Tts, pipeline: dict = SPEAKER):
        client = _client(_route(pipeline), llm=llm, speaker=tts)
        return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def _answer(self, ws, response: dict | None = None) -> list[dict]:
        _say(ws, "Is dinner ready?")
        ws.send_json({"type": "response.create", **({"response": response} if response else {})})
        return _until(ws, "response.done")

    def test_a_pipeline_with_tts_speaks_by_default(self):
        with self._session(_Llm(), _Tts()) as ws:
            session = _open(ws)
        assert session["output_modalities"] == ["audio"]
        assert session["audio"]["output"] == {
            "format": {"type": "audio/pcm", "rate": 24000},
            "voice": None,
            "speed": 1.0,
        }

    def test_a_pipeline_without_tts_answers_in_text_and_a_transcription_session_has_no_output(self):
        with self._session(_Llm(), _Tts(), pipeline=ASSISTANT) as ws:
            assistant = _open(ws)
        with self._session(_Llm(), _Tts(), pipeline=SCRIBE) as ws:
            scribe = _open(ws)
        assert assistant["output_modalities"] == ["text"]
        assert scribe["audio"]["output"] is None

    def test_the_session_defaults_choose_text_on_a_pipeline_with_tts(self):
        quiet = {**SPEAKER, "session_defaults": {"output_modalities": ["text"]}}
        with self._session(_Llm(), _Tts(), pipeline=quiet) as ws:
            assert _open(ws)["output_modalities"] == ["text"]

    def test_the_answer_streams_as_its_transcript_and_its_audio(self):
        llm, tts = _Llm(_answer(*KITCHEN)), _Tts()
        with self._session(llm, tts) as ws:
            session = _open(ws)
            events = self._answer(ws)
        types = [event["type"] for event in events]
        # the audio goes out alongside the transcript, as each piece is spoken
        assert [kind for kind in _types(events) if kind != "response.output_audio.delta"] == [
            "response.created",
            "response.output_item.added",
            "conversation.item.added",
            "response.content_part.added",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        deltas = [i for i, kind in enumerate(types) if kind == "response.output_audio.delta"]
        assert types.index("response.content_part.added") < deltas[0]
        assert deltas[-1] < types.index("response.output_audio.done")
        text = "".join(KITCHEN)
        transcript = "".join(e["delta"] for e in events if e["type"] == "response.output_audio_transcript.delta")
        assert transcript == text
        assert [request.input for request in tts.requests] == [KITCHEN[0].strip(), KITCHEN[1]]
        assert [(r.voice, r.speed, r.response_format) for r in tts.requests] == [(None, 1.0, "wav")] * 2
        assert tts.sessions == [session["id"]] * 2
        # 10 ms per character, 16-bit at 24 kHz
        assert len(_audio(events)) == sum(len(r.input) for r in tts.requests) * 240 * 2
        # half a second per delta at most
        sizes = [len(base64.b64decode(e["delta"])) for e in events if e["type"] == "response.output_audio.delta"]
        assert max(sizes) <= 24000
        assert not [event for event in events if "output_text" in event["type"]]
        added = next(e for e in events if e["type"] == "response.content_part.added")
        assert added["part"] == {"type": "audio", "transcript": ""}
        done = events[-1]["response"]
        assert done["status"] == "completed"
        assert done["output_modalities"] == ["audio"]
        assert done["audio"] == {"output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": None}}
        assert done["output"][0]["content"] == [{"type": "output_audio", "transcript": text}]
        part_done = next(e for e in events if e["type"] == "response.content_part.done")
        assert part_done["part"] == {"type": "audio", "transcript": text}

    def test_the_first_piece_is_spoken_alone_and_the_waiting_sentences_in_one_call(self):
        words = ("Sure, I can help with that. ", "The kitchen lights are on. ", "The oven is hot. Dinner is at seven.")
        tts = _Tts(delay=0.3)
        with self._session(_Llm(_answer(*words)), tts) as ws:
            _open(ws)
            self._answer(ws)
        assert [request.input for request in tts.requests] == [
            "Sure, I can help with that.",
            "The kitchen lights are on. The oven is hot. Dinner is at seven.",
        ]

    def test_markdown_is_spoken_without_its_symbols_and_kept_in_the_transcript(self):
        tts = _Tts()
        with self._session(_Llm(_answer("**Sure**, the `kitchen` lights are on.")), tts) as ws:
            _open(ws)
            events = self._answer(ws)
        assert [request.input for request in tts.requests] == ["Sure, the kitchen lights are on."]
        assert (
            events[-1]["response"]["output"][0]["content"][0]["transcript"] == "**Sure**, the `kitchen` lights are on."
        )

    def test_the_next_response_sees_the_spoken_answer_as_text(self):
        llm = _Llm(_answer(*KITCHEN), _answer("Yes."))
        with self._session(llm, _Tts()) as ws:
            _open(ws)
            self._answer(ws)
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert llm.requests[1].input[-1] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(KITCHEN)}],
        }

    def test_the_sessions_voice_speed_and_format_reach_the_model_and_the_client(self):
        tts = _Tts()
        with self._session(_Llm(_answer(*KITCHEN)), tts) as ws:
            _open(ws)
            output = {"voice": {"id": "af_bella"}, "speed": 1.25, "format": {"type": "audio/pcmu"}}
            ws.send_json({"type": "session.update", "session": {"audio": {"output": output}}})
            updated = ws.receive_json()["session"]["audio"]["output"]
            events = self._answer(ws)
        assert updated == {"format": {"type": "audio/pcmu"}, "voice": {"id": "af_bella"}, "speed": 1.25}
        assert [(r.voice, r.speed) for r in tts.requests] == [("af_bella", 1.25)] * 2
        # G.711: one byte a sample at 8 kHz
        assert len(_audio(events)) == sum(len(r.input) for r in tts.requests) * 80
        assert events[-1]["response"]["audio"]["output"]["format"] == {"type": "audio/pcmu"}

    def test_a_response_takes_its_own_voice_and_format_over_the_sessions(self):
        tts = _Tts()
        with self._session(_Llm(_answer(*KITCHEN)), tts) as ws:
            _open(ws)
            output = {"voice": "marin", "format": {"type": "audio/pcm", "rate": 16000}}
            events = self._answer(ws, {"audio": {"output": output}})
        assert [request.voice for request in tts.requests] == ["marin"] * 2
        assert len(_audio(events)) == sum(len(r.input) for r in tts.requests) * 160 * 2
        assert events[-1]["response"]["audio"]["output"] == output

    def test_a_text_response_on_a_pipeline_with_tts_stays_text(self):
        tts = _Tts()
        with self._session(_Llm(_answer(*KITCHEN)), tts) as ws:
            _open(ws)
            events = self._answer(ws, {"output_modalities": ["text"]})
        assert tts.requests == []
        assert "response.output_text.delta" in _types(events)
        assert not _audio(events)

    @pytest.mark.parametrize(
        "output",
        [{"speed": 2.0}, {"speed": 0.1}, {"voice": 5}, {"format": {"type": "audio/pcm", "rate": 96000}}],
    )
    def test_an_output_setting_out_of_range_is_refused(self, output):
        with self._session(_Llm(), _Tts()) as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"audio": {"output": output}}})
            error = ws.receive_json()["error"]
        assert error["code"] == "invalid_value"
        assert error["param"].startswith("session.audio.output.")

    def test_a_failed_piece_fails_the_response_and_keeps_its_transcript(self):
        llm, tts = _Llm(_answer(*KITCHEN)), _Tts(fail=0)
        with self._session(llm, tts) as ws:
            _open(ws)
            events = self._answer(ws)
        types = _types(events)
        assert types[-6:] == [
            "error",
            "response.output_audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        assert "response.output_audio.delta" not in types
        assert next(event for event in events if event["type"] == "error")["error"]["message"] == "The voice broke."
        done = events[-1]["response"]
        assert (done["status"], done["status_details"]["error"]["type"]) == ("failed", "server_error")
        assert done["output"][0]["content"][0]["transcript"]
        assert done["output"][0]["status"] == "incomplete"
        assert len(tts.requests) == 1

    def test_a_tts_model_that_stopped_serving_fails_the_response(self):
        with _client(_route(SPEAKER), llm=_Llm(_answer(*KITCHEN))).websocket_connect(
            "/v1/realtime?model=speaker"
        ) as ws:
            _open(ws)
            events = self._answer(ws)
        error = next(event for event in events if event["type"] == "error")["error"]
        assert error["message"] == "Pipeline model 'kokoro' didn't speak the answer."
        assert events[-1]["response"]["status"] == "failed"

    def _speaking(self, tts: _Tts) -> None:
        """Waits for the first call to *tts*."""
        deadline = time.monotonic() + 5
        while not tts.requests:
            assert time.monotonic() < deadline
            time.sleep(0.01)

    def test_cancelling_stops_the_answer_and_its_speech(self):
        llm = _Llm(_answer("The kitchen lights are on now. And", finish=None), _answer("Ok."))
        tts = _Tts(hold=True)
        with self._session(llm, tts) as ws:
            _open(ws)
            _say(ws, "Is dinner ready?")
            ws.send_json({"type": "response.create"})
            self._speaking(tts)
            ws.send_json({"type": "response.cancel"})
            events = _until(ws, "response.done")
        assert _types(events)[-6:] == [
            "response.output_audio_transcript.done",
            "response.output_audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        assert events[-1]["response"]["status"] == "cancelled"
        assert tts.cancelled == 1
        assert llm.closed_early == [True]
        transcript = events[-1]["response"]["output"][0]["content"][0]["transcript"]
        assert transcript == "The kitchen lights are on now. And"

    def test_leaving_mid_answer_stops_its_speech(self):
        llm = _Llm(_answer("The kitchen lights are on now. And", finish=None))
        tts = _Tts(hold=True)
        with self._session(llm, tts) as ws:
            _open(ws)
            _say(ws, "Is dinner ready?")
            ws.send_json({"type": "response.create"})
            self._speaking(tts)
        assert tts.cancelled == 1
        assert llm.closed_early == [True]


def _tone(seconds: float, rate: int = 24000) -> bytes:
    count = int(seconds * rate)
    return (np.sin(2 * np.pi * 440 * np.arange(count) / rate) * 8000).astype("<i2").tobytes()


def _quiet(seconds: float, rate: int = 24000) -> bytes:
    return bytes(int(seconds * rate) * 2)


def _stream(ws, *parts: bytes, step: float = 0.1, rate: int = 24000) -> None:
    """Appends *parts* in *step* seconds of audio each, as a client streaming its microphone does."""
    audio = b"".join(parts)
    size = int(step * rate) * 2
    for start in range(0, len(audio), size):
        _append(ws, audio[start : start + size])


SERVER_VAD = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "create_response": True,
    "interrupt_response": True,
    "idle_timeout_ms": None,
}


class TestServerVad:
    def _session(self, stt: _Stt, llm: _Llm | None = None, pipeline: dict = ASSISTANT, vad: FakeVad | None = None):
        client = _client(_route(pipeline), llm=llm, transcriber=stt, vad=vad)
        return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def _settle(self, ws, llm: _Llm | None = None) -> None:
        """Waits until the session has handled everything sent so far."""
        ws.send_json({"type": "session.update", "session": {}})
        _until(ws, "session.updated")
        time.sleep(0.1)

    def test_a_session_with_stt_detects_turns_by_default(self):
        with self._session(_Stt()) as ws:
            assistant = _open(ws)
        with self._session(_Stt(), pipeline=SCRIBE) as ws:
            scribe = _open(ws)
        with self._session(_Stt(), pipeline={"name": "chat", "llm": "qwen"}) as ws:
            chat = _open(ws)
        assert assistant["audio"]["input"]["turn_detection"] == SERVER_VAD
        assert scribe["audio"]["input"]["turn_detection"] == SERVER_VAD
        assert chat["audio"]["input"]["turn_detection"] is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (None, None),
            (
                {"type": "server_vad", "silence_duration_ms": 800, "create_response": False},
                {**SERVER_VAD, "silence_duration_ms": 800, "create_response": False},
            ),
        ],
    )
    def test_the_pipelines_defaults_set_the_turn_detection(self, given, expected):
        pipeline = {**ASSISTANT, "session_defaults": {"audio": {"input": {"turn_detection": given}}}}
        with self._session(_Stt(), pipeline=pipeline) as ws:
            session = _open(ws)
        assert session["audio"]["input"]["turn_detection"] == expected

    def test_a_turn_is_found_committed_transcribed_and_answered(self):
        stt, llm = _Stt("Lights on."), _Llm(_answer("Done."))
        with self._session(stt, llm) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            events = _until(ws, "response.done")
        kinds = [event["type"] for event in events]
        assert kinds[:7] == [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.added",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.done",
            "response.created",
        ]
        started, stopped, committed = events[:3]
        assert (started["audio_start_ms"], stopped["audio_end_ms"]) == (700, 2500)
        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        assert _wav(stt.audio[0]) == (24000, round(1.8 * 24000))
        assert events[-1]["response"]["status"] == "completed"
        assert llm.requests[0].input[-1]["content"] == [{"type": "input_text", "text": "Lights on."}]

    def test_turns_streamed_faster_than_real_time_are_each_found(self):
        stt = _Stt("One.", "Two.")
        with self._session(stt) as ws:
            _open(ws)
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"turn_detection": {**SERVER_VAD, "create_response": False}}}},
                }
            )
            _until(ws, "session.updated")
            _stream(ws, _quiet(1), _tone(1), _quiet(1), _tone(1), _quiet(1), step=5)
            events = _until(ws, "conversation.item.done") + _until(ws, "conversation.item.done")
        stops = [event["audio_end_ms"] for event in events if event["type"] == "input_audio_buffer.speech_stopped"]
        assert stops == [2500, 4500]
        transcripts = [e["transcript"] for e in events if e["type"].endswith("transcription.completed")]
        assert transcripts == ["One.", "Two."]

    def test_a_turn_that_says_nothing_gets_no_answer(self):
        llm = _Llm(_answer("Hm?"))
        with self._session(_Stt(""), llm) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            _until(ws, "conversation.item.done")
            self._settle(ws)
        assert llm.requests == []

    def test_without_interrupt_response_a_turn_ending_while_the_model_answers_gets_no_answer_of_its_own(self):
        llm = _Llm(_answer("Let me think", finish=None))
        stt = _Stt("Lights on.", "Thanks.")
        detection = {"type": "server_vad", "interrupt_response": False}
        pipeline = {**ASSISTANT, "session_defaults": {"audio": {"input": {"turn_detection": detection}}}}
        with self._session(stt, llm, pipeline=pipeline) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            _until(ws, "response.output_text.delta")
            _stream(ws, _tone(1), _quiet(1))
            events = _until(ws, "conversation.item.done")
            self._settle(ws)
            assert "conversation.item.input_audio_transcription.completed" in [e["type"] for e in events]
            ws.send_json({"type": "response.cancel"})
            _until(ws, "response.done")
        assert len(llm.requests) == 1

    def test_without_create_response_turns_are_committed_but_not_answered(self):
        llm = _Llm(_answer("Done."))
        pipeline = {
            **ASSISTANT,
            "session_defaults": {
                "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}}
            },
        }
        with self._session(_Stt("Lights on."), llm, pipeline=pipeline) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            _until(ws, "conversation.item.done")
            self._settle(ws)
        assert llm.requests == []

    def test_a_commit_during_a_turn_ends_it(self):
        stt = _Stt("Lights")
        with self._session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1))
            started = _until(ws, "input_audio_buffer.speech_started")[-1]
            ws.send_json({"type": "input_audio_buffer.commit"})
            stopped, committed = _until(ws, "input_audio_buffer.committed")[-2:]
        assert stopped["type"] == "input_audio_buffer.speech_stopped"
        assert stopped["item_id"] == committed["item_id"] == started["item_id"]

    def test_clearing_drops_the_turn_in_progress(self):
        stt = _Stt()
        with self._session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1))
            _until(ws, "input_audio_buffer.speech_started")
            ws.send_json({"type": "input_audio_buffer.clear"})
            _until(ws, "input_audio_buffer.cleared")
            _stream(ws, _quiet(1))
            self._settle(ws)
        assert stt.requests == []

    def test_turning_detection_off_leaves_the_turns_to_the_client(self):
        stt = _Stt()
        with self._session(stt) as ws:
            _open(ws)
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": None}}}})
            updated = _until(ws, "session.updated")[-1]
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            self._settle(ws)
        assert updated["session"]["audio"]["input"]["turn_detection"] is None
        assert stt.requests == []

    def test_speech_detection_that_fails_is_reported_once_and_a_commit_still_works(self):
        vad = FakeVad()
        vad.failing = True
        stt = _Stt("Lights on.")
        with self._session(stt, vad=vad) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            error = ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.commit"})
            events = _until(ws, "conversation.item.done")
        assert (error["error"]["type"], error["error"]["code"]) == ("server_error", "server_error")
        assert not [event for event in events if event["type"] == "error"]
        assert [e["transcript"] for e in events if e["type"].endswith("transcription.completed")] == ["Lights on."]

    def test_a_transcription_session_transcribes_its_turns_and_answers_none(self):
        with self._session(_Stt("Note one."), pipeline=SCRIBE) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            events = _until(ws, "conversation.item.done")
            self._settle(ws)
        assert [e["transcript"] for e in events if e["type"].endswith("transcription.completed")] == ["Note one."]

    def test_a_turn_found_while_another_is_transcribed_is_committed(self):
        stt = _Stt("One.", "Two.", delay=0.5)
        pipeline = {**SCRIBE, "session_defaults": {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}}
        with self._session(stt, pipeline=pipeline) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1), _tone(1), _quiet(1))
            events = _until(ws, "conversation.item.done") + _until(ws, "conversation.item.done")
        assert not [event for event in events if event["type"] == "error"]
        transcripts = [e["transcript"] for e in events if e["type"].endswith("transcription.completed")]
        assert transcripts == ["One.", "Two."]
        stops = [event["audio_end_ms"] for event in events if event["type"] == "input_audio_buffer.speech_stopped"]
        assert stops == [2500, 4500]
        assert _wav(stt.audio[1]) == (24000, round(1.8 * 24000))


# 620 characters: 6.2 s of the fake tts model's audio
LONG_ANSWER = "The kitchen lights are on now. " * 20
HALF = {**SPEAKER, "duplex": "half"}


def _messages(request: ResponsesRequest) -> list[str]:
    return [part["text"] for message in request.input for part in message["content"]]


class TestBargeIn:
    def _session(self, stt: _Stt, llm: _Llm, pipeline: dict = ASSISTANT, tts: _Tts | None = None):
        client = _client(_route(pipeline), llm=llm, transcriber=stt, speaker=tts)
        return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def _settle(self, ws) -> list[dict]:
        """The events the session sends once it has handled everything sent so far."""
        time.sleep(0.1)
        ws.send_json({"type": "session.update", "session": {}})
        return _until(ws, "session.updated")

    def _spoken_answer(self, ws) -> None:
        _say(ws, "Read me the news.")
        ws.send_json({"type": "response.create"})
        _until(ws, "response.done")

    @pytest.mark.parametrize("second", ["In the kitchen.", ""], ids=["says-something", "says-nothing"])
    def test_talking_over_the_answer_cancels_it_and_the_new_turn_is_answered(self, second):
        llm = _Llm(_answer("Let me think", finish=None), _answer("The kitchen it is."))
        stt = _Stt("Lights on.", second)
        with self._session(stt, llm) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            _until(ws, "response.output_text.delta")
            _stream(ws, _tone(1), _quiet(1))
            cut = _until(ws, "response.done")
            answered = _until(ws, "response.done")
        kinds = [event["type"] for event in cut]
        assert kinds.index("input_audio_buffer.speech_started") < kinds.index("response.done")
        done = cut[-1]["response"]
        assert (done["status"], done["status_details"]["reason"]) == ("cancelled", "turn_detected")
        assert answered[-1]["response"]["status"] == "completed"
        assert llm.closed_early == [True, False]
        # a turn that says nothing is left out, and the answer it cut off is given again
        assert _messages(llm.requests[1]) == ["Lights on.", "Let me think", *([second] if second else [])]

    def test_an_answer_out_of_the_conversation_goes_on(self):
        llm = _Llm(_answer("Thinking", finish=None))
        with self._session(_Stt("Lights on."), llm) as ws:
            _open(ws)
            ws.send_json({"type": "response.create", "response": {"conversation": "none"}})
            _until(ws, "response.output_text.delta")
            _stream(ws, _quiet(1), _tone(1), _quiet(1))
            heard = _until(ws, "conversation.item.done") + self._settle(ws)
            ws.send_json({"type": "response.cancel"})
            done = _until(ws, "response.done")[-1]["response"]
        assert "response.done" not in [event["type"] for event in heard]
        assert done["status_details"]["reason"] == "client_cancelled"
        assert len(llm.requests) == 1

    @pytest.mark.parametrize("second", ["and the oven too.", ""], ids=["goes-on", "says-nothing"])
    def test_a_turn_the_user_talks_on_after_is_answered_with_the_next(self, second):
        llm = _Llm(_answer("Both are on."))
        stt = _Stt("Lights on", second, delay=0.2)
        with self._session(stt, llm) as ws:
            _open(ws)
            # the pause ends the first turn, and the user talks again before it is transcribed
            _stream(ws, _quiet(1), _tone(1), _quiet(0.6), _tone(1))
            _until(ws, "conversation.item.input_audio_transcription.completed")
            self._settle(ws)
            unanswered = list(llm.requests)
            _stream(ws, _quiet(1))
            _until(ws, "response.done")
        assert unanswered == []
        assert len(llm.requests) == 1
        assert _messages(llm.requests[0]) == (["Lights on", second] if second else ["Lights on"])

    def test_while_an_answer_plays_a_short_sound_starts_no_turn(self):
        llm = _Llm(_answer(LONG_ANSWER), _answer("Ok."))
        stt = _Stt("Wait.")
        with self._session(stt, llm, pipeline=SPEAKER, tts=_Tts()) as ws:
            _open(ws)
            self._spoken_answer(ws)
            # a cough, then speech that lasts
            _stream(ws, _quiet(0.5), _tone(0.3), _quiet(1))
            coughed = self._settle(ws)
            _stream(ws, _tone(1), _quiet(1))
            spoke = _until(ws, "response.done")
        assert "input_audio_buffer.speech_started" not in [event["type"] for event in coughed]
        (started,) = [event for event in spoke if event["type"] == "input_audio_buffer.speech_started"]
        # after the cough, with its padding
        assert started["audio_start_ms"] == 1500
        assert len(stt.requests) == 1

    def test_in_half_duplex_what_it_hears_while_the_answer_plays_is_dropped(self):
        stt = _Stt()
        with self._session(stt, _Llm(_answer(LONG_ANSWER)), pipeline=HALF, tts=_Tts()) as ws:
            _open(ws)
            self._spoken_answer(ws)
            _stream(ws, _tone(1), _quiet(1))
            events = self._settle(ws)
        assert "input_audio_buffer.speech_started" not in [event["type"] for event in events]
        assert stt.requests == []

    def test_in_half_duplex_a_commit_while_the_answer_plays_holds_silence(self):
        manual = {**HALF, "session_defaults": {"audio": {"input": {"turn_detection": None}}}}
        stt = _Stt("")
        with self._session(stt, _Llm(_answer(LONG_ANSWER)), pipeline=manual, tts=_Tts()) as ws:
            _open(ws)
            self._spoken_answer(ws)
            _append(ws, _tone(1))
            _commit(ws)
        with wave.open(io.BytesIO(stt.audio[0])) as wav:
            assert wav.readframes(wav.getnframes()) == _quiet(1)

    def test_in_half_duplex_the_user_is_heard_once_the_answer_is_cancelled(self):
        llm = _Llm(_answer(LONG_ANSWER, finish=None), _answer("Ok."))
        stt = _Stt("Stop.")
        with self._session(stt, llm, pipeline=HALF, tts=_Tts()) as ws:
            _open(ws)
            _say(ws, "Read me the news.")
            ws.send_json({"type": "response.create"})
            _until(ws, "response.output_audio.delta")
            ws.send_json({"type": "response.cancel"})
            _until(ws, "response.done")
            _stream(ws, _quiet(0.5), _tone(1), _quiet(1))
            events = _until(ws, "conversation.item.done")
        assert "input_audio_buffer.speech_started" in [event["type"] for event in events]
        assert [e["transcript"] for e in events if e["type"].endswith("transcription.completed")] == ["Stop."]


TRAIN = "The train leaves at seven forty. You don't need to book a seat, but it's cheaper if you do."


def _truncate(item_id: str, audio_end_ms: int, content_index: int = 0) -> dict:
    return {
        "type": "conversation.item.truncate",
        "item_id": item_id,
        "content_index": content_index,
        "audio_end_ms": audio_end_ms,
    }


class TestTruncate:
    def _session(self, llm: _Llm, pipeline: dict = SPEAKER):
        client = _client(_route(pipeline), llm=llm, speaker=_Tts())
        return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")

    def _spoken_answer(self, ws) -> str:
        """The item id of a spoken answer of TRAIN: 320 ms for its first sentence, 580 ms for its second."""
        _say(ws, "When is the train?")
        ws.send_json({"type": "response.create"})
        return _until(ws, "response.done")[-1]["response"]["output"][0]["id"]

    @pytest.mark.parametrize(
        ("audio_end_ms", "heard"),
        [
            (0, ""),
            (320, "The train leaves at seven forty."),
            # half of the second sentence's audio: half its words, rounded down
            (610, "The train leaves at seven forty. You don't need to book a"),
            (900, TRAIN),
        ],
    )
    def test_truncating_keeps_the_words_heard(self, audio_end_ms, heard):
        llm = _Llm(_answer(TRAIN), _answer("Ok."))
        with self._session(llm) as ws:
            _open(ws)
            item_id = self._spoken_answer(ws)
            ws.send_json(_truncate(item_id, audio_end_ms))
            truncated = ws.receive_json()
            ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
            retrieved = ws.receive_json()["item"]
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        assert {key: value for key, value in truncated.items() if key != "event_id"} == {
            "type": "conversation.item.truncated",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms,
        }
        assert retrieved["content"] == [{"type": "output_audio", "transcript": heard}]
        expected = [{"type": "output_text", "text": heard}] if heard else None
        answer = [item for item in llm.requests[1].input if item.get("role") == "assistant"]
        assert (answer[0]["content"] if answer else None) == expected

    def test_a_second_truncate_measures_against_the_first(self):
        with self._session(_Llm(_answer(TRAIN))) as ws:
            _open(ws)
            item_id = self._spoken_answer(ws)
            ws.send_json(_truncate(item_id, 610))
            ws.receive_json()
            ws.send_json(_truncate(item_id, 700))
            error = ws.receive_json()["error"]
            ws.send_json(_truncate(item_id, 320))
            truncated = ws.receive_json()
            ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
            retrieved = ws.receive_json()["item"]
        assert (error["code"], error["param"]) == ("invalid_value", "audio_end_ms")
        assert "at 610 ms" in error["message"]
        assert truncated["type"] == "conversation.item.truncated"
        assert retrieved["content"][0]["transcript"] == "The train leaves at seven forty."

    @pytest.mark.parametrize(
        ("target", "content_index", "audio_end_ms", "code", "param"),
        [
            ("missing", 0, 0, "item_not_found", "item_id"),
            ("user", 0, 0, "invalid_value", "item_id"),
            ("answer", 1, 0, "invalid_value", "content_index"),
            ("answer", 0, 901, "invalid_value", "audio_end_ms"),
            ("answer", 0, -1, "invalid_value", "audio_end_ms"),
        ],
    )
    def test_a_truncate_it_cannot_do_is_refused(self, target, content_index, audio_end_ms, code, param):
        with self._session(_Llm(_answer(TRAIN))) as ws:
            _open(ws)
            user = _say(ws, "Hello.")["item"]["id"]
            answer = self._spoken_answer(ws)
            item_id = {"missing": "item_missing", "user": user, "answer": answer}[target]
            ws.send_json(_truncate(item_id, audio_end_ms, content_index))
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == (code, param)

    def test_a_text_answer_holds_no_audio_to_truncate(self):
        with self._session(_Llm(_answer(TRAIN)), pipeline=ASSISTANT) as ws:
            _open(ws)
            item_id = self._spoken_answer(ws)
            ws.send_json(_truncate(item_id, 0))
            error = ws.receive_json()["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "content_index")

    def test_truncating_an_answer_still_spoken_cancels_it(self):
        llm = _Llm(_answer("The kitchen lights are on now. And", finish=None))
        with self._session(llm) as ws:
            _open(ws)
            _say(ws, "Are the lights on?")
            ws.send_json({"type": "response.create"})
            item_id = _until(ws, "response.output_audio.delta")[-1]["item_id"]
            # a third of the first sentence's 300 ms
            ws.send_json(_truncate(item_id, 100))
            events = _until(ws, "conversation.item.truncated")
            ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
            retrieved = ws.receive_json()["item"]
        kinds = [event["type"] for event in events]
        assert kinds.index("response.done") < kinds.index("conversation.item.truncated")
        done = next(event for event in events if event["type"] == "response.done")["response"]
        assert (done["status"], done["status_details"]["reason"]) == ("cancelled", "client_cancelled")
        assert retrieved["content"][0]["transcript"] == "The kitchen"
        assert llm.closed_early == [True]


def test_g711_tables_match_audioop():
    from modelship.infer.pipelines.assistant.audio import to_pcm16

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
