"""`/v1/realtime` sessions in the gateway: opening on a pipeline, session.update, and events not supported yet."""

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from modelship.openai.realtime import serve_session

ASSISTANT = {"name": "assistant", "stt": "whisper", "llm": "qwen", "session_defaults": {"instructions": "Be brief."}}
SCRIBE = {"name": "scribe", "stt": "whisper"}


def _client(pipelines: dict, known=("whisper", "qwen"), serving=("whisper", "qwen")) -> TestClient:
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        await serve_session(websocket, pipelines, list(known), set(serving))

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


class TestOpening:
    def test_a_known_pipeline_opens_with_its_defaults(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            event = ws.receive_json()
        assert event["type"] == "session.created"
        assert event["event_id"].startswith("event_")
        session = event["session"]
        assert session["type"] == "realtime"
        assert session["object"] == "realtime.session"
        assert session["model"] == "assistant"
        assert session["instructions"] == "Be brief."
        assert session["id"].startswith("sess_")

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


class TestEvents:
    def _open(self, ws) -> dict:
        return ws.receive_json()["session"]

    def test_session_update_answers_with_the_updated_session(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Speak French.", "output_modalities": ["text"]},
                }
            )
            event = ws.receive_json()
        assert event["type"] == "session.updated"
        assert event["session"]["instructions"] == "Speak French."
        assert event["session"]["output_modalities"] == ["text"]

    def test_session_fields_not_used_yet_are_ignored(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_json({"type": "session.update", "session": {"audio": {"output": {"voice": "marin"}}}})
            event = ws.receive_json()
        assert event["type"] == "session.updated"
        assert "audio" not in event["session"]

    def test_the_session_type_cannot_change(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_json({"type": "session.update", "event_id": "e1", "session": {"type": "transcription"}})
            error = ws.receive_json()["error"]
        assert (error["param"], error["event_id"]) == ("session.type", "e1")

    def test_an_invalid_field_names_its_param(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_json({"type": "session.update", "session": {"output_modalities": ["video"]}})
            error = ws.receive_json()["error"]
        assert error["code"] == "invalid_value"
        assert error["param"].startswith("session.output_modalities")

    def test_other_events_are_not_supported_yet_and_the_session_stays_open(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_json({"type": "response.create", "event_id": "e2"})
            error = ws.receive_json()["error"]
            ws.send_json({"type": "session.update", "session": {}})
            still_open = ws.receive_json()
        assert (error["code"], error["event_id"]) == ("unsupported_event", "e2")
        assert error["message"] == "response.create is not supported yet."
        assert still_open["type"] == "session.updated"

    @pytest.mark.parametrize("frame", ["not json", "[1, 2]", '{"no": "type"}'])
    def test_a_malformed_event_is_an_error(self, frame):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_text(frame)
            event = ws.receive_json()
        assert event["type"] == "error"

    def test_a_binary_frame_is_an_error(self):
        with _client(_route(ASSISTANT)).websocket_connect("/v1/realtime?model=assistant") as ws:
            self._open(ws)
            ws.send_bytes(b"\x00\x01")
            event = ws.receive_json()
        assert event["error"]["message"] == "Send events as JSON text frames."
