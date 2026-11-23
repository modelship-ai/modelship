"""Pipelines on a real cluster: deployed with their models, listed in /v1/models, opening a /v1/realtime session,
and a text conversation the pipeline's LLM answers. `test_realtime_session.py` and the deploy unit tests cover the
same logic against fakes."""

import json

import httpx
import pytest
import yaml
from websockets.sync.client import connect

from modelship.utils.config_schema import PipelineConfig
from tests.conftest import MODEL_CONFIGS, serve_apps

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"
REALTIME = "ws://localhost:8000/modelship/v1/realtime"

STT = MODEL_CONFIGS["stt-cpp-model"]
VOICE = {"name": "voice", "stt": STT["name"], "session_defaults": {"instructions": "Be brief."}}
GHOST = {"name": "ghost", "stt": "not-deployed"}
LLM = MODEL_CONFIGS["chat-llama-server-plain"]
CHAT = {"name": "chat", "llm": LLM["name"], "session_defaults": {"instructions": "Answer in one short sentence."}}


def _audio_app(raw: dict) -> str:
    return PipelineConfig.model_validate(raw).deployment_name("modelship")


def _listed() -> dict[str, str]:
    models = httpx.get(f"{OPENAI_API_BASE}/models", timeout=10).json()["data"]
    return {card["id"]: card["owned_by"] for card in models}


def _first_event(query: str) -> dict:
    with connect(f"{REALTIME}{query}", open_timeout=30) as ws:
        return json.loads(ws.recv(timeout=30))


@pytest.mark.integration
class TestPipelines:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime") / "models.yaml"
        config.write_text(yaml.dump({"models": [STT], "pipelines": [VOICE, GHOST]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-pipelines")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    def test_each_pipeline_with_stt_has_its_audio_app(self):
        apps = serve_apps()
        assert apps[_audio_app(VOICE)]["status"] == "RUNNING"
        assert apps[_audio_app(GHOST)]["status"] == "RUNNING"

    def test_models_lists_the_pipelines_beside_the_models(self):
        listed = _listed()
        assert listed[STT["name"]] == "modelship"
        assert listed["voice"] == "modelship-pipeline"

    def test_a_session_opens_on_a_pipeline_with_its_defaults(self):
        event = _first_event("?model=voice")
        assert event["type"] == "session.created"
        assert event["session"]["type"] == "transcription"
        assert event["session"]["instructions"] == "Be brief."

    def test_a_session_updates(self):
        with connect(f"{REALTIME}?model=voice", open_timeout=30) as ws:
            ws.recv(timeout=30)
            ws.send(json.dumps({"type": "session.update", "session": {"instructions": "Be terse."}}))
            event = json.loads(ws.recv(timeout=30))
        assert event["type"] == "session.updated"
        assert event["session"]["instructions"] == "Be terse."

    def test_a_pipeline_whose_model_is_missing_deploys_and_its_session_is_refused(self):
        event = _first_event("?model=ghost")
        assert event["type"] == "error"
        assert "uses model 'not-deployed'" in event["error"]["message"]

    def test_a_reconcile_without_the_pipelines_removes_them(self, model_deployer):
        model_deployer.deploy(STT["name"])
        assert _audio_app(VOICE) not in serve_apps()
        assert "voice" not in _listed()
        assert _first_event("?model=voice")["error"]["code"] == "model_not_found"


def _until(ws, kind: str) -> list[dict]:
    """Events up to and including the first of *kind*."""
    events = []
    while True:
        events.append(json.loads(ws.recv(timeout=120)))
        if events[-1]["type"] == kind:
            return events


def _send(ws, event: dict) -> None:
    ws.send(json.dumps(event))


def _say(ws, text: str) -> None:
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    _send(ws, {"type": "conversation.item.create", "item": item})
    _until(ws, "conversation.item.done")


def _text(response: dict) -> str:
    return "".join(part["text"] for item in response["output"] for part in item["content"])


@pytest.mark.integration
class TestTextConversation:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime-chat") / "models.yaml"
        config.write_text(yaml.dump({"models": [LLM], "pipelines": [CHAT]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-chat")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    @pytest.fixture
    def ws(self):
        with connect(f"{REALTIME}?model=chat", open_timeout=30) as ws:
            _until(ws, "conversation.created")
            yield ws

    def test_the_answer_streams_as_text_and_the_next_turn_sees_it(self, ws):
        _say(ws, "My name is Ada.")
        _send(ws, {"type": "response.create"})
        events = _until(ws, "response.done")
        first = events[-1]["response"]
        deltas = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        _say(ws, "What is my name?")
        _send(ws, {"type": "response.create"})
        second = _until(ws, "response.done")[-1]["response"]
        assert (first["status"], second["status"]) == ("completed", "completed")
        assert deltas and deltas == _text(first)
        assert "ada" in _text(second).lower(), _text(second)
        assert second["usage"]["input_tokens"] > first["usage"]["input_tokens"]

    def test_an_out_of_band_answer_stays_out_of_the_conversation(self, ws):
        hello = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}
        _send(ws, {"type": "response.create", "response": {"conversation": "none", "input": [hello]}})
        events = _until(ws, "response.done")
        assert not [event for event in events if event["type"].startswith("conversation.")]
        assert events[-1]["response"]["conversation_id"] is None
        assert _text(events[-1]["response"])

    def test_a_cancelled_answer_ends_cancelled_and_the_session_answers_again(self, ws):
        _say(ws, "Count from 1 to 300, one number per line.")
        _send(ws, {"type": "response.create", "response": {"instructions": "Do exactly as asked."}})
        _until(ws, "response.output_text.delta")
        _send(ws, {"type": "response.cancel"})
        cancelled = _until(ws, "response.done")[-1]["response"]
        _send(ws, {"type": "response.create", "response": {"max_output_tokens": 16}})
        again = _until(ws, "response.done")[-1]["response"]
        assert cancelled["status"] == "cancelled"
        assert again["status"] in ("completed", "incomplete")
        assert _text(again)
