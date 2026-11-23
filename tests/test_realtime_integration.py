"""Pipelines on a real cluster: deployed with their models, listed in /v1/models, and opening a /v1/realtime
session. `test_realtime_session.py` and the deploy unit tests cover the same logic against fakes."""

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
