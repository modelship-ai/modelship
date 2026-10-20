import contextlib
import os
import secrets
import subprocess
import tempfile

import httpx
import pytest
import yaml

from modelship.deploy.serve_utils import local_ray_clusters
from openai import OpenAI
from tests.conftest import MODEL_CONFIGS
from tests.test_cluster_join import _poll, _terminate_process_group
from tests.test_responses_store_integration import _redis_server

_GCS_PORT = 6586
_DASHBOARD_PORT = 6587
_API_PORT = 6588
_API = f"http://127.0.0.1:{_API_PORT}/modelship/v1"
_IDENTITY_HEADER = "X-Mship-Test-Identity"

_MODEL = {**MODEL_CONFIGS["chat-capable"], "num_gpus": 0.4, "num_replicas": 2}

pytestmark = [pytest.mark.integration, pytest.mark.vllm, pytest.mark.state_store]


def _model_listed() -> bool:
    try:
        models = httpx.get(f"{_API}/models", timeout=10).json()["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False
    return _MODEL["name"] in {model["id"] for model in models}


def _stop(head: subprocess.Popen) -> None:
    _terminate_process_group(head)
    assert _poll(lambda: not local_ray_clusters(), deadline_s=120), "the head did not stop"


@pytest.fixture(scope="module", params=["memory", "redis"])
def client(request, tmp_path_factory):
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")
    with contextlib.ExitStack() as stack:
        store = stack.enter_context(_redis_server()) if request.param == "redis" else "memory://"
        ray_tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="mship-af-"))
        tmp_dir = tmp_path_factory.mktemp(f"affinity-{request.param}")
        config = tmp_dir / "models.yaml"
        config.write_text(yaml.dump({"models": [_MODEL]}))
        log_path = tmp_dir / "head.log"
        args = ["start", "--config", str(config), "--gcs-port", str(_GCS_PORT), "--prune-ray-sessions", "false"]
        args += ["--ray-dashboard-port", str(_DASHBOARD_PORT), "--openai-api-port", str(_API_PORT)]
        env = {
            **os.environ,
            "RAY_TMPDIR": ray_tmp,
            "PYTHONUNBUFFERED": "1",
            "MSHIP_STATE_STORE": store,
            "MSHIP_TRUSTED_IDENTITY_HEADER": _IDENTITY_HEADER,
            "MSHIP_GATEWAY_MIN_REPLICAS": "2",
            "MSHIP_GATEWAY_MAX_REPLICAS": "2",
        }
        with open(log_path, "a") as log_file:
            head = subprocess.Popen(
                ["uv", "run", "python", "-m", "modelship.launcher", *args],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        stack.callback(_stop, head)

        def model_routed() -> bool:
            if head.poll() is not None:
                pytest.fail(f"mship start exited with {head.returncode}.\n{log_path.read_text()[-3000:]}")
            return _model_listed()

        assert _poll(model_routed, deadline_s=600), f"the model never became routable.\n{log_path.read_text()[-3000:]}"
        yield OpenAI(base_url=_API, api_key="not-needed", max_retries=0, timeout=120)


def _instructions(n: int) -> str:
    return f"Conversation {n}, {secrets.token_hex(8)}. " + "Answer every question in one short sentence. " * 40


def test_a_chats_later_turns_reuse_the_earlier_prompt(client):
    for n in range(8):
        messages = [{"role": "system", "content": _instructions(n)}, {"role": "user", "content": "Name a colour."}]
        previous = None
        for question in ("Name a fruit.", "Name an animal.", "Name a city."):
            reply = client.chat.completions.create(
                model=_MODEL["name"],
                messages=messages,
                max_tokens=8,
                temperature=0,
                extra_headers={_IDENTITY_HEADER: "alice"},
            )
            if previous is not None:
                details = reply.usage.prompt_tokens_details
                assert details is not None and details.cached_tokens >= previous.usage.prompt_tokens - 16
            messages += [{"role": "assistant", "content": reply.choices[0].message.content}]
            messages += [{"role": "user", "content": question}]
            previous = reply


def test_a_stored_conversations_later_turns_reuse_the_earlier_prompt(client):
    for n in range(8):
        instructions = _instructions(n)
        previous = None
        for question in ("Name a colour.", "Name a fruit.", "Name an animal."):
            response = client.responses.create(
                model=_MODEL["name"],
                input=question,
                instructions=instructions,
                previous_response_id=previous.id if previous is not None else None,
                max_output_tokens=16,
                temperature=0,
                extra_headers={_IDENTITY_HEADER: "alice"},
            )
            if previous is not None:
                assert response.usage.input_tokens_details.cached_tokens >= previous.usage.input_tokens - 16
            previous = response
