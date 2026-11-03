import contextlib
import json
import re
import socket
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from websockets.sync.client import connect

from tests.conftest import MODEL_CONFIGS, OPENAI_API_BASE
from tests.test_cluster_join import _poll
from tests.test_dropped_request_integration import _replica

VLLM = {**MODEL_CONFIGS["chat-capable"], "vllm_engine_kwargs": {"max_model_len": 32768}}
LLAMA = MODEL_CONFIGS["chat-llama-server-plain"]
PROMPT = "Repeat the word 'hello' forever, separated by spaces. Never stop."
MAX_TOKENS = {VLLM["name"]: 30000, LLAMA["name"]: 4000}


def _vllm_running() -> float:
    replica_id, _ = _replica(VLLM["name"])
    metrics = httpx.get("http://localhost:8079/metrics", timeout=10).text
    pattern = rf'^ray_vllm_num_requests_running\{{[^}}]*ReplicaId="{replica_id}"[^}}]*\}} (\S+)$'
    return sum(float(value) for value in re.findall(pattern, metrics, re.M))


def _llama_processing() -> int:
    _, pid = _replica(LLAMA["name"])
    args = next(a for a in (c.cmdline() for c in psutil.Process(pid).children(recursive=True)) if "--api-key" in a)
    port, key = args[args.index("--port") + 1], args[args.index("--api-key") + 1]
    slots = httpx.get(f"http://127.0.0.1:{port}/slots", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    return sum(slot["is_processing"] for slot in slots.json())


ENGINES = {VLLM["name"]: (_vllm_running, 30), LLAMA["name"]: (_llama_processing, 10)}


def _chat(model: str, stream: bool = False) -> dict:
    messages = [{"role": "user", "content": PROMPT}]
    return {"model": model, "messages": messages, "max_tokens": MAX_TOKENS[model], "temperature": 0, "stream": stream}


def _response(model: str) -> dict:
    return {"model": model, "input": PROMPT, "max_output_tokens": MAX_TOKENS[model], "temperature": 0}


@contextlib.contextmanager
def _plain(model: str):
    url = urlsplit(f"{OPENAI_API_BASE}/chat/completions")
    body = json.dumps(_chat(model)).encode()
    head = (
        f"POST {url.path} HTTP/1.1\r\nHost: {url.netloc}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    )
    with socket.create_connection((url.hostname, url.port)) as sock:
        sock.sendall(head.encode() + body)
        yield


@contextlib.contextmanager
def _stream(model: str):
    body = _chat(model, stream=True)
    with httpx.stream("POST", f"{OPENAI_API_BASE}/chat/completions", json=body, timeout=60) as resp:
        lines = resp.iter_lines()
        next(line for line in lines if line.startswith("data: "))
        yield


@contextlib.contextmanager
def _websocket(model: str):
    with connect(f"{OPENAI_API_BASE.replace('http', 'ws', 1)}/responses", max_queue=None) as ws:
        ws.send(json.dumps({"type": "response.create", **_response(model)}))
        ws.recv()
        yield


@contextlib.contextmanager
def _background(model: str):
    resp = httpx.post(f"{OPENAI_API_BASE}/responses", json={**_response(model), "background": True}, timeout=30)
    resp.raise_for_status()
    try:
        yield
    finally:
        httpx.post(f"{OPENAI_API_BASE}/responses/{resp.json()['id']}/cancel", timeout=30).raise_for_status()


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.llama_server
class TestTheEngineStops:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy_raw([VLLM, LLAMA])

    @pytest.mark.parametrize("model", [VLLM["name"], LLAMA["name"]])
    @pytest.mark.parametrize(
        "drop", [_plain, _stream, _websocket, _background], ids=["plain", "stream", "websocket", "background"]
    )
    def test_when_the_request_is_dropped(self, model, drop):
        engine, idle_within_s = ENGINES[model]
        with drop(model):
            assert _poll(lambda: engine() >= 1, deadline_s=30), "the request never reached the engine"
        assert _poll(lambda: engine() == 0, deadline_s=idle_within_s), "the engine kept generating"
