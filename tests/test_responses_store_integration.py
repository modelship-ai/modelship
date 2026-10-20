import concurrent.futures
import contextlib
import json
import os
import shutil
import socket
import subprocess
import tempfile

import httpx
import pytest
import redis
import yaml

from modelship.deploy.serve_utils import local_ray_clusters
from openai import NotFoundError, OpenAI
from tests.conftest import MODEL_CONFIGS
from tests.test_cluster_join import _poll, _terminate_process_group
from tests.test_mcp_e2e import serve_dice

_GCS_PORT = 6496
_DASHBOARD_PORT = 6497
_API_PORT = 6498
_API = f"http://127.0.0.1:{_API_PORT}/modelship/v1"
_IDENTITY_HEADER = "X-Mship-Test-Identity"

_MODEL = MODEL_CONFIGS["chat-llama-server-plain"]

pytestmark = [pytest.mark.integration, pytest.mark.llama_server, pytest.mark.state_store]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pings(client: redis.Redis) -> bool:
    try:
        return bool(client.ping())
    except redis.exceptions.ConnectionError:
        return False


@contextlib.contextmanager
def _redis_server():
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("redis-server is not on PATH")
    port = _free_port()
    with tempfile.TemporaryDirectory() as data_dir:
        server = subprocess.Popen(
            [binary, "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no", "--dir", data_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            with redis.Redis(port=port) as client:
                if not _poll(lambda: _pings(client), deadline_s=10):
                    pytest.fail("redis-server did not start")
            yield f"redis://127.0.0.1:{port}/0"
        finally:
            server.terminate()
            server.wait(timeout=10)


def _api_answers() -> bool:
    try:
        httpx.get(f"{_API}/models", timeout=2)
    except httpx.HTTPError:
        return False
    return True


def _model_listed() -> bool:
    try:
        models = httpx.get(f"{_API}/models", timeout=10).json()["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False
    return _MODEL["name"] in {model["id"] for model in models}


class _Head:
    def __init__(self, tmp_dir, backend: str, redis_url: str | None):
        self.backend = backend
        self.redis_url = redis_url
        self.client = OpenAI(base_url=_API, api_key="not-needed", max_retries=0, timeout=120)
        self._config = tmp_dir / "models.yaml"
        self._config.write_text(yaml.dump({"models": [_MODEL]}))
        self._log_path = tmp_dir / "head.log"
        self._proc: subprocess.Popen | None = None
        self._ray_tmp: str | None = None

    def start(self) -> None:
        self._ray_tmp = tempfile.mkdtemp(prefix="mship-rs-")
        args = [
            "start",
            "--config",
            str(self._config),
            "--gcs-port",
            str(_GCS_PORT),
            "--ray-dashboard-port",
            str(_DASHBOARD_PORT),
            "--openai-api-port",
            str(_API_PORT),
            "--prune-ray-sessions",
            "false",
        ]
        env = {
            **os.environ,
            "RAY_TMPDIR": self._ray_tmp,
            "PYTHONUNBUFFERED": "1",
            "MSHIP_STATE_STORE": self.redis_url or "memory://",
            "MSHIP_TRUSTED_IDENTITY_HEADER": _IDENTITY_HEADER,
        }
        with open(self._log_path, "a") as log_file:
            self._proc = subprocess.Popen(
                ["uv", "run", "python", "-m", "modelship.launcher", *args],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        def model_routed() -> bool:
            assert self._proc is not None
            if self._proc.poll() is not None:
                pytest.fail(f"mship start exited with {self._proc.returncode}.\n{self._log_tail()}")
            return _model_listed()

        assert _poll(model_routed, deadline_s=300), f"the model never became routable.\n{self._log_tail()}"

    def stop(self) -> None:
        if self._proc is not None:
            _terminate_process_group(self._proc)
            self._proc = None
            assert _poll(lambda: not local_ray_clusters() and not _api_answers(), deadline_s=120), (
                f"the head did not stop.\n{self._log_tail()}"
            )
        if self._ray_tmp is not None:
            shutil.rmtree(self._ray_tmp, ignore_errors=True)
            self._ray_tmp = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def _log_tail(self) -> str:
        return self._log_path.read_text()[-3000:] if self._log_path.exists() else "<no log>"


@pytest.fixture(scope="module", params=["memory", "redis"])
def head(request, tmp_path_factory):
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")
    with contextlib.ExitStack() as stack:
        redis_url = stack.enter_context(_redis_server()) if request.param == "redis" else None
        head = _Head(tmp_path_factory.mktemp(f"responses-{request.param}"), request.param, redis_url)
        stack.callback(head.stop)
        head.start()
        yield head


def _create(head: _Head, text: str, previous_response_id: str | None = None, identity: str = "alice", **kwargs):
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    return head.client.responses.create(
        model=_MODEL["name"],
        input=text,
        max_output_tokens=16,
        temperature=0,
        extra_headers={_IDENTITY_HEADER: identity},
        **kwargs,
    )


def _input_items(response_id: str, identity: str = "alice") -> list[dict]:
    resp = httpx.get(f"{_API}/responses/{response_id}/input_items", headers={_IDENTITY_HEADER: identity}, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def _user_texts(items: list[dict]) -> list[str]:
    return [item["content"] for item in items if item.get("role") == "user"]


def _item_ids(response_id: str) -> list[str]:
    return [item["id"] for item in _input_items(response_id)]


def test_a_chain_carries_every_turn_and_stores_each_once(head):
    first = _create(head, "My name is Alex.")
    second = _create(head, "I live in Berlin.", first.id)
    third = _create(head, "Where do I live?", second.id)

    assert first.usage.input_tokens < second.usage.input_tokens < third.usage.input_tokens
    items = _input_items(third.id)
    assert _user_texts(items) == ["My name is Alex.", "I live in Berlin.", "Where do I live?"]
    assert all(item.get("id") for item in items)
    assert [item["id"] for item in items if item.get("role") == "assistant"] == [
        *(o.id for o in first.output),
        *(o.id for o in second.output),
    ]
    fetched = httpx.get(f"{_API}/responses/{third.id}", headers={_IDENTITY_HEADER: "alice"}, timeout=30).json()
    assert [o["id"] for o in fetched["output"]] == [o.id for o in third.output]

    if head.redis_url is not None:
        with redis.Redis.from_url(head.redis_url, decode_responses=True) as client:
            [log] = client.keys(f"modelship/state/responses-log/*/{first.id}")
            assert client.xlen(log) == 3
            [record] = client.keys(f"modelship/state/responses/*/{third.id}")
            assert json.loads(client.get(record))["response"]["output"] == []


def test_branches_from_one_response_stay_apart(head):
    root = _create(head, "My name is Alex.")
    a = _create(head, "Say A.", root.id)
    b = _create(head, "Say A.", root.id)
    assert a.usage.input_tokens == b.usage.input_tokens

    after_a = _create(head, "And now?", a.id)
    after_b = _create(head, "And now?", b.id)
    a_ids, b_ids = _item_ids(after_a.id), _item_ids(after_b.id)
    assert a.output[0].id in a_ids and a.output[0].id not in b_ids
    assert b.output[0].id in b_ids and b.output[0].id not in a_ids

    older = _create(head, "Say C.", root.id)
    assert _user_texts(_input_items(older.id)) == ["My name is Alex.", "Say C."]


def test_concurrent_continuations_from_one_response_each_keep_their_own_turn(head):
    root = _create(head, "My name is Alex.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        branches = list(pool.map(lambda n: _create(head, f"Branch {n}.", root.id), range(4)))

    assert len({branch.id for branch in branches}) == 4
    for n, branch in enumerate(branches):
        assert _user_texts(_input_items(branch.id)) == ["My name is Alex.", f"Branch {n}."]
        assert _create(head, "Go on.", branch.id).previous_response_id == branch.id


def test_streamed_and_background_turns_continue(head):
    root = _create(head, "My name is Alex.")
    streamed = None
    for event in _create(head, "Say S.", root.id, stream=True):
        if event.type in ("response.completed", "response.incomplete"):
            streamed = event.response
    assert streamed is not None

    queued = _create(head, "Say B.", streamed.id, background=True)

    def finished() -> bool:
        status = head.client.responses.retrieve(queued.id, extra_headers={_IDENTITY_HEADER: "alice"}).status
        return status not in ("queued", "in_progress")

    assert _poll(finished, deadline_s=120)
    last = _create(head, "Go on.", queued.id)
    assert _user_texts(_input_items(last.id)) == ["My name is Alex.", "Say S.", "Say B.", "Go on."]


def test_deleting_a_middle_response_leaves_its_children_whole(head):
    first = _create(head, "My name is Alex.")
    second = _create(head, "I live in Berlin.", first.id)
    third = _create(head, "I like tea.", second.id)

    headers = {_IDENTITY_HEADER: "alice"}
    assert httpx.delete(f"{_API}/responses/{second.id}", headers=headers, timeout=30).status_code == 200
    assert httpx.get(f"{_API}/responses/{second.id}", headers=headers, timeout=30).status_code == 404
    with pytest.raises(NotFoundError):
        _create(head, "Go on.", second.id)

    fourth = _create(head, "Go on.", third.id)
    assert _user_texts(_input_items(fourth.id)) == ["My name is Alex.", "I live in Berlin.", "I like tea.", "Go on."]


def test_another_identity_cannot_read_or_continue(head):
    mine = _create(head, "My name is Alex.", identity="alice")
    with pytest.raises(NotFoundError):
        _create(head, "Go on.", mine.id, identity="bob")
    assert httpx.get(f"{_API}/responses/{mine.id}", headers={_IDENTITY_HEADER: "bob"}, timeout=30).status_code == 404
    assert _create(head, "Go on.", mine.id, identity="alice").previous_response_id == mine.id


@pytest.fixture(scope="module")
def dice():
    seen: list[str] = []
    with serve_dice(_free_port(), seen) as server_url:
        yield server_url, seen


def test_an_mcp_tool_list_is_fetched_once_per_conversation(head, dice):
    server_url, seen = dice
    seen.clear()
    tool = {"type": "mcp", "server_label": "dice", "server_url": server_url, "require_approval": "never"}
    approved = {**tool, "require_approval": "always"}
    headers = {_IDENTITY_HEADER: "alice"}

    def respond(text, previous_response_id=None, tools=(tool,), **kwargs):
        return head.client.responses.create(
            model=_MODEL["name"],
            input=text,
            previous_response_id=previous_response_id,
            tools=list(tools),
            max_output_tokens=128,
            max_tool_calls=1,
            temperature=0,
            extra_headers=headers,
            **kwargs,
        )

    def retrieve(response_id):
        return head.client.responses.retrieve(response_id, extra_headers=headers)

    first = respond("Roll 2 dice.", tool_choice="required")
    queued = respond("Roll 3 dice.", first.id, tool_choice="required", background=True)
    assert _poll(lambda: retrieve(queued.id).status not in ("queued", "in_progress"), deadline_s=120)
    second = retrieve(queued.id)
    third = respond("Roll 4 dice.", second.id, tools=(approved,), tool_choice="required")
    [approval] = [item for item in third.output if item.type == "mcp_approval_request"]
    approval_response = {"type": "mcp_approval_response", "approval_request_id": approval.id, "approve": True}
    fourth = respond([approval_response], third.id, tools=(approved,))
    fifth = respond("Roll 5 dice.", fourth.id, tools=({**tool, "allowed_tools": ["roll"]},), tool_choice="required")

    turns = [[item.type for item in turn.output] for turn in (first, second, third, fourth, fifth)]
    assert ["mcp_list_tools" in types for types in turns] == [True, False, False, False, True], turns
    assert ["mcp_call" in types for types in turns] == [True, True, False, True, True], turns
    assert seen.count("server/discover") == seen.count("tools/list") == 4, seen


def test_a_restarted_head_keeps_conversations_only_on_redis(head):
    first = _create(head, "My name is Alex.")
    second = _create(head, "I live in Berlin.", first.id)
    head.restart()

    if head.backend == "memory":
        with pytest.raises(NotFoundError):
            _create(head, "Go on.", second.id)
        return
    after = _create(head, "Go on.", second.id)
    assert _user_texts(_input_items(after.id)) == ["My name is Alex.", "I live in Berlin.", "Go on."]
