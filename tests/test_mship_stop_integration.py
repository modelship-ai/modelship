import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import psutil
import pytest
import yaml

from modelship.deploy.local_nodes import local_nodes
from modelship.deploy.serve_utils import local_ray_clusters
from tests.conftest import MODEL_CONFIGS
from tests.test_cluster_join import _poll, _terminate_process_group

_GCS_PORT = 6580
_DASHBOARD_PORT = 6581
_API_PORT = 6582
_GATEWAY = f"http://127.0.0.1:{_API_PORT}/modelship"

_MODEL = MODEL_CONFIGS["chat-llama-server-plain"]

_MSHIP = ["uv", "run", "python", "-m", "modelship.launcher"]

pytestmark = [pytest.mark.integration, pytest.mark.llama_server, pytest.mark.mship_stop]


class _Cluster:
    def __init__(self, tmp_path: Path):
        self._tmp_path = tmp_path
        self._config = tmp_path / "models.yaml"
        self._config.write_text(yaml.dump({"models": [_MODEL]}))
        self._procs: list[subprocess.Popen] = []
        self._ray_tmps: list[str] = []

    def _spawn(self, name: str, args: list[str], **env: str) -> subprocess.Popen:
        ray_tmp = tempfile.mkdtemp(prefix=f"mship-stop-{name}-")
        self._ray_tmps.append(ray_tmp)
        env = {**os.environ, "RAY_TMPDIR": ray_tmp, "PYTHONUNBUFFERED": "1", **env}
        env.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
        with open(self._tmp_path / f"{name}.log", "w") as log_file:
            proc = subprocess.Popen(
                [*_MSHIP, *args], env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )
        self._procs.append(proc)
        return proc

    def head(self, *node_args: str) -> subprocess.Popen:
        ports = ["--gcs-port", str(_GCS_PORT), "--ray-dashboard-port", str(_DASHBOARD_PORT)]
        args = ["start", "--config", str(self._config), *node_args, *ports, "--openai-api-port", str(_API_PORT)]
        head = self._spawn("head", [*args, "--prune-ray-sessions", "false"])

        def gateway_healthy() -> bool:
            if head.poll() is not None:
                pytest.fail(f"mship start exited with {head.returncode}.\n{self.logs()}")
            try:
                return httpx.get(f"{_GATEWAY}/health", timeout=5).status_code == 200
            except httpx.HTTPError:
                return False

        assert _poll(gateway_healthy, deadline_s=120), f"the gateway never became healthy.\n{self.logs()}"
        return head

    def join(self, **env: str) -> subprocess.Popen:
        args = ["join", "--gcs-address", f"127.0.0.1:{_GCS_PORT}", "--node-num-cpus", "2", "--node-num-gpus", "0"]
        return self._spawn("join", [*args, "--prune-ray-sessions", "false"], **env)

    def wait_for_the_model(self) -> None:
        assert _poll(_model_routed, deadline_s=300), f"the model never became routable.\n{self.logs()}"

    def log(self, name: str) -> str:
        path = self._tmp_path / f"{name}.log"
        return path.read_text() if path.exists() else ""

    def logs(self) -> str:
        return "\n".join(f"--- {name}:\n{self.log(name)[-3000:]}" for name in ("head", "join"))

    def close(self) -> None:
        for proc in reversed(self._procs):
            _terminate_process_group(proc)
        for ray_tmp in self._ray_tmps:
            shutil.rmtree(ray_tmp, ignore_errors=True)


@pytest.fixture
def cluster(tmp_path):
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")
    cluster = _Cluster(tmp_path)
    try:
        yield cluster
    finally:
        cluster.close()


def _model_routed() -> bool:
    try:
        models = httpx.get(f"{_GATEWAY}/v1/models", timeout=5).json()["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False
    return _MODEL["name"] in {model["id"] for model in models}


def _running(predicate) -> list[psutil.Process]:
    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        with contextlib.suppress(psutil.Error):
            if predicate(proc.info["name"], proc.info["cmdline"] or []) and proc.status() != psutil.STATUS_ZOMBIE:
                found.append(proc)
    return found


def _alive(procs: list[psutil.Process]) -> list[psutil.Process]:
    alive = []
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                alive.append(proc)
    return alive


def _llama_servers() -> list[psutil.Process]:
    return _running(lambda _name, argv: bool(argv) and os.path.basename(argv[0]).startswith("llama"))


def _ours(name: str, argv: list[str]) -> bool:
    if name in ("raylet", "gcs_server"):
        return True
    return bool(argv) and (argv[0].startswith("ray::") or os.path.basename(argv[0]).startswith("llama"))


def _left() -> list[psutil.Process]:
    return _running(_ours)


def _mship_stop(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*_MSHIP, "stop", *args], capture_output=True, text=True, timeout=300)


def test_stop_shuts_a_head_down_and_leaves_nothing(cluster):
    head = cluster.head()
    cluster.wait_for_the_model()
    assert _llama_servers()

    result = _mship_stop()

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert head.wait(timeout=30) == 0
    assert _poll(lambda: not _left(), deadline_s=30), _left()


def test_stop_stops_the_worker_first_and_kills_a_head_that_does_not_stop(cluster):
    cluster.head("--node-num-cpus", "0", "--node-num-gpus", "0")
    cluster.join()
    cluster.wait_for_the_model()
    next(node.owner for node in local_nodes()[0] if node.role == "head").suspend()

    result = _mship_stop("--timeout", "10")

    output = f"{result.stdout}{result.stderr}"
    assert result.returncode == 0, output
    assert output.index("Stopping the worker node") < output.index("Stopping the head node")
    assert "did not stop within 10.0s; killing it" in output
    assert "leaving the Ray cluster" in cluster.log("join")
    assert _poll(lambda: not _left(), deadline_s=30), _left()


def test_a_worker_that_loses_its_head_kills_its_models_backends(cluster):
    cluster.head("--node-num-cpus", "0", "--node-num-gpus", "0")
    join = cluster.join(RAY_gcs_rpc_server_reconnect_timeout_s="5")
    cluster.wait_for_the_model()
    backends = _llama_servers()
    assert backends

    next(node.owner for node in local_nodes()[0] if node.role == "head").kill()

    assert join.wait(timeout=120) == 1, cluster.logs()
    assert _poll(lambda: not _alive(backends), deadline_s=20), cluster.logs()
