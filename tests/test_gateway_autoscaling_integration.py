"""Same-box integration test for gateway autoscaling: load past its target scales the gateway out, and the
new replica serves the deployed model.

Its own throwaway `mship start` on distinct ports + RAY_TMPDIR; only ever signals its own process, never `ray stop`.
"""

import os
import shutil
import subprocess
import tempfile
import threading

import httpx
import pytest
import yaml

from modelship.deploy.serve_utils import local_ray_clusters
from tests.conftest import MODEL_CONFIGS
from tests.test_cluster_join import _poll, _terminate_process_group

# Off Ray's 6379, modelship's defaults (6380/8265/8000), test_cluster_join's and test_node_logging's.
_GCS_PORT = 6493
_DASHBOARD_PORT = 6494
_API_PORT = 6495
_DASHBOARD = f"http://127.0.0.1:{_DASHBOARD_PORT}"
_API = f"http://127.0.0.1:{_API_PORT}/modelship/v1"

_MODEL = MODEL_CONFIGS["chat-llama-server-plain"]
_LOAD_THREADS = 4


def _apps() -> dict:
    try:
        return httpx.get(f"{_DASHBOARD}/api/serve/applications/", timeout=10).json().get("applications", {})
    except (httpx.HTTPError, ValueError):
        return {}


def _gateway() -> dict:
    return _apps().get("modelship", {}).get("deployments", {}).get("modelship", {})


def _running_gateway_replicas() -> set[str]:
    return {replica["replica_id"] for replica in _gateway().get("replicas", []) if replica["state"] == "RUNNING"}


def _model_listed() -> bool:
    try:
        models = httpx.get(f"{_API}/models", timeout=10).json()["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False
    return _MODEL["name"] in {model["id"] for model in models}


def _chat(max_tokens: int) -> httpx.Response:
    return httpx.post(
        f"{_API}/chat/completions",
        json={
            "model": _MODEL["name"],
            "messages": [{"role": "user", "content": "Count to fifty."}],
            "max_tokens": max_tokens,
        },
        timeout=120,
    )


def _load(stop: threading.Event, errors: list) -> None:
    while not stop.is_set():
        try:
            _chat(max_tokens=64).raise_for_status()
        except httpx.HTTPError as exc:
            errors.append(exc)


def _tail(log_path) -> str:
    return log_path.read_text()[-3000:] if log_path.exists() else "<no log>"


@pytest.fixture
def autoscaling_cluster(tmp_path):
    """A head serving one model, with a gateway that scales from 1 to 2 replicas past 1 ongoing request each.
    Yields the head's log path."""
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")

    config = tmp_path / "models.yaml"
    config.write_text(yaml.dump({"models": [_MODEL]}))
    log_path = tmp_path / "head.log"
    # A short /tmp-rooted dir: AF_UNIX socket paths have a length limit.
    ray_tmp = tempfile.mkdtemp(prefix="mship-as-")
    args = [
        "start",
        "--config",
        str(config),
        "--gateway-min-replicas",
        "1",
        "--gateway-max-replicas",
        "2",
        "--gateway-target-ongoing-requests",
        "1",
        "--ray-port",
        str(_GCS_PORT),
        "--ray-dashboard-port",
        str(_DASHBOARD_PORT),
        "--openai-api-port",
        str(_API_PORT),
        "--prune-ray-sessions",
        "false",
    ]
    with open(log_path, "w") as log_file:
        head = subprocess.Popen(
            ["uv", "run", "python", "-m", "modelship.launcher", *args],
            env={**os.environ, "RAY_TMPDIR": ray_tmp, "PYTHONUNBUFFERED": "1"},
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:

        def model_routed() -> bool:
            if head.poll() is not None:
                pytest.fail(f"mship start exited with {head.returncode}.\n{_tail(log_path)}")
            return _model_listed()

        assert _poll(model_routed, deadline_s=300), f"the model never became routable.\n{_tail(log_path)}"
        yield log_path
    finally:
        _terminate_process_group(head)
        shutil.rmtree(ray_tmp, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.gateway_autoscaling
def test_load_past_the_target_scales_the_gateway_out_and_the_new_replica_serves_the_model(autoscaling_cluster):
    config = _gateway()["deployment_config"]
    assert (config["autoscaling_config"]["min_replicas"], config["autoscaling_config"]["max_replicas"]) == (1, 2)
    assert config["autoscaling_config"]["target_ongoing_requests"] == 1
    assert config["graceful_shutdown_timeout_s"] == 600
    first = _running_gateway_replicas()
    assert len(first) == 1, _gateway()

    stop = threading.Event()
    errors: list[Exception] = []
    threads = [threading.Thread(target=_load, args=(stop, errors), daemon=True) for _ in range(_LOAD_THREADS)]
    for thread in threads:
        thread.start()
    try:
        scaled = _poll(lambda: len(_running_gateway_replicas()) == 2, deadline_s=180)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=150)
    assert scaled, f"the gateway never scaled out: {_gateway()}\n{_tail(autoscaling_cluster)}"
    assert not errors, f"requests failed while the gateway scaled out: {errors[:3]}"
    assert first < _running_gateway_replicas()

    # The proxy spreads these over both replicas; a replica without the routing table would omit the model or 404.
    assert all(_model_listed() for _ in range(20)), "a gateway replica does not list the model"
    for _ in range(10):
        _chat(max_tokens=5).raise_for_status()
