"""Same-box integration test for per-node logging: a replica on a joined node logs at the
joiner's level and target, in the head's format, with the head's metrics toggle.

Its own throwaway `mship start` head and `mship join` node, on distinct ports + RAY_TMPDIR;
only ever signals its own processes, never `ray stop`.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import httpx
import pytest
import yaml
from ray.util.state import list_actors, list_nodes

from modelship.deploy.serve_utils import local_ray_clusters
from tests.conftest import MODEL_CONFIGS
from tests.test_cluster_join import _poll, _terminate_process_group

# Off Ray's 6379, modelship's defaults (6380/8265/8000) and test_cluster_join's 6480/6481.
_GCS_PORT = 6490
_DASHBOARD_PORT = 6491
_API_PORT = 6492
_DASHBOARD = f"http://127.0.0.1:{_DASHBOARD_PORT}"

_MODEL = MODEL_CONFIGS["chat-llama-server-plain"]

_MSHIP = ["uv", "run", "python", "-m", "modelship.launcher"]

# Unset in both processes, so the replica's values can only come from the flags.
_LOGGING_ENV = (
    "MSHIP_LOG_LEVEL",
    "MSHIP_LOG_FORMAT",
    "MSHIP_LOG_TARGET",
    "MSHIP_METRICS",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "VLLM_LOGGING_LEVEL",
)


class _SyslogListener:
    """Collects the datagrams sent to a UDP syslog target on 127.0.0.1."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self.datagrams: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                self.datagrams.append(self._sock.recv(65535).decode(errors="replace"))
            except TimeoutError:
                continue

    def json_records(self) -> list[dict]:
        records = []
        for datagram in list(self.datagrams):
            # SysLogHandler prefixes <priority> and appends a NUL.
            payload = re.sub(r"^<\d+>", "", datagram).rstrip("\x00")
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        return records

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()


def _spawn(args: list[str], ray_tmp: str, log_path: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k not in _LOGGING_ENV}
    env |= {"RAY_TMPDIR": ray_tmp, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as log_file:
        return subprocess.Popen(
            [*_MSHIP, *args], env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )


def _model_app_running() -> bool:
    try:
        apps = httpx.get(f"{_DASHBOARD}/api/serve/applications/", timeout=10).json().get("applications", {})
    except (httpx.HTTPError, ValueError):
        return False
    return any(
        name.startswith(f"modelship.{_MODEL['name']}-") and app["status"] == "RUNNING" for name, app in apps.items()
    )


def _process_env(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(entry.split("=", 1) for entry in raw.decode(errors="replace").split("\0") if "=" in entry)


def _tails(*log_paths: Path) -> str:
    return "\n".join(f"--- {path.name}:\n{path.read_text()[-3000:]}" for path in log_paths if path.exists())


@pytest.fixture
def two_node_cluster(tmp_path):
    """A head at the default level on the console, with JSON format and metrics off, and a joiner
    logging at DEBUG to a syslog listener. Yields (listener, head log, joiner log)."""
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")

    config = tmp_path / "models.yaml"
    config.write_text(yaml.dump({"models": [_MODEL]}))
    head_log, join_log = tmp_path / "head.log", tmp_path / "joiner.log"
    # Short /tmp-rooted dirs: AF_UNIX socket paths have a length limit.
    head_tmp = tempfile.mkdtemp(prefix="mship-nl-head-")
    join_tmp = tempfile.mkdtemp(prefix="mship-nl-join-")
    listener = _SyslogListener()
    procs: list[subprocess.Popen] = []
    try:
        # No CPUs on the head, so the replica can only land on the joiner.
        head = _spawn(
            [
                "start",
                "--config",
                str(config),
                "--log-format",
                "json",
                "--no-metrics",
                "--node-num-cpus",
                "0",
                "--node-num-gpus",
                "0",
                "--gcs-port",
                str(_GCS_PORT),
                "--ray-dashboard-port",
                str(_DASHBOARD_PORT),
                "--openai-api-port",
                str(_API_PORT),
                "--prune-ray-sessions",
                "false",
            ],
            head_tmp,
            head_log,
        )
        procs.append(head)

        def head_ready() -> bool:
            if head.poll() is not None:
                pytest.fail(f"mship start exited with {head.returncode}.\n{_tails(head_log)}")
            try:
                return httpx.get(f"http://127.0.0.1:{_API_PORT}/modelship/health", timeout=5).status_code == 200
            except httpx.HTTPError:
                return False

        assert _poll(head_ready, deadline_s=120), f"the head's gateway never became healthy.\n{_tails(head_log)}"

        procs.append(
            _spawn(
                [
                    "join",
                    "--gcs-address",
                    f"127.0.0.1:{_GCS_PORT}",
                    "--log-level",
                    "debug",
                    "--log-target",
                    f"syslog://127.0.0.1:{listener.port}",
                    "--node-num-cpus",
                    "2",
                    "--node-num-gpus",
                    "0",
                    "--prune-ray-sessions",
                    "false",
                ],
                join_tmp,
                join_log,
            )
        )
        yield listener, head_log, join_log
    finally:
        for proc in reversed(procs):
            _terminate_process_group(proc)
        listener.close()
        shutil.rmtree(head_tmp, ignore_errors=True)
        shutil.rmtree(join_tmp, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.node_logging
def test_a_replica_on_the_joiner_uses_its_nodes_logging_and_the_heads_format_and_metrics(two_node_cluster):
    listener, head_log, join_log = two_node_cluster
    assert _poll(_model_app_running, deadline_s=300), f"the model never ran.\n{_tails(head_log, join_log)}"

    head_node = next(node.node_id for node in list_nodes(address=_DASHBOARD) if node.is_head_node)
    actors = list_actors(address=_DASHBOARD, filters=[("state", "=", "ALIVE")], limit=1000)
    replica_class = f"ServeReplica:modelship.{_MODEL['name']}-"
    replica = next(actor for actor in actors if actor.class_name.startswith(replica_class))
    assert replica.node_id != head_node

    env = _process_env(replica.pid)
    assert (env.get("MSHIP_METRICS"), env.get("VLLM_LOGGING_LEVEL")) == ("false", "DEBUG")

    records = listener.json_records()
    from_replica = [record for record in records if record.get("pid") == replica.pid]
    assert any(record["logger"].startswith("modelship.") for record in from_replica), listener.datagrams[-20:]
    assert any(record["level"] == "DEBUG" for record in from_replica), from_replica[-20:]

    head_pids = {actor.pid for actor in actors if actor.node_id == head_node}
    assert not head_pids & {record.get("pid") for record in records}
