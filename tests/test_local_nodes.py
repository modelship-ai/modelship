import contextlib
import os
import subprocess
import sys
import time

import psutil
import pytest

from modelship.deploy.local_nodes import _role, _stop, kill_groups, local_nodes

_WORKER = "import subprocess, time; subprocess.Popen(['sleep', '600']); time.sleep(600)"

_RAYLET = """
import os, signal, subprocess, sys, time
worker = subprocess.Popen([sys.argv[1], "-c", sys.argv[2]], process_group=0)
def stop(*_):
    os.killpg(worker.pid, signal.SIGKILL)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(600)
"""

_OWNER = """
import signal, subprocess, sys, time
raylet = subprocess.Popen([sys.argv[1], "-c", sys.argv[2], sys.executable, sys.argv[3]])
def stop(*_):
    raylet.terminate()
    raylet.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop if sys.argv[4] == "graceful" else signal.SIG_IGN)
time.sleep(600)
"""


@pytest.fixture
def raylet_binary(tmp_path):
    path = tmp_path / "raylet"
    path.symlink_to(sys.executable)
    return str(path)


@pytest.fixture
def spawn(raylet_binary):
    trees = []

    def _spawn(command: str, on_sigterm: str = "graceful") -> psutil.Process:
        fake = [sys.executable, "-c", _OWNER, raylet_binary, _RAYLET, _WORKER, on_sigterm]
        proc = subprocess.Popen([*fake, "-m", "modelship.launcher", command], start_new_session=True)
        owner = psutil.Process(proc.pid)
        deadline = time.monotonic() + 30
        while len(owner.children(recursive=True)) < 3:
            assert time.monotonic() < deadline, "the fake node did not come up"
            time.sleep(0.1)
        trees.append([owner, *owner.children(recursive=True)])
        return owner

    yield _spawn
    for tree in trees:
        for proc in tree:
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()


def _node_of(owner: psutil.Process):
    return next(node for node in local_nodes()[0] if node.owner == owner)


def _alive(procs: list[psutil.Process]) -> list[psutil.Process]:
    return [proc for proc in procs if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE]


@pytest.mark.parametrize(
    ("argv", "role"),
    [
        (["/env/bin/python", "-m", "modelship.launcher", "start", "--config", "m.yaml"], "head"),
        (["/env/bin/python", "-m", "modelship.launcher", "join", "--gcs-address", "head"], "worker"),
        (["/env/bin/python", "-m", "modelship.launcher", "deploy"], None),
        (["/env/bin/ray", "start", "--head"], None),
    ],
)
def test_role_reads_the_launcher_command(argv, role):
    assert _role(argv) == role


def test_finds_a_node_by_its_raylets_owner(spawn):
    owner = spawn("start")
    node = _node_of(owner)
    assert node.role == "head"
    assert node.raylet.name() == "raylet"


def test_lists_workers_before_heads(spawn):
    head, worker = spawn("start"), spawn("join")
    ours = [node.owner for node in local_nodes()[0] if node.owner in (head, worker)]
    assert ours == [worker, head]


def test_a_raylet_no_launcher_runs_is_foreign(raylet_binary):
    proc = subprocess.Popen([raylet_binary, "-c", "import time; time.sleep(600)"])
    try:
        deadline = time.monotonic() + 10
        while psutil.Process(proc.pid).name() != "raylet":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert proc.pid in [raylet.pid for raylet in local_nodes()[1]]
    finally:
        proc.kill()
        proc.wait()


def test_stops_a_node_that_shuts_down_on_sigterm(spawn):
    owner = spawn("start")
    tree = [owner, *owner.children(recursive=True)]
    assert _stop([_node_of(owner)], timeout_s=10) == []
    assert _alive(tree) == []


def test_kills_a_node_that_ignores_sigterm_and_its_workers_groups(spawn):
    owner = spawn("start", on_sigterm="ignore")
    tree = [owner, *owner.children(recursive=True)]
    started = time.monotonic()
    assert _stop([_node_of(owner)], timeout_s=1) == []
    assert _alive(tree) == []
    assert time.monotonic() - started < 30


def test_imports_no_ray():
    code = "import sys, modelship.deploy.local_nodes; assert 'ray' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_kill_groups_spares_processes_older_than_since():
    proc = subprocess.Popen(["sleep", "600"], process_group=0)
    sleeper = psutil.Process(proc.pid)
    try:
        assert os.getpgid(proc.pid) == proc.pid
        kill_groups({proc.pid}, since=sleeper.create_time() + 1)
        assert _alive([sleeper]) == [sleeper]
        kill_groups({proc.pid}, since=0)
        assert _alive([sleeper]) == []
    finally:
        proc.kill()
        proc.wait()
