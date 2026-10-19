"""This machine's modelship nodes and the process groups of their Ray workers."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

import psutil

from modelship.logging import get_logger

logger = get_logger("startup")

_ROLES = {"join": "worker", "start": "head"}


@dataclass(frozen=True)
class LocalNode:
    role: str
    owner: psutil.Process
    raylet: psutil.Process


def _role(argv: list[str]) -> str | None:
    for i in range(len(argv) - 2):
        if argv[i : i + 2] == ["-m", "modelship.launcher"]:
            return _ROLES.get(argv[i + 2])
    return None


def local_nodes() -> tuple[list[LocalNode], list[psutil.Process]]:
    """This machine's modelship nodes, workers first, and the raylets no `mship start` or `mship join` owns."""
    nodes: list[LocalNode] = []
    foreign: list[psutil.Process] = []
    for raylet in psutil.process_iter(["name"]):
        if raylet.info["name"] != "raylet":
            continue
        try:
            owner = raylet.parent()
            role = _role(owner.cmdline()) if owner else None
        except psutil.Error:
            continue
        if owner is None or role is None:
            foreign.append(raylet)
        else:
            nodes.append(LocalNode(role, owner, raylet))
    return sorted(nodes, key=lambda node: node.role != "worker"), foreign


def worker_groups(raylet: psutil.Process) -> set[int]:
    """Process groups led by the raylet's workers."""
    groups = set()
    for worker in raylet.children():
        with contextlib.suppress(OSError):
            if os.getpgid(worker.pid) == worker.pid:
                groups.add(worker.pid)
    return groups


def kill_groups(groups: set[int], since: float) -> list[psutil.Process]:
    """SIGTERM, then SIGKILL, every process in *groups* started at or after *since*."""
    members = []
    for proc in psutil.process_iter(["create_time"]):
        with contextlib.suppress(OSError):
            if os.getpgid(proc.pid) in groups and (proc.info["create_time"] or 0) >= since:
                members.append(proc)
    return _terminate(members)


def _running(procs: list[psutil.Process]) -> list[psutil.Process]:
    alive = []
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                alive.append(proc)
    return alive


def _wait(procs: list[psutil.Process], timeout_s: float) -> list[psutil.Process]:
    deadline = time.monotonic() + timeout_s
    while (alive := _running(procs)) and time.monotonic() < deadline:
        time.sleep(0.2)
    return alive


def _terminate(procs: list[psutil.Process]) -> list[psutil.Process]:
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            proc.terminate()
    alive = _wait(procs, 5)
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    return _wait(alive, 5)


def _stop(nodes: list[LocalNode], timeout_s: float) -> list[psutil.Process]:
    trees = []
    for node in nodes:
        logger.info("Stopping the %s node (pid %s)...", node.role, node.owner.pid)
        with contextlib.suppress(psutil.Error):
            trees += [node.owner, *node.owner.children(recursive=True)]
        with contextlib.suppress(psutil.Error):
            node.owner.terminate()
    hung = _wait([node.owner for node in nodes], timeout_s)
    for node in nodes:
        if node.owner not in hung:
            continue
        logger.warning(
            "The %s node (pid %s) did not stop within %ss; killing it.", node.role, node.owner.pid, timeout_s
        )
        with contextlib.suppress(psutil.Error):
            survivors = kill_groups(worker_groups(node.raylet), node.owner.create_time())
            trees += [proc for proc in survivors if proc not in trees]
        with contextlib.suppress(psutil.Error):
            node.owner.kill()
    return _terminate(_wait(trees, 10))


def stop_local_nodes(timeout_s: float) -> int:
    """Stops every modelship node on this machine, workers first; 0 once none of their processes is left."""
    nodes, foreign = local_nodes()
    for raylet in foreign:
        logger.warning(
            "Leaving the Ray node with raylet pid %s alone: neither `mship start` nor `mship join` started it.",
            raylet.pid,
        )
    if not nodes:
        logger.info("No modelship node is running on this machine.")
        return 0
    survivors = []
    for role in ("worker", "head"):
        if same_role := [node for node in nodes if node.role == role]:
            survivors += _stop(same_role, timeout_s)
    for proc in survivors:
        with contextlib.suppress(psutil.Error):
            logger.error("Still running: pid %s (%s)", proc.pid, proc.name())
    if survivors:
        return 1
    logger.info("Stopped.")
    return 0
