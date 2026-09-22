#!/usr/bin/env python3
"""CI check: the chart's Ray pods run `mship start`/`mship join` under KubeRay's
overwrite-container-cmd, with worker sizing env derived from the pod's resources."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "modelship"


def _cluster(values: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"redis": {"address": "cache:6379"}} | values, f)
    out = subprocess.run(
        ["helm", "template", "modelship", str(_CHART_DIR), "-f", f.name], capture_output=True, text=True, check=True
    )
    return next(d for d in yaml.safe_load_all(out.stdout) if d and d.get("kind") == "RayCluster")


def _group(name: str, **extra) -> dict:
    return {"name": name, "replicas": 1, "minReplicas": 1, "maxReplicas": 1, **extra}


def _containers(cluster: dict) -> tuple[dict, list[dict]]:
    spec = cluster["spec"]
    head = spec["headGroupSpec"]["template"]["spec"]["containers"][0]
    return head, [g["template"]["spec"]["containers"][0] for g in spec["workerGroupSpecs"]]


def _env(container: dict) -> dict[str, dict]:
    return {e["name"]: e for e in container.get("env", [])}


def _field_ref(container: dict, name: str) -> str | None:
    entry = _env(container).get(name)
    if not entry or "valueFrom" not in entry:
        return None
    return entry["valueFrom"]["resourceFieldRef"]["resource"]


def _port_names(container: dict) -> set[str]:
    return {p["name"] for p in container["ports"]}


def main() -> int:
    cluster = _cluster(
        {
            "workerGroups": [
                _group("req", resources={"requests": {"cpu": "4", "memory": "8Gi"}}),
                _group("lim", resources={"limits": {"cpu": "2"}}, env=[{"name": "MSHIP_NODE_MEMORY", "value": "4Gi"}]),
                _group(
                    "own",
                    resources={"limits": {"cpu": "2", "memory": "4Gi"}},
                    env=[{"name": "MSHIP_NODE_NUM_CPUS", "value": "1"}],
                ),
                _group("bare"),
            ]
        }
    )
    head, (req, lim, own, bare) = _containers(cluster)
    assert cluster["metadata"]["annotations"]["ray.io/overwrite-container-cmd"] == "true"
    assert cluster["metadata"]["finalizers"] == ["ray.io/gcs-ft-redis-cleanup-finalizer"]
    assert cluster["spec"]["headGroupSpec"]["rayStartParams"] == {}
    assert all(g["rayStartParams"] == {} for g in cluster["spec"]["workerGroupSpecs"])
    assert head["args"][:5] == ["start", "--ray-port", "$(RAY_PORT)", "--ray-dashboard-host", "0.0.0.0"], head["args"]
    assert head["args"][5:] == [
        "--gateway-name",
        "modelship",
        "--gateway-replicas",
        "1",
        "--openai-api-port",
        "8000",
        "--metrics-port",
        "8079",
    ], head["args"]
    for worker in (req, lim, own, bare):
        assert worker["args"] == ["join", "--cluster", "$(RAY_ADDRESS)", "--metrics-port", "8079"], worker["args"]
    assert all("lifecycle" not in c for c in (head, req, lim, own, bare))
    assert all("metrics" in _port_names(c) for c in (head, req, lim, own, bare))

    assert _field_ref(req, "MSHIP_NODE_NUM_CPUS") == "requests.cpu"
    assert _field_ref(req, "MSHIP_NODE_MEMORY") == "requests.memory"
    assert _field_ref(lim, "MSHIP_NODE_NUM_CPUS") == "limits.cpu"
    assert _env(lim)["MSHIP_NODE_MEMORY"] == {"name": "MSHIP_NODE_MEMORY", "value": "4Gi"}
    assert [e["name"] for e in own["env"]].count("MSHIP_NODE_NUM_CPUS") == 1
    assert _env(own)["MSHIP_NODE_NUM_CPUS"]["value"] == "1"
    assert _field_ref(own, "MSHIP_NODE_MEMORY") == "limits.memory"
    assert not {"MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_MEMORY"} & (_env(bare).keys() | _env(head).keys())

    head, (worker,) = _containers(_cluster({"metrics": {"enabled": False}, "workerGroups": [_group("w")]}))
    assert head["args"][-1] == "--no-metrics" and "--metrics-port" not in head["args"], head["args"]
    assert worker["args"] == ["join", "--cluster", "$(RAY_ADDRESS)"], worker["args"]
    assert "metrics" not in _port_names(head) | _port_names(worker)

    print("OK: head runs mship start, workers run mship join, sizing env follows pod resources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
