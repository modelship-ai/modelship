#!/usr/bin/env python3
"""CI check: the chart's Ray pods run `mship start`/`mship join` under KubeRay's
overwrite-container-cmd, with worker sizing env derived from the pod's resources, a
fixed metrics port and one Redis namespace for Ray and modelship."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "modelship"


def _render(values: dict) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"redis": {"address": "cache:6379"}} | values, f)
    return subprocess.run(
        ["helm", "template", "modelship", str(_CHART_DIR), "-f", f.name], capture_output=True, text=True
    )


def _cluster(values: dict) -> dict:
    out = _render(values)
    assert out.returncode == 0, out.stderr
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


def _ports(container: dict) -> dict[str, int]:
    return {p["name"]: p["containerPort"] for p in container["ports"]}


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
    assert all(_ports(c)["metrics"] == 8079 for c in (head, req, lim, own, bare))

    assert _field_ref(req, "MSHIP_NODE_NUM_CPUS") == "requests.cpu"
    assert _field_ref(req, "MSHIP_NODE_MEMORY") == "requests.memory"
    assert _field_ref(lim, "MSHIP_NODE_NUM_CPUS") == "limits.cpu"
    assert _env(lim)["MSHIP_NODE_MEMORY"] == {"name": "MSHIP_NODE_MEMORY", "value": "4Gi"}
    assert [e["name"] for e in own["env"]].count("MSHIP_NODE_NUM_CPUS") == 1
    assert _env(own)["MSHIP_NODE_NUM_CPUS"]["value"] == "1"
    assert _field_ref(own, "MSHIP_NODE_MEMORY") == "limits.memory"
    assert not {"MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_MEMORY"} & (_env(bare).keys() | _env(head).keys())

    assert cluster["spec"]["gcsFaultToleranceOptions"]["externalStorageNamespace"] == "modelship"
    for c in (head, req, lim, own, bare):
        assert _env(c)["MSHIP_STATE_STORE"]["value"] == "redis://cache:6379/0?namespace=modelship"
    custom = _cluster({"redis": {"address": "cache:6379", "externalStorageNamespace": "edge.a_1"}})
    assert custom["spec"]["gcsFaultToleranceOptions"]["externalStorageNamespace"] == "edge.a_1"
    custom_head = custom["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    assert _env(custom_head)["MSHIP_STATE_STORE"]["value"].endswith("?namespace=edge.a_1")
    out = _render({"redis": {"address": "cache:6379", "externalStorageNamespace": "a/b"}})
    assert out.returncode != 0 and "must be letters, digits" in out.stderr, out.stderr

    head, (worker,) = _containers(_cluster({"metrics": {"enabled": False}, "workerGroups": [_group("w")]}))
    assert head["args"][-1] == "--no-metrics" and "--metrics-port" not in head["args"], head["args"]
    assert worker["args"] == ["join", "--cluster", "$(RAY_ADDRESS)"], worker["args"]
    assert _ports(head)["metrics"] == _ports(worker)["metrics"] == 8079
    out = _render({"metrics": {"enabled": False}, "podMonitor": {"enabled": True}})
    assert out.returncode != 0 and "podMonitor.enabled needs metrics.enabled" in out.stderr, out.stderr

    for values in ({}, {"workerGroups": None}):
        assert "workerGroupSpecs" not in _cluster(values)["spec"]

    print("OK: head runs mship start, workers run mship join, sizing env follows pod resources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
