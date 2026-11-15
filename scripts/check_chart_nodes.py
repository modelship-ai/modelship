#!/usr/bin/env python3
"""CI check: the chart's Ray pods run `mship start`/`mship join` under KubeRay's
overwrite-container-cmd, with worker sizing env derived from the pod's resources, a
fixed metrics port, one Redis namespace for Ray and modelship, a head that turns Ready
only once its gateway answers, and workers that are Ready only while their Serve proxy answers."""

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


def _pod_specs(cluster: dict) -> tuple[dict, list[dict]]:
    spec = cluster["spec"]
    return spec["headGroupSpec"]["template"]["spec"], [g["template"]["spec"] for g in spec["workerGroupSpecs"]]


def _check_variants() -> None:
    cluster = _cluster(
        {
            "image": {"variant": "cpu"},
            "workerGroups": [
                _group("default"),
                _group("vulkan", image={"variant": "vulkan"}),
                _group("cuda", image={"variant": "cuda"}),
                _group("thin", image={"variant": "thin"}),
            ],
        }
    )
    tag = yaml.safe_load((_CHART_DIR / "values.yaml").read_text())["image"]["tag"]
    head, workers = _containers(cluster)
    assert head["image"] == f"ghcr.io/modelship-ai/modelship:{tag}", head["image"]
    tags = [w["image"].rsplit(":", 1)[1] for w in workers]
    assert tags == [f"{tag}-cpu", f"{tag}-vulkan", f"{tag}-cuda", tag], tags
    for values in (
        {"image": {"variant": "vulkun"}, "workerGroups": [_group("w")]},
        {"workerGroups": [_group("w", image={"variant": "rocm"})]},
        {"head": {"image": {"variant": "gpu"}}},
    ):
        out = _render(values)
        assert out.returncode != 0 and "is not one of cuda, cpu, vulkan, thin" in out.stderr, out.stderr


def _check_security_contexts() -> None:
    base = {"runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000}
    head, (plain, gpu) = _pod_specs(
        _cluster({"workerGroups": [_group("plain"), _group("gpu", podSecurityContext={"supplementalGroups": [992]})]})
    )
    assert head["securityContext"] == plain["securityContext"] == base, (head, plain)
    assert gpu["securityContext"] == base | {"supplementalGroups": [992]}, gpu["securityContext"]

    shared = {"podSecurityContext": base | {"supplementalGroups": [992]}}
    head, (plain, own) = _pod_specs(
        _cluster(
            shared | {"workerGroups": [_group("plain"), _group("own", podSecurityContext={"supplementalGroups": [44]})]}
        )
    )
    assert head["securityContext"] == plain["securityContext"] == base | {"supplementalGroups": [992]}
    assert own["securityContext"] == base | {"supplementalGroups": [44]}, own["securityContext"]


def main() -> int:
    cluster = _cluster(
        {
            "workerGroups": [
                _group("req", resources={"requests": {"cpu": "4", "memory": "8Gi"}}),
                _group("lim", resources={"limits": {"cpu": "2"}}),
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
    assert head["args"][:6] == [
        "start",
        "--gcs-port",
        "$(RAY_PORT)",
        "--ray-dashboard-host",
        "0.0.0.0",
        "--enable-ray-auth",
    ], head["args"]
    assert head["args"][6:] == [
        "--gateway-name",
        "modelship",
        "--gateway-min-replicas",
        "1",
        "--gateway-max-replicas",
        "4",
        "--gateway-target-ongoing-requests",
        "64",
        "--gateway-max-ongoing-requests",
        "1024",
        "--openai-api-port",
        "8000",
        "--metrics-port",
        "8079",
    ], head["args"]
    for worker in (req, lim, own, bare):
        assert worker["args"] == ["join", "--gcs-address", "$(RAY_ADDRESS)", "--metrics-port", "8079"], worker["args"]
    assert head["startupProbe"] == {
        "httpGet": {"path": "/modelship/health", "port": "serve"},
        "periodSeconds": 5,
        "timeoutSeconds": 5,
        "failureThreshold": 120,
    }, head["startupProbe"]
    # KubeRay injects only the probes left unset.
    assert not {"readinessProbe", "livenessProbe"} & head.keys()
    for worker in (req, lim, own, bare):
        assert worker["readinessProbe"] == {
            "httpGet": {"path": "/-/healthz", "port": "serve"},
            "periodSeconds": 2,
            "timeoutSeconds": 2,
            "failureThreshold": 2,
        }, worker["readinessProbe"]
        assert not {"startupProbe", "livenessProbe"} & worker.keys()
    slugged, _ = _containers(_cluster({"gateway": {"name": "Edge+API"}, "workerGroups": [_group("w")]}))
    assert slugged["startupProbe"]["httpGet"]["path"] == "/edge-api/health", slugged["startupProbe"]
    assert all("lifecycle" not in c for c in (head, req, lim, own, bare))
    assert all(_ports(c)["metrics"] == 8079 for c in (head, req, lim, own, bare))

    assert _field_ref(req, "MSHIP_NODE_NUM_CPUS") == "requests.cpu"
    assert _field_ref(lim, "MSHIP_NODE_NUM_CPUS") == "limits.cpu"
    assert [e["name"] for e in own["env"]].count("MSHIP_NODE_NUM_CPUS") == 1
    assert _env(own)["MSHIP_NODE_NUM_CPUS"]["value"] == "1"
    assert "MSHIP_NODE_NUM_CPUS" not in _env(bare).keys() | _env(head).keys()

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
    assert worker["args"] == ["join", "--gcs-address", "$(RAY_ADDRESS)"], worker["args"]
    assert _ports(head)["metrics"] == _ports(worker)["metrics"] == 8079
    out = _render({"metrics": {"enabled": False}, "podMonitor": {"enabled": True}})
    assert out.returncode != 0 and "podMonitor.enabled needs metrics.enabled" in out.stderr, out.stderr

    for values in ({}, {"workerGroups": None}):
        assert "workerGroupSpecs" not in _cluster(values)["spec"]

    _check_variants()
    _check_security_contexts()

    print(
        "OK: head runs mship start and waits for its gateway, workers run mship join and are Ready only while "
        "their Serve proxy answers, sizing env follows pod resources, variants map to image tags, worker groups "
        "extend the pod security context"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
