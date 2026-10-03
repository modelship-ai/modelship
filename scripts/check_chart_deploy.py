#!/usr/bin/env python3
"""CI check: each chart revision deploys through its own Job and ConfigMap (not Helm hooks) running
`mship deploy --ray-dashboard-url` against the head Service, with models.yaml mounted byte for byte,
the Ray token from its Secret, and a podFailurePolicy that fails the Job only on exits 1 and 2."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "modelship"

_MODELS_YAML = """models:
  - name: "qwen"   # 'single' "double" $HOME `date` \\back\\slash ; && | > ünïcode
    model: "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
    loader: llama_server
"""


def _render(values: dict) -> dict[str, dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"redis": {"address": "cache:6379"}} | values, f, allow_unicode=True)
    out = subprocess.run(
        ["helm", "template", "rel", str(_CHART_DIR), "-f", f.name], capture_output=True, text=True, check=True
    )
    return {f"{d['kind']}/{d['metadata']['name']}": d for d in yaml.safe_load_all(out.stdout) if d}


def main() -> int:
    docs = _render(
        {
            "models": {"config": _MODELS_YAML},
            "gateway": {"name": "LLM API"},
            "deploy": {"reconcile": True, "extraArgs": ["--replace-strategy", "stop_start"]},
        }
    )
    assert not any(k.startswith("RayJob/") for k in docs), sorted(docs)
    config_map, job = docs["ConfigMap/rel-deploy-1"], docs["Job/rel-deploy-1"]
    for doc in (config_map, job):
        assert "helm.sh/hook" not in doc["metadata"].get("annotations", {}), doc["metadata"]
        assert doc["metadata"]["labels"]["app.kubernetes.io/component"] == "deploy"
    assert config_map["data"] == {"models.yaml": _MODELS_YAML}, config_map["data"]

    spec = job["spec"]
    assert spec["backoffLimit"] == 3
    assert "ttlSecondsAfterFinished" not in spec
    assert spec["podFailurePolicy"]["rules"] == [
        {"action": "FailJob", "onExitCodes": {"operator": "In", "values": [1, 2]}},
        {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]},
    ], spec["podFailurePolicy"]
    pod = spec["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    (container,) = pod["containers"]
    assert container["args"] == [
        "deploy",
        "--ray-dashboard-url",
        "http://rel-head-svc:8265",
        "--config",
        "/etc/mship/models.yaml",
        "--gateway-name",
        "LLM API",
        "--replace-strategy",
        "blue_green",
        "--wait",
        "--reconcile",
        "--replace-strategy",
        "stop_start",
    ], container["args"]
    head = docs["RayCluster/rel"]["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    assert container["image"] == head["image"]
    assert container["env"] == [
        {"name": "MSHIP_RAY_AUTH_TOKEN", "valueFrom": {"secretKeyRef": {"name": "rel", "key": "auth_token"}}}
    ], container["env"]
    assert container["volumeMounts"] == [{"name": "models", "mountPath": "/etc/mship", "readOnly": True}]
    assert pod["volumes"] == [{"name": "models", "configMap": {"name": "rel-deploy-1"}}], pod["volumes"]

    job = _render({"rayAuth": {"existingSecret": "ray-token"}, "deploy": {"ttlSecondsAfterFinished": 30}})[
        "Job/rel-deploy-1"
    ]
    assert job["spec"]["ttlSecondsAfterFinished"] == 30
    token = job["spec"]["template"]["spec"]["containers"][0]["env"][0]["valueFrom"]["secretKeyRef"]
    assert token == {"name": "ray-token", "key": "auth_token"}, token

    print("OK: each revision deploys through its own Job running `mship deploy --ray-dashboard-url`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
