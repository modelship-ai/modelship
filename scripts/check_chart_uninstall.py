#!/usr/bin/env python3
"""CI check: the chart's pre-delete hook runs `modelship.uninstall` on the release's own
RayCluster and RayJob, with the same Redis namespace and credentials as the Ray pods."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

_CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "modelship"
_HOOK_ANNOTATIONS = {"helm.sh/hook": "pre-delete", "helm.sh/hook-delete-policy": "before-hook-creation,hook-succeeded"}


def _render(*args: str) -> dict[str, dict]:
    out = subprocess.run(
        ["helm", "template", "rel", str(_CHART_DIR), "--set", "redis.address=cache:6379", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return {f"{d['kind']}/{d['metadata']['name']}": d for d in yaml.safe_load_all(out.stdout) if d}


def _env(container: dict) -> dict[str, dict]:
    return {e["name"]: e for e in container["env"]}


def main() -> int:
    docs = _render()
    hook = {k: d for k, d in docs.items() if d["metadata"].get("annotations", {}).get("helm.sh/hook") == "pre-delete"}
    assert sorted(hook) == [
        "Job/rel-modelship-uninstall",
        "Role/rel-modelship-uninstall",
        "RoleBinding/rel-modelship-uninstall",
        "ServiceAccount/rel-modelship-uninstall",
    ], sorted(hook)
    for key, doc in hook.items():
        annotations = doc["metadata"]["annotations"]
        assert annotations.items() >= _HOOK_ANNOTATIONS.items(), (key, annotations)
        assert int(annotations["helm.sh/hook-weight"]) == (0 if key.startswith("Job/") else -10), (key, annotations)

    assert hook["Role/rel-modelship-uninstall"]["rules"] == [
        {
            "apiGroups": ["ray.io"],
            "resources": ["rayclusters"],
            "resourceNames": ["rel-modelship"],
            "verbs": ["get", "delete"],
        },
        {
            "apiGroups": ["ray.io"],
            "resources": ["rayjobs"],
            "resourceNames": ["rel-modelship-deploy"],
            "verbs": ["delete"],
        },
    ]
    pod = hook["Job/rel-modelship-uninstall"]["spec"]["template"]["spec"]
    job = pod["containers"][0]
    assert pod["serviceAccountName"] == "rel-modelship-uninstall"
    assert job["command"] == ["python", "-m", "modelship.uninstall", "rel-modelship", "rel-modelship-deploy"]

    cluster = docs["RayCluster/rel-modelship"]
    head = cluster["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    assert job["image"] == head["image"]
    assert _env(job)["MSHIP_STATE_STORE"] == _env(head)["MSHIP_STATE_STORE"]
    namespace = cluster["spec"]["gcsFaultToleranceOptions"]["externalStorageNamespace"]
    assert _env(job)["MSHIP_STATE_STORE"]["value"].endswith(f"?namespace={namespace}")
    assert "MSHIP_REDIS_PASSWORD" not in _env(job)

    job = _render("--set", "redis.password=pw")["Job/rel-modelship-uninstall"]["spec"]["template"]["spec"]
    secret_ref = _env(job["containers"][0])["MSHIP_REDIS_PASSWORD"]["valueFrom"]["secretKeyRef"]
    assert secret_ref == {"name": "rel-modelship-secrets", "key": "REDIS_PASSWORD"}, secret_ref

    print("OK: the pre-delete hook deletes this release's RayCluster and RayJob, then its state namespace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
