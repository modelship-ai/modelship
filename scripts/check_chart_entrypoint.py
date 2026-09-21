#!/usr/bin/env python3
"""CI check: the chart's deploy RayJob entrypoint hands models.yaml to `mship deploy`
byte for byte. Replays the three shells it passes through, with `ray` and `mship`
stubbed: KubeRay's submitter bash (entrypoint pasted unquoted), `ray job submit`
(re-joins its argv with list2cmdline), and the head's job supervisor (shell=True).
"""

from __future__ import annotations

import os
import stat
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

_RAY_STUB = f"""#!{sys.executable}
import sys
from subprocess import list2cmdline
args = sys.argv[sys.argv.index("--") + 1:]
open(sys.argv[0] + ".entrypoint", "w").write(list2cmdline(args))
"""

_MSHIP_STUB = """#!/bin/sh
out="$(dirname "$0")/mship.out"
printf '%s\\n' "$*" > "$out.args"
while [ $# -gt 0 ]; do [ "$1" = --config ] && cp "$2" "$out.config"; shift; done
"""


def _render(values: dict) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(values, f, allow_unicode=True)
    return subprocess.run(
        ["helm", "template", "modelship", str(_CHART_DIR), "-f", f.name], capture_output=True, text=True
    )


def _entrypoint(values: dict) -> str:
    out = _render(values)
    out.check_returncode()
    for doc in yaml.safe_load_all(out.stdout):
        if doc and doc.get("kind") == "RayJob":
            return doc["spec"]["entrypoint"]
    raise RuntimeError("no RayJob in rendered chart output")


def main() -> int:
    base = {"redis": {"address": "cache:6379"}}
    entrypoint = _entrypoint(base | {"models": {"config": _MODELS_YAML}})

    with tempfile.TemporaryDirectory() as tmp:
        bindir = Path(tmp)
        for name, body in (("ray", _RAY_STUB), ("mship", _MSHIP_STUB)):
            stub = bindir / name
            stub.write_text(body)
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        env = os.environ | {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}

        subprocess.run(
            ["bash", "-ce", f"ray job submit --address http://head:8265 -- {entrypoint} ;"], env=env, check=True
        )
        subprocess.run((bindir / "ray.entrypoint").read_text(), shell=True, env=env, check=True)

        args = (bindir / "mship.out.args").read_text().split()
        config = (bindir / "mship.out.config").read_text(encoding="utf-8")
    Path(args[args.index("--config") + 1]).unlink(missing_ok=True)

    if config != _MODELS_YAML:
        print(f"FAIL: mship received a different models.yaml:\n{config}")
        return 1
    if args[:3] != ["deploy", "--gateway-name", "modelship"]:
        print(f"FAIL: unexpected mship arguments: {args}")
        return 1
    if _render(base | {"gateway": {"name": "a b"}}).returncode == 0:
        print("FAIL: a gateway name with a space rendered")
        return 1
    print("OK: models.yaml reaches `mship deploy` intact through the RayJob entrypoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
