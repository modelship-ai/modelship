"""Ray cluster-auth env resolution, deliberately free of any Ray import."""

from __future__ import annotations

import os


def resolve_ray_auth_env() -> None:
    """Translate MSHIP_RAY_AUTH/MSHIP_RAY_AUTH_TOKEN into Ray's own
    RAY_AUTH_MODE/RAY_AUTH_TOKEN. Must run after argv is folded into the MSHIP_*
    vars (apply_args_to_env) and before the first `import ray` — Ray's
    RAY_AUTH_MODE check latches at import time, so setting it later has no
    effect on this process's own ray.init()/Node() calls."""
    token = os.environ.get("MSHIP_RAY_AUTH_TOKEN")
    if token or os.environ.get("MSHIP_RAY_AUTH", "none").lower() == "token":
        os.environ.setdefault("RAY_AUTH_MODE", "token")
    if token:
        os.environ.setdefault("RAY_AUTH_TOKEN", token)
