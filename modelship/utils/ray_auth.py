"""Ray cluster-auth env resolution, deliberately free of any Ray import."""

from __future__ import annotations

import ipaddress
import os


def auth_enabled() -> bool:
    return os.environ.get("MSHIP_RAY_AUTH", "false").lower() == "true"


def token_env_without_auth() -> list[str]:
    """The token-auth settings in the env while MSHIP_RAY_AUTH is off."""
    if auth_enabled():
        return []
    found = [name for name in ("MSHIP_RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN") if os.environ.get(name)]
    if os.environ.get("RAY_AUTH_MODE", "").lower() == "token":
        found.append("RAY_AUTH_MODE=token")
    return found


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_ray_auth_env() -> None:
    """Translate MSHIP_RAY_AUTH/MSHIP_RAY_AUTH_TOKEN into Ray's own
    RAY_AUTH_MODE/RAY_AUTH_TOKEN. Must run after argv is folded into the MSHIP_*
    vars (apply_args_to_env) and before the first `import ray` — Ray's
    RAY_AUTH_MODE check latches at import time, so setting it later has no
    effect on this process's own ray.init()/Node() calls."""
    token = os.environ.get("MSHIP_RAY_AUTH_TOKEN")
    if token or auth_enabled():
        os.environ.setdefault("RAY_AUTH_MODE", "token")
    else:
        # Unset, Ray enables token auth on a new local cluster, and on an attach that finds a token file.
        os.environ.setdefault("RAY_AUTH_MODE", "disabled")
    if token:
        os.environ.setdefault("RAY_AUTH_TOKEN", token)
