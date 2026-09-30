"""The gateway's autoscaling settings, read from the environment."""

from __future__ import annotations

import os

GATEWAY_SIZING_ENV_VARS = (
    "MSHIP_GATEWAY_MIN_REPLICAS",
    "MSHIP_GATEWAY_MAX_REPLICAS",
    "MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS",
    "MSHIP_GATEWAY_MAX_ONGOING_REQUESTS",
)


def gateway_sizing() -> dict:
    """The gateway deployment's autoscaling_config and max_ongoing_requests, from this process's env."""
    min_replicas = _positive_number_env("MSHIP_GATEWAY_MIN_REPLICAS", 1, int)
    max_replicas = _positive_number_env("MSHIP_GATEWAY_MAX_REPLICAS", 4, int)
    if max_replicas < min_replicas:
        raise ValueError(
            f"MSHIP_GATEWAY_MAX_REPLICAS ({max_replicas}) must be >= MSHIP_GATEWAY_MIN_REPLICAS ({min_replicas})"
        )
    return {
        "autoscaling_config": {
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "target_ongoing_requests": _positive_number_env("MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS", 64.0, float),
        },
        "max_ongoing_requests": _positive_number_env("MSHIP_GATEWAY_MAX_ONGOING_REQUESTS", 1024, int),
    }


def _positive_number_env[T: (int, float)](name: str, default: T, kind: type[T]) -> T:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive {'integer' if kind is int else 'number'}, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {raw!r}")
    return value
