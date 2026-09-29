"""Env vars forwarded into an actor's runtime_env: read in the actor's process, set only
on the process that creates it. Never secrets — runtime_env is plain-text cluster metadata.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Every modelship actor's logging and metrics, set once by `mship start`; an empty OTLP endpoint exports nothing.
CLUSTER_ENV_DEFAULTS = {
    "MSHIP_LOG_LEVEL": "INFO",
    "MSHIP_LOG_FORMAT": "text",
    "MSHIP_LOG_TARGET": "console",
    "MSHIP_METRICS": "true",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "",
}

MODEL_ENV_VARS = ("MSHIP_GATEWAY_NAME", "MSHIP_PREFLIGHT")

DEPLOY_COORDINATOR_ENV_VARS = ("MSHIP_DEPLOY_SWITCH_TIMEOUT_S", "MSHIP_DEPLOY_CANCEL_GRACE_S")

GATEWAY_ENV_VARS = (
    "MSHIP_GATEWAY_NAME",
    "MSHIP_TRUSTED_IDENTITY_HEADER",
    "MSHIP_MAX_REQUEST_BODY_BYTES",
    "MSHIP_MCP_ALLOWED_HOSTS",
    "MSHIP_MCP_REQUIRE_HTTPS",
    "MSHIP_RESPONSES_TTL_S",
    "MSHIP_RESPONSES_STALE_S",
    "MSHIP_RESPONSES_STREAM_BUFFER_TTL_S",
)

MEMORY_STORE_ENV_VARS = ("MSHIP_STATE_SWEEP_INTERVAL_S",)


def build_env_vars(names: Iterable[str]) -> dict[str, str]:
    """The creating process's values for *names*, skipping the unset ones."""
    return {name: os.environ[name] for name in names if os.environ.get(name) is not None}


def cluster_env_vars() -> dict[str, str]:
    """The creating process's logging and metrics settings, the defaults included."""
    return {name: cluster_env_value(name) for name in CLUSTER_ENV_DEFAULTS}


def cluster_env_value(name: str) -> str:
    """This process's value for the cluster setting *name*, or its default."""
    return os.environ.get(name) or CLUSTER_ENV_DEFAULTS[name]
