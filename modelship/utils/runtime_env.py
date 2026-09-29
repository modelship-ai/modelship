"""Env vars forwarded into an actor's runtime_env: read in the actor's process, set only
on the process that creates it. Never secrets — runtime_env is plain-text cluster metadata.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Set once by `mship start` and forwarded to every modelship actor.
CLUSTER_ENV_DEFAULTS = {"MSHIP_LOG_FORMAT": "text", "MSHIP_METRICS": "true"}

# Set by each node's `mship start` or `mship join`; never forwarded, so actors inherit their node's.
# An empty OTLP endpoint exports nothing.
NODE_ENV_DEFAULTS = {"MSHIP_LOG_LEVEL": "INFO", "MSHIP_LOG_TARGET": "console", "OTEL_EXPORTER_OTLP_ENDPOINT": ""}

_ENV_DEFAULTS = CLUSTER_ENV_DEFAULTS | NODE_ENV_DEFAULTS

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
    """The creating process's cluster-wide settings, the defaults included."""
    return {name: env_setting(name) for name in CLUSTER_ENV_DEFAULTS}


def env_setting(name: str) -> str:
    """This process's value for the cluster or node setting *name*, or its default when unset."""
    return os.environ.get(name, _ENV_DEFAULTS[name])
