"""Env vars forwarded into an actor's runtime_env: read in the actor's process, set only
on the process that creates it. Never secrets — runtime_env is plain-text cluster metadata.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Every modelship actor: its logging and metrics.
COMMON_ENV_VARS = (
    "MSHIP_LOG_LEVEL",
    "MSHIP_LOG_FORMAT",
    "MSHIP_LOG_TARGET",
    "MSHIP_METRICS",
)

MODEL_ENV_VARS = (*COMMON_ENV_VARS, "MSHIP_GATEWAY_NAME", "MSHIP_PREFLIGHT")

GATEWAY_ENV_VARS = (
    *COMMON_ENV_VARS,
    "MSHIP_GATEWAY_NAME",
    "MSHIP_TRUSTED_IDENTITY_HEADER",
    "MSHIP_MAX_REQUEST_BODY_BYTES",
    "MSHIP_MCP_ALLOWED_HOSTS",
    "MSHIP_MCP_REQUIRE_HTTPS",
    "MSHIP_RESPONSES_TTL_S",
    "MSHIP_RESPONSES_STALE_S",
    "MSHIP_RESPONSES_STREAM_BUFFER_TTL_S",
)

MEMORY_STORE_ENV_VARS = (*COMMON_ENV_VARS, "MSHIP_STATE_SWEEP_INTERVAL_S")


def build_env_vars(names: Iterable[str]) -> dict[str, str]:
    """The creating process's values for *names*, skipping the unset ones."""
    return {name: os.environ[name] for name in names if os.environ.get(name) is not None}
