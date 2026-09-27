"""Deployment teardown, kept free of serve_utils' gateway and probe imports."""

import contextlib
import time

import ray
from ray import serve
from ray.exceptions import GetTimeoutError, RayActorError

from modelship.logging import get_logger
from modelship.utils.config_schema import parse_deployment_name

logger = get_logger("startup")

# Covers the gateway coordinator's 10 s grace period plus Serve's graceful replica shutdown.
_RETIRE_TIMEOUT_S = 60.0
# Pause before asking a restarting gateway coordinator again.
_RETRY_INTERVAL_S = 1.0


def delete_model_apps(timeout_s: float) -> None:
    """Deletes every model app without blocking on Serve, then waits up to *timeout_s* for them to go."""
    try:
        left = [name for name in serve.status().applications if parse_deployment_name(name) is not None]
    except Exception:
        logger.exception("Could not list the model deployments to delete")
        return
    for name in left:
        try:
            serve.delete(name, _blocking=False)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)
    deadline = time.monotonic() + timeout_s
    while left and time.monotonic() < deadline:
        time.sleep(_RETRY_INTERVAL_S)
        with contextlib.suppress(Exception):
            present = serve.status().applications
            left = [name for name in left if name in present]
    if left:
        logger.warning("%d deployment(s) still being removed after %.0f s: %s", len(left), timeout_s, ", ".join(left))


def delete_apps_quietly(app_names) -> None:
    """Best-effort serve.delete for cleanup paths — never raises."""
    for name in app_names:
        try:
            logger.info("Deleting deployment: %s", name)
            serve.delete(name)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)


def wait_for_retired_apps(gateway_coordinator, gateway_name: str) -> None:
    """Waits up to `_RETIRE_TIMEOUT_S` for the gateway coordinator to delete this gateway's
    unused apps, warning about any it hasn't."""
    deadline = time.monotonic() + _RETIRE_TIMEOUT_S
    left: list[str] | None = None
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            retiring = ray.get(gateway_coordinator.get_retiring.remote(gateway_name), timeout=remaining)
        except GetTimeoutError:
            break
        except RayActorError:
            time.sleep(_RETRY_INTERVAL_S)
            continue
        if not retiring:
            return
        if left is None:
            logger.info(
                "Waiting up to %.0f s for %d retired deployment(s) to be removed: %s",
                _RETIRE_TIMEOUT_S,
                len(retiring),
                ", ".join(retiring),
            )
        left = retiring
    if left:
        logger.warning(
            "%d retired deployment(s) still being removed after %.0f s: %s",
            len(left),
            _RETIRE_TIMEOUT_S,
            ", ".join(left),
        )
    else:
        logger.warning("Could not confirm that retired deployments were removed within %.0f s", _RETIRE_TIMEOUT_S)
