"""Deployment teardown, kept free of serve_utils' gateway and probe imports."""

import contextlib
import time

from ray import serve

from modelship.logging import get_logger
from modelship.utils.config_schema import parse_deployment_name

logger = get_logger("startup")

_POLL_S = 1.0


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
        time.sleep(_POLL_S)
        with contextlib.suppress(Exception):
            present = serve.status().applications
            left = [name for name in left if name in present]
    if left:
        logger.warning("%d deployment(s) still being removed after %.0f s: %s", len(left), timeout_s, ", ".join(left))
