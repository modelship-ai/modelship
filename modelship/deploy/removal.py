"""Deployment teardown, kept free of serve_utils' gateway and probe imports."""

from ray import serve

from modelship.logging import get_logger

logger = get_logger("startup")


def delete_apps_quietly(app_names) -> None:
    """Best-effort serve.delete for cleanup paths — never raises."""
    for name in app_names:
        try:
            logger.info("Deleting deployment: %s", name)
            serve.delete(name)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)
