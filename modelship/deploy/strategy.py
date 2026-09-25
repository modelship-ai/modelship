import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import ray
from ray import serve
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview, LoggingConfig

from modelship.deploy.actor_options import build_deployment_options
from modelship.deploy.removal import delete_apps_quietly
from modelship.infer.deploy_leases import gateway_lease
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger
from modelship.utils.config_schema import parse_deployment_name

logger = get_logger("startup")

_DEPLOY_RETRY_SLEEP_S = 2.0
_MAX_TRANSIENT_FAILURES = 3
_POLL_SECONDS = 2.0
_PENDING_LOG_EVERY_N_POLLS = 30  # with a 2s poll, log what's outstanding at first, then once a minute
_TIMEOUT_ENV = "MSHIP_DEPLOY_TIMEOUT_S"
_DEFAULT_TIMEOUT_SECONDS = 600.0


@dataclass
class DeployPlan:
    """Result of diffing models.yaml against the cluster."""

    models_to_add: list[ModelshipModelConfig]
    # this gateway's apps no desired model targets: replaced versions and dropped models
    stale_apps: list[str]


# Serve has stopped starting replicas for these; a redeploy replaces the app.
_NOT_LIVE = (ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.DELETING)


def compute_deploy_plan(
    desired_conf: ModelshipConfig,
    app_statuses: dict[str, ApplicationStatus],
    gateway_name: str,
) -> DeployPlan:
    """Diff the desired effective set against what's live. Deployment names are
    `{gateway}.{model}-{fingerprint}`, so a set comparison detects config drift and
    tells this gateway's apps from any other."""
    targets = {c.deployment_name(gateway_name) for c in desired_conf.models}
    stale_apps = sorted(
        name
        for name in app_statuses
        if name not in targets and (parsed := parse_deployment_name(name)) is not None and parsed[0] == gateway_name
    )
    if stale_apps:
        logger.info("%d deployment(s) to retire: %s", len(stale_apps), stale_apps)

    # An app already live under its fingerprint is kept, so re-runs are idempotent.
    models_to_add: list[ModelshipModelConfig] = []
    for c in desired_conf.models:
        status = app_statuses.get(c.deployment_name(gateway_name))
        if status is None or status in _NOT_LIVE:
            models_to_add.append(c)
    if models_to_add:
        logger.info(
            "%d deployment(s) to add: %s",
            len(models_to_add),
            [c.deployment_name(gateway_name) for c in models_to_add],
        )
    return DeployPlan(models_to_add=models_to_add, stale_apps=stale_apps)


@dataclass
class DeployContext:
    coordinator: Any
    gateway_name: str
    serve_logging_config: LoggingConfig
    deployed_this_run: dict[str, str]


@dataclass
class DeployOutcome:
    ready: list[ModelshipModelConfig]
    # each paired with Serve's pending reason (may be empty), or why it failed
    still_pending: list[tuple[ModelshipModelConfig, str]]
    fatally_failed: list[tuple[ModelshipModelConfig, str]]
    # deleted by something other than this deploy after Serve had reported them
    removed_elsewhere: list[ModelshipModelConfig] = field(default_factory=list)


@dataclass
class _Pending:
    config: ModelshipModelConfig
    failures: int = 0
    last_error: str = ""
    # set while waiting out the backoff before being submitted again
    retry_at: float | None = None
    # set once Serve reports the app, not being deleted
    seen: bool = False


def deploy_timeout_seconds() -> float:
    raw = os.environ.get(_TIMEOUT_ENV)
    try:
        return float(raw) if raw else _DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        logger.warning("Ignoring %s=%r, not a number; using %s", _TIMEOUT_ENV, raw, _DEFAULT_TIMEOUT_SECONDS)
        return _DEFAULT_TIMEOUT_SECONDS


def submit_deploy(config: ModelshipModelConfig, ctx: DeployContext) -> None:
    """Hand one model to Serve and return; its replicas come up afterwards, and
    pend rather than fail when the cluster has no room for them yet."""
    deployment_name = config.deployment_name(ctx.gateway_name)
    deploy_opts = build_deployment_options(config)

    # Mutually exclusive, enforced at config validation — pass Serve exactly one.
    if config.autoscaling_config is not None:
        scaling_opts: dict = {"autoscaling_config": config.autoscaling_config.to_serve_dict()}
    else:
        scaling_opts = {"num_replicas": config.num_replicas}

    logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
    ctx.deployed_this_run[deployment_name] = config.name
    serve.run_many(
        [
            serve.RunTarget(
                target=ModelDeployment.options(
                    name=deployment_name,
                    max_constructor_retry_count=1,
                    logging_config=ctx.serve_logging_config,
                    **scaling_opts,
                    **deploy_opts,
                ).bind(config),
                name=deployment_name,
                route_prefix=None,
            )
        ],
        wait_for_applications_running=False,
    )


def _app_statuses() -> dict[str, ApplicationStatusOverview] | None:
    """Serve's application statuses; None when they can't be read."""
    try:
        return dict(serve.status().applications)
    except Exception:
        logger.exception("Could not read Serve status; retrying next poll")
        return None


def run_deploy_loop(
    models: list[ModelshipModelConfig],
    ctx: DeployContext,
) -> DeployOutcome:
    """Submits every model and polls until each is up, failed or out of time. A failed deploy is retried
    with a doubling backoff, up to `_MAX_TRANSIENT_FAILURES` times, unless its replica reported it fatal."""
    pending = {config.deployment_name(ctx.gateway_name): _Pending(config) for config in models}
    ready: list[ModelshipModelConfig] = []
    fatally_failed: list[tuple[ModelshipModelConfig, str]] = []
    removed_elsewhere: list[ModelshipModelConfig] = []
    statuses: dict[str, ApplicationStatusOverview] = {}

    # removal blocks until the app is torn down, so it runs beside the polling
    with ThreadPoolExecutor(thread_name_prefix="deploy-removal") as removals:

        def give_up(name: str, reason: str) -> None:
            ctx.deployed_this_run.pop(name, None)
            fatally_failed.append((pending.pop(name).config, reason))
            removals.submit(_remove_given_up, name, ctx.gateway_name)

        def fail(name: str, message: str) -> None:
            if (reason := _record_failure(name, pending[name], ctx, message)) is not None:
                give_up(name, reason)

        for name, item in list(pending.items()):
            if (error := _submit(item.config, ctx)) is not None:
                fail(name, error)

        deadline = time.monotonic() + deploy_timeout_seconds()
        polls = 0
        while pending and time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            polls += 1
            now = time.monotonic()
            for name, item in list(pending.items()):
                if item.retry_at is not None and now >= item.retry_at:
                    item.retry_at = None
                    if (error := _submit(item.config, ctx)) is not None:
                        fail(name, error)

            if (read := _app_statuses()) is None:
                continue
            statuses = read
            for name, item in list(pending.items()):
                app = statuses.get(name)
                if item.retry_at is not None:
                    continue
                if app is None or app.status == ApplicationStatus.DELETING:
                    if item.seen:
                        logger.warning(
                            "Model '%s' was removed before it came up: deployment %s was deleted",
                            item.config.name,
                            name,
                        )
                        ctx.deployed_this_run.pop(name, None)
                        removed_elsewhere.append(pending.pop(name).config)
                    continue
                item.seen = True
                if app.status == ApplicationStatus.RUNNING:
                    logger.info("Model ready: %s (deployment: %s)", item.config.name, name)
                    ready.append(pending.pop(name).config)
                # UNHEALTHY follows RUNNING while Serve replaces a replica, so it stays pending
                elif app.status == ApplicationStatus.DEPLOY_FAILED:
                    fail(name, app.message)

            if pending and (polls == 1 or polls % _PENDING_LOG_EVERY_N_POLLS == 0):
                _log_pending(pending, statuses)

        for name, item in list(pending.items()):
            if item.retry_at is not None:
                logger.error(
                    "Giving up on model '%s' (deployment=%s): deploy timeout reached before its next attempt",
                    item.config.name,
                    name,
                )
                give_up(name, item.last_error)

    still_pending = [(item.config, _pending_reason(name, statuses)) for name, item in pending.items()]
    return DeployOutcome(ready, still_pending, fatally_failed, removed_elsewhere)


def _submit(config: ModelshipModelConfig, ctx: DeployContext) -> str | None:
    """Submits the app under the gateway's lease unless it is live by then; the error text if that fails."""
    name = config.deployment_name(ctx.gateway_name)
    try:
        with gateway_lease(ctx.gateway_name):
            if _live(name):
                logger.info("%s was submitted by another deploy; waiting for it", name)
                return None
            submit_deploy(config, ctx)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _live(name: str) -> bool:
    """Whether Serve has the app, neither failed nor being deleted."""
    app = serve.status().applications.get(name)
    return app is not None and app.status not in _NOT_LIVE


def _remove_given_up(name: str, gateway_name: str) -> None:
    """Deletes a given-up app under the gateway's lease, unless another deploy has made it live again."""
    try:
        with gateway_lease(gateway_name):
            if not _live(name):
                delete_apps_quietly([name])
    except Exception:
        logger.exception("Could not remove deployment %s", name)


def _record_failure(name: str, item: _Pending, ctx: DeployContext, message: str) -> str | None:
    """None to resubmit after a backoff, else why the model is given up on: a fatal
    error from its replica, or its attempts running out."""
    try:
        fatal_err = ray.get(ctx.coordinator.pop_fatal_error.remote(name), timeout=2.0)
    except Exception:
        fatal_err = None
    if fatal_err is not None:
        logger.error("Skipping model '%s' permanently (deployment=%s): %s", item.config.name, name, fatal_err)
        return str(fatal_err)

    item.failures += 1
    item.last_error = message
    if item.failures >= _MAX_TRANSIENT_FAILURES:
        logger.error(
            "Giving up on model '%s' after %d failed attempt(s) (deployment=%s): %s",
            item.config.name,
            item.failures,
            name,
            message,
        )
        return message

    logger.warning("Deploy failed for %s (deployment=%s); will retry: %s", item.config.name, name, message)
    # resubmitting over the failed app replaces it
    item.retry_at = time.monotonic() + _DEPLOY_RETRY_SLEEP_S * 2**item.failures
    return None


def _pending_reason(name: str, statuses: dict[str, ApplicationStatusOverview]) -> str:
    """Serve's deployment message, else its app message; empty when Serve gives neither."""
    app = statuses.get(name)
    if app is None:
        return ""
    return next((d.message for d in app.deployments.values() if d.message), app.message)


def _log_pending(pending: dict[str, _Pending], statuses: dict[str, ApplicationStatusOverview]) -> None:
    waiting = []
    for name, item in pending.items():
        reason = _pending_reason(name, statuses)
        waiting.append(f"{item.config.name} ({reason})" if reason else item.config.name)
    logger.info("Waiting on %d model(s): %s", len(pending), ", ".join(waiting))
