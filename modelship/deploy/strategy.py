import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import ray
from ray import serve
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview, LoggingConfig

from modelship.deploy.actor_options import build_deployment_options
from modelship.deploy.removal import remove_apps
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger

logger = get_logger("startup")

_DEPLOY_RETRY_SLEEP_S = 2.0
_MAX_TRANSIENT_FAILURES = 3
_POLL_SECONDS = 2.0
_PENDING_LOG_EVERY_N_POLLS = 30  # with a 2s poll, log what's outstanding once a minute
_TIMEOUT_ENV = "MSHIP_DEPLOY_TIMEOUT_S"
_DEFAULT_TIMEOUT_SECONDS = 600.0


@dataclass
class DeployPlan:
    """Result of diffing models.yaml against the cluster."""

    models_to_add: list[ModelshipModelConfig]
    apps_to_remove: list[str]
    # Dropped deployments with no live app to delete — only their stale coordinator
    # registry entry needs clearing, or the gateway routes to a ghost.
    registry_only_drop: list[str]


def compute_deploy_plan(
    desired_conf: ModelshipConfig,
    existing_apps: set[str],
    prev_effective_names: set[str],
    gateway_name: str,
) -> DeployPlan:
    """Diff the desired effective set against what's live. The merge verb already
    folded additive/reconcile into `desired_conf`, so this always reconciles
    live -> desired. Deployment names are `{model}-{fingerprint}`, so a set
    comparison detects renames and config drift.

    Removal is `prev_effective_names & existing_apps`: only deployments THIS
    gateway previously managed are removed, never untracked ones or another
    gateway's. An empty prev-effective set removes nothing."""

    # Sort key: footprint desc, whole-GPU before fractional, larger fraction first.
    def _gpu_footprint(c: ModelshipModelConfig) -> tuple[int, bool, float]:
        world_size = (
            c.vllm_engine_kwargs.tensor_parallel_size * c.vllm_engine_kwargs.pipeline_parallel_size
            if c.vllm_engine_kwargs
            else 1
        )
        footprint = max(world_size, math.ceil(c.num_gpus))
        fractional = 0 < c.num_gpus < 1
        return (footprint, not fractional, c.num_gpus)

    sorted_models = sorted(desired_conf.models, key=_gpu_footprint, reverse=True)

    desired_names = {c.deployment_name(gateway_name) for c in sorted_models}

    # Split the dropped set by liveness: live ones get serve.delete + a registry
    # drop, the rest a registry-only drop.
    dropped = prev_effective_names - desired_names
    apps_to_remove = sorted(dropped & existing_apps)
    registry_only_drop = sorted(dropped - existing_apps)
    if apps_to_remove:
        logger.info("Reconcile: %d deployment(s) to remove: %s", len(apps_to_remove), apps_to_remove)
    if registry_only_drop:
        logger.info(
            "Reconcile: %d stale registry entr(ies) to drop (no live app): %s",
            len(registry_only_drop),
            registry_only_drop,
        )

    # Already live under its fingerprint -> skip, so re-runs are idempotent and a
    # matching untracked deployment is adopted rather than redeployed.
    models_to_add = [c for c in sorted_models if c.deployment_name(gateway_name) not in existing_apps]
    if models_to_add:
        logger.info(
            "%d deployment(s) to add: %s",
            len(models_to_add),
            [c.deployment_name(gateway_name) for c in models_to_add],
        )
    return DeployPlan(
        models_to_add=models_to_add,
        apps_to_remove=apps_to_remove,
        registry_only_drop=registry_only_drop,
    )


@dataclass
class DeployContext:
    coordinator: Any
    replica_coordinator: Any
    gateway_name: str
    serve_logging_config: LoggingConfig
    deployed_this_run: dict[str, str]


@dataclass
class _Pending:
    config: ModelshipModelConfig
    failures: int = 0
    last_error: str = ""
    # >0 while waiting out the backoff before being submitted again
    retry_at: float = 0.0


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
    # Before submitting, so the first replica to load finds itself declared.
    ray.get(ctx.replica_coordinator.declare_deployment.remote(ctx.gateway_name, deployment_name, config.name))
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


def _app_statuses() -> dict[str, ApplicationStatusOverview]:
    try:
        return dict(serve.status().applications)
    except Exception:
        logger.exception("Could not read Serve status; retrying next poll")
        return {}


def run_deploy_loop(
    models: list[ModelshipModelConfig],
    ctx: DeployContext,
) -> tuple[list[tuple[ModelshipModelConfig, str]], list[tuple[ModelshipModelConfig, str]]]:
    """Submit every model, then report what each one did.

    Ray places the replicas, so a model the cluster has no room for pends and
    publishes its demand instead of being held back here. A model that fails to
    come up is retried `_MAX_TRANSIENT_FAILURES` times with a doubling backoff
    unless its replica reported a fatal error, which is permanent.

    Returns (still_pending, fatally_failed), pairing each config with a reason; a
    model still waiting to retry at the deadline is failed. The caller keeps both
    in the effective config, so a later deploy retries."""
    pending = {config.deployment_name(ctx.gateway_name): _Pending(config) for config in models}
    fatally_failed: list[tuple[ModelshipModelConfig, str]] = []
    statuses: dict[str, ApplicationStatusOverview] = {}

    # removal blocks until the app is torn down, so it runs beside the polling
    with ThreadPoolExecutor(thread_name_prefix="deploy-removal") as removals:

        def give_up(name: str, reason: str) -> None:
            ctx.deployed_this_run.pop(name, None)
            fatally_failed.append((pending.pop(name).config, reason))
            removals.submit(remove_apps, [name], ctx.replica_coordinator, ctx.gateway_name)

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
                if item.retry_at and now >= item.retry_at:
                    item.retry_at = 0.0
                    if (error := _submit(item.config, ctx)) is not None:
                        fail(name, error)

            statuses = _app_statuses()
            for name, item in list(pending.items()):
                app = statuses.get(name)
                if app is None or item.retry_at:
                    continue
                if app.status == ApplicationStatus.RUNNING:
                    logger.info("Model ready: %s (deployment: %s)", item.config.name, name)
                    del pending[name]
                elif app.status in (ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY):
                    fail(name, app.message)

            if pending and polls % _PENDING_LOG_EVERY_N_POLLS == 0:
                _log_pending(pending, statuses)

        for name, item in list(pending.items()):
            if item.retry_at:
                logger.error(
                    "Giving up on model '%s' (deployment=%s): deploy timeout reached before its next attempt",
                    item.config.name,
                    name,
                )
                give_up(name, item.last_error)

    still_pending = [(item.config, _pending_reason(name, statuses)) for name, item in pending.items()]
    return still_pending, fatally_failed


def _submit(config: ModelshipModelConfig, ctx: DeployContext) -> str | None:
    try:
        submit_deploy(config, ctx)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


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
    app = statuses.get(name)
    return app.message if app and app.message else "waiting to be scheduled"


def _log_pending(pending: dict[str, _Pending], statuses: dict[str, ApplicationStatusOverview]) -> None:
    logger.info(
        "Waiting on %d model(s): %s",
        len(pending),
        ", ".join(f"{item.config.name} ({_pending_reason(name, statuses)})" for name, item in pending.items()),
    )
