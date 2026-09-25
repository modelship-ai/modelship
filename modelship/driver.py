import os
import signal
import sys
import time

# Module scope stays Ray/HF-free: run() sets env vars those latch at import.
from modelship.logging import configure_logging, get_lib_log_config, get_logger, propagate_lib_log_env
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.ray_auth import resolve_ray_auth_env

propagate_lib_log_env()

logger = get_logger("startup")
_DEFAULT_GATEWAY_NAME = "modelship"


def run(command: str, argv: list[str] | None = None) -> None:
    args = parse_args(command, argv)
    apply_args_to_env(args)
    # After argv, before Ray starts: raylets inherit the roots replicas expand cache paths from.
    base_cache = os.environ.setdefault("MSHIP_CACHE_DIR", resolve_cache_root())
    node_cache = os.environ.setdefault("MSHIP_NODE_CACHE_DIR", resolve_node_cache_root())
    # huggingface_hub latches HF_HOME at import.
    os.environ.setdefault("HF_HOME", f"{base_cache}/huggingface")
    os.environ.setdefault("VLLM_CACHE_ROOT", f"{node_cache}/vllm")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", f"{node_cache}/flashinfer")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Before `import ray`: RAY_AUTH_MODE latches at import.
    resolve_ray_auth_env()

    import ray  # noqa: F401 — Ray sets up its loggers at import; ours go on top.

    configure_logging()
    if command == "start":
        _start(args)
    elif command == "join":
        _join()
    else:
        _deploy(args)


def _start(args) -> None:
    from modelship.deploy.serve_utils import local_ray_clusters, start_gateway, start_head, start_serve
    from modelship.state import reject_inline_password

    gateway_name, route_prefix, _ = _gateway_from_env()
    reject_inline_password(os.environ.get("MSHIP_STATE_STORE", ""))
    if running := local_ray_clusters():
        sys.exit(
            f"error: a Ray cluster is already running on this machine (GCS at {', '.join(sorted(running))}). "
            "Change its models with `mship deploy`, or stop it before starting a new one."
        )
    lib_level, serve_logging_config = _serve_logging()

    # deployment_name -> model_name created by this run; read by _cleanup.
    deployed_this_run: dict[str, str] = {}

    def _cleanup(sig, _frame) -> None:
        logger.info("Shutting down (signal %s)...", sig)
        _stop_head(deployed_this_run)
        sys.exit(0)

    # Before start_head, which spawns processes a signal must still clean up.
    _on_signals(_cleanup)

    try:
        start_head(lib_level)
        # ray.init replaces the SIGTERM handler with its own.
        _on_signals(_cleanup)
        _log_cluster()
        _log_join_hint()
        _log_gpus()
        start_serve(serve_logging_config)
        # First, so /health and /readyz answer while models load.
        start_gateway(gateway_name, serve_logging_config, route_prefix)
        _apply(args, gateway_name, serve_logging_config, deployed_this_run)
    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, shutting down...")
        _stop_head(deployed_this_run)
        raise

    # Resident even on fatal failures; a signal stops the head via _cleanup.
    signal.pause()


def _stop_head(deployed_this_run: dict[str, str]) -> None:
    """Delete this run's deployments, then stop Serve and Ray. With a Redis-backed GCS
    every app stays, for the next head to restore."""
    from modelship.deploy.removal import delete_apps_quietly
    from modelship.deploy.serve_utils import shutdown_ray

    if os.environ.get("RAY_REDIS_ADDRESS"):
        logger.info("GCS is stored in Redis; leaving the Serve apps for the next head.")
        shutdown_ray(keep_serve=True)
        return
    logger.info("Cleaning up deployments from this run...")
    delete_apps_quietly(reversed(deployed_this_run))
    shutdown_ray()


def _join() -> None:
    from modelship.deploy.serve_utils import join_cluster, leave_ray_cluster, supervise_join_node

    def _leave(sig, _frame) -> None:
        logger.info("Shutting down (signal %s), leaving the Ray cluster...", sig)
        leave_ray_cluster()
        sys.exit(0)

    _on_signals(_leave)

    try:
        join_cluster(os.environ["MSHIP_CLUSTER"])
    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        leave_ray_cluster()
        raise
    # Node() replaces the SIGTERM handler with one that exits 1.
    _on_signals(_leave)
    _log_gpus()
    # Exits non-zero when a node process dies unexpectedly.
    supervise_join_node()


def _deploy(args) -> None:
    from modelship.deploy.removal import wait_for_retired_apps
    from modelship.deploy.serve_utils import (
        attach_cluster,
        get_existing_apps,
        local_ray_clusters,
        start_gateway,
        start_serve,
    )
    from modelship.infer.gateway_coordinator import get_or_create_gateway_coordinator
    from modelship.state import reject_inline_password

    gateway_name, route_prefix, explicit_gateway = _gateway_from_env()
    reject_inline_password(os.environ.get("MSHIP_STATE_STORE", ""))
    if not local_ray_clusters():
        sys.exit(
            "error: no Ray cluster is running on this machine. Start one with `mship start`, "
            "or run deploy on a node of a running cluster."
        )
    lib_level, serve_logging_config = _serve_logging()
    attach_cluster(lib_level)
    _log_cluster()
    # A no-op when Serve already runs; the first call on a cluster sets it up.
    start_serve(serve_logging_config)

    create_gateway = gateway_name not in get_existing_apps()
    if create_gateway and not explicit_gateway:
        sys.exit(
            f"error: no gateway {gateway_name!r} on this cluster. Pass --gateway-name NAME to deploy to "
            f"another gateway, or --gateway-name {gateway_name} to create this one."
        )

    def _stop(sig, _frame) -> None:
        logger.info("Stopping (signal %s); models this deploy submitted keep coming up.", sig)
        sys.exit(0)

    # Neither a signal nor a failure deletes what this run submitted.
    _on_signals(_stop)
    if create_gateway:
        start_gateway(gateway_name, serve_logging_config, route_prefix)
    fatally_failed = _apply(args, gateway_name, serve_logging_config, {})

    # The deploy is done; a signal now only stops the wait.
    _on_signals(lambda sig, _frame: sys.exit(1 if fatally_failed else 0))
    wait_for_retired_apps(get_or_create_gateway_coordinator(), gateway_name)

    if fatally_failed:
        # No resident /readyz to report it, so fail via the exit code.
        logger.error("Exiting non-zero: %d model(s) fatally failed to deploy.", len(fatally_failed))
        sys.exit(1)


def _apply(args, gateway_name: str, serve_logging_config, deployed_this_run: dict[str, str]) -> list:
    """Reconcile this gateway's models to the desired set, recording each deployment
    created in *deployed_this_run*. Returns the (config, reason) pairs that fatally failed."""
    import ray

    from modelship.deploy.actor_options import build_deployment_options, total_gpu_reservation
    from modelship.deploy.config import resolve_all_model_sources, resolve_input_models
    from modelship.deploy.effective_config import merge, read_effective, resolve_mode, to_config
    from modelship.deploy.removal import delete_apps_quietly
    from modelship.deploy.serve_utils import get_app_statuses
    from modelship.deploy.strategy import DeployContext, DeployOutcome, compute_deploy_plan, run_deploy_loop
    from modelship.infer.deploy_coordinator import get_or_create_coordinator
    from modelship.infer.deploy_leases import gateway_lease
    from modelship.infer.gateway_coordinator import get_or_create_gateway_coordinator
    from modelship.metrics import DEPLOY_DURATION_SECONDS, DEPLOY_MODELS_CHANGED_TOTAL
    from modelship.openai.compaction_crypto import ensure_key_seeded
    from modelship.state import MemoryStateStore, get_state_store

    # mode only picks the merge (additive=union, reconcile=replace); the deploy always reconciles.
    mode = resolve_mode(reconcile=args.reconcile)
    store = get_state_store()
    if isinstance(getattr(store, "inner", store), MemoryStateStore):
        logger.warning(
            "Effective config is backed by a cluster-scoped (non-durable) memory state store; it "
            "survives deploys and deploy/gateway coordinator restarts but NOT cluster loss. Set MSHIP_STATE_STORE "
            "to redis:// for self-heal after cluster loss."
        )
    ensure_key_seeded(store)

    if args.config is None and args.model is None and mode == "reconcile":
        input_raw = None
        logger.info("Self-heal: reconciling to persisted effective config (no --config/--model given).")
    elif (input_raw := resolve_input_models(args)) is None:
        logger.info(
            "No --config/--model given and no default config/models.yaml found — keeping this gateway's "
            "effective model set."
        )

    # Detached actors: the deploy coordinator and the gateway coordinator.
    # With this gateway the only app, no replica can hold a lease, so a new deploy coordinator grants at once.
    coordinator = get_or_create_coordinator(startup_window=set(get_app_statuses()) != {gateway_name})
    get_or_create_gateway_coordinator()
    deploy_started = time.monotonic()

    # Nothing else plans, submits or deletes this gateway's apps during this block.
    with gateway_lease(gateway_name) as lease:
        app_statuses = get_app_statuses()
        if app_statuses:
            logger.info("Found existing deployments: %s", ", ".join(sorted(app_statuses)))
        effective_raw = read_effective(store, gateway_name)
        desired_raw = effective_raw if input_raw is None else merge(effective_raw, input_raw, gateway_name, mode)
        yml_conf = to_config(desired_raw)
        logger.debug("Deploying effective config (%s mode, %d model(s)): %s", mode, len(desired_raw), yml_conf)

        # Log-only and optimistic: fractions can sum under the GPU total yet not pack.
        # Reservations come from build_deployment_options, as Ray Serve's do.
        gpu_demand = sum(
            (m.autoscaling_config.max_replicas if m.autoscaling_config else m.num_replicas)
            * total_gpu_reservation(build_deployment_options(m))
            for m in yml_conf.models
        )
        cluster_gpus = ray.cluster_resources().get("GPU", 0)
        if gpu_demand > cluster_gpus:
            logger.warning(
                "Configured models need at least %.2f GPU(s) at full scale; cluster has %.2f total. "
                "Deploys exceeding available capacity will pend until more nodes join.",
                gpu_demand,
                cluster_gpus,
            )

        plan = compute_deploy_plan(yml_conf, app_statuses, gateway_name)
        # Pins sources on the driver so auth/missing-repo errors fail before any replica starts.
        resolve_all_model_sources(yml_conf)
        # Before any submit: the gateway coordinator deletes this gateway's apps the effective config doesn't target.
        # Includes models that later fail, so the next deploy retries them.
        lease.write_effective(desired_raw)
        logger.info("Effective config for gateway %r now has %d model(s).", gateway_name, len(desired_raw))

        # stop_start: remove old apps first to free their resources.
        if getattr(args, "replace_strategy", "blue_green") == "stop_start":
            delete_apps_quietly(plan.stale_apps)

    outcome = DeployOutcome(ready=[], still_pending=[], fatally_failed=[])
    if plan.models_to_add:
        ctx = DeployContext(
            coordinator=coordinator,
            gateway_name=gateway_name,
            serve_logging_config=serve_logging_config,
            deployed_this_run=deployed_this_run,
        )
        outcome = run_deploy_loop(plan.models_to_add, ctx)
    fatally_failed = outcome.fatally_failed

    logger.info(
        "Deploy complete: %d model(s) up, %d still coming up, %d failed.",
        len(outcome.ready),
        len(outcome.still_pending),
        len(fatally_failed),
    )
    for config, reason in outcome.still_pending:
        logger.warning(
            "Model '%s' is still coming up and will land on its own%s", config.name, f": {reason}" if reason else ""
        )

    DEPLOY_DURATION_SECONDS.observe(time.monotonic() - deploy_started, tags={"gateway": gateway_name})
    for action, count in (
        ("add", len(outcome.ready)),
        ("remove", len(plan.stale_apps)),
        ("fail", len(fatally_failed)),
    ):
        if count:
            DEPLOY_MODELS_CHANGED_TOTAL.inc(count, tags={"gateway": gateway_name, "action": action})

    if fatally_failed:
        logger.error(
            "%d model(s) failed to deploy — fix config and redeploy (they remain in the effective config "
            "and will be retried on the next deploy/self-heal):",
            len(fatally_failed),
        )
        for cfg, reason in fatally_failed:
            logger.error("  - %s: %s", cfg.name, reason)
    return fatally_failed


def _on_signals(handler) -> None:
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _gateway_from_env() -> tuple[str, str, bool]:
    """(name, route prefix, whether a name was given). Pins MSHIP_GATEWAY_NAME, which is
    forwarded to replicas and tags metrics."""
    from modelship.deploy.serve_utils import gateway_route_prefix

    explicit = "MSHIP_GATEWAY_NAME" in os.environ
    name = os.environ.get("MSHIP_GATEWAY_NAME", _DEFAULT_GATEWAY_NAME)
    if "." in name:
        sys.exit(f"error: --gateway-name {name!r} must not contain '.'")
    os.environ["MSHIP_GATEWAY_NAME"] = name
    return name, gateway_route_prefix(name), explicit


def _serve_logging():
    from ray.serve.schema import LoggingConfig

    # One level above the app's; Serve's system actors and Ray's driver logger ignore setLevel.
    lib_level, lib_level_name = get_lib_log_config()
    return lib_level, LoggingConfig(log_level=lib_level_name)


def _log_cluster() -> None:
    import ray

    alive_nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
    total = ray.cluster_resources()
    available = ray.available_resources()
    logger.info(
        "Connected to Ray: %d node(s), %s GPU / %s CPU total (%s GPU / %s CPU schedulable now).",
        alive_nodes,
        total.get("GPU", 0),
        total.get("CPU", 0),
        available.get("GPU", 0),
        available.get("CPU", 0),
    )


def _log_join_hint() -> None:
    import ray

    # From the runtime context: Ray may bind a port other than the intended one.
    gcs_address = ray.get_runtime_context().gcs_address
    intended_port = os.environ.get("RAY_GCS_SERVER_PORT")
    actual_port = gcs_address.rsplit(":", 1)[-1]
    if intended_port and actual_port != intended_port:
        logger.warning(
            "Ray's GCS bound port %s, not the intended %s (RAY_GCS_SERVER_PORT) — pin --ray-port to a "
            "free port so the address `mship join --cluster` takes stays stable across restarts.",
            actual_port,
            intended_port,
        )
    token = " --token=<token>" if os.environ.get("RAY_AUTH_MODE") == "token" else ""
    token_hint = "the token is in ~/.ray/auth_token on this machine; " if token else ""
    logger.info(
        "To add a machine to this cluster, run on it: mship join --cluster=%s%s (%ssee docs/multi-node-docker.md).",
        gcs_address,
        token,
        token_hint,
    )


def _log_gpus() -> None:
    from modelship.preflight import detect_gpus

    # This node's physical GPUs, not Ray's cluster tally.
    for gpu in detect_gpus():
        logger.info(
            "This node sees GPU %d: %s (uuid=%s, %.2f GiB)",
            gpu.index,
            gpu.name,
            gpu.uuid or "unknown",
            gpu.sizing_total_bytes / 1024**3,
        )
