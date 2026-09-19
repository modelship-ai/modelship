import os
import signal
import sys
import time

# Module scope stays Ray/HF-free: main() sets env vars those latch at import.
from modelship.logging import configure_logging, get_lib_log_config, get_logger, propagate_lib_log_env
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.ray_auth import resolve_ray_auth_env

propagate_lib_log_env()

logger = get_logger("startup")
_DEFAULT_GATEWAY_NAME = "modelship"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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

    import ray
    from ray.serve.schema import LoggingConfig

    from modelship.deploy.actor_options import build_deployment_options, total_gpu_reservation
    from modelship.deploy.config import (
        resolve_all_model_sources,
        resolve_input_models,
    )
    from modelship.deploy.effective_config import (
        deployment_names,
        merge,
        read_effective,
        resolve_mode,
        to_config,
        write_effective,
    )
    from modelship.deploy.removal import delete_apps_quietly, remove_apps
    from modelship.deploy.serve_utils import (
        connect_ray,
        gateway_route_prefix,
        get_existing_apps,
        leave_ray_cluster,
        make_operator_id,
        seed_expected_models,
        shutdown_ray,
        start_gateway,
        start_serve,
        supervise_join_node,
    )
    from modelship.deploy.strategy import DeployContext, compute_deploy_plan, run_deploy_loop
    from modelship.infer.deploy_coordinator import create_operator_probe, get_or_create_coordinator
    from modelship.infer.replica_coordinator import get_or_create_replica_coordinator
    from modelship.metrics import DEPLOY_DURATION_SECONDS, DEPLOY_MODELS_CHANGED_TOTAL
    from modelship.openai.compaction_crypto import ensure_key_seeded
    from modelship.preflight import detect_gpus
    from modelship.state import MemoryStateStore, get_state_store

    # Read before the env write below overwrites it.
    explicit_gateway = "MSHIP_GATEWAY_NAME" in os.environ

    configure_logging()
    gateway_name = os.environ.get("MSHIP_GATEWAY_NAME", _DEFAULT_GATEWAY_NAME)
    # Forwarded to replicas via runtime_env; metrics.py tags with it.
    os.environ["MSHIP_GATEWAY_NAME"] = gateway_name
    # Validate before connect_ray.
    route_prefix = gateway_route_prefix(gateway_name)
    # apply_args_to_env folded --use-existing-ray-cluster/--address into these.
    joined_cluster = bool(os.environ.get("MSHIP_ADDRESS"))
    owns_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() != "true" and not joined_cluster
    # One level above the app's; Serve's system actors and Ray's driver logger ignore setLevel.
    lib_level, lib_level_name = get_lib_log_config()
    serve_logging_config = LoggingConfig(log_level=lib_level_name)

    # Before connect_ray, which spawns processes a signal must still clean up.
    def _early_cleanup(sig, _frame) -> None:
        logger.info("Shutting down (signal %s) during connect...", sig)
        if joined_cluster:
            leave_ray_cluster()
        elif owns_cluster:
            shutdown_ray()
        sys.exit(0)

    signal.signal(signal.SIGINT, _early_cleanup)
    signal.signal(signal.SIGTERM, _early_cleanup)

    connect_ray(lib_level)

    alive_nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
    total_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    logger.info(
        "Connected to Ray: %d node(s), %s GPU / %s CPU total (%s GPU / %s CPU schedulable now).",
        alive_nodes,
        total_resources.get("GPU", 0),
        total_resources.get("CPU", 0),
        available_resources.get("GPU", 0),
        available_resources.get("CPU", 0),
    )

    # gcs_address from the runtime context: Ray may bind a port other than the intended one.
    if owns_cluster:
        gcs_address = ray.get_runtime_context().gcs_address
        intended_port = os.environ.get("RAY_GCS_SERVER_PORT")
        actual_port = gcs_address.rsplit(":", 1)[-1]
        if intended_port and actual_port != intended_port:
            logger.warning(
                "Ray's GCS bound port %s, not the intended %s (RAY_GCS_SERVER_PORT) — pin "
                "--ray-port to a free port so a join address stays stable across head restarts.",
                actual_port,
                intended_port,
            )
        join_cmd = f"docker run ... --address={gcs_address}"
        if os.environ.get("RAY_AUTH_MODE") == "token":
            join_cmd += " --token=<token>"
            token_hint = " Retrieve the token with: docker exec <this-container> cat ~/.ray/auth_token"
        else:
            token_hint = ""
        logger.info(
            "To join this cluster as an additional compute node from another machine: %s%s "
            "(see docs/multi-node-docker.md).",
            join_cmd,
            token_hint,
        )

    # This node's physical GPUs, not Ray's cluster tally.
    for gpu in detect_gpus():
        logger.info(
            "This node sees GPU %d: %s (uuid=%s, %.2f GiB)",
            gpu.index,
            gpu.name,
            gpu.uuid or "unknown",
            gpu.sizing_total_bytes / 1024**3,
        )

    start_serve(serve_logging_config)

    existing_apps = get_existing_apps()
    fresh_install = gateway_name not in existing_apps
    if existing_apps:
        logger.info("Found existing deployments: %s", ", ".join(sorted(existing_apps)))
    if fresh_install:
        logger.info("No existing gateway found — treating as fresh install.")

    # A join creates a gateway only when named explicitly.
    create_gateway = owns_cluster or explicit_gateway
    phantom_gateway = fresh_install and not create_gateway
    if phantom_gateway:
        logger.warning(
            "Join: gateway %r not found and no explicit --gateway-name was given — skipping gateway "
            "creation to avoid silently starting a second gateway. This node still contributes compute "
            "to the cluster; pass --gateway-name explicitly to also create a gateway here.",
            gateway_name,
        )

    # mode only picks the merge (additive=union, reconcile=replace); the deploy always reconciles.
    mode = resolve_mode(reconcile=args.reconcile)
    store = get_state_store()
    if isinstance(getattr(store, "inner", store), MemoryStateStore):
        logger.warning(
            "Effective config is backed by a cluster-scoped (non-durable) memory state store; it "
            "survives deploys and coordinator restarts but NOT cluster loss. Set MSHIP_STATE_STORE "
            "to redis:// for self-heal after cluster loss."
        )
    ensure_key_seeded(store)
    effective_raw = read_effective(store, gateway_name)

    # No input on reconcile/join: reuse the effective config.
    if args.config is None and args.model is None and (mode == "reconcile" or joined_cluster):
        desired_raw = effective_raw
        logger.info(
            "Self-heal: reconciling to persisted effective config (no --config/--model given)."
            if not joined_cluster
            else "Join: no config given — contributing resources, reconciling to effective set."
        )
    elif (input_raw := resolve_input_models(args)) is None:
        desired_raw = effective_raw
        logger.info(
            "No --config/--model given and no default config/models.yaml found — bootstrapping an "
            "empty coordinator; it will wait for capacity/models via a later config or join."
        )
    else:
        desired_raw = merge(effective_raw, input_raw, gateway_name, mode)
    yml_conf = to_config(desired_raw)
    logger.debug("Deploying effective config (%s mode, %d model(s)): %s", mode, len(desired_raw), yml_conf)

    # Log-only and optimistic: fractions can sum under the GPU total yet not pack.
    # Reservations come from build_deployment_options, as Ray Serve's do.
    gpu_demand = sum(
        (m.autoscaling_config.max_replicas if m.autoscaling_config else m.num_replicas)
        * total_gpu_reservation(build_deployment_options(m))
        for m in yml_conf.models
    )
    cluster_gpus = total_resources.get("GPU", 0)
    if gpu_demand > cluster_gpus:
        logger.warning(
            "Configured models need at least %.2f GPU(s) at full scale; cluster has %.2f total. "
            "Deploys exceeding available capacity will pend until more nodes join.",
            gpu_demand,
            cluster_gpus,
        )

    # Detached actors: the cross-operator deploy lock and the ownership registry.
    coordinator = get_or_create_coordinator()
    replica_coord = get_or_create_replica_coordinator()
    # Removal is scoped to the prior effective set, so an empty one removes nothing.
    plan = compute_deploy_plan(
        yml_conf,
        existing_apps,
        deployment_names(effective_raw, gateway_name),
        gateway_name,
    )
    apps_to_remove = list(plan.apps_to_remove)
    removed_count = len(apps_to_remove)
    deploy_started = time.monotonic()

    # deployment_name -> model_name created by this run; read by _cleanup.
    deployed_this_run: dict[str, str] = {}

    def _cleanup(sig, _frame) -> None:
        if joined_cluster:
            # Leave only: deployments may run on other nodes; Serve reschedules this node's replicas.
            logger.info("Shutting down (signal %s), leaving the joined Ray cluster...", sig)
            leave_ray_cluster()
        else:
            logger.info("Shutting down (signal %s), cleaning up deployments from this run...", sig)
            delete_apps_quietly(reversed(deployed_this_run))
            if fresh_install and owns_cluster:
                shutdown_ray()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # First, so /health and /readyz answer while models load.
        if fresh_install and create_gateway:
            start_gateway(gateway_name, serve_logging_config, route_prefix)

        # Pins sources on the driver so auth/missing-repo errors fail before any replica starts.
        resolve_all_model_sources(yml_conf)

        if not phantom_gateway:
            seed_expected_models(replica_coord, gateway_name, yml_conf)

        # No live Serve app to delete; drop straight from the registry.
        if plan.registry_only_drop:
            try:
                ray.get(
                    [replica_coord.unregister_deployment.remote(gateway_name, name) for name in plan.registry_only_drop]
                )
            except Exception:
                logger.exception("Failed to drop stale registry entries: %s", plan.registry_only_drop)

        # stop_start: remove old apps first to free their resources.
        if args.replace_strategy == "stop_start":
            remove_apps(apps_to_remove, replica_coord, gateway_name)
            apps_to_remove = []

        # Driver-owned: Ray releases the coordinator lock if this process dies.
        operator_id = make_operator_id()
        probe = create_operator_probe()
        logger.info("Operator id=%s; coordinator acquired.", operator_id)

        ctx = DeployContext(
            coordinator=coordinator,
            replica_coordinator=replica_coord,
            probe=probe,
            operator_id=operator_id,
            gateway_name=gateway_name,
            serve_logging_config=serve_logging_config,
            deployed_this_run=deployed_this_run,
        )
        pass_count, fatally_failed = run_deploy_loop(plan.models_to_add, ctx)

        logger.info(
            "Deploy complete. %d new deployment(s) from this run (over %d pass(es)).",
            len(deployed_this_run),
            pass_count,
        )

        # blue_green: routing cut over at registration; delete the drained old app.
        if apps_to_remove:
            remove_apps(apps_to_remove, replica_coord, gateway_name)

        # Includes fatally-failed models, so the next deploy retries them.
        if not phantom_gateway:
            write_effective(store, gateway_name, desired_raw)

        DEPLOY_DURATION_SECONDS.observe(time.monotonic() - deploy_started, tags={"gateway": gateway_name})
        for action, count in (
            ("add", len(deployed_this_run)),
            ("remove", removed_count),
            ("fail", len(fatally_failed)),
        ):
            if count:
                DEPLOY_MODELS_CHANGED_TOTAL.inc(count, tags={"gateway": gateway_name, "action": action})

        if fatally_failed:
            if not phantom_gateway:
                failed_names = {cfg.name for cfg, _ in fatally_failed}
                seed_expected_models(replica_coord, gateway_name, yml_conf, exclude=failed_names)
            logger.error(
                "%d model(s) failed to deploy — fix config and redeploy (they remain in the effective config "
                "and will be retried on the next deploy/self-heal):",
                len(fatally_failed),
            )
            for cfg, reason in fatally_failed:
                logger.error("  - %s: %s", cfg.name, reason)

        if fresh_install and owns_cluster:
            # Stay resident even on fatal failures; _cleanup deletes deployments before stopping Ray.
            signal.pause()
        elif joined_cluster:
            # Exits non-zero when a node process dies unexpectedly.
            supervise_join_node()
        elif fatally_failed:
            # Deploy-and-exit: no resident /readyz, so fail via the exit code.
            logger.error("Exiting non-zero: %d model(s) fatally failed to deploy.", len(fatally_failed))
            sys.exit(1)

    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, cleaning up deployments from this run...")
        delete_apps_quietly(reversed(deployed_this_run))
        if fresh_install and owns_cluster:
            shutdown_ray()
        elif joined_cluster:
            leave_ray_cluster()
        raise


if __name__ == "__main__":
    main()
