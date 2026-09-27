import os
import signal
import sys

# Module scope stays Ray/HF-free: run() sets env vars those latch at import.
from modelship.logging import configure_logging, get_lib_log_config, get_logger, propagate_lib_log_env
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.ray_auth import resolve_ray_auth_env

propagate_lib_log_env()

logger = get_logger("startup")
_DEFAULT_GATEWAY_NAME = "modelship"
_STOP_DELETE_TIMEOUT_S = 30.0


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
    elif command == "stop":
        _cancel(args)
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

    def _cleanup(sig, _frame) -> None:
        logger.info("Shutting down (signal %s)...", sig)
        _stop_head()
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
        _send(args, gateway_name, serve_logging_config)
    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, shutting down...")
        _stop_head()
        raise

    # Resident; a signal stops the head via _cleanup.
    signal.pause()


def _stop_head() -> None:
    """Delete every model app, then stop Serve and Ray. With a Redis-backed GCS
    every app stays, for the next head to restore."""
    from modelship.deploy.removal import delete_model_apps
    from modelship.deploy.serve_utils import shutdown_ray

    if os.environ.get("RAY_REDIS_ADDRESS"):
        logger.info("GCS is stored in Redis; leaving the Serve apps for the next head.")
        shutdown_ray(keep_serve=True)
        return
    logger.info("Deleting the model deployments...")
    delete_model_apps(_STOP_DELETE_TIMEOUT_S)
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
    import ray
    from ray.exceptions import ObjectLostError, RayActorError

    from modelship.deploy.serve_utils import (
        attach_cluster,
        get_existing_apps,
        local_ray_clusters,
        start_gateway,
        start_serve,
    )
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
    if create_gateway:
        start_gateway(gateway_name, serve_logging_config, route_prefix)
    receipt = _send(args, gateway_name, serve_logging_config)
    request_id = receipt["id"]

    def _stop_waiting(sig, _frame) -> None:
        logger.info(
            "Stopped waiting (signal %s); deploy %s keeps running. Cancel it with `mship stop --deploy-id %s`.",
            sig,
            request_id,
            request_id,
        )
        sys.exit(130)

    _on_signals(_stop_waiting)
    try:
        outcome: dict = ray.get(receipt["outcome"])
    except (RayActorError, ObjectLostError):
        logger.error("Deploy %s was lost: the deploy coordinator restarted. Run the deploy again.", request_id)
        sys.exit(1)
    _log_outcome(outcome)
    if outcome["state"] != "succeeded":
        sys.exit(1)


def _send(args, gateway_name: str, serve_logging_config) -> dict:
    """Queues this invocation's models on the gateway; returns the deploy coordinator's receipt."""
    import ray

    from modelship.deploy.actor_options import deploy_env_vars
    from modelship.deploy.config import resolve_input_models
    from modelship.deploy.ledger import DeployRequest
    from modelship.deploy.serve_utils import get_app_statuses
    from modelship.infer.deploy_coordinator import get_or_create_coordinator
    from modelship.infer.gateway_coordinator import get_or_create_gateway_coordinator
    from modelship.openai.compaction_crypto import ensure_key_seeded
    from modelship.state import MemoryStateStore, get_state_store

    store = get_state_store()
    if isinstance(getattr(store, "inner", store), MemoryStateStore):
        logger.warning(
            "Deploy versions are kept in a cluster-scoped (non-durable) memory state store; they survive "
            "deploys and deploy/gateway coordinator restarts but NOT cluster loss. Set MSHIP_STATE_STORE "
            "to redis:// for self-heal after cluster loss."
        )
    ensure_key_seeded(store)

    models = None if args.config is None and args.model is None and args.reconcile else resolve_input_models(args)
    if models is None:
        logger.info("No models given: redeploying gateway %r's committed models that are missing.", gateway_name)
    mode = "bare" if models is None else ("reconcile" if args.reconcile else "additive")
    request = DeployRequest(
        gateway=gateway_name,
        mode=mode,
        strategy=getattr(args, "replace_strategy", "blue_green"),
        models=models,
        serve_logging_config=serve_logging_config,
        env=deploy_env_vars(),
    )

    # Detached actors: the deploy coordinator and the gateway coordinator.
    # With this gateway the only app, no replica can hold a lease, so a new deploy coordinator grants at once.
    coordinator = get_or_create_coordinator(startup_window=set(get_app_statuses()) != {gateway_name})
    get_or_create_gateway_coordinator()
    receipt: dict = ray.get(coordinator.submit.remote(request))
    behind = f", queued behind deploy {receipt['behind']}" if receipt["behind"] else ""
    logger.info("Deploy %s sent to gateway %r (%s%s).", receipt["id"], gateway_name, mode, behind)
    return receipt


def _log_outcome(outcome: dict) -> None:
    for model, result in sorted(outcome["models"].items()):
        logger.info("  %s: %s", model, result)
    if outcome["state"] == "succeeded":
        version = f"; the gateway is on version {outcome['version']}" if outcome["version"] else ""
        logger.info("Deploy %s succeeded%s.", outcome["id"], version)
    else:
        logger.error("Deploy %s %s: %s", outcome["id"], outcome["state"], outcome["reason"])


def _cancel(args) -> None:
    import ray

    from modelship.deploy.serve_utils import attach_cluster, local_ray_clusters
    from modelship.infer.deploy_coordinator import find_coordinator

    if not local_ray_clusters():
        sys.exit("error: no Ray cluster is running on this machine.")
    lib_level, _ = _serve_logging()
    attach_cluster(lib_level)
    if (coordinator := find_coordinator()) is None:
        sys.exit(f"error: no deploy {args.deploy_id} on this cluster.")
    result: dict = ray.get(coordinator.cancel.remote(args.deploy_id))
    if not result["cancelled"]:
        sys.exit(f"error: {result['message']}.")
    logger.info("%s.", result["message"].capitalize())


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
