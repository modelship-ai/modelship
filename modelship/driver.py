import functools
import os
import signal
import sys

# Module scope stays Ray/HF-free: run() sets env vars those latch at import.
from modelship.logging import (
    configure_logging,
    get_lib_log_config,
    get_logger,
    propagate_lib_log_env,
    serve_logging_config,
)
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.ray_auth import auth_enabled, is_loopback, resolve_ray_auth_env, token_env_without_auth

propagate_lib_log_env()

logger = get_logger("startup")
_DEFAULT_GATEWAY_NAME = "modelship"
_STOP_DELETE_TIMEOUT_S = 30.0


def run(command: str, argv: list[str] | None = None) -> None:
    args = parse_args(command, argv)
    apply_args_to_env(args)
    if command == "deploy" and (url := os.environ.get("MSHIP_RAY_DASHBOARD_URL")):
        from modelship.remote import run as run_remote

        run_remote(url, list(argv or []), args.config)
        return
    # After argv, before Ray starts: raylets inherit the roots replicas expand cache paths from.
    base_cache = os.environ.setdefault("MSHIP_CACHE_DIR", resolve_cache_root())
    node_cache = os.environ.setdefault("MSHIP_NODE_CACHE_DIR", resolve_node_cache_root())
    # huggingface_hub latches HF_HOME at import.
    os.environ.setdefault("HF_HOME", f"{base_cache}/huggingface")
    os.environ.setdefault("VLLM_CACHE_ROOT", f"{node_cache}/vllm")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", f"{node_cache}/flashinfer")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if command == "start" and (stray := token_env_without_auth()):
        sys.exit(f"error: {', '.join(stray)} is set but Ray auth is off; pass --enable-ray-auth, or unset it.")
    # Before `import ray`: RAY_AUTH_MODE latches at import.
    resolve_ray_auth_env()

    import ray  # noqa: F401 — Ray sets up its loggers at import; ours go on top.

    configure_logging()
    if command == "start":
        _start(args)
    elif command == "join":
        _join()
    elif args.cancel is not None:
        _cancel(args)
    else:
        _deploy(args)


def _start(args) -> None:
    from modelship.deploy.gateway_sizing import gateway_sizing
    from modelship.deploy.serve_utils import local_ray_clusters, start_gateway, start_head, start_serve
    from modelship.infer.deploy_coordinator import get_or_create_coordinator
    from modelship.infer.gateway_coordinator import get_or_create_gateway_coordinator
    from modelship.state import reject_inline_password, state_store_env_var
    from modelship.utils.runtime_env import cluster_env_vars

    gateway_name, route_prefix, _ = _gateway_from_env()
    reject_inline_password(os.environ.get("MSHIP_STATE_STORE", ""))
    sizing = gateway_sizing()
    if running := local_ray_clusters():
        sys.exit(
            f"error: a Ray cluster is already running on this machine (GCS at {', '.join(sorted(running))}). "
            "Change its models with `mship deploy`, or stop it before starting a new one."
        )
    lib_level, serve_logging_config = _serve_logging()
    # Inherited by job shells on this head, which run `mship deploy` for a remote one.
    os.environ["MSHIP_ENGINE_PYTHON"] = sys.executable
    dashboard_host = os.environ.get("MSHIP_RAY_DASHBOARD_HOST", "127.0.0.1")
    if not auth_enabled() and not is_loopback(dashboard_host):
        logger.warning(
            "Ray's dashboard listens on %s without token auth: anyone who can reach it can run code on this "
            "cluster. Pass --enable-ray-auth.",
            dashboard_host,
        )

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
        _prepare_state_store()
        # First, so /health and /readyz answer while models load.
        start_gateway(
            gateway_name, serve_logging_config, route_prefix, cluster_env_vars() | state_store_env_var(), sizing
        )
        get_or_create_gateway_coordinator()
        _send(args, gateway_name, get_or_create_coordinator())
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


def _prepare_state_store() -> None:
    """Seeds the head's state store with the compaction key, warning when it's the memory one."""
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


def _join() -> None:
    from modelship.deploy.serve_utils import join_cluster, leave_ray_cluster, supervise_join_node

    def _leave(sig, _frame) -> None:
        logger.info("Shutting down (signal %s), leaving the Ray cluster...", sig)
        leave_ray_cluster()
        sys.exit(0)

    _on_signals(_leave)

    try:
        join_cluster(os.environ["MSHIP_GCS_ADDRESS"])
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

    from modelship.deploy.serve_utils import (
        attach_cluster,
        get_existing_apps,
        local_ray_clusters,
        start_gateway,
        start_serve,
    )
    from modelship.infer.deploy_coordinator import find_coordinator
    from modelship.remote import EXIT_NO_DEPLOY_COORDINATOR

    gateway_name, route_prefix, explicit_gateway = _gateway_from_env()
    if not local_ray_clusters():
        sys.exit(
            "error: no Ray cluster is running on this machine. Start one with `mship start`, "
            "or run deploy on a node of a running cluster."
        )
    lib_level, _ = _serve_logging()
    attach_cluster(lib_level)
    _log_cluster()
    if (coordinator := find_coordinator()) is None:
        print("error: no deploy coordinator on this cluster; `mship start` creates it.", file=sys.stderr)
        sys.exit(EXIT_NO_DEPLOY_COORDINATOR)
    head: dict = ray.get(coordinator.cluster_settings.remote())
    # A no-op when Serve already runs; the first call on a cluster sets it up.
    start_serve(head["serve_logging_config"])

    create_gateway = gateway_name not in get_existing_apps()
    if create_gateway and not explicit_gateway:
        sys.exit(
            f"error: no gateway {gateway_name!r} on this cluster. Pass --gateway-name NAME to deploy to "
            f"another gateway, or --gateway-name {gateway_name} to create this one."
        )
    if create_gateway:
        start_gateway(gateway_name, head["serve_logging_config"], route_prefix, head["env"], head["gateway_sizing"])
    receipt = _send(args, gateway_name, coordinator)
    request_id = receipt["id"]
    if not args.wait:
        logger.info("Follow it in the head's log; cancel it with `mship deploy --cancel %s`.", request_id)
        return

    outcome = _wait_for_outcome(
        receipt["outcome"],
        stopped=f"deploy {request_id} keeps running. Cancel it with `mship deploy --cancel {request_id}`.",
        lost=f"Deploy {request_id} was lost: the deploy coordinator restarted. Run the deploy again.",
    )
    if outcome["state"] != "succeeded":
        sys.exit(1)


def _send(args, gateway_name: str, coordinator) -> dict:
    """Queues this invocation's models on the gateway; returns the deploy coordinator's receipt."""
    import ray

    from modelship.deploy.actor_options import deploy_env_vars
    from modelship.deploy.config import resolve_input_models
    from modelship.deploy.ledger import DeployRequest

    from_job = getattr(args, "config_from_job", False)
    bare = args.config is None and args.model is None and not from_job and args.reconcile
    models = None if bare else resolve_input_models(args)
    if models is None:
        logger.info("No models given: redeploying gateway %r's committed models that are missing.", gateway_name)
    mode = "bare" if models is None else ("reconcile" if args.reconcile else "additive")
    request = DeployRequest(
        gateway=gateway_name,
        mode=mode,
        strategy=getattr(args, "replace_strategy", None) or "blue_green",
        models=models,
        env=deploy_env_vars(),
    )

    receipt: dict = ray.get(coordinator.submit.remote(request))
    behind = f", queued behind deploy {receipt['behind']}" if receipt["behind"] else ""
    logger.info("Deploy %s sent to gateway %r (%s%s).", receipt["id"], gateway_name, mode, behind)
    return receipt


def _wait_for_outcome(ref, stopped: str, lost: str) -> dict:
    """Waits for a deploy's outcome and logs it. A signal stops the wait, logging *stopped*; a lost deploy
    coordinator exits 1, logging *lost*."""
    import ray
    from ray.exceptions import ObjectLostError, RayActorError

    def _stop_waiting(sig, _frame) -> None:
        logger.info("Stopped waiting (signal %s); %s", sig, stopped)
        sys.exit(130)

    _on_signals(_stop_waiting)
    try:
        outcome: dict = ray.get(ref)
    except (RayActorError, ObjectLostError):
        logger.error(lost)
        sys.exit(1)
    _log_outcome(outcome)
    return outcome


def _log_outcome(outcome: dict) -> None:
    for model, result in sorted(outcome["models"].items()):
        logger.info("  %s: %s", model, result)
    if outcome["state"] == "succeeded":
        version = f"; the gateway is on version {outcome['version']}" if outcome["version"] else ""
        logger.info("Deploy %s succeeded%s.", outcome["id"], version)
    elif outcome["state"] == "cancelled":
        logger.info("Deploy %s cancelled.", outcome["id"])
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
    deploy_id = args.cancel
    if (coordinator := find_coordinator()) is None:
        sys.exit(f"error: no deploy {deploy_id} on this cluster.")
    result: dict = ray.get(coordinator.cancel.remote(deploy_id))
    if not result["cancelled"]:
        sys.exit(f"error: {result['message']}.")
    logger.info("%s.", result["message"].capitalize())
    if not args.wait:
        return
    outcome = _wait_for_outcome(
        result["outcome"],
        stopped=f"deploy {deploy_id} keeps rolling back.",
        lost=f"Deploy {deploy_id} was lost: the deploy coordinator restarted, which rolls back what wasn't committed.",
    )
    if outcome["state"] != "cancelled":
        sys.exit(1)


def _on_signals(handler) -> None:
    @functools.wraps(handler)
    def once(sig, frame) -> None:
        # A repeat signal would re-enter the handler mid-teardown.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        handler(sig, frame)

    signal.signal(signal.SIGINT, once)
    signal.signal(signal.SIGTERM, once)


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
    # One level above the app's; Serve's system actors and Ray's driver logger ignore setLevel.
    return get_lib_log_config()[0], serve_logging_config()


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
            "Ray's GCS bound port %s, not the intended %s (RAY_GCS_SERVER_PORT) — pin --gcs-port to a "
            "free port so the address `mship join --gcs-address` takes stays stable across restarts.",
            actual_port,
            intended_port,
        )
    token_hint = ""
    if os.environ.get("RAY_AUTH_MODE") == "token":
        if os.environ.get("MSHIP_RAY_AUTH_TOKEN"):
            where = "the one this head was started with"
        elif os.environ.get("RAY_AUTH_TOKEN"):
            where = "RAY_AUTH_TOKEN in this head's environment"
        else:
            where = "~/.ray/auth_token on this machine"
        token_hint = f", with MSHIP_RAY_AUTH_TOKEN set to the cluster's token ({where})"
    logger.info(
        "To add a machine to this cluster, run on it: mship join --gcs-address=%s%s (see docs/multi-node-docker.md).",
        gcs_address,
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
