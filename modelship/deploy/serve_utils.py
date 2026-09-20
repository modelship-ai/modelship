from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from ray import serve
from ray._common.utils import get_ray_temp_dir
from ray.serve.config import HTTPOptions, ProxyLocation
from ray.serve.schema import LoggingConfig

from modelship.deploy.capabilities import node_capability_resources
from modelship.infer.infer_config import ModelshipConfig
from modelship.logging import get_logger
from modelship.openai.api import ModelshipAPI
from modelship.preflight import detect_available_ram_bytes, detect_gpus
from modelship.state import state_store_env_var
from modelship.utils import parse_memory_bytes, rand_suffix
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.runtime_env import GATEWAY_ENV_VARS, build_env_vars

if TYPE_CHECKING:
    from ray._private.node import Node

logger = get_logger("startup")
_DEFAULT_OPENAI_API_PORT = 8000
# Not 6379 (ray-start-head's default) — that collides with the docs-recommended
# same-host Redis state store (MSHIP_STATE_STORE=redis://) under --network=host.
_DEFAULT_RAY_GCS_PORT = 6380
_GATEWAY_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def gateway_route_prefix(gateway_name: str) -> str:
    """URL-safe HTTP path this gateway is mounted under, e.g. "llm-api" -> "/llm-api"."""
    slug = _GATEWAY_SLUG_RE.sub("-", gateway_name.lower()).strip("-")
    if not slug:
        raise ValueError(
            f"--gateway-name {gateway_name!r} has no URL-safe characters ([a-z0-9_-]) to route "
            "on. Pick a name that isn't purely symbols/whitespace."
        )
    return f"/{slug}"


# This container's own Ray node when it joined another cluster (connect_ray's
# join branch), created in-process via ray._private.node.Node(head=False). None
# on the other two branches (own-head, existing-cluster). Module-level so
# mship_deploy can supervise it (stay-resident) and leave_ray_cluster can tear
# it down — the raylet/agent subprocesses are owned by this Node object, not a
# `ray start` wrapper we babysit.
_join_node: Node | None = None


def join_node() -> Node | None:
    return _join_node


# Ray names each head's session dir `session_<timestamp>_<pid>` under its temp
# root and never cleans them up; the trailing group captures the owning pid.
_RAY_SESSION_DIR_RE = re.compile(r"^session_.*_(\d+)$")


def make_operator_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{rand_suffix(4)}"


def get_existing_apps() -> set[str]:
    """Return the set of currently deployed Serve app names."""
    try:
        return set(serve.status().applications.keys())
    except Exception:
        return set()


def shutdown_ray() -> None:
    """Shut down Ray Serve and Ray. Logs but swallows errors."""
    for label, fn in (("serve.shutdown()", serve.shutdown), ("ray.shutdown()", ray.shutdown)):
        try:
            fn()
        except Exception:
            logger.exception("%s failed", label)


# Buffer below the free-RAM snapshot, since usage can drift before Ray allocates against it.
_AUTO_NODE_MEMORY_HEADROOM = 0.9


def _resolve_node_memory_kwargs() -> dict[str, int]:
    """Split this node's memory budget into Ray's object_store_memory and schedulable
    'memory' resource via Ray's own resolve_object_store_memory, so the split matches
    however Ray derives these from a total.

    Total is MSHIP_NODE_MEMORY if set, else a host-wide free-RAM probe
    (detect_available_ram_bytes) scaled by _AUTO_NODE_MEMORY_HEADROOM. Both keys are
    always set together — leaving 'memory' to auto-derive would fall back to Ray's
    own estimate instead. Empty if MSHIP_NODE_MEMORY is unset and the probe returns 0."""
    total = os.environ.get("MSHIP_NODE_MEMORY")
    if total:
        total_bytes = parse_memory_bytes(total)
    else:
        available = detect_available_ram_bytes()
        if not available:
            return {}
        total_bytes = int(available * _AUTO_NODE_MEMORY_HEADROOM)
        logger.info("Auto-detected node memory budget: %.1f GiB free host RAM", total_bytes / 1024**3)

    from ray._private.utils import resolve_object_store_memory

    object_store_memory = resolve_object_store_memory(total_bytes)
    return {"memory": total_bytes - object_store_memory, "object_store_memory": object_store_memory}


def _resolve_node_num_gpus() -> int | None:
    """GPU count for this node (own-head and join paths alike). Explicit
    MSHIP_NODE_NUM_GPUS always wins; else metal -> 1 (no Ray Apple plugin), cpu ->
    0 (closes the cpu-image-with-`--gpus` hole), else None so Ray autodetects the
    real device count instead of pinning multi-GPU boxes to 1."""
    if gpus := os.environ.get("MSHIP_NODE_NUM_GPUS"):
        return int(gpus)
    accelerator = detect_accelerator()
    if accelerator == "metal":
        return 1
    if accelerator == "cpu":
        return 0
    return None


def _own_cluster_init_kwargs() -> dict[str, object]:
    """ray.init kwargs to start our own head. Resources auto-detect when
    MSHIP_NODE_NUM_*/MSHIP_NODE_MEMORY are unset."""
    kwargs: dict[str, object] = {
        "include_dashboard": True,
        "dashboard_host": os.environ.get("MSHIP_RAY_DASHBOARD", "127.0.0.1"),
        "resources": node_capability_resources(),
    }
    if dashboard_port := os.environ.get("MSHIP_RAY_DASHBOARD_PORT"):
        kwargs["dashboard_port"] = int(dashboard_port)
    if cpus := os.environ.get("MSHIP_NODE_NUM_CPUS"):
        kwargs["num_cpus"] = int(cpus)
    if (num_gpus := _resolve_node_num_gpus()) is not None:
        kwargs["num_gpus"] = num_gpus
    if node_memory := _resolve_node_memory_kwargs():
        kwargs["_memory"] = node_memory["memory"]
        kwargs["object_store_memory"] = node_memory["object_store_memory"]
    if os.environ.get("MSHIP_METRICS", "true").lower() == "true":
        # _metrics_export_port is a private ray.init kwarg; guarded by a connect_ray test.
        kwargs["_metrics_export_port"] = int(os.environ.get("RAY_METRICS_EXPORT_PORT", "8079"))
    return kwargs


def _pid_alive(pid: int) -> bool:
    """True if a process with *pid* currently exists. Used to avoid deleting a
    still-running head's session dir."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — treat as alive (keep it).
        return True
    except OSError:
        return True
    return True


def prune_ray_sessions() -> None:
    """Delete stale `session_<timestamp>_<pid>` dirs Ray leaves under its temp
    root (default /tmp/ray). Ray never cleans them up, so they accumulate across
    container/process restarts and slowly fill the disk on long-lived self-hosted
    boxes.

    Called when we start our OWN head, or when we join another cluster as an
    additional node (connect_ray's own-cluster and join branches) — never on
    the existing-cluster (KubeRay) branch, whose temp root we don't own. Called
    before ray.init/ray start creates this run's session — so this run's dir
    doesn't exist yet and can't be removed. A session whose owning pid is still
    alive is kept: on a single machine a second non---use-existing deploy joins
    this machine's running head (ray.init reads /tmp/ray/ray_current_cluster), so
    a live head may be present even on the own-cluster path. The `session_latest`
    symlink and non-session files (e.g. ray_current_cluster) never match and are
    left alone.

    Best-effort: pruning never aborts startup — any failure is logged as a
    warning and the deploy proceeds. Set MSHIP_PRUNE_RAY_SESSIONS=false to disable
    (e.g. to keep a crashed session's logs for debugging)."""
    if os.environ.get("MSHIP_PRUNE_RAY_SESSIONS", "true").lower() != "true":
        return
    try:
        temp_root = Path(get_ray_temp_dir())
        if not temp_root.is_dir():
            return
        removed = 0
        for entry in temp_root.iterdir():
            match = _RAY_SESSION_DIR_RE.match(entry.name)
            if not match or entry.is_symlink() or not entry.is_dir():
                continue
            if _pid_alive(int(match.group(1))):
                continue
            try:
                shutil.rmtree(entry)
                removed += 1
            except OSError:
                logger.warning("Failed to prune stale Ray session dir %s (continuing).", entry, exc_info=True)
        if removed:
            logger.info("Pruned %d stale Ray session dir(s) under %s", removed, temp_root)
    except Exception:
        # Cleanup is never worth failing a deploy over — warn and carry on.
        logger.warning("Ray session pruning failed; continuing without it.", exc_info=True)


def _join_ray_cluster(address: str) -> Node:
    """Start THIS container's Ray node in-process and join the cluster whose GCS
    is at `address` (host:port). Uses ray._private.node.Node(head=False) directly
    — the same path `ray start --address` takes internally — rather than shelling
    out to `ray start --block`, so there's no wrapper subprocess to supervise or
    process-group-signal: the raylet/agent subprocesses are owned by this Node
    object and torn down in-process by leave_ray_cluster.

    Binds to a few Ray-internal APIs (Node, RayParams, two services helpers,
    write_ray_address) — the stable public surface for a worker join is only the
    `ray start` CLI. Guarded by TestConnectRayJoin so a Ray bump that moves any
    of them fails loudly. Auth, if the head runs --ray-auth=token, rides via
    RAY_AUTH_MODE/RAY_AUTH_TOKEN already in this process's env (resolve_ray_auth_env,
    before `import ray`) — a bad/missing token surfaces as an AuthenticationError
    from ensure_token_if_auth_enabled or the GCS handshake in Node().
    """
    global _join_node
    from ray._private import services
    from ray._private.authentication.authentication_token_setup import ensure_token_if_auth_enabled
    from ray._private.node import Node
    from ray._private.parameter import RayParams
    from ray._private.utils import write_ray_address

    bootstrap = services.canonicalize_bootstrap_address(address)
    if bootstrap is None:
        raise RuntimeError(f"Could not resolve the Ray head address {address!r} to join.")

    cpus = os.environ.get("MSHIP_NODE_NUM_CPUS")
    num_gpus = _resolve_node_num_gpus()
    node_memory = _resolve_node_memory_kwargs()
    # Unlike the head, a joining node never needs a fixed metrics port: nothing external
    # targets it directly, and the head's PrometheusServiceDiscoveryWriter already picks up
    # whatever port Ray actually binds (via GCS) every few seconds. Leaving this None lets
    # Ray assign an ephemeral port, which also avoids colliding with the head's own fixed
    # port when both share a host network namespace (e.g. Docker --network=host).
    ray_params = RayParams(
        gcs_address=bootstrap,
        node_ip_address=services.get_node_ip_address(bootstrap),
        num_cpus=int(cpus) if cpus else None,
        num_gpus=num_gpus,
        memory=node_memory.get("memory"),
        object_store_memory=node_memory.get("object_store_memory"),
        metrics_export_port=None,
        resources=node_capability_resources(),
    )

    # Fail early and clearly if auth is on but no token is available locally,
    # mirroring `ray start`'s own preflight (the real rejection of a *wrong*
    # token still comes from the GCS handshake in Node()).
    ensure_token_if_auth_enabled(create_token_if_missing=False)

    logger.info("Joining Ray cluster at %s ...", address)
    # Node() blocks until the node's processes are up, so there's no readiness
    # poll to race a signal against. shutdown_at_exit/spawn_reaper mirror what
    # `ray start --block` passes, so an ungraceful driver death still tears the
    # node down.
    node = Node(ray_params, head=False, shutdown_at_exit=True, spawn_reaper=True)
    node.check_version_info()
    # Write the local discovery marker so the driver's ray.init(address="auto")
    # below finds THIS node (ray start does this itself; Node() does not).
    write_ray_address(bootstrap, node.get_temp_dir_path())
    _join_node = node
    logger.info("Joined Ray cluster at %s.", address)
    return node


# Return codes Ray treats as a graceful subprocess exit (SIGTERM is how the node
# is asked to stop); anything else means a core process died unexpectedly.
_GRACEFUL_EXIT_CODES = {0, signal.SIGTERM, -signal.SIGTERM, 128 + signal.SIGTERM}


def supervise_join_node() -> None:
    """Block, supervising the joined node like `ray start --block`: poll for any
    core subprocess dying with an unexpected code, and if one does, kill the rest
    and exit nonzero so Docker's restart policy revives the node instead of it
    lingering as a zombie that contributes nothing. Returns only via sys.exit;
    a normal SIGTERM interrupts the sleep and is handled by the caller's signal
    handler (leave_ray_cluster), never reaching the failure path here."""
    node = _join_node
    assert node is not None, "supervise_join_node called before a successful join"
    while True:
        time.sleep(1)
        unexpected = [(t, p) for t, p in node.dead_processes() if p.returncode not in _GRACEFUL_EXIT_CODES]
        if unexpected:
            for proc_type, proc in unexpected:
                logger.error("Joined node subprocess %s exited unexpectedly (code %s).", proc_type, proc.returncode)
            node.kill_all_processes(check_alive=False, allow_graceful=False)
            logger.error("Joined Ray node lost a core process; exiting for restart.")
            sys.exit(1)


def leave_ray_cluster() -> None:
    """Disconnect the driver, then stop ONLY the local node this process created
    via _join_ray_cluster. Never shutdown_ray() for a joined node — that tears
    down the whole remote cluster, which this process doesn't own — and never
    `ray stop`, which kills every Ray process on the machine, not just ours."""
    try:
        ray.shutdown()
    except Exception:
        logger.exception("ray.shutdown() failed while leaving the cluster")
    if _join_node is not None:
        try:
            _join_node.kill_all_processes(check_alive=False, allow_graceful=True)
        except Exception:
            logger.exception("Failed to stop the joined Ray node")


def _validate_node_gpu_reservation() -> None:
    """Refuse to start (own-head or join) if MSHIP_NODE_NUM_GPUS reserves more GPUs
    than this container can actually see. Ray takes the reservation on faith and
    advertises it cluster-wide; an inflated value means a later actor gets handed a
    CUDA_VISIBLE_DEVICES index that doesn't exist in this container — which crashes
    at model load, far from this misconfiguration, on whatever node happens to draw
    the actor. Catching it here (at node startup, before any Ray process forms)
    turns that into an immediate, legible error instead.

    Deliberately not a same-host/cross-node check — reserving FEWER GPUs than are
    visible (fencing a subset for another co-located container) is a legitimate,
    unrelated configuration and is left alone; only the reservation-vs-visible
    relationship on THIS node is checked."""
    reserved = os.environ.get("MSHIP_NODE_NUM_GPUS")
    if not reserved:
        return
    visible = len(detect_gpus())
    if int(reserved) > visible:
        raise RuntimeError(
            f"--node-num-gpus={reserved} (MSHIP_NODE_NUM_GPUS) exceeds the {visible} GPU(s) this "
            "container can actually see. Ray would advertise phantom capacity, and a replica "
            "scheduled against it would fail at model load with a CUDA device error, not here. "
            "Check that `docker run --gpus ...` exposes at least this many devices to this "
            "container."
        )


def connect_ray(lib_level: int) -> None:
    use_existing_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    join_address = os.environ.get("MSHIP_ADDRESS")
    if use_existing_cluster and join_address:
        raise RuntimeError(
            "--address/MSHIP_ADDRESS and --use-existing-ray-cluster/MSHIP_USE_EXISTING_RAY_CLUSTER "
            "are mutually exclusive — pick one."
        )
    if not use_existing_cluster:
        # KubeRay sizes the pod itself (rayStartParams/pod resources); MSHIP_NODE_NUM_GPUS
        # isn't read on that branch at all, so there's nothing to validate against.
        _validate_node_gpu_reservation()
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")

    if use_existing_cluster:
        # We don't own the cluster: attach to the running one. The driver must run
        # ON a cluster node (Docker co-located / k8s RayJob / bare-metal node) —
        # "auto" finds the local raylet + GCS. A driver cannot attach via a remote
        # GCS address from off-cluster.
        ray.init(address="auto", ignore_reinit_error=True, logging_level=lib_level)
    elif join_address:
        # Join an existing cluster as an additional node: create this container's
        # own node in-process (Node(head=False)), then attach the driver to it.
        # A bad/missing token surfaces as an auth error from the join itself.
        # RAY_AUTH_MODE/RAY_AUTH_TOKEN are already resolved upfront (before
        # `import ray`, see resolve_ray_auth_env) — setting them now would be too
        # late for this process's latched auth singleton anyway.
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        prune_ray_sessions()
        _join_ray_cluster(join_address)  # assigns _join_node itself, see its docstring
        # address="auto", NOT a bare init: bare ray.init() silently forms a NEW
        # local cluster if discovery fails — a split-brain node. "auto" raises
        # instead, and is the same attach call the existing-cluster branch uses.
        ray.init(address="auto", ignore_reinit_error=True, logging_level=lib_level)
    else:
        # We own the cluster: start a local head sized from MSHIP_NODE_NUM_* (what
        # start_ray.sh used to do). mship_deploy stays alive as the operator and
        # tears it down on exit (owns_cluster in mship_deploy).
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        # RAY_GCS_SERVER_PORT is the only hook ray.init() exposes for this — it's not a
        # kwarg. Left unset, Ray picks a random GCS port every head start, which breaks a
        # joiner's --address across restarts — so pin a stable default here rather than
        # requiring every operator to pass --ray-port (setdefault: an operator's explicit
        # RAY_GCS_SERVER_PORT always wins over both).
        ray_port = os.environ.get("MSHIP_RAY_PORT", str(_DEFAULT_RAY_GCS_PORT))
        os.environ.setdefault("RAY_GCS_SERVER_PORT", ray_port)
        # RAY_AUTH_MODE/RAY_AUTH_TOKEN are already resolved upfront (before
        # `import ray`, see resolve_ray_auth_env). A no-auth cluster already
        # running here surfaces as Ray's own AuthenticationError on ray.init()
        # below — no local guard needed.
        # Reclaim disk from prior runs' leftover session dirs before this run's
        # session is created. Skipped on the existing-cluster branch (KubeRay /
        # an operator we don't own manages its own temp root).
        prune_ray_sessions()
        ray.init(ignore_reinit_error=True, logging_level=lib_level, **_own_cluster_init_kwargs())
    # ray.init re-sets ray.* loggers, so re-pin them after init.
    logging.getLogger("ray").setLevel(lib_level)
    logging.getLogger("ray._private.worker").setLevel(lib_level)


def start_serve(serve_logging_config: LoggingConfig) -> None:
    port = int(os.environ.get("MSHIP_OPENAI_API_PORT", str(_DEFAULT_OPENAI_API_PORT)))
    serve.start(
        proxy_location=ProxyLocation.EveryNode,
        http_options=HTTPOptions(host="0.0.0.0", port=port),
        logging_config=serve_logging_config,
    )


def _positive_int_env(name: str, default: int) -> int:
    """Read an env var as an int >= 1, failing fast with a clear message.

    Ray Serve rejects num_replicas / max_ongoing_requests < 1 deep in
    deployment, so validate up front to surface the misconfiguration plainly.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def start_gateway(gateway_name: str, serve_logging_config: LoggingConfig, route_prefix: str) -> None:
    logger.info("Starting API gateway...")
    gateway_replicas = _positive_int_env("MSHIP_GATEWAY_REPLICAS", 1)
    gateway_max_ongoing = _positive_int_env("MSHIP_GATEWAY_MAX_ONGOING", 1024)
    # A replica can land on any node, so these come from here, not that node's env.
    # MSHIP_GATEWAY_NAME is pinned from the arg so metrics stamping stays correct.
    env_vars = build_env_vars(GATEWAY_ENV_VARS) | state_store_env_var()
    env_vars["MSHIP_GATEWAY_NAME"] = gateway_name
    serve.run(
        ModelshipAPI.options(
            name=gateway_name,
            num_replicas=gateway_replicas,
            max_ongoing_requests=gateway_max_ongoing,
            ray_actor_options={"num_cpus": 0, "runtime_env": {"env_vars": env_vars}},
            logging_config=serve_logging_config,
        ).bind(gateway_name),
        name=gateway_name,
        route_prefix=route_prefix,
    )
    logger.info(
        "Gateway up at %s — %s/health and %s/readyz now serving. (replicas=%d, max_ongoing=%d)",
        route_prefix,
        route_prefix.rstrip("/"),
        route_prefix.rstrip("/"),
        gateway_replicas,
        gateway_max_ongoing,
    )


def seed_expected_models(
    replica_coordinator, gateway_name: str, yml_conf: ModelshipConfig, exclude: set[str] | None = None
) -> None:
    # Record the full desired set on the replica coordinator (the gateway's
    # readiness baseline) — already-deployed models also count toward "ready".
    # Bumping the generation makes every replica adopt it via its watch loop.
    names = [c.name for c in yml_conf.models if c.name not in (exclude or set())]
    try:
        ray.get(replica_coordinator.set_expected.remote(gateway_name, names))
    except Exception:
        logger.exception("Failed to seed expected model list on coordinator (non-fatal).")
