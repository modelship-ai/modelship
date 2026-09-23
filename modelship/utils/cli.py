"""CLI argument parsing for the engine's start, join and deploy commands."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from modelship.utils import parse_memory_bytes
from modelship.utils.config_schema import ModelLoader, ModelUsecase
from modelship.utils.model_flags import (
    MODEL_ARG_KEYS,
    add_generated_model_args,
    apply_generated_args,
    set_generated_options,
)
from modelship.utils.model_ref import parse_model_ref

# Marks a `--model` path as a file rather than a directory when it doesn't
# exist yet, so name inference can still take the stem.
_WEIGHT_SUFFIXES = (".gguf", ".safetensors", ".bin")

_GGUF_REPO_SUFFIX = "-gguf"

# A trailing quant token in a weight filename: Q4_K_M, Q8_0, IQ4_XS, F16, BF16.
_QUANT_SEGMENT = re.compile(r"^(?:i?q\d+|bf\d+|f\d+)(?:[_-]\w+)*$", re.IGNORECASE)

# argparse attribute -> the env var it sets. Downstream code reads only os.environ.
_ARG_TO_ENV: dict[str, str] = {
    "cache_dir": "MSHIP_CACHE_DIR",
    "node_cache_dir": "MSHIP_NODE_CACHE_DIR",
    "state_store": "MSHIP_STATE_STORE",
    "log_format": "MSHIP_LOG_FORMAT",
    "log_target": "MSHIP_LOG_TARGET",
    "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
    "api_keys": "MSHIP_API_KEYS",
    "trusted_identity_header": "MSHIP_TRUSTED_IDENTITY_HEADER",
    "gateway_name": "MSHIP_GATEWAY_NAME",
    "cluster": "MSHIP_CLUSTER",
    "token": "MSHIP_RAY_AUTH_TOKEN",
    "ray_auth": "MSHIP_RAY_AUTH",
    "ray_port": "MSHIP_RAY_PORT",
    "ray_dashboard_host": "MSHIP_RAY_DASHBOARD_HOST",
    "ray_dashboard_port": "MSHIP_RAY_DASHBOARD_PORT",
    "metrics_port": "MSHIP_METRICS_PORT",
    "node_num_cpus": "MSHIP_NODE_NUM_CPUS",
    "node_num_gpus": "MSHIP_NODE_NUM_GPUS",
    "node_memory": "MSHIP_NODE_MEMORY",
    "prune_ray_sessions": "MSHIP_PRUNE_RAY_SESSIONS",
    "max_request_body_bytes": "MSHIP_MAX_REQUEST_BODY_BYTES",
    "gateway_replicas": "MSHIP_GATEWAY_REPLICAS",
    "openai_api_port": "MSHIP_OPENAI_API_PORT",
    "responses_ttl_s": "MSHIP_RESPONSES_TTL_S",
    "state_sweep_interval_s": "MSHIP_STATE_SWEEP_INTERVAL_S",
    "deploy_timeout": "MSHIP_DEPLOY_TIMEOUT_S",
}

# store_true flags -> (env var, value when passed).
_SWITCH_TO_ENV: dict[str, tuple[str, str]] = {
    "no_metrics": ("MSHIP_METRICS", "false"),
    "no_preflight": ("MSHIP_PREFLIGHT", "false"),
}

_MODEL_USAGE = "[options] [--model REF [--<block>.<key> VALUE ...]]"
_USAGE = {
    "start": f"mship start {_MODEL_USAGE}",
    "join": "mship join --cluster HOST:PORT [options]",
    "deploy": f"mship deploy {_MODEL_USAGE}",
}
_DESCRIPTION = {
    "start": "Start a cluster on this machine: its head node, the API gateway and any models given. Stays running.",
    "join": "Add this machine to a running cluster as a worker node. Stays running.",
    "deploy": "Change the models of the cluster running on this machine, then exit.",
}


def parse_args(command: str, argv: list[str] | None = None) -> argparse.Namespace:
    # Explicit usage: the generated model flags make argparse's own usage line
    # ~90 lines, which it reprints on every error.
    parser = argparse.ArgumentParser(prog=f"mship {command}", usage=_USAGE[command], description=_DESCRIPTION[command])
    if command == "join":
        _add_join_args(parser)
    if command in ("start", "join"):
        _add_node_args(parser)
    if command == "start":
        _add_head_args(parser)
    if command in ("start", "deploy"):
        _add_auth_arg(parser)
        _add_cluster_args(parser)
    if command == "deploy":
        _add_token_arg(parser)
        parser.add_argument(
            "--replace-strategy",
            choices=["blue_green", "stop_start"],
            default="blue_green",
            help=(
                "How to replace a model whose config changed. blue_green (default): deploy "
                "new alongside old, then drop old (no request loss, peak resource = old+new). "
                "stop_start: drop old first, then deploy new (brief unavailability, no overlap)."
            ),
        )
    _add_logging_args(parser)
    if command in ("start", "deploy"):
        _add_model_args(parser)

    args = parser.parse_args(argv)
    if command == "join" and not (args.cluster or os.environ.get("MSHIP_CLUSTER")):
        parser.error("--cluster is required: the head's address as HOST:PORT (env: MSHIP_CLUSTER)")
    if command in ("start", "deploy"):
        _check_model_args(parser, args)
    return args


def _check_model_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    tuning = set_generated_options(args)
    if tuning and args.model is None:
        shown = ", ".join(tuning[:3]) + (", ..." if len(tuning) > 3 else "")
        if args.config is not None:
            parser.error(
                f"{shown}: the model tuning flags only apply to --model. Set these keys "
                "on the model's entry in the config file instead."
            )
        parser.error(f"{shown}: the model tuning flags configure the model --model deploys; pass --model.")
    if args.model is not None and args.config is not None:
        # Rejected here rather than in resolve_input_models so both entry points fail
        # before the driver starts a Ray head.
        parser.error(
            "--model and --config are mutually exclusive: --model deploys a single model, "
            "--config deploys the set in a models.yaml. Add the model to the file instead."
        )


def _add_join_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cluster",
        help=(
            "The head node's address as HOST:PORT, e.g. mship-head:6380 — the head's --ray-port "
            "(env: MSHIP_CLUSTER). Reachable only from inside the cluster's private network."
        ),
    )
    _add_token_arg(parser)


def _add_token_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--token",
        help=(
            "Ray auth token of a cluster started with --ray-auth=token (env: MSHIP_RAY_AUTH_TOKEN); "
            "read it on the head with `cat ~/.ray/auth_token`."
        ),
    )


def _add_auth_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ray-auth",
        choices=["token", "none"],
        help=(
            "Ray cluster authentication (env: MSHIP_RAY_AUTH, default: none). With 'token', start "
            "makes the cluster require the bearer token Ray generates at ~/.ray/auth_token, for the "
            "dashboard and cluster-internal RPC, and deploy sends it."
        ),
    )


def _add_node_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", help="Model cache directory (env: MSHIP_CACHE_DIR)")
    parser.add_argument(
        "--node-cache-dir",
        help="Node-local compile/JIT cache directory; not for shared storage (env: MSHIP_NODE_CACHE_DIR)",
    )
    parser.add_argument(
        "--node-num-cpus", type=int, help="CPUs this node reserves (env: MSHIP_NODE_NUM_CPUS, default: auto-detect)"
    )
    parser.add_argument(
        "--node-num-gpus", type=int, help="GPUs this node reserves (env: MSHIP_NODE_NUM_GPUS, default: auto-detect)"
    )
    parser.add_argument(
        "--node-memory",
        type=parse_memory_bytes,
        help=(
            "This node's total memory budget, e.g. '8Gi' (env: MSHIP_NODE_MEMORY, default: "
            "auto-detect). Split into Ray's object_store_memory (30%%) and schedulable 'memory' "
            "resource (70%%) the same way Ray splits an auto-detected total. Set this explicitly "
            "when co-locating multiple modelship node containers on one physical host without "
            "per-container cgroup memory limits — otherwise each node auto-detects the full "
            "host's RAM independently, and the cluster total double/triple-counts the same "
            "physical memory (mirrors --node-num-cpus/--node-num-gpus for the memory dimension). "
            "Under Docker, also pass `--shm-size` >= the derived object_store_memory (~30%% of "
            "this value) — Docker's 64MB default /dev/shm is far below that, and Ray silently "
            "falls back to slower disk-backed storage instead of erroring when it doesn't fit."
        ),
    )
    parser.add_argument(
        "--prune-ray-sessions",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to delete stale dead-pid Ray session dirs under the temp root at node "
            "startup (env: MSHIP_PRUNE_RAY_SESSIONS, default: true)"
        ),
    )
    parser.add_argument(
        "--api-keys",
        help="Comma-separated API keys gateway replicas on this node accept (env: MSHIP_API_KEYS)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help=(
            "Port for this node's Prometheus metrics (env: MSHIP_METRICS_PORT, default: 8079 on start, "
            "random on join; the head lists every node's port for Prometheus service discovery)"
        ),
    )


def _add_head_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ray-port",
        type=int,
        help=(
            "Port for Ray's GCS server, the address `mship join --cluster` takes "
            "(env: MSHIP_RAY_PORT, default: 6380). Change this if 6380 is already taken on "
            "the host — e.g. avoid 6379, which the docs-recommended same-host Redis state "
            "store (MSHIP_STATE_STORE=redis://) may also want under --network=host."
        ),
    )
    parser.add_argument(
        "--ray-dashboard-host",
        help=(
            "Bind address for Ray's dashboard (env: MSHIP_RAY_DASHBOARD_HOST, default: 127.0.0.1). Its job API "
            "runs arbitrary code: bind beyond loopback only on a private network, with --ray-auth=token."
        ),
    )
    parser.add_argument(
        "--ray-dashboard-port",
        type=int,
        help=(
            "Port for Ray's dashboard (env: MSHIP_RAY_DASHBOARD_PORT, default: 8265, Ray's own "
            "default). Set it when running multiple modelship heads on one host under "
            "--network=host, so each head's dashboard gets a distinct port."
        ),
    )


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-store",
        help=(
            "State-store connection URI for the effective config, deploy coordinator and "
            "/v1/responses conversations (env: MSHIP_STATE_STORE, default: memory://). Schemes: "
            "memory:// | redis://host:port/db (rediss:// for TLS). No password in the URI — set "
            "MSHIP_REDIS_PASSWORD on every node instead. memory:// is cluster-scoped but dies with "
            "the cluster; redis:// survives it."
        ),
    )
    parser.add_argument(
        "--gateway-name",
        help="Name for the API gateway app (env: MSHIP_GATEWAY_NAME, default: modelship)",
    )
    parser.add_argument(
        "--gateway-replicas",
        type=int,
        help="Number of API gateway replicas (env: MSHIP_GATEWAY_REPLICAS, default: 1)",
    )
    parser.add_argument(
        "--openai-api-port",
        type=int,
        help="Port for the OpenAI-compatible API (env: MSHIP_OPENAI_API_PORT, default: 8000)",
    )
    parser.add_argument(
        "--trusted-identity-header",
        help=(
            "Header name (e.g. X-Consumer-Id) whose value a fronting credentials layer has "
            "already resolved and authorized; modelship trusts it unconditionally for log "
            "correlation and future state-keying (env: MSHIP_TRUSTED_IDENTITY_HEADER). "
            "Only enable when modelship is reachable exclusively from that layer — see "
            "docs/model-configuration.md."
        ),
    )
    parser.add_argument(
        "--max-request-body-bytes", type=int, help="Max request body size in bytes (env: MSHIP_MAX_REQUEST_BODY_BYTES)"
    )
    parser.add_argument(
        "--responses-ttl-s",
        type=float,
        help=(
            "TTL in seconds for stored /v1/responses conversation state; <=0 disables "
            "expiry (env: MSHIP_RESPONSES_TTL_S, default: 2592000 = 30 days)"
        ),
    )
    parser.add_argument(
        "--deploy-timeout",
        type=float,
        help=(
            "Seconds to wait for models to come up before reporting the rest as still "
            "pending; they keep deploying (env: MSHIP_DEPLOY_TIMEOUT_S, default: 600)"
        ),
    )
    parser.add_argument(
        "--state-sweep-interval-s",
        type=float,
        help=(
            "Interval in seconds between expired-key sweeps in the in-memory state store "
            "(env: MSHIP_STATE_SWEEP_INTERVAL_S, default: 300)"
        ),
    )
    parser.add_argument("--no-metrics", action="store_true", default=None, help="Disable metrics (env: MSHIP_METRICS)")


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-format", choices=["text", "json"], help="Log format (env: MSHIP_LOG_FORMAT)")
    parser.add_argument(
        "--log-target",
        help="Log target: 'console' (default) or syslog URI e.g. syslog://host:514, syslog+tcp://host:514 (env: MSHIP_LOG_TARGET)",
    )
    parser.add_argument(
        "--otel-endpoint",
        help="OpenTelemetry OTLP endpoint e.g. http://collector:4317 (env: OTEL_EXPORTER_OTLP_ENDPOINT)",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to models.yaml config file (default: config/models.yaml)")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        default=None,
        help=(
            "Disable preflight hardware-based auto-sizing; models run on loader/library "
            "defaults plus explicit config (env: MSHIP_PREFLIGHT). Useful for benchmarking."
        ),
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        default=False,
        help=(
            "Diff models.yaml against the cluster: add new models, remove dropped ones, "
            "replace those whose config changed (matched by name + fingerprint). "
            "With no --config, reconciles the live cluster to this gateway's persisted "
            "effective config only (self-heal after cluster loss)."
        ),
    )
    _add_single_model_args(parser)


def _add_single_model_args(parser: argparse.ArgumentParser) -> None:
    """The models.yaml root-level scalars, as flags for a single-model deploy.

    None is `required`: unset flags are dropped from the raw dict, leaving the
    schema to raise requiredness for both surfaces.
    """
    group = parser.add_argument_group(
        "single-model deploy",
        "Deploy one model with no config file; use --config for several. The nested "
        "tuning blocks are generated from the same schema, one flag per key, named for "
        "its config path: --llama-server-config.n-ctx 8192. Their values are read as "
        "YAML, i.e. the text you'd write after the colon in models.yaml.",
    )
    group.add_argument(
        "--model",
        help=(
            "Model reference: an HF repo id, a repo id with a file selector "
            "(repo:*Q4_K_M.gguf), or a local file/directory path. Mutually exclusive "
            "with --config."
        ),
    )
    group.add_argument("--name", help="Model name clients call it by (default: inferred from --model)")
    group.add_argument("--usecase", choices=[u.value for u in ModelUsecase])
    group.add_argument("--loader", choices=[loader.value for loader in ModelLoader])
    group.add_argument("--num-gpus", type=float, help="GPUs to reserve; a fraction < 1 shares one GPU")
    group.add_argument("--num-cpus", type=float, help="CPUs to reserve (default: 0.1)")
    group.add_argument("--num-replicas", type=int, help="Fixed replica count (default: 1)")
    group.add_argument("--max-ongoing-requests", type=int, help="Per-replica concurrency cap")
    add_generated_model_args(parser)


def model_from_args(args: argparse.Namespace) -> dict | None:
    """The raw models.yaml entry `--model` describes, or None when it's absent.

    Unset flags are omitted, not defaulted — validators branch on
    `model_fields_set`, so a materialized default validates differently.
    """
    if args.model is None:
        return None

    raw: dict = {"name": args.name if args.name is not None else infer_model_name(args.model)}
    for key in MODEL_ARG_KEYS:
        if key == "name":
            continue
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    apply_generated_args(args, raw)
    return raw


def infer_model_name(model: str) -> str:
    """Model name for a `--model` ref given no `--name`: the source's basename, or
    its stem for a weight file, minus GGUF and quant decoration. The selector is
    ignored; it picks a quant rather than identifying the model."""
    ref = parse_model_ref(model)
    base = os.path.basename(ref.source.rstrip("/"))

    if ref.is_local and (Path(ref.source).is_file() or Path(base).suffix.lower() in _WEIGHT_SUFFIXES):
        base = _strip_quant(Path(base).stem)
    elif base.lower().endswith(_GGUF_REPO_SUFFIX):
        base = base[: -len(_GGUF_REPO_SUFFIX)]

    name = _sanitize_name(base)
    if not name:
        raise ValueError(f"cannot infer a model name from --model {model!r}; pass --name explicitly.")
    return name


def _strip_quant(stem: str) -> str:
    """Drop trailing quant segments from a weight-file stem: qwen3-8b.Q4_K_M -> qwen3-8b."""
    parts = stem.split(".")
    while len(parts) > 1 and _QUANT_SEGMENT.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def _sanitize_name(base: str) -> str:
    """Lowercase and reduce to characters safe in a Serve app name and an OpenAI
    `model` field. Capped short of the fingerprint the deployment name appends."""
    name = re.sub(r"[^a-z0-9._-]+", "-", base.lower())
    return re.sub(r"-{2,}", "-", name).strip("-._")[:63].strip("-._")


def apply_args_to_env(args: argparse.Namespace) -> None:
    """Write CLI args into os.environ. CLI takes precedence over pre-set env vars;
    flags another command owns are absent from *args* and skipped."""
    for attr, env_var in _ARG_TO_ENV.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[env_var] = str(val)
    for attr, (env_var, value) in _SWITCH_TO_ENV.items():
        if getattr(args, attr, None) is True:
            os.environ[env_var] = value
