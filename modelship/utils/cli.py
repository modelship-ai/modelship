"""CLI argument parsing for the engine's start, join, deploy and stop commands."""

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
    "log_level": "MSHIP_LOG_LEVEL",
    "log_format": "MSHIP_LOG_FORMAT",
    "log_target": "MSHIP_LOG_TARGET",
    "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
    "trusted_identity_header": "MSHIP_TRUSTED_IDENTITY_HEADER",
    "gateway_name": "MSHIP_GATEWAY_NAME",
    "gcs_address": "MSHIP_GCS_ADDRESS",
    "gcs_port": "MSHIP_GCS_PORT",
    "ray_dashboard_url": "MSHIP_RAY_DASHBOARD_URL",
    "ray_dashboard_host": "MSHIP_RAY_DASHBOARD_HOST",
    "ray_dashboard_port": "MSHIP_RAY_DASHBOARD_PORT",
    "metrics_port": "MSHIP_METRICS_PORT",
    "node_num_cpus": "MSHIP_NODE_NUM_CPUS",
    "node_num_gpus": "MSHIP_NODE_NUM_GPUS",
    "node_memory": "MSHIP_NODE_MEMORY",
    "prune_ray_sessions": "MSHIP_PRUNE_RAY_SESSIONS",
    "max_request_body_bytes": "MSHIP_MAX_REQUEST_BODY_BYTES",
    "gateway_min_replicas": "MSHIP_GATEWAY_MIN_REPLICAS",
    "gateway_max_replicas": "MSHIP_GATEWAY_MAX_REPLICAS",
    "gateway_target_ongoing_requests": "MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS",
    "gateway_max_ongoing_requests": "MSHIP_GATEWAY_MAX_ONGOING_REQUESTS",
    "openai_api_port": "MSHIP_OPENAI_API_PORT",
    "responses_ttl_s": "MSHIP_RESPONSES_TTL_S",
    "state_sweep_interval_s": "MSHIP_STATE_SWEEP_INTERVAL_S",
}

# store_true flags -> (env var, value when passed).
_SWITCH_TO_ENV: dict[str, tuple[str, str]] = {
    "no_metrics": ("MSHIP_METRICS", "false"),
    "no_preflight": ("MSHIP_PREFLIGHT", "false"),
    "enable_ray_auth": ("MSHIP_RAY_AUTH", "true"),
}

_MODEL_USAGE = "[options] [--model REF [--<block>.<key> VALUE ...]]"
_USAGE = {
    "start": f"mship start {_MODEL_USAGE}",
    "join": "mship join --gcs-address HOST[:PORT] [options]",
    "deploy": f"mship deploy [--ray-dashboard-url URL] {_MODEL_USAGE}\n"
    "       mship deploy [--ray-dashboard-url URL] --cancel ID [--wait]",
}
_DESCRIPTION = {
    "start": "Start a cluster on this machine: its head node, the API gateway and any models given. Stays running.",
    "join": "Add this machine to a running cluster as a worker node. Stays running.",
    "deploy": "Change the models of the cluster running on this machine, or with --ray-dashboard-url, of the cluster "
    "whose Ray dashboard that is: sends the change and exits, or with --wait, waits for it to succeed or fail. With "
    "--cancel ID, cancels a deploy instead, rolling back what it has done so far.",
}


def parse_args(command: str, argv: list[str] | None = None) -> argparse.Namespace:
    # Explicit usage: the generated model flags make argparse's own usage line
    # ~90 lines, which it reprints on every error.
    parser = argparse.ArgumentParser(
        prog=f"mship {command}", usage=_USAGE[command], description=_DESCRIPTION[command], allow_abbrev=False
    )
    if command == "join":
        _add_join_args(parser)
    if command in ("start", "join"):
        _add_node_args(parser)
        _add_node_logging_args(parser)
    if command == "start":
        _add_head_args(parser)
        _add_logging_metrics_args(parser)
        _add_auth_arg(parser)
    if command in ("start", "deploy"):
        _add_cluster_args(parser)
    if command == "deploy":
        parser.add_argument(
            "--ray-dashboard-url",
            metavar="URL",
            help=(
                "Deploy to the cluster whose Ray dashboard is at URL, e.g. http://head:8265, instead of the one on "
                "this machine; sends MSHIP_RAY_AUTH_TOKEN (env: MSHIP_RAY_DASHBOARD_URL)"
            ),
        )
        parser.add_argument("--config-from-job", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument(
            "--cancel",
            metavar="ID",
            help="Cancel this deploy instead, rolling back what it has done so far (the id `mship deploy` printed)",
        )
        parser.add_argument(
            "--wait",
            action="store_true",
            help=(
                "Wait for the deploy to succeed or fail, or with --cancel, to be rolled back, and exit with its "
                "outcome. A signal only stops the wait."
            ),
        )
        parser.add_argument(
            "--replace-strategy",
            choices=["blue_green", "stop_start"],
            help=(
                "How to replace a model whose config changed. blue_green (default): deploy "
                "new alongside old, then drop old (no request loss, peak resource = old+new). "
                "stop_start: drop old first, then deploy new (brief unavailability, no overlap)."
            ),
        )
    if command in ("start", "deploy"):
        _add_model_args(parser)

    args = parser.parse_args(argv)
    if command == "join" and not (args.gcs_address or os.environ.get("MSHIP_GCS_ADDRESS")):
        parser.error("--gcs-address is required: the head's GCS address as HOST[:PORT] (env: MSHIP_GCS_ADDRESS)")
    url = args.ray_dashboard_url or os.environ.get("MSHIP_RAY_DASHBOARD_URL") if command == "deploy" else None
    if url and not re.match(r"https?://", url):
        parser.error(f"--ray-dashboard-url takes a URL, e.g. http://{url}; got {url!r}.")
    if command == "deploy" and args.config_from_job and (args.config is not None or args.model is not None):
        parser.error("--config-from-job takes no --config or --model.")
    if command == "deploy" and args.cancel is not None:
        _check_cancel_args(parser, args)
    elif command in ("start", "deploy"):
        _check_model_args(parser, args)
    return args


def _check_cancel_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    given = [
        flag
        for flag, value in (
            ("--config", args.config),
            ("--reconcile", args.reconcile),
            ("--replace-strategy", args.replace_strategy),
            ("--no-preflight", args.no_preflight),
        )
        if value
    ]
    given += [f"--{key.replace('_', '-')}" for key in MODEL_ARG_KEYS if getattr(args, key) is not None]
    given += set_generated_options(args)
    if given:
        parser.error(f"--cancel takes no model options: {', '.join(given)}.")


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
        "--gcs-address",
        help=(
            "The head's GCS address as HOST[:PORT], e.g. mship-head:6380; PORT is the head's --gcs-port "
            "(env: MSHIP_GCS_ADDRESS, default port: 6380). Reachable only from inside the cluster's private network."
        ),
    )


def _add_auth_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-ray-auth",
        action="store_true",
        default=None,
        help=(
            "Require Ray's token auth for the dashboard and cluster-internal RPC (env: MSHIP_RAY_AUTH=true). "
            "The token is MSHIP_RAY_AUTH_TOKEN when set; otherwise Ray generates it at ~/.ray/auth_token."
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
        "--metrics-port",
        type=int,
        help=(
            "Port for this node's Prometheus metrics (env: MSHIP_METRICS_PORT, default: 8079 on start, "
            "random on join; the head lists every node's port for Prometheus service discovery)"
        ),
    )


def _add_head_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gcs-port",
        type=int,
        help=(
            "Port for Ray's GCS server, the address `mship join --gcs-address` takes "
            "(env: MSHIP_GCS_PORT, default: 6380). Change this if 6380 is already taken on "
            "the host — e.g. avoid 6379, which the docs-recommended same-host Redis state "
            "store (MSHIP_STATE_STORE=redis://) may also want under --network=host."
        ),
    )
    parser.add_argument(
        "--ray-dashboard-host",
        help=(
            "Bind address for Ray's dashboard (env: MSHIP_RAY_DASHBOARD_HOST, default: 127.0.0.1). Its job API "
            "runs arbitrary code: bind beyond loopback only on a private network, with --enable-ray-auth."
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
    parser.add_argument(
        "--gateway-min-replicas",
        type=int,
        help="Fewest replicas each API gateway scales down to (env: MSHIP_GATEWAY_MIN_REPLICAS, default: 1)",
    )
    parser.add_argument(
        "--gateway-max-replicas",
        type=int,
        help="Most replicas each API gateway scales up to (env: MSHIP_GATEWAY_MAX_REPLICAS, default: 4)",
    )
    parser.add_argument(
        "--gateway-target-ongoing-requests",
        type=float,
        help=(
            "Ongoing requests per gateway replica that autoscaling aims for; a streamed response counts until it "
            "ends (env: MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS, default: 64)"
        ),
    )
    parser.add_argument(
        "--gateway-max-ongoing-requests",
        type=int,
        help=(
            "Most requests one gateway replica handles at once; more wait in the proxy "
            "(env: MSHIP_GATEWAY_MAX_ONGOING_REQUESTS, default: 1024)"
        ),
    )
    parser.add_argument(
        "--state-store",
        help=(
            "State-store connection URI for the gateways' deploy versions and /v1/responses "
            "conversations (env: MSHIP_STATE_STORE, default: memory://). Schemes: "
            "memory:// | redis://host:port/db (rediss:// for TLS). No password in the URI — set "
            "MSHIP_REDIS_PASSWORD on every node instead. memory:// is cluster-scoped but dies with "
            "the cluster; redis:// survives it."
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


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gateway-name",
        help="Name for the API gateway app (env: MSHIP_GATEWAY_NAME, default: modelship)",
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


def _add_logging_metrics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        default=None,
        help="Disable metrics on the whole cluster (env: MSHIP_METRICS)",
    )
    parser.add_argument(
        "--log-format", choices=["text", "json"], help="Log format for the whole cluster (env: MSHIP_LOG_FORMAT)"
    )


def _add_node_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level of this node's actors; library levels follow it (env: MSHIP_LOG_LEVEL, default: INFO)",
    )
    parser.add_argument(
        "--log-target",
        help=(
            "Log target of this node's actors: 'console' (default) or syslog URI e.g. syslog://host:514, "
            "syslog+tcp://host:514 (env: MSHIP_LOG_TARGET)"
        ),
    )
    parser.add_argument(
        "--otel-endpoint",
        help=(
            "OpenTelemetry OTLP endpoint of this node's actors e.g. http://collector:4317 "
            "(env: OTEL_EXPORTER_OTLP_ENDPOINT)"
        ),
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
            "With no --config, redeploys this gateway's committed models that are missing "
            "(self-heal after cluster loss)."
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
