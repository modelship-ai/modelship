"""Ray Serve actor option construction for model deployments.

Centralises the GPU-allocation decisions for model deployments. Multi-slot
vLLM deploys always use a Ray Serve placement group (one whole-GPU bundle
per slot) that vLLM inherits via its ray distributed executor.
"""

from __future__ import annotations

import os
import platform

from modelship.deploy.capabilities import deployment_capability_resources
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root

logger = get_logger("startup")

# Forwarded from the driver to each replica's runtime_env: logging vars, the gateway
# name (metrics.py stamps every metric with it), MSHIP_METRICS so --no-metrics on
# the driver also disables metrics in the replicas (else they'd default to on),
# MSHIP_PREFLIGHT so --no-preflight on the driver also disables it in the replicas
# (preflight runs inside each loader's actor __init__, not on the driver), and the
# /v1/responses state-store tuning read inside the gateway replica's own process
# (state.responses.ttl_seconds / state.memory._sweep_interval_s), not the driver's.
_PASSTHROUGH_ENV_VARS = (
    "MSHIP_LOG_LEVEL",
    "MSHIP_LOG_FORMAT",
    "MSHIP_LOG_TARGET",
    "MSHIP_GATEWAY_NAME",
    "MSHIP_METRICS",
    "MSHIP_PREFLIGHT",
    "MSHIP_RESPONSES_TTL_S",
    "MSHIP_STATE_SWEEP_INTERVAL_S",
)


def build_passthrough_env_vars() -> dict[str, str]:
    """Driver→replica env vars (logging, gateway name, metrics) read off the
    driver's environment. Shared by model and gateway deployments so both
    replicas inherit the same logging/metrics config."""
    return {var: os.environ[var] for var in _PASSTHROUGH_ENV_VARS if os.environ.get(var) is not None}


def build_cache_env_vars() -> dict[str, str]:
    """Resolve cache dirs: weights under MSHIP_CACHE_DIR, compile caches under MSHIP_NODE_CACHE_DIR.

    Also forwards HF_TOKEN/HF_HUB_OFFLINE when set on the driver, so an actor
    downloading a gated/offline model has the same auth."""
    base_cache = resolve_cache_root()
    node_cache = resolve_node_cache_root()
    env_vars = {
        "HF_HOME": os.environ.get("HF_HOME", f"{base_cache}/huggingface"),
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{node_cache}/vllm"),
        # flashinfer appends .cache/flashinfer/<version>/<arch>
        "FLASHINFER_WORKSPACE_BASE": os.environ.get("FLASHINFER_WORKSPACE_BASE", f"{node_cache}/flashinfer"),
        # Triton JITs kernels at import for some archs
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", f"{node_cache}/triton"),
        # vLLM's usage-stats thread writes usage_stats.json/do_not_track here
        "VLLM_CONFIG_ROOT": os.environ.get("VLLM_CONFIG_ROOT", f"{node_cache}/vllm-config"),
        # Default download dir for the whispercpp loader's pywhispercpp-managed
        # built-in model names (a bare `model:` like `base.en`).
        "MSHIP_WHISPERCPP_CACHE_DIR": os.environ.get("MSHIP_WHISPERCPP_CACHE_DIR", f"{base_cache}/whispercpp"),
    }
    for var in ("HF_TOKEN", "HF_HUB_OFFLINE"):
        if os.environ.get(var) is not None:
            env_vars[var] = os.environ[var]
    return env_vars


def _world_size(config: ModelshipModelConfig) -> int:
    if config.loader != ModelLoader.vllm:
        return 1
    tp = config.vllm_engine_kwargs.tensor_parallel_size
    pp = config.vllm_engine_kwargs.pipeline_parallel_size
    return tp * pp


def total_gpu_reservation(deploy_opts: dict) -> float:
    """Sum the GPU units this deployment (actor + any PG bundles) will consume.

    Used by the coordinator's resource tracker, which can't read the PG
    bundle list as a single scalar.
    """
    return _total_reservation(deploy_opts, "GPU", "num_gpus")


def total_cpu_reservation(deploy_opts: dict) -> float:
    """Sum the CPU units this deployment (actor + any PG bundles) will consume.

    For multi-slot deploys the outer actor sits in bundle 0 and its CPU
    request is satisfied from that bundle's reservation, so summing the
    bundles gives the correct total — same shape as the GPU helper.
    """
    return _total_reservation(deploy_opts, "CPU", "num_cpus")


def _total_reservation(deploy_opts: dict, bundle_key: str, actor_key: str) -> float:
    if "placement_group_bundles" in deploy_opts:
        return float(sum(b.get(bundle_key, 0) for b in deploy_opts["placement_group_bundles"]))
    return float(deploy_opts.get("ray_actor_options", {}).get(actor_key, 0) or 0)


def build_deployment_options(config: ModelshipModelConfig) -> dict:
    """Return a kwargs dict for `Deployment.options(**...)`.

    Always contains ``ray_actor_options``; for multi-slot vLLM deploys also
    contains ``placement_group_bundles`` and ``placement_group_strategy`` so
    Ray Serve allocates one whole-GPU bundle per slot and vLLM's ray executor
    inherits the PG. When the model config sets ``max_ongoing_requests`` it is
    forwarded as the per-replica Ray Serve concurrency cap.
    """
    env_vars = build_cache_env_vars()
    env_vars.update(build_passthrough_env_vars())

    runtime_env: dict = {"env_vars": env_vars}

    capability_resources = deployment_capability_resources(config)

    # sherpa_onnx never touches CUDA or CoreML (CPU only); ggml-backed loaders are
    # CPU-only off Darwin, where forcing 0 would mislead Ray into co-scheduling
    # another GPU actor onto the device Metal is actually using.
    force_zero_gpu = config.loader == ModelLoader.sherpa_onnx or (
        config.loader in (ModelLoader.stable_diffusion_cpp, ModelLoader.whispercpp) and platform.system() != "Darwin"
    )
    if force_zero_gpu:
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': %s loader has no GPU backend here.",
                config.num_gpus,
                config.name,
                config.loader.value,
            )
        opts: dict = {
            "ray_actor_options": {
                "num_gpus": 0,
                "num_cpus": config.num_cpus,
                "runtime_env": runtime_env,
                "resources": capability_resources,
            }
        }
    else:
        world_size = _world_size(config)
        if world_size == 1:
            # Single slot: scalar Ray allocation. Fractional num_gpus (0 < n < 1)
            # lets Ray pack other actors onto the same physical GPU.
            opts = {
                "ray_actor_options": {
                    "num_gpus": config.num_gpus,
                    "num_cpus": config.num_cpus,
                    "runtime_env": runtime_env,
                    "resources": capability_resources,
                }
            }
        else:
            # Multi-slot: one PG bundle per slot, STRICT_PACK keeps them on the
            # same node (NVLink). Outer actor sits in bundle 0 with 0 GPU; vLLM's
            # ray executor reuses the PG via get_current_placement_group() and
            # pins each worker actor to its bundle. Each bundle requests a whole
            # GPU, so Ray spreads across distinct physical GPUs. The capability
            # resource is requested on every bundle (not the outer actor) since
            # the bundles are what pin the deploy to a capable node.
            bundles = [{"GPU": 1, "CPU": config.num_cpus, **capability_resources} for _ in range(world_size)]
            opts = {
                "ray_actor_options": {"num_gpus": 0, "num_cpus": config.num_cpus, "runtime_env": runtime_env},
                "placement_group_bundles": bundles,
                "placement_group_strategy": "STRICT_PACK",
            }

    # Per-model Ray Serve concurrency cap; only override the default when set.
    # The reservation helpers read only the GPU/CPU keys, so this is inert there.
    max_ongoing = config.max_ongoing_requests
    if max_ongoing is None and config.loader == ModelLoader.llama_server:
        parallel = config.llama_server_config.parallel if config.llama_server_config else 1
        max_ongoing = parallel

    if max_ongoing is not None:
        opts["max_ongoing_requests"] = max_ongoing
    return opts
