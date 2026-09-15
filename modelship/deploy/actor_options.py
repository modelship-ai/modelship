"""Ray Serve actor options for model deployments: GPU allocation and runtime_env."""

from __future__ import annotations

import os
import platform

from modelship.deploy.capabilities import deployment_capability_resources
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("startup")

# Read in the replica's process; flags only set them on the driver, so they're forwarded.
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
    """The driver's set _PASSTHROUGH_ENV_VARS, for model and gateway replicas."""
    return {var: os.environ[var] for var in _PASSTHROUGH_ENV_VARS if os.environ.get(var) is not None}


def build_cache_env_vars() -> dict[str, str]:
    """Cache paths as ${root}/subdir placeholders that Ray expands on each replica's node.

    Also forwards the driver's HF_TOKEN/HF_HUB_OFFLINE."""
    env_vars = {
        "HF_HOME": "${MSHIP_CACHE_DIR}/huggingface",
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
        "VLLM_CACHE_ROOT": "${MSHIP_NODE_CACHE_DIR}/vllm",
        # flashinfer appends .cache/flashinfer/<version>/<arch>
        "FLASHINFER_WORKSPACE_BASE": "${MSHIP_NODE_CACHE_DIR}/flashinfer",
        # Triton JITs kernels at import for some archs
        "TRITON_CACHE_DIR": "${MSHIP_NODE_CACHE_DIR}/triton",
        # vLLM writes usage_stats.json here
        "VLLM_CONFIG_ROOT": "${MSHIP_NODE_CACHE_DIR}/vllm-config",
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
    """GPUs this deployment consumes: its PG bundles, else the actor's num_gpus."""
    return _total_reservation(deploy_opts, "GPU", "num_gpus")


def total_cpu_reservation(deploy_opts: dict) -> float:
    """CPUs this deployment consumes; the outer actor draws from bundle 0, so the bundles cover it."""
    return _total_reservation(deploy_opts, "CPU", "num_cpus")


def _total_reservation(deploy_opts: dict, bundle_key: str, actor_key: str) -> float:
    if "placement_group_bundles" in deploy_opts:
        return float(sum(b.get(bundle_key, 0) for b in deploy_opts["placement_group_bundles"]))
    return float(deploy_opts.get("ray_actor_options", {}).get(actor_key, 0) or 0)


def build_deployment_options(config: ModelshipModelConfig) -> dict:
    """kwargs for `Deployment.options(**...)`."""
    env_vars = build_cache_env_vars()
    env_vars.update(build_passthrough_env_vars())

    runtime_env: dict = {"env_vars": env_vars}

    capability_resources = deployment_capability_resources(config)

    # CPU-only loaders. ggml ones use Metal on Darwin, where num_gpus stays so Ray doesn't co-schedule onto it.
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
            # Scalar num_gpus; a fraction lets Ray pack other actors onto the same GPU.
            opts = {
                "ray_actor_options": {
                    "num_gpus": config.num_gpus,
                    "num_cpus": config.num_cpus,
                    "runtime_env": runtime_env,
                    "resources": capability_resources,
                }
            }
        else:
            # One whole-GPU bundle per slot, STRICT_PACK onto one node; vLLM's ray executor reuses the PG via
            # get_current_placement_group(). The outer actor sits in bundle 0 with no GPU, so capability
            # resources go on the bundles, which pick the node.
            bundles = [{"GPU": 1, "CPU": config.num_cpus, **capability_resources} for _ in range(world_size)]
            opts = {
                "ray_actor_options": {"num_gpus": 0, "num_cpus": config.num_cpus, "runtime_env": runtime_env},
                "placement_group_bundles": bundles,
                "placement_group_strategy": "STRICT_PACK",
            }

    # Serve's per-replica concurrency cap; llama_server defaults it to its parallel slots.
    max_ongoing = config.max_ongoing_requests
    if max_ongoing is None and config.loader == ModelLoader.llama_server:
        parallel = config.llama_server_config.parallel if config.llama_server_config else 1
        max_ongoing = parallel

    if max_ongoing is not None:
        opts["max_ongoing_requests"] = max_ongoing
    return opts
