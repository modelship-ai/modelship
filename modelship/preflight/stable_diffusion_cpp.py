from __future__ import annotations

import math
import os
from typing import Any

from modelship.infer.infer_config import ModelshipModelConfig, StableDiffusionCppConfig
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile, gpu_share_bytes
from modelship.utils.variant import current_variant

logger = get_logger("preflight.stable_diffusion_cpp")

_GIB = 1024**3

# Below this much *free* system RAM, default to tiled VAE decode. The VAE decode is
# the memory high-water mark for SD/SDXL generation, so tiling it trades a little
# speed for a markedly lower peak — the right default on small NAS-class hosts.
# Measured against free (not total) RAM because the image model deploys after the
# generate model + satellites, so what's left is what matters.
_VAE_TILING_RAM_THRESHOLD_BYTES = 8 * _GIB

# GPU memory sd.cpp's `--max-vram` budget doesn't count: the CUDA context and ggml-cuda's pool, the Vulkan and
# Metal drivers' own allocations.
_UNCOUNTED_GIB = {"cuda": 1.0, "vulkan": 0.25, "metal": 0.25}
# Compute buffers on top of the weights at full speed; below it sd.cpp cuts graphs or tiles the VAE.
_COMPUTE_HEADROOM_GIB = 0.5
# Least budget passed: `--max-vram 0` means no budget.
_MIN_BUDGET_GIB = 0.25


class StableDiffusionCppPreflight:
    """CPU: VAE tiling on low-RAM hosts. Fractional `num_gpus`: a `--max-vram` budget inside the share, and
    `--auto-fit off` (weights stay on the GPU) when they fit it. A whole GPU keeps sd.cpp's own fitting."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if config.num_gpus <= 0:
            return self._recommend_cpu(config, hw)
        if config.num_gpus < 1:
            return self._recommend_share(config, hw)
        return {}

    def _recommend_cpu(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if hw.ram_bytes <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        ram_basis = hw.sizing_ram_bytes
        fallback = " [total fallback]" if not hw.available_ram_bytes else ""
        rec: dict[str, Any] = {}
        if ram_basis < _VAE_TILING_RAM_THRESHOLD_BYTES:
            rec["vae_tiling"] = True
            logger.info(
                "preflight stable_diffusion_cpp '%s': ram_avail=%.2f GiB%s < %.0f GiB → recommend vae_tiling=True",
                config.name,
                ram_basis / _GIB,
                fallback,
                _VAE_TILING_RAM_THRESHOLD_BYTES / _GIB,
            )
        return rec

    def _recommend_share(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if not hw.gpus:
            logger.warning(
                "preflight stable_diffusion_cpp '%s': no GPU visible; leaving sd.cpp's defaults", config.name
            )
            return {}
        gpu = hw.gpus[0]
        share = gpu_share_bytes(config, gpu)
        if gpu.available_bytes < share:
            logger.warning(
                "preflight stable_diffusion_cpp '%s': %.2f GiB free on GPU %d, less than its %.2f GiB share",
                config.name,
                gpu.available_bytes / _GIB,
                gpu.index,
                share / _GIB,
            )

        uncounted = _UNCOUNTED_GIB.get(current_variant(), 1.0)
        budget = math.floor((share / _GIB - uncounted) * 100) / 100
        if budget < _MIN_BUDGET_GIB:
            logger.warning(
                "preflight stable_diffusion_cpp '%s': a %.2f GiB share leaves %.2f GiB after %.2f GiB sd.cpp "
                "doesn't count; budgeting %.2f GiB",
                config.name,
                share / _GIB,
                budget,
                uncounted,
                _MIN_BUDGET_GIB,
            )
            budget = _MIN_BUDGET_GIB

        weights = _weights_bytes(config) / _GIB
        fits = weights > 0 and weights + _COMPUTE_HEADROOM_GIB <= budget
        if weights == 0:
            logger.warning(
                "preflight stable_diffusion_cpp '%s': model size unreadable; sd.cpp's auto-fit places the weights "
                "within a %.2f GiB budget",
                config.name,
                budget,
            )
        elif fits:
            logger.info(
                "preflight stable_diffusion_cpp '%s': %.2f GiB of weights fit a %.2f GiB budget → auto_fit off",
                config.name,
                weights,
                budget,
            )
        else:
            logger.warning(
                "preflight stable_diffusion_cpp '%s': %.2f GiB of weights + %.2f GiB of compute don't fit a "
                "%.2f GiB budget; sd.cpp's auto-fit places what doesn't fit in RAM",
                config.name,
                weights,
                _COMPUTE_HEADROOM_GIB,
                budget,
            )
        return {"max_vram": budget, "auto_fit": not fits}


def _weights_bytes(config: ModelshipModelConfig) -> int:
    """The model files' size; 0 when none can be read."""
    sd = config.stable_diffusion_cpp_config or StableDiffusionCppConfig()
    paths = [
        config._resolved_path,
        sd.diffusion_model_path,
        sd.clip_l_path,
        sd.clip_g_path,
        sd.t5xxl_path,
        sd.vae_path,
    ]
    total = 0
    for path in paths:
        if path and os.path.isfile(path):
            total += os.path.getsize(path)
    return total
