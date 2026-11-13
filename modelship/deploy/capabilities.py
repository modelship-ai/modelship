"""Node/deployment capability resources: `mship_<loader>_<device>` alongside Ray's native `GPU`, so a deploy only
schedules onto a node whose variant runs that loader on that device.

Must stay ray-free at import time: modelship/launcher.py imports it before resolve_ray_auth_env runs.
"""

from __future__ import annotations

import json
import os
import platform

from modelship.utils.variant import current_variant

RESOURCE_PREFIX = "mship_"
GPU = "gpu"
CPU = "cpu"

VARIANT_LOADERS: dict[str, dict[str, frozenset[str]]] = {
    "cuda": {
        "vllm": frozenset({GPU}),
        "diffusers": frozenset({GPU, CPU}),
        "llama_server": frozenset({GPU, CPU}),
        "stable_diffusion_cpp": frozenset({CPU}),
        "whispercpp": frozenset({CPU}),
        "sherpa_onnx": frozenset({CPU}),
    },
    "cpu": {
        "vllm": frozenset({CPU}),
        "llama_server": frozenset({CPU}),
        "stable_diffusion_cpp": frozenset({CPU}),
        "whispercpp": frozenset({CPU}),
        "sherpa_onnx": frozenset({CPU}),
    },
    "vulkan": {
        "vllm": frozenset({CPU}),
        "llama_server": frozenset({GPU, CPU}),
        "stable_diffusion_cpp": frozenset({CPU}),
        "whispercpp": frozenset({CPU}),
        "sherpa_onnx": frozenset({CPU}),
    },
    "metal": {
        "llama_server": frozenset({GPU, CPU}),
        "stable_diffusion_cpp": frozenset({GPU, CPU}),
        "whispercpp": frozenset({GPU, CPU}),
        "sherpa_onnx": frozenset({CPU}),
    },
    "thin": {},
}


def variant_loaders(variant: str, system: str, machine: str) -> dict[str, frozenset[str]]:
    loaders = dict(VARIANT_LOADERS[variant])
    if variant in ("cpu", "vulkan") and system != "Linux":
        del loaders["vllm"]
    if variant == "cuda" and machine != "x86_64":
        loaders["llama_server"] = frozenset({CPU})
    return loaders


def runs_on_gpu(loader: str) -> bool:
    return any(GPU in loaders.get(loader, ()) for loaders in VARIANT_LOADERS.values())


def node_capability_resources() -> dict[str, float]:
    override = os.environ.get("MSHIP_NODE_CAPABILITIES")
    if override:
        return {str(name): float(qty) for name, qty in json.loads(override).items()}
    loaders = variant_loaders(current_variant(), platform.system(), platform.machine())
    return {f"{RESOURCE_PREFIX}{loader}_{device}": 1 for loader, devices in loaders.items() for device in devices}


def deployment_capability_resources(loader: str, num_gpus: float) -> dict[str, float]:
    device = GPU if num_gpus > 0 and runs_on_gpu(loader) else CPU
    return {f"{RESOURCE_PREFIX}{loader}_{device}": 0.001}
