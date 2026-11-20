"""Node labels and deployment label selectors: a node labels itself with its variant and OS, and a deploy selects
the variants that run its loader on its device, so it only schedules onto a node that can run it.

Must stay ray-free at import time: modelship/launcher.py imports it before resolve_ray_auth_env runs.
"""

from __future__ import annotations

import os
import platform
import re
import socket

from modelship.utils.variant import current_variant

LABEL_PREFIX = "modelship/"
VARIANT_LABEL = f"{LABEL_PREFIX}variant"
OS_LABEL = f"{LABEL_PREFIX}os"
HOSTNAME_LABEL = f"{LABEL_PREFIX}hostname"
GPU = "gpu"
CPU = "cpu"

VARIANT_LOADERS: dict[str, dict[str, frozenset[str]]] = {
    "cuda": {
        "vllm": frozenset({GPU}),
        "diffusers": frozenset({GPU, CPU}),
        "llama_server": frozenset({GPU, CPU}),
        "stable_diffusion_cpp": frozenset({GPU, CPU}),
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
        "stable_diffusion_cpp": frozenset({GPU, CPU}),
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


# Loaders pinned for Linux only on every variant that lists them.
LINUX_ONLY_LOADERS = frozenset({"vllm"})

_INVALID_VALUE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def runs_on_gpu(loader: str) -> bool:
    return any(GPU in loaders.get(loader, ()) for loaders in VARIANT_LOADERS.values())


def capable_variants(loader: str, num_gpus: float) -> tuple[str, ...]:
    device = GPU if num_gpus > 0 and runs_on_gpu(loader) else CPU
    return tuple(variant for variant, loaders in VARIANT_LOADERS.items() if device in loaders.get(loader, ()))


def runs_on(loader: str, num_gpus: float, variant: str, system: str) -> bool:
    return variant in capable_variants(loader, num_gpus) and (loader not in LINUX_ONLY_LOADERS or system == "Linux")


def deployment_label_selector(loader: str, num_gpus: float, node_selector: dict[str, str]) -> dict[str, str]:
    """Ray label selector for a deploy: the variants that run *loader* on its device, then *node_selector*,
    whose keys replace the computed ones."""
    selector = {VARIANT_LABEL: f"in({','.join(capable_variants(loader, num_gpus))})"}
    if loader in LINUX_ONLY_LOADERS:
        selector[OS_LABEL] = "linux"
    return selector | node_selector


def node_labels() -> dict[str, str]:
    labels = {VARIANT_LABEL: current_variant(), OS_LABEL: platform.system().lower()}
    hostname = _INVALID_VALUE_CHARS.sub("-", socket.gethostname().split(".")[0])[:63].strip("-_.")
    if hostname:
        labels[HOSTNAME_LABEL] = hostname
    return labels | operator_node_labels()


def operator_node_labels() -> dict[str, str]:
    """MSHIP_NODE_LABELS as `key=value,key=value`; Ray validates the keys and values."""
    labels = {}
    for item in filter(None, (part.strip() for part in os.environ.get("MSHIP_NODE_LABELS", "").split(","))):
        key, sep, value = (part.strip() for part in item.partition("="))
        if not sep or not key:
            raise ValueError(f"--node-labels (MSHIP_NODE_LABELS): {item!r} is not key=value.")
        if key.startswith(LABEL_PREFIX):
            raise ValueError(f"--node-labels (MSHIP_NODE_LABELS): {key!r} uses the reserved prefix {LABEL_PREFIX}.")
        labels[key] = value
    return labels
