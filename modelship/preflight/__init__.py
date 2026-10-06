"""Hardware-aware preflight estimator framework.

Each loader registers a `BasePreflight` implementation that produces a
recommendation dict (loader-config field names → values) sized for the
actor's assigned hardware. The caller merges the recommendation with the
user-supplied config so user values always win:

    final = {**recommendation, **user_overrides}

When user overrides conflict with the recommendation, a warning is logged
naming both values, and the user value is used. The runtime catch in
`ModelDeployment.__init__` remains the safety net for cases the estimator
can't model (multimodal, LoRA, speculative decoding, etc.).
"""

from modelship.preflight.base import (
    BasePreflight,
    GPUInfo,
    HardwareProfile,
    ModelNotSizedError,
    detect_available_ram_bytes,
    detect_gpus,
    detect_ram_bytes,
    discover_hardware,
    gpu_share_bytes,
    gpu_share_fraction,
    merge_with_user_overrides,
    run_preflight,
)

__all__ = [
    "BasePreflight",
    "GPUInfo",
    "HardwareProfile",
    "ModelNotSizedError",
    "detect_available_ram_bytes",
    "detect_gpus",
    "detect_ram_bytes",
    "discover_hardware",
    "gpu_share_bytes",
    "gpu_share_fraction",
    "merge_with_user_overrides",
    "run_preflight",
]
