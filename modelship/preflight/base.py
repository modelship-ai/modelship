from __future__ import annotations

import contextlib
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.utils.variant import current_variant

logger = get_logger("preflight")


class ModelNotSizedError(Exception):
    """Preflight could not fit the model into the memory its node has left for it. Fails the load."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    available_bytes: int
    name: str
    uuid: str | None = None  # "GPU-<uuid>"; None when the probe can't read it
    total_bytes: int = 0  # device capacity; 0 when the probe couldn't read it

    @property
    def sizing_total_bytes(self) -> int:
        """Total capacity when known, else free."""
        return self.total_bytes or self.available_bytes


@dataclass(frozen=True)
class HardwareProfile:
    """Per-actor view of the hardware Ray assigned. GPU indices are CUDA-visible,
    i.e. already filtered through `CUDA_VISIBLE_DEVICES`."""

    gpus: list[GPUInfo] = field(default_factory=list)
    ram_bytes: int = 0
    available_ram_bytes: int = 0

    @property
    def sizing_ram_bytes(self) -> int:
        """Free RAM when the probe read it, else total."""
        return self.available_ram_bytes or self.ram_bytes


class BasePreflight(Protocol):
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        """Return a dict keyed on the loader config's field names. Empty dict
        means no recommendation (estimator can't reason about this config)."""
        ...


_REGISTRY: dict[ModelLoader, BasePreflight] = {}


def register(loader: ModelLoader, impl: BasePreflight) -> None:
    _REGISTRY[loader] = impl


def get_preflight(loader: ModelLoader) -> BasePreflight | None:
    return _REGISTRY.get(loader)


def discover_hardware() -> HardwareProfile:
    """Snapshot the hardware available to this deployment."""
    return HardwareProfile(
        gpus=detect_gpus(),
        ram_bytes=detect_ram_bytes(),
        available_ram_bytes=detect_available_ram_bytes(),
    )


def detect_gpus() -> list[GPUInfo]:
    variant = current_variant()
    if variant == "cuda":
        node = _nvml_gpus()
        return _visible_gpus(node, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if variant == "metal":
        return _metal_gpus()
    return []


def detect_ram_bytes() -> int:
    """Total RAM available to *this* process, honoring a container memory cap.

    psutil reads /proc/meminfo, which the kernel does not namespace per container,
    so under a cap it reports the host's RAM. The real ceiling is in the cgroup
    pseudo-files; take the tighter of the two. 0 when neither is readable."""
    host_total = 0
    try:
        import psutil

        host_total = int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("preflight: psutil total-RAM probe failed", exc_info=True)
    return _tighter_ram(host_total, _cgroup_memory_limit_bytes(), what="memory limit")


def detect_available_ram_bytes() -> int:
    """RAM currently *free* for new allocations, honoring a container memory cap.

    Same host-vs-cgroup reconciliation as `detect_ram_bytes`, on headroom rather
    than the ceiling. `psutil.virtual_memory().available` counts reclaimable page
    cache as free but is not cgroup-namespaced, so under a cap it overestimates."""
    host_available = 0
    try:
        import psutil

        host_available = int(psutil.virtual_memory().available)
    except Exception:
        logger.debug("preflight: psutil available-RAM probe failed", exc_info=True)
    return _tighter_ram(host_available, _cgroup_memory_available_bytes(), what="memory headroom")


def _tighter_ram(host_bytes: int, cgroup_bytes: int | None, *, what: str) -> int:
    """Take the tighter of a host psutil reading and the cgroup's, falling back to
    whichever is readable, 0 if neither is."""
    if cgroup_bytes is None:  # uncapped or unreadable cgroup
        return host_bytes
    if host_bytes <= 0:
        logger.debug("preflight: psutil unavailable; using cgroup %s %.2f GiB", what, cgroup_bytes / 1024**3)
        return cgroup_bytes
    if cgroup_bytes < host_bytes:
        logger.debug(
            "preflight: cgroup %s %.2f GiB binds (host reports %.2f GiB)",
            what,
            cgroup_bytes / 1024**3,
            host_bytes / 1024**3,
        )
    return min(host_bytes, cgroup_bytes)


# cgroup v1's "unlimited" sentinel: PAGE_COUNTER_MAX (LONG_MAX rounded down to the
# page size); some kernels report LONG_MAX instead. Both are >= this value.
_CGROUP_V1_UNLIMITED = 0x7FFFFFFFFFFFF000


def _cgroup_memory_limit_bytes(
    paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ),
) -> int | None:
    """The container's memory ceiling from cgroup, or None if unlimited, not
    containerized, or unreadable. The v1 sentinel is resolved here rather than by
    the caller's `min()`, so the value stays safe when psutil is unavailable.
    `paths` is a parameter only so tests can point it at temp files."""
    for path in paths:
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":  # cgroup v2 "unlimited"
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _CGROUP_V1_UNLIMITED:  # cgroup v1 "unlimited" / nonsensical
            return None
        return value
    return None


def _cgroup_memory_available_bytes(
    usage_paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.current",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # cgroup v1
    ),
    stat_paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.stat",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.stat",  # cgroup v1
    ),
) -> int | None:
    """Free RAM inside the container's memory cgroup, or None when uncapped or
    unreadable.

    `limit - current + reclaimable`: current usage counts page cache, which the
    kernel evicts under pressure, so the evictable part is added back. Unreadable
    memory.stat means reclaimable 0, i.e. smaller headroom. `*_paths` are
    parameters only so tests can use temp files."""
    limit = _cgroup_memory_limit_bytes()
    if limit is None:  # uncapped — defer to the host (psutil) reading
        return None
    current = _read_first_int(usage_paths)
    if current is None:
        return None
    reclaimable = _cgroup_reclaimable_cache_bytes(stat_paths) or 0
    return max(0, limit - current + reclaimable)


def _read_first_int(paths: tuple[str, ...]) -> int | None:
    """Read the first readable path as a single integer, else None."""
    for path in paths:
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _cgroup_reclaimable_cache_bytes(stat_paths: tuple[str, ...]) -> int | None:
    """Sum the evictable file-cache from memory.stat. None when no memory.stat is
    readable, 0 when it lists no cache keys.

    cgroup v1 lists both the hierarchical `total_*_file` and per-cgroup `*_file`
    lines, so summing every key double-counts; prefer the `total_*` pair and fall
    back to `inactive_file`/`active_file`, which is all v2 has."""
    for path in stat_paths:
        try:
            with open(path) as f:
                raw = f.read()
        except OSError:
            continue
        values: dict[str, int] = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    values[parts[0]] = int(parts[1])
        if "total_inactive_file" in values:  # cgroup v1 — use the hierarchical pair only
            return values.get("total_inactive_file", 0) + values.get("total_active_file", 0)
        return values.get("inactive_file", 0) + values.get("active_file", 0)
    return None


def _nvml_gpus() -> list[GPUInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        logger.warning("preflight: NVML unavailable; sizing without GPU figures", exc_info=True)
        return []
    try:
        gpus: list[GPUInfo] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
            gpus.append(
                GPUInfo(
                    index=i,
                    available_bytes=int(mem.free),
                    name=pynvml.nvmlDeviceGetName(handle),
                    uuid=pynvml.nvmlDeviceGetUUID(handle),
                    total_bytes=int(mem.total) - int(mem.reserved),
                )
            )
        return gpus
    except Exception:
        logger.warning("preflight: NVML read failed; sizing without GPU figures", exc_info=True)
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _visible_gpus(node: list[GPUInfo], devices: str | None) -> list[GPUInfo]:
    if not devices:
        return node
    visible: list[GPUInfo] = []
    for entry in filter(None, (d.strip() for d in devices.split(","))):
        if entry.isdigit():
            gpu = node[int(entry)] if int(entry) < len(node) else None
        else:
            gpu = next((g for g in node if g.uuid and g.uuid.startswith(entry)), None)
        if gpu is None:
            break
        visible.append(replace(gpu, index=len(visible)))
    return visible


def _metal_gpus() -> list[GPUInfo]:
    try:
        limit = _metal_working_set_limit()
    except Exception:
        logger.warning("preflight: Metal probe failed; sizing without GPU figures", exc_info=True)
        return []
    available = max(0, min(limit - _metal_bytes_in_use(), detect_available_ram_bytes()))
    name = _sysctl_str("machdep.cpu.brand_string") or "Apple GPU"
    return [GPUInfo(index=0, available_bytes=available, name=name, uuid=None, total_bytes=limit)]


def _metal_working_set_limit() -> int:
    import ctypes

    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned no device")
    send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    release = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    try:
        return int(send(device, objc.sel_registerName(b"recommendedMaxWorkingSetSize")))
    finally:
        release(device, objc.sel_registerName(b"release"))


def _metal_bytes_in_use() -> int:
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except Exception:
        logger.debug("preflight: ioreg failed; counting no Metal memory in use", exc_info=True)
        return 0
    match = re.search(r'"In use system memory"=(\d+)', out)
    return int(match.group(1)) if match else 0


def _sysctl_str(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_share_fraction(config: ModelshipModelConfig) -> float:
    """Declared fraction of one GPU for a fractional num_gpus, else 1.0."""
    ng = config.num_gpus
    return ng if 0 < ng < 1 else 1.0


def gpu_share_bytes(config: ModelshipModelConfig, gpu: GPUInfo) -> float:
    """This deploy's declared share of `gpu`'s total capacity, in bytes."""
    return gpu_share_fraction(config) * gpu.sizing_total_bytes


def run_preflight(config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
    """Look up the loader's estimator and run it. Returns `{}` if no estimator
    is registered, the estimator declines (no resolved path, missing config,
    etc.) or it fails. Raises only `ModelNotSizedError`."""
    if os.environ.get("MSHIP_PREFLIGHT", "true").lower() == "false":
        logger.info(
            "preflight disabled via MSHIP_PREFLIGHT=false for '%s'; using loader defaults + user config",
            config.name,
        )
        return {}

    # Register on first call so importing this module doesn't pull in vllm etc.
    _ensure_registered()

    impl = get_preflight(config.loader)
    if impl is None:
        return {}
    try:
        return impl.recommend(config, hw)
    except ModelNotSizedError:
        raise
    except Exception:
        logger.exception("preflight estimator raised for '%s'; ignoring recommendation", config.name)
        return {}


def merge_with_user_overrides(
    recommendation: dict[str, Any],
    user_overrides: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    """`final = {**recommendation, **user_overrides}` with a warning logged
    for every key the user overrode to a different value."""
    for key, rec_value in recommendation.items():
        if key in user_overrides and user_overrides[key] != rec_value:
            logger.warning(
                "preflight: '%s' suggested %s=%r based on hardware budget, "
                "user config specifies %r — proceeding with user value",
                model_name,
                key,
                rec_value,
                user_overrides[key],
            )
    return {**recommendation, **user_overrides}


def _ensure_registered() -> None:
    if ModelLoader.llama_server not in _REGISTRY:
        try:
            from modelship.preflight.llama_cpp import LlamaServerPreflight

            register(ModelLoader.llama_server, LlamaServerPreflight())
        except Exception:
            logger.debug("preflight: LlamaServerPreflight registration skipped", exc_info=True)
    if ModelLoader.stable_diffusion_cpp not in _REGISTRY:
        try:
            from modelship.preflight.stable_diffusion_cpp import StableDiffusionCppPreflight

            register(ModelLoader.stable_diffusion_cpp, StableDiffusionCppPreflight())
        except Exception:
            logger.debug("preflight: StableDiffusionCppPreflight registration skipped", exc_info=True)
    if ModelLoader.vllm in _REGISTRY:
        return
    try:
        from modelship.preflight.vllm import VllmPreflight

        register(ModelLoader.vllm, VllmPreflight())
    except Exception:
        logger.debug("preflight: VllmPreflight registration skipped", exc_info=True)
