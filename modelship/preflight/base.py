from __future__ import annotations

import contextlib
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("preflight")


class ModelNotSizedError(Exception):
    """Preflight could not fit the model into the memory its node has left for it. Fails the load."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    available_bytes: int  # free VRAM when the probe read it, else capacity
    name: str
    uuid: str | None = None  # "GPU-<uuid>"; None when the probe can't read it
    kind: str = "cuda"  # "cuda" | "rocm" | "xpu" | "mps"
    total_bytes: int = 0  # device capacity; 0 when the probe couldn't read it

    @property
    def sizing_total_bytes(self) -> int:
        """Total capacity when known, else free."""
        return self.total_bytes or self.available_bytes


@dataclass(frozen=True)
class HardwareProfile:
    """Per-actor view of the hardware Ray assigned. GPU indices are CUDA-visible,
    i.e. already filtered through `CUDA_VISIBLE_DEVICES`; an "mps" GPUInfo is
    always index 0."""

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


def discover_hardware(*, read_free_memory: bool = False) -> HardwareProfile:
    """Snapshot the hardware available to this deployment. `read_free_memory` is
    passed through to `detect_gpus`."""
    return HardwareProfile(
        gpus=detect_gpus(read_free_memory=read_free_memory),
        ram_bytes=detect_ram_bytes(),
        available_ram_bytes=detect_available_ram_bytes(),
    )


def detect_gpus(*, read_free_memory: bool = False) -> list[GPUInfo]:
    """GPUs visible to this process.

    `torch.cuda` first, which honors `CUDA_VISIBLE_DEVICES`; the node-level probes
    are fallbacks for when the actor owns no GPU directly, as with vLLM's ray
    backend where worker sub-actors hold them. Metal is checked last so no CUDA
    host reaches it.

    `read_free_memory=False` (the default) leaves `available_bytes` carrying
    capacity. True fills in free VRAM: from NVML on CUDA (no context), from
    `hipMemGetInfo` on ROCm (one HIP context per device). The node fallbacks
    report free either way."""
    gpus = _torch_cuda_discover(read_free_memory=read_free_memory)
    if read_free_memory:
        gpus = _join_nvml_free(gpus)
    if not gpus:
        gpus = _pynvml_node_discover()
        if gpus:
            logger.debug(
                "preflight: actor has no direct GPU ownership; using node-level pynvml view (%d GPU(s))", len(gpus)
            )
    if not gpus:
        gpus = _rocm_smi_node_discover()
    if not gpus:
        gpus = _apple_metal_discover()
    return gpus


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


def _torch_cuda_discover(*, read_free_memory: bool = False) -> list[GPUInfo]:
    try:
        import torch
    except Exception:
        logger.debug("preflight: torch import failed", exc_info=True)
        return []
    try:
        if not torch.cuda.is_available():
            return []
        # ROCm PyTorch maps torch.cuda onto HIP; torch.version.hip distinguishes it.
        kind = "rocm" if torch.version.hip is not None else "cuda"
        gpus: list[GPUInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            available = int(props.total_memory)
            if read_free_memory and kind == "rocm":
                # CUDA takes free from `_join_nvml_free`; no NVML equivalent for AMD.
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    available = int(free)
                except Exception:
                    logger.debug("preflight: hipMemGetInfo failed on device %d; using capacity", i, exc_info=True)
            # `props.uuid` is absent on older torch builds.
            uuid = f"GPU-{props.uuid}" if getattr(props, "uuid", None) is not None else None
            gpus.append(
                GPUInfo(
                    index=i,
                    available_bytes=available,
                    name=props.name,
                    uuid=uuid,
                    kind=kind,
                    total_bytes=int(props.total_memory),
                )
            )
        return gpus
    except Exception:
        logger.debug("preflight: torch.cuda probe failed", exc_info=True)
        return []


def _join_nvml_free(gpus: list[GPUInfo]) -> list[GPUInfo]:
    """Fill `available_bytes` with NVML's free VRAM, matched by uuid.

    NVML needs no CUDA context but enumerates the whole node, so the uuid join is
    what lines it up with the `CUDA_VISIBLE_DEVICES`-filtered list. `total_bytes`
    keeps torch's figure — NVML reports nameplate capacity, several hundred MiB
    above the driver-usable total. Unmatched entries keep their capacity."""
    if not any(g.kind == "cuda" and g.uuid for g in gpus):
        return gpus
    free_by_uuid = {g.uuid: g.available_bytes for g in _pynvml_node_discover() if g.uuid}
    if not free_by_uuid:
        logger.debug("preflight: NVML reported no GPUs; free VRAM falls back to capacity")
        return gpus
    joined = []
    for gpu in gpus:
        free = free_by_uuid.get(gpu.uuid) if gpu.kind == "cuda" and gpu.uuid else None
        if free is None:
            logger.debug("preflight: no NVML match for GPU %d (%s); using capacity", gpu.index, gpu.uuid)
        joined.append(gpu if free is None else replace(gpu, available_bytes=free))
    return joined


def _pynvml_node_discover() -> list[GPUInfo]:
    """The physical node's GPUs via NVML, ignoring `CUDA_VISIBLE_DEVICES`.

    `pynvml` is the module name of NVIDIA's `nvidia-ml-py`, pinned transitively by
    vllm/torch; the deprecated third-party package of that name registers it too."""
    try:
        import pynvml
    except Exception:
        logger.debug("preflight: pynvml not installed; node GPU discovery unavailable")
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        logger.debug("preflight: nvmlInit failed; node GPU discovery unavailable", exc_info=True)
        return []
    try:
        gpus: list[GPUInfo] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            gpus.append(
                GPUInfo(index=i, available_bytes=int(mem.free), name=name, uuid=uuid, total_bytes=int(mem.total))
            )
        return gpus
    except Exception:
        logger.debug("preflight: pynvml node discovery failed", exc_info=True)
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _card_sort_key(card: str) -> tuple[int, int | str]:
    """Sorts rocm-smi's "card0".."card10" keys numerically, not lexicographically
    (plain string sort would put "card10" before "card2")."""
    match = re.search(r"\d+$", card)
    return (0, int(match.group())) if match else (1, card)


def _rocm_smi_node_discover() -> list[GPUInfo]:
    """AMD GPUs via `rocm-smi --json`, for a torch-less process (pynvml above is
    NVIDIA-only). Best-effort: any failure returns []."""
    import json
    import shutil
    import subprocess

    binary = shutil.which("rocm-smi")
    if binary is None:
        return []
    try:
        out = subprocess.run(
            [binary, "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        data = json.loads(out.stdout)
    except Exception:
        logger.debug("preflight: rocm-smi node discovery failed", exc_info=True)
        return []
    gpus: list[GPUInfo] = []
    for i, card in enumerate(sorted(data, key=_card_sort_key)):
        info = data[card]
        try:
            total = int(info["VRAM Total Memory (B)"])
            used = int(info["VRAM Total Used Memory (B)"])
        except (KeyError, ValueError, TypeError):
            continue
        name = info.get("Card series") or info.get("Card model") or card
        gpus.append(
            GPUInfo(
                index=i,
                available_bytes=max(0, total - used),
                name=str(name),
                uuid=None,
                kind="rocm",
                total_bytes=total,
            )
        )
    return gpus


# Fraction of total RAM assumed usable by Metal when `iogpu.wired_limit_mb` reads 0
# (its default). macOS's real cap is undocumented, so this is a guess — the
# torch.mps path below reports the OS's own figure and is preferred.
_METAL_DEFAULT_WORKING_SET_FRACTION = 0.7


def _apple_metal_discover() -> list[GPUInfo]:
    """Apple Silicon's unified-memory GPU as a single synthetic GPUInfo (index 0,
    kind="mps"). Intel Macs are excluded; torch MPS is Apple-Silicon-only.

    Prefers `torch.mps.recommended_max_memory()`, an OS-reported figure, and falls
    back to sysctl + psutil so the probe works without torch installed."""
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []

    try:
        import psutil

        total_ram = psutil.virtual_memory().total
        available_ram = psutil.virtual_memory().available

        try:
            import torch

            cap = int(torch.mps.recommended_max_memory()) if torch.backends.mps.is_available() else None
        except Exception:
            cap = None

        if cap is None:
            wired_limit_mb = _sysctl_int("iogpu.wired_limit_mb")
            cap = (
                wired_limit_mb * 1024 * 1024 if wired_limit_mb else int(total_ram * _METAL_DEFAULT_WORKING_SET_FRACTION)
            )

        available_bytes = min(cap, available_ram)
        name = _sysctl_str("machdep.cpu.brand_string") or "Apple GPU"
        return [GPUInfo(index=0, available_bytes=available_bytes, name=name, uuid=None, kind="mps", total_bytes=cap)]
    except Exception:
        logger.debug("preflight: apple metal probe failed", exc_info=True)
        return []


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True)
        return int(out.stdout.strip())
    except Exception:
        return None


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
