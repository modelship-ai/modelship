"""Checks that run before anything is downloaded."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil

from .variants import Variant

CUDA_TOOLKIT_DOCS = "https://docs.model-ship.ai/install-native/#platform-prerequisites"


class GateError(Exception):
    pass


def check_platform(variant: Variant) -> None:
    if platform.system() == "Windows":
        raise GateError("error: Windows is not supported.\nRun the Docker images, or use WSL2 with a Linux install.")
    if _is_musl():
        raise GateError(
            "error: musl-based distros (Alpine) are not supported.\n"
            "ray publishes no musllinux wheels, so the engine cannot be installed.\n"
            "Use a glibc distro, or run the Docker images, which are Debian-based."
        )
    if variant.serves_models and platform.system() == "Linux":
        glibc = os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
        if tuple(int(part) for part in glibc.split(".")[:2]) < (2, 39):
            raise GateError(
                f"error: --{variant.name} needs glibc 2.39 or newer; this host has {glibc}.\n"
                "Upgrade to Ubuntu 24.04, Debian 13, RHEL 10 or later, or run the Docker images."
            )


def _is_musl() -> bool:
    if platform.system() != "Linux":
        return False
    if "musl" in (platform.libc_ver()[0] or "").lower():
        return True
    # libc_ver() reports the Python binary's libc, not the host's.
    return bool(_ld_musl_present())


def _ld_musl_present() -> bool:
    for directory in ("/lib", "/usr/lib"):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        if any(e.startswith("ld-musl-") for e in entries):
            return True
    return False


def _nvidia_driver() -> tuple[int, int] | None:
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return None
    except (OSError, AttributeError):
        return None
    try:
        count = ctypes.c_uint()
        cuda = ctypes.c_int()
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return None
        if nvml.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(cuda)) != 0:
            return None
        return count.value, cuda.value
    except AttributeError:
        return None
    finally:
        nvml.nvmlShutdown()


def check_hardware(required: str | None) -> None:
    if required == "cuda":
        _check_nvidia_driver()
    elif required == "metal" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise GateError(
            f"error: --metal needs Apple Silicon, found {platform.system()}/{platform.machine()}.\n"
            "Use --cuda for an NVIDIA host or --cpu for a CPU-only node."
        )


def _check_nvidia_driver() -> None:
    driver = _nvidia_driver()
    if driver is None or driver[0] == 0:
        raise GateError(
            "error: --cuda needs an NVIDIA GPU, and none was found.\n"
            "The NVIDIA driver's NVML (libnvidia-ml.so.1) must load and list at least one device.\n"
            "Install the NVIDIA driver first, or use --cpu for a CPU-only node."
        )
    if driver[1] < 13000:
        raise GateError(
            f"error: --cuda needs a driver that supports CUDA 13.0, and this one supports CUDA "
            f"{driver[1] // 1000}.{driver[1] % 1000 // 10}.\nUpdate the NVIDIA driver."
        )


def find_nvcc() -> str | None:
    """CUDA_HOME/CUDA_PATH, then PATH, then /usr/local/cuda — torch's own order in
    `cpp_extension._find_cuda_home`. The CUDA apt packages leave nvcc off PATH."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = (os.environ.get(var) or "").strip()
        if root and os.path.exists(candidate := os.path.join(root, "bin", "nvcc")):
            return candidate
    if found := shutil.which("nvcc"):
        return found
    return "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None


def cuda_toolkit_gaps() -> list[str]:
    """What flashinfer's JIT needs beyond the driver."""
    gaps = []
    if find_nvcc() is None:
        gaps.append("nvcc (CUDA toolkit)")
    if shutil.which("ninja") is None:
        gaps.append("ninja")
    return gaps


def check_toolchain(variant_name: str) -> None:
    if variant_name != "cuda" or not (gaps := cuda_toolkit_gaps()):
        return
    raise GateError(
        f"error: --cuda is missing {' and '.join(gaps)}.\n"
        "vLLM and Diffusers JIT-compile kernels at model load and fail without them.\n"
        f"See {CUDA_TOOLKIT_DOCS}"
    )
