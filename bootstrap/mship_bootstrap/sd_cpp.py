"""sd-server provisioning.

Must run *after* the engine environment exists: the cuda wrapper puts torch's
bundled `nvidia/cu13/lib` on the loader path, where the CUDA build finds
libcudart and libcublas.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import NamedTuple

from . import paths
from .fetch import fetch_and_extract_archive
from .variants import Variant

# Own-CI builds, one per platform and backend; see .github/workflows/sd-cpp-build.yml.
_SD_CPP_TAG = "master-1106-532d2c1"
_SD_CPP_RELEASE = "sdcpp-master-1106-532d2c1"
_ENGINE_BUILDS_REPO = "modelship-ai/engine-builds"

# Line-anchored: sd-cpp-build.yml's pin job rewrites these by sed.
_SHA256_LINUX_X64_CPU = "a6519372a2e372867f417e8f201b6190e9a391f372e230d2753031d63146ccea"
_SHA256_LINUX_ARM64_CPU = "b99c8b54d9f9d743671a6764ef4c8b1db1d1abbd52adf125259e4841fec0dee6"
_SHA256_LINUX_X64_VULKAN = "206122c08663c17b58c35901045dfaef6db6075207bd30f9b97e00f443028bd2"
_SHA256_LINUX_ARM64_VULKAN = "88bf8386a029252a316bb43f8113d9e93491384f4e46d95afbabda804f7f4939"
_SHA256_LINUX_X64_CUDA13 = "1729a023a6a10677ce41885e7b7d09786f4cac988a55d15bd1ab2922529496d5"
_SHA256_LINUX_ARM64_CUDA13 = "dcdef794faa70594751a913c012894bb4205bdb2da1c0f0407f203c371ce1bd7"
_SHA256_MACOS_ARM64_METAL = "8b81606c839c5dc25de2fd500ed62a42195144586092bc3c94f54e6825c7066c"

# Static because the engine environment is pinned to one CPython minor.
_TORCH_CUDA_LIBS = os.path.join("lib", "python3.12", "site-packages", "nvidia", "cu13", "lib")


class _Build(NamedTuple):
    """`lib_env` is the loader-path env var the wrapper script exports."""

    name: str
    sha256: str
    lib_env: str


_BUILDS: dict[tuple[str, str, str], _Build] = {
    ("cpu", "Linux", "x86_64"): _Build("linux-x64-cpu", _SHA256_LINUX_X64_CPU, "LD_LIBRARY_PATH"),
    ("cpu", "Linux", "aarch64"): _Build("linux-arm64-cpu", _SHA256_LINUX_ARM64_CPU, "LD_LIBRARY_PATH"),
    ("vulkan", "Linux", "x86_64"): _Build("linux-x64-vulkan", _SHA256_LINUX_X64_VULKAN, "LD_LIBRARY_PATH"),
    ("vulkan", "Linux", "aarch64"): _Build("linux-arm64-vulkan", _SHA256_LINUX_ARM64_VULKAN, "LD_LIBRARY_PATH"),
    ("cuda", "Linux", "x86_64"): _Build("linux-x64-cuda13", _SHA256_LINUX_X64_CUDA13, "LD_LIBRARY_PATH"),
    ("cuda", "Linux", "aarch64"): _Build("linux-arm64-cuda13", _SHA256_LINUX_ARM64_CUDA13, "LD_LIBRARY_PATH"),
    ("metal", "Darwin", "arm64"): _Build("macos-arm64-metal", _SHA256_MACOS_ARM64_METAL, "DYLD_LIBRARY_PATH"),
    # The Metal build runs `--backend cpu` without touching the GPU.
    ("cpu", "Darwin", "arm64"): _Build("macos-arm64-metal", _SHA256_MACOS_ARM64_METAL, "DYLD_LIBRARY_PATH"),
}


def _url(build: _Build) -> str:
    return (
        f"https://github.com/{_ENGINE_BUILDS_REPO}/releases/download/{_SD_CPP_RELEASE}/"
        f"sd-server-{_SD_CPP_TAG}-{build.name}.tar.gz"
    )


class SdCppError(Exception):
    pass


def provision(variant: Variant) -> str | None:
    """Downloads the build if needed. Returns the wrapper path."""
    if explicit := os.environ.get("MSHIP_SD_SERVER_BIN"):
        return explicit

    build = _BUILDS.get((variant.name, platform.system(), platform.machine()))
    if build is None:
        raise SdCppError(
            f"error: there is no stable-diffusion.cpp build for --{variant.name} on "
            f"{platform.system()}/{platform.machine()}."
        )

    try:
        return _resolve(variant, build, fetch=True)
    except Exception as e:
        raise SdCppError(f"error: stable-diffusion.cpp {_SD_CPP_TAG} provisioning failed: {e}") from e


def locate(variant: Variant) -> str | None:
    """Same, but never downloads: for `deploy`, which installs nothing."""
    if explicit := os.environ.get("MSHIP_SD_SERVER_BIN"):
        return explicit

    build = _BUILDS.get((variant.name, platform.system(), platform.machine()))
    if build is None:
        return None

    try:
        return _resolve(variant, build, fetch=False)
    except Exception as e:
        print(f"warning: sd-server lookup failed: {e}", file=sys.stderr)
        return None


def _resolve(variant: Variant, build: _Build, *, fetch: bool) -> str | None:
    release_dir = os.path.join(paths.builds_dir(variant.name), "stable-diffusion.cpp", _SD_CPP_RELEASE)
    archive_path = os.path.join(release_dir, "archive.tar.gz")
    extract_dir = os.path.join(release_dir, "extracted")
    wrapper_path = os.path.join(release_dir, "sd-server.sh")
    binary = os.path.join(extract_dir, "sd-server")
    cuda = variant.name == "cuda"

    if not fetch:
        # Read-only: deploy may run as a UID that doesn't own the build.
        if not (os.path.isfile(binary) and os.path.isfile(wrapper_path)):
            print(
                f"warning: no stable-diffusion.cpp {_SD_CPP_TAG} build under {release_dir}; "
                f"the stable_diffusion_cpp loader will not work. Run: mship bootstrap --{variant.name}",
                file=sys.stderr,
            )
            return None
        for path in (wrapper_path, binary):
            if not os.access(path, os.X_OK):
                print(
                    f"warning: {path} is not executable by this user; the stable_diffusion_cpp loader will not work.",
                    file=sys.stderr,
                )
                return None
        return wrapper_path

    if cuda and not os.path.isdir(torch_cuda_libs := _torch_cuda_libs(variant)):
        raise SdCppError(
            f"{torch_cuda_libs} is missing; the stable-diffusion.cpp CUDA build loads torch's libcudart and "
            "libcublas from there"
        )

    if not os.path.isfile(binary):
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"mship: fetching stable-diffusion.cpp {_SD_CPP_TAG}", flush=True)
        fetch_and_extract_archive(_url(build), build.sha256, archive_path, extract_dir, flatten=True, keep_archive=True)

    # Also repairs an existing build.
    os.chmod(binary, 0o755)
    # Bakes in a venv-specific library path.
    _write_wrapper(wrapper_path, extract_dir, build.lib_env, variant, cuda=cuda)
    return wrapper_path


def _torch_cuda_libs(variant: Variant) -> str:
    return os.path.join(paths.venv_dir(variant.name), _TORCH_CUDA_LIBS)


def _write_wrapper(wrapper_path: str, extract_dir: str, lib_env: str, variant: Variant, *, cuda: bool) -> None:
    lib_dirs = [extract_dir, _torch_cuda_libs(variant)] if cuda else [extract_dir]
    search_path = ":".join(lib_dirs)
    content = (
        f'#!/bin/sh\nexport {lib_env}="{search_path}${{{lib_env}:+:${lib_env}}}"\nexec "{extract_dir}/sd-server" "$@"\n'
    )
    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, 0o755)
