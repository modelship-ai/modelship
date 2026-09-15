"""llama-server provisioning.

Must run *after* the engine environment exists: the cuda wrapper puts torch's
bundled `nvidia/cu13/lib` on the loader path, without which `libggml-cuda.so`
cannot resolve libcudart/libcublas and ggml skips the backend silently.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import NamedTuple

from . import paths
from .fetch import fetch_and_extract_archive
from .variants import Variant

# Own-CI builds for every platform; see .github/workflows/llama-cpp-build.yml.
_LLAMA_CPP_TAG = "b10375"
_LLAMA_CPP_BUILDS_REPO = "modelship-ai/llama-cpp-builds"
_LLAMA_CPP_RELEASE_URL = f"https://github.com/{_LLAMA_CPP_BUILDS_REPO}/releases/download/llamacpp-{_LLAMA_CPP_TAG}"

# Line-anchored: llama-cpp-build.yml's pin job rewrites these by sed.
_SHA256_LINUX_X64 = "64625921d1257485a82cc7eee6de58075d5f81a1b588e3e2817cf9632ffc8090"
_SHA256_LINUX_ARM64 = "122186a168c10c9510b6e43c670515206d3a4ca7f5c10ef9fa4708fbea77a9de"
_SHA256_MACOS_ARM64_METAL = "b3f66fc4f82fbaaa70a3d18c37d1e9cbddc65133cf226b1695dc8c2cd20b4545"
_SHA256_CUDA_X64 = "693d45d45b42902a2746f89e51e7caa62bffa22059673db0255c5b029755256a"

# dlopen'd ggml backend, layered over the linux-x64 tarball.
_CUDA_ADDON_URL = f"{_LLAMA_CPP_RELEASE_URL}/libggml-cuda-{_LLAMA_CPP_TAG}-linux-x64-cuda13.tar.gz"
_CUDA_BACKEND_SO = "libggml-cuda.so"

# Static because the engine environment is pinned to one CPython minor.
_TORCH_CUDA_LIBS = os.path.join("lib", "python3.12", "site-packages", "nvidia", "cu13", "lib")


class _Asset(NamedTuple):
    """`lib_env` is the loader-path env var the wrapper script exports."""

    url: str
    sha256: str
    lib_env: str


_ASSETS: dict[tuple[str, str], _Asset] = {
    ("Darwin", "arm64"): _Asset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-macos-arm64-metal.tar.gz",
        sha256=_SHA256_MACOS_ARM64_METAL,
        lib_env="DYLD_LIBRARY_PATH",
    ),
    ("Linux", "x86_64"): _Asset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-x64.tar.gz",
        sha256=_SHA256_LINUX_X64,
        lib_env="LD_LIBRARY_PATH",
    ),
    ("Linux", "aarch64"): _Asset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-arm64.tar.gz",
        sha256=_SHA256_LINUX_ARM64,
        lib_env="LD_LIBRARY_PATH",
    ),
}


def provision(variant: Variant) -> str | None:
    """Downloads the build if needed. Returns the wrapper path, or None when there
    is no prebuilt asset for this platform. Never fatal — only the llama_server
    loader needs it."""
    return _resolve_or_warn(variant, fetch=True)


def locate(variant: Variant) -> str | None:
    """Same, but never downloads: for `deploy`, which installs nothing."""
    return _resolve_or_warn(variant, fetch=False)


def _resolve_or_warn(variant: Variant, *, fetch: bool) -> str | None:
    if explicit := os.environ.get("MSHIP_LLAMA_SERVER_BIN"):
        return explicit

    asset = _ASSETS.get((platform.system(), platform.machine()))
    if asset is None:
        return None

    try:
        return _resolve(variant, asset, fetch=fetch)
    except Exception as e:
        print(f"warning: llama-server provisioning failed: {e}", file=sys.stderr)
        return None


def _resolve(variant: Variant, asset: _Asset, *, fetch: bool) -> str | None:
    tag_dir = os.path.join(paths.builds_dir(variant.name), "llama.cpp", _LLAMA_CPP_TAG)
    archive_path = os.path.join(tag_dir, "archive.tar.gz")
    extract_dir = os.path.join(tag_dir, "extracted")
    wrapper_path = os.path.join(tag_dir, "llama-server.sh")

    binary = os.path.join(extract_dir, "llama")
    cuda = variant.name == "cuda" and (platform.system(), platform.machine()) == ("Linux", "x86_64")
    built = os.path.isfile(binary) and (not cuda or os.path.isfile(os.path.join(extract_dir, _CUDA_BACKEND_SO)))

    if not fetch:
        # Read-only: deploy may run as a UID that doesn't own the build.
        if built and os.path.isfile(wrapper_path):
            return wrapper_path
        print(
            f"warning: no llama.cpp {_LLAMA_CPP_TAG} build under {tag_dir}; "
            f"the llama_server loader will not work. Run: mship bootstrap --{variant.name}",
            file=sys.stderr,
        )
        return None

    if not built:
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"mship: fetching llama.cpp {_LLAMA_CPP_TAG}", flush=True)
        fetch_and_extract_archive(asset.url, asset.sha256, archive_path, extract_dir, flatten=True, keep_archive=True)
        os.chmod(binary, 0o755)
        if cuda:
            _install_cuda_backend(tag_dir, extract_dir)

    # Bakes in a venv-specific library path.
    _write_wrapper(wrapper_path, extract_dir, asset.lib_env, variant, cuda=cuda)
    return wrapper_path


def _install_cuda_backend(tag_dir: str, extract_dir: str) -> None:
    """fetch_and_extract_archive swaps whole directories, so it can't merge into
    extract_dir."""
    addon_dir = os.path.join(tag_dir, "cuda")
    shutil.rmtree(addon_dir, ignore_errors=True)
    fetch_and_extract_archive(
        _CUDA_ADDON_URL,
        _SHA256_CUDA_X64,
        os.path.join(tag_dir, "cuda.tar.gz"),
        addon_dir,
        flatten=True,
        keep_archive=True,
    )
    os.replace(os.path.join(addon_dir, _CUDA_BACKEND_SO), os.path.join(extract_dir, _CUDA_BACKEND_SO))
    shutil.rmtree(addon_dir, ignore_errors=True)


def _write_wrapper(wrapper_path: str, extract_dir: str, lib_env: str, variant: Variant, *, cuda: bool) -> None:
    lib_dirs = [extract_dir]
    if cuda:
        torch_cuda_libs = os.path.join(paths.venv_dir(variant.name), _TORCH_CUDA_LIBS)
        if os.path.isdir(torch_cuda_libs):
            lib_dirs.append(torch_cuda_libs)
        else:
            print(
                f"warning: {torch_cuda_libs} is missing; llama.cpp will fall back to CPU",
                file=sys.stderr,
            )
    search_path = ":".join(lib_dirs)
    content = (
        f'#!/bin/sh\nexport {lib_env}="{search_path}${{{lib_env}:+:${lib_env}}}"\nexec "{extract_dir}/llama" "$@"\n'
    )
    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, 0o755)


def warn_if_no_cuda_device(wrapper_path: str) -> None:
    try:
        result = subprocess.run(
            [wrapper_path, "serve", "--list-devices"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return
    if "CUDA" not in result.stdout:
        print(
            "warning: llama-server reports no CUDA device; GGUF models will run on CPU",
            file=sys.stderr,
        )
