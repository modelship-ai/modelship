"""llama-server provisioning.

Must run *after* the engine environment exists: the cuda wrapper puts torch's
bundled `nvidia/cu13/lib` on the loader path, without which `libggml-cuda.so`
cannot resolve libcudart/libcublas and ggml skips the backend silently.
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

# Own-CI builds for every platform; see .github/workflows/llama-cpp-build.yml.
_LLAMA_CPP_TAG = "b11769"
_LLAMA_CPP_RELEASE = "llamacpp-b11769-r3"
_LLAMA_CPP_BUILDS_REPO = "modelship-ai/llama-cpp-builds"
_LLAMA_CPP_RELEASE_URL = f"https://github.com/{_LLAMA_CPP_BUILDS_REPO}/releases/download/{_LLAMA_CPP_RELEASE}"

# Line-anchored: llama-cpp-build.yml's pin job rewrites these by sed.
_SHA256_LINUX_X64 = "ee6b95778b7346338c9fac1425670b2149c203bbfc3093bf9454254882b00b0b"
_SHA256_LINUX_ARM64 = "a9e9310845d50f4646c59148b5e5007c86bf045666f2aa67f8dcb0f87a5dba5a"
_SHA256_MACOS_ARM64_METAL = "ebea3ff4e81b6120573afc01a26a413bf115899c1fc66f98a08895171ec9b0f2"
_SHA256_CUDA_X64 = "7e6cd1a3d91f49b0d85a60c46e7d51deb2d9dd72ee8a4ac5772d5dae4bc3870e"
_SHA256_CUDA_ARM64 = "e1d58a74753f7b4f3a87b8b26e36f2c52275e58b0a1a27635f9a1b0795c51bf1"
_SHA256_VULKAN_X64 = "e950e0b60c87ac7f1020bd525c0a5634b0e8cd250cb87c6455424088672bd7c0"
_SHA256_VULKAN_ARM64 = "6f98c11402c186594a8c64a556c78b1df2afa1220caf14865650596c10330957"

# Static because the engine environment is pinned to one CPython minor.
_TORCH_CUDA_LIBS = os.path.join("lib", "python3.12", "site-packages", "nvidia", "cu13", "lib")


class _ServerBuild(NamedTuple):
    """`lib_env` is the loader-path env var the wrapper script exports."""

    url: str
    sha256: str
    lib_env: str


_SERVER_BUILDS: dict[tuple[str, str], _ServerBuild] = {
    ("Darwin", "arm64"): _ServerBuild(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-macos-arm64-metal.tar.gz",
        sha256=_SHA256_MACOS_ARM64_METAL,
        lib_env="DYLD_LIBRARY_PATH",
    ),
    ("Linux", "x86_64"): _ServerBuild(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-x64.tar.gz",
        sha256=_SHA256_LINUX_X64,
        lib_env="LD_LIBRARY_PATH",
    ),
    ("Linux", "aarch64"): _ServerBuild(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-arm64.tar.gz",
        sha256=_SHA256_LINUX_ARM64,
        lib_env="LD_LIBRARY_PATH",
    ),
}


class _GgmlBackend(NamedTuple):
    """Dlopen'd by llama.cpp; layered over its platform's server build."""

    url: str
    sha256: str
    so_name: str


_GGML_BACKENDS: dict[tuple[str, str, str], _GgmlBackend] = {
    ("cuda", "Linux", "x86_64"): _GgmlBackend(
        url=f"{_LLAMA_CPP_RELEASE_URL}/libggml-cuda-{_LLAMA_CPP_TAG}-linux-x64-cuda13.tar.gz",
        sha256=_SHA256_CUDA_X64,
        so_name="libggml-cuda.so",
    ),
    ("cuda", "Linux", "aarch64"): _GgmlBackend(
        url=f"{_LLAMA_CPP_RELEASE_URL}/libggml-cuda-{_LLAMA_CPP_TAG}-linux-arm64-cuda13.tar.gz",
        sha256=_SHA256_CUDA_ARM64,
        so_name="libggml-cuda.so",
    ),
    ("vulkan", "Linux", "x86_64"): _GgmlBackend(
        url=f"{_LLAMA_CPP_RELEASE_URL}/libggml-vulkan-{_LLAMA_CPP_TAG}-linux-x64.tar.gz",
        sha256=_SHA256_VULKAN_X64,
        so_name="libggml-vulkan.so",
    ),
    ("vulkan", "Linux", "aarch64"): _GgmlBackend(
        url=f"{_LLAMA_CPP_RELEASE_URL}/libggml-vulkan-{_LLAMA_CPP_TAG}-linux-arm64.tar.gz",
        sha256=_SHA256_VULKAN_ARM64,
        so_name="libggml-vulkan.so",
    ),
}


class LlamaCppError(Exception):
    pass


def provision(variant: Variant) -> str | None:
    """Downloads the build if needed. Returns the wrapper path."""
    if explicit := os.environ.get("MSHIP_LLAMA_SERVER_BIN"):
        return explicit

    server = _SERVER_BUILDS.get((platform.system(), platform.machine()))
    if server is None:
        raise LlamaCppError(f"error: there is no llama.cpp build for {platform.system()}/{platform.machine()}.")

    try:
        return _resolve(variant, server, fetch=True)
    except Exception as e:
        raise LlamaCppError(f"error: llama.cpp {_LLAMA_CPP_TAG} provisioning failed: {e}") from e


def locate(variant: Variant) -> str | None:
    """Same, but never downloads: for `deploy`, which installs nothing."""
    if explicit := os.environ.get("MSHIP_LLAMA_SERVER_BIN"):
        return explicit

    server = _SERVER_BUILDS.get((platform.system(), platform.machine()))
    if server is None:
        return None

    try:
        return _resolve(variant, server, fetch=False)
    except Exception as e:
        print(f"warning: llama-server lookup failed: {e}", file=sys.stderr)
        return None


def _resolve(variant: Variant, server: _ServerBuild, *, fetch: bool) -> str | None:
    release_dir = os.path.join(paths.builds_dir(variant.name), "llama.cpp", _LLAMA_CPP_RELEASE)
    archive_path = os.path.join(release_dir, "archive.tar.gz")
    extract_dir = os.path.join(release_dir, "extracted")
    wrapper_path = os.path.join(release_dir, "llama-server.sh")

    binary = os.path.join(extract_dir, "llama")
    backend = _GGML_BACKENDS.get((variant.name, platform.system(), platform.machine()))
    cuda = variant.name == "cuda" and backend is not None
    built = os.path.isfile(binary) and (backend is None or os.path.isfile(os.path.join(extract_dir, backend.so_name)))

    if not fetch:
        # Read-only: deploy may run as a UID that doesn't own the build.
        if not (built and os.path.isfile(wrapper_path)):
            print(
                f"warning: no llama.cpp {_LLAMA_CPP_TAG} build under {release_dir}; "
                f"the llama_server loader will not work. Run: mship bootstrap --{variant.name}",
                file=sys.stderr,
            )
            return None
        for path in (wrapper_path, binary):
            if not os.access(path, os.X_OK):
                print(
                    f"warning: {path} is not executable by this user; the llama_server loader will not work.",
                    file=sys.stderr,
                )
                return None
        return wrapper_path

    if cuda and not os.path.isdir(torch_cuda_libs := _torch_cuda_libs(variant)):
        raise LlamaCppError(
            f"{torch_cuda_libs} is missing; llama.cpp's CUDA backend loads torch's libcudart and libcublas from there"
        )

    if not built:
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"mship: fetching llama.cpp {_LLAMA_CPP_TAG}", flush=True)
        fetch_and_extract_archive(server.url, server.sha256, archive_path, extract_dir, flatten=True, keep_archive=True)
        if backend is not None:
            _install_ggml_backend(backend, variant, release_dir, extract_dir)

    # Also repairs an existing build.
    os.chmod(binary, 0o755)
    # Bakes in a venv-specific library path.
    _write_wrapper(wrapper_path, extract_dir, server.lib_env, variant, cuda=cuda)
    return wrapper_path


def _install_ggml_backend(backend: _GgmlBackend, variant: Variant, release_dir: str, extract_dir: str) -> None:
    """fetch_and_extract_archive swaps whole directories, so it can't merge into
    extract_dir."""
    backend_dir = os.path.join(release_dir, variant.name)
    shutil.rmtree(backend_dir, ignore_errors=True)
    fetch_and_extract_archive(
        backend.url,
        backend.sha256,
        os.path.join(release_dir, f"{variant.name}.tar.gz"),
        backend_dir,
        flatten=True,
        keep_archive=True,
    )
    os.replace(os.path.join(backend_dir, backend.so_name), os.path.join(extract_dir, backend.so_name))
    shutil.rmtree(backend_dir, ignore_errors=True)


def _torch_cuda_libs(variant: Variant) -> str:
    return os.path.join(paths.venv_dir(variant.name), _TORCH_CUDA_LIBS)


def _write_wrapper(wrapper_path: str, extract_dir: str, lib_env: str, variant: Variant, *, cuda: bool) -> None:
    lib_dirs = [extract_dir, _torch_cuda_libs(variant)] if cuda else [extract_dir]
    search_path = ":".join(lib_dirs)
    content = (
        f'#!/bin/sh\nexport {lib_env}="{search_path}${{{lib_env}:+:${lib_env}}}"\nexec "{extract_dir}/llama" "$@"\n'
    )
    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, 0o755)
