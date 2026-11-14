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
_LLAMA_CPP_TAG = "b11769"
_LLAMA_CPP_RELEASE = "llamacpp-b11769-r2"
_LLAMA_CPP_BUILDS_REPO = "modelship-ai/llama-cpp-builds"
_LLAMA_CPP_RELEASE_URL = f"https://github.com/{_LLAMA_CPP_BUILDS_REPO}/releases/download/{_LLAMA_CPP_RELEASE}"

# Line-anchored: llama-cpp-build.yml's pin job rewrites these by sed.
_SHA256_LINUX_X64 = "a5f36a720f2408065cf8efd05a7be991ffce2f8273ddb89e96e1eb56bb0b8535"
_SHA256_LINUX_ARM64 = "01493ada9f64aec240a81f5d5cbf1e26233d69aed8d5f3d1b88a398932728d5a"
_SHA256_MACOS_ARM64_METAL = "2228672fbe71732affbacfe32b4e74141e69f46c2b4d1e2619a9ce1c4df941b7"
_SHA256_CUDA_X64 = "798760ec372c278342b9a2744ee4a8244765d83005d819cf9445fa40dc7a7711"
_SHA256_VULKAN_X64 = "71816c34e197b7eb17dda6a87d85748b8e2450f19f50c4306358a2e64aa02aad"
_SHA256_VULKAN_ARM64 = "259551bcb59ea592f2f918028a7b507f690953bdd7996fdf1e6096854f8cc3a5"

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
