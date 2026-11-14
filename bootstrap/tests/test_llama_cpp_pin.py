"""The bootstrapper's pinned llama.cpp tag must not drift from what
llama-cpp-build.yml actually builds; images carry no pin of their own, fetching through `mship bootstrap` instead."""

import re
from pathlib import Path

import yaml

from mship_bootstrap import llama_cpp

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "llama-cpp-build.yml"


def _build_workflow() -> dict:
    return yaml.safe_load(_BUILD_WORKFLOW.read_text())


def test_dockerfile_carries_no_pin_of_its_own():
    """A reintroduced ARG would be a second source of truth the pin job no longer
    rewrites."""
    assert "LLAMA_CPP_TAG" not in (_REPO_ROOT / "Dockerfile").read_text()


def test_asset_url_embeds_pinned_tag():
    for asset in llama_cpp._ASSETS.values():
        assert llama_cpp._LLAMA_CPP_TAG in asset.url
        assert asset.url.startswith(f"https://github.com/{llama_cpp._LLAMA_CPP_BUILDS_REPO}/")


def test_release_is_a_build_of_the_pinned_tag():
    assert re.fullmatch(rf"llamacpp-{llama_cpp._LLAMA_CPP_TAG}(-r\d+)?", llama_cpp._LLAMA_CPP_RELEASE)
    for url in (*(asset.url for asset in llama_cpp._ASSETS.values()), llama_cpp._CUDA_ADDON_URL):
        assert f"/releases/download/{llama_cpp._LLAMA_CPP_RELEASE}/" in url


def test_cuda_addon_comes_from_the_same_build():
    """The backend is dlopened beside the CPU tarball's libggml-base.so; a
    different tag is an ABI mismatch."""
    assert llama_cpp._LLAMA_CPP_TAG in llama_cpp._CUDA_ADDON_URL
    assert llama_cpp._CUDA_ADDON_URL.startswith(f"https://github.com/{llama_cpp._LLAMA_CPP_BUILDS_REPO}/")


def test_build_workflow_matrix_names():
    built = {m["name"] for m in _build_workflow()["jobs"]["server"]["strategy"]["matrix"]["include"]}
    assert built == {"linux-x64", "linux-arm64", "macos-arm64-metal"}


def test_makefile_triggers_the_build_workflow():
    makefile = (_REPO_ROOT / "Makefile").read_text()
    assert f"gh workflow run {_BUILD_WORKFLOW.name}" in makefile


def test_pin_job_rewrites_the_pinned_file():
    pin = _build_workflow()["jobs"]["pin"]
    assert pin["needs"] == "publish"
    steps = " ".join(step.get("run", "") for step in pin["steps"])
    assert "bootstrap/mship_bootstrap/llama_cpp.py" in steps
    assert "s|^_LLAMA_CPP_RELEASE = .*|" in steps


def test_cuda_backend_shares_the_linux_x64_runner():
    """A newer runner image would raise the backend's glibc floor above the
    linux-x64 binaries it is dlopened beside."""
    workflow = _build_workflow()
    x64 = next(m for m in workflow["jobs"]["server"]["strategy"]["matrix"]["include"] if m["name"] == "linux-x64")
    assert workflow["jobs"]["cuda-backend"]["runs-on"] == x64["os"]


def test_vulkan_jobs_share_the_server_runners():
    workflow = _build_workflow()
    servers = {m["name"]: m["os"] for m in workflow["jobs"]["server"]["strategy"]["matrix"]["include"]}
    for job in ("vulkan-backend", "verify-vulkan"):
        rows = {m["name"]: m["os"] for m in workflow["jobs"][job]["strategy"]["matrix"]["include"]}
        assert rows == {name: servers[name] for name in ("linux-x64", "linux-arm64")}
