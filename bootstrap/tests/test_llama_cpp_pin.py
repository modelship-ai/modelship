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


def test_server_build_url_embeds_pinned_tag():
    for server in llama_cpp._SERVER_BUILDS.values():
        assert llama_cpp._LLAMA_CPP_TAG in server.url
        assert server.url.startswith(f"https://github.com/{llama_cpp._LLAMA_CPP_BUILDS_REPO}/")


def test_release_is_a_build_of_the_pinned_tag():
    assert re.fullmatch(rf"llamacpp-{llama_cpp._LLAMA_CPP_TAG}(-r\d+)?", llama_cpp._LLAMA_CPP_RELEASE)
    for url in (
        *(a.url for a in llama_cpp._SERVER_BUILDS.values()),
        *(a.url for a in llama_cpp._GGML_BACKENDS.values()),
    ):
        assert f"/releases/download/{llama_cpp._LLAMA_CPP_RELEASE}/" in url


def test_backends_come_from_the_same_build():
    """A backend is dlopened beside the server tarball's libggml-base.so; a
    different tag is an ABI mismatch."""
    for backend in llama_cpp._GGML_BACKENDS.values():
        assert llama_cpp._LLAMA_CPP_TAG in backend.url
        assert backend.url.startswith(f"https://github.com/{llama_cpp._LLAMA_CPP_BUILDS_REPO}/")


def test_every_backend_has_a_server_build_to_layer_over():
    assert {(system, machine) for _, system, machine in llama_cpp._GGML_BACKENDS} <= set(llama_cpp._SERVER_BUILDS)


def test_vulkan_backends_are_what_the_workflow_packages():
    built = {m["name"] for m in _build_workflow()["jobs"]["vulkan-backend"]["strategy"]["matrix"]["include"]}
    names = {a.url.rsplit("/", 1)[1] for (variant, *_), a in llama_cpp._GGML_BACKENDS.items() if variant == "vulkan"}
    assert names == {f"libggml-vulkan-{llama_cpp._LLAMA_CPP_TAG}-{name}.tar.gz" for name in built}


def test_pin_job_rewrites_every_pinned_hash():
    """sed exits 0 on a pattern that matches nothing, so a constant missing here keeps its old hash."""
    pinned = re.findall(r"^(_SHA256_\w+) = ", Path(llama_cpp.__file__).read_text(), re.MULTILINE)
    steps = " ".join(step.get("run", "") for step in _build_workflow()["jobs"]["pin"]["steps"])
    assert pinned
    for name in pinned:
        assert f"s|^{name} = .*|" in steps


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


def test_pin_job_branches_from_the_current_default_branch():
    checkout = next(s for s in _build_workflow()["jobs"]["pin"]["steps"] if s.get("name") == "Checkout modelship")
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"


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
