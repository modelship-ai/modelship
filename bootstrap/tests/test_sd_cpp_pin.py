"""The bootstrapper's pinned stable-diffusion.cpp build must not drift from what sd-cpp-build.yml builds."""

import re
from pathlib import Path

import yaml

from mship_bootstrap import sd_cpp

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "sd-cpp-build.yml"


def _build_workflow() -> dict:
    return yaml.safe_load(_BUILD_WORKFLOW.read_text())


def _pin_steps() -> str:
    return " ".join(step.get("run", "") for step in _build_workflow()["jobs"]["pin"]["steps"])


def test_release_is_a_build_of_the_pinned_tag():
    assert re.fullmatch(rf"sdcpp-{re.escape(sd_cpp._SD_CPP_TAG)}(-r\d+)?", sd_cpp._SD_CPP_RELEASE)


def test_urls_point_at_the_pinned_release():
    for build in sd_cpp._BUILDS.values():
        url = sd_cpp._url(build)
        assert url.startswith(f"https://github.com/{sd_cpp._ENGINE_BUILDS_REPO}/releases/download/")
        assert f"/{sd_cpp._SD_CPP_RELEASE}/sd-server-{sd_cpp._SD_CPP_TAG}-{build.name}.tar.gz" in url


def test_builds_are_what_the_workflow_packages():
    built = {m["name"] for m in _build_workflow()["jobs"]["build"]["strategy"]["matrix"]["include"]}
    assert {b.name for b in sd_cpp._BUILDS.values()} == built


def test_every_build_has_the_hash_constant_the_pin_job_derives():
    """The pin job names each constant `_SHA256_` + the build name upper-cased, `-` → `_`."""
    source = Path(sd_cpp.__file__).read_text()
    for build in sd_cpp._BUILDS.values():
        name = "_SHA256_" + build.name.upper().replace("-", "_")
        match = re.search(rf'^{name} = "([0-9a-f]{{64}})"$', source, re.MULTILINE)
        assert match, name
        assert match.group(1) == build.sha256


def test_no_hash_is_a_placeholder():
    for build in sd_cpp._BUILDS.values():
        assert build.sha256 != "0" * 64, build.name


def test_pin_job_rewrites_the_pinned_file():
    pin = _build_workflow()["jobs"]["pin"]
    assert pin["needs"] == "publish"
    steps = _pin_steps()
    assert "PIN=bootstrap/mship_bootstrap/sd_cpp.py" in steps
    assert "s|^_SD_CPP_TAG = .*|" in steps
    assert "s|^_SD_CPP_RELEASE = .*|" in steps
    assert "s|^${VAR} = .*|" in steps


def test_pin_job_branches_from_the_tip_of_the_runs_branch():
    steps = _build_workflow()["jobs"]["pin"]["steps"]
    checkout = next(s for s in steps if s.get("name") == "Checkout modelship")
    assert checkout["with"]["ref"] == "${{ github.ref_name }}"
    pr = next(s for s in steps if s.get("name") == "Open pin-bump PR")
    assert '--base "${{ github.ref_name }}"' in pr["run"]


def test_publishes_to_the_repo_the_bootstrapper_fetches_from():
    assert _build_workflow()["env"]["ENGINE_BUILDS_REPO"] == sd_cpp._ENGINE_BUILDS_REPO


def test_makefile_triggers_the_build_workflow():
    makefile = (_REPO_ROOT / "Makefile").read_text()
    assert f'gh workflow run {_BUILD_WORKFLOW.name} --ref "$$BRANCH"' in makefile
