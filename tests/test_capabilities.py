import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from modelship.deploy.capabilities import (
    VARIANT_LOADERS,
    deployment_capability_resources,
    node_capability_resources,
    runs_on_gpu,
    variant_loaders,
)
from modelship.utils.config_schema import ModelLoader
from modelship.utils.variant import VARIANTS, VariantError

_PINS = Path(__file__).resolve().parents[1] / "bootstrap" / "mship_bootstrap" / "pins"
_LOADER_PACKAGES = {
    "vllm": "vllm",
    "diffusers": "diffusers",
    "stable_diffusion_cpp": "stable-diffusion-cpp-python",
    "whispercpp": "pywhispercpp",
    "sherpa_onnx": "sherpa-onnx",
}


def _pins(variant: str) -> dict[str, str]:
    pinned = {}
    for line in (_PINS / f"{variant}.txt").read_text().splitlines():
        if match := re.match(r"([A-Za-z0-9_.-]+)==\S+(?:\s*;\s*([^\\]+?))?\s*\\?$", line):
            pinned[match.group(1).lower()] = match.group(2) or ""
    return pinned


def _node(variant: str, system: str = "Linux", machine: str = "x86_64") -> dict[str, float]:
    with (
        patch.dict(os.environ, {"MSHIP_VARIANT": variant}, clear=True),
        patch("platform.system", return_value=system),
        patch("platform.machine", return_value=machine),
    ):
        return node_capability_resources()


class TestVariantLoaders:
    def test_every_variant_has_pins_and_a_table(self):
        assert set(VARIANT_LOADERS) == set(VARIANTS) == {p.stem for p in _PINS.glob("*.txt")}

    def test_every_loader_runs_somewhere(self):
        assert {loader for loaders in VARIANT_LOADERS.values() for loader in loaders} == set(ModelLoader)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("loader", _LOADER_PACKAGES)
    def test_a_python_loader_is_listed_where_its_package_is_pinned(self, variant, loader):
        assert (loader in VARIANT_LOADERS[variant]) == (_LOADER_PACKAGES[loader] in _pins(variant))

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("loader", _LOADER_PACKAGES)
    def test_a_linux_only_pin_is_left_out_on_macos(self, variant, loader):
        if "sys_platform == 'linux'" not in _pins(variant).get(_LOADER_PACKAGES[loader], ""):
            return
        assert loader in variant_loaders(variant, "Linux", "x86_64")
        assert loader not in variant_loaders(variant, "Darwin", "arm64")

    def test_llama_server_is_cpu_only_on_arm64_cuda(self):
        assert variant_loaders("cuda", "Linux", "x86_64")["llama_server"] == {"gpu", "cpu"}
        assert variant_loaders("cuda", "Linux", "aarch64")["llama_server"] == {"cpu"}


class TestNodeCapabilityResources:
    def test_cuda_node(self):
        assert _node("cuda") == {
            "mship_vllm_gpu": 1,
            "mship_diffusers_gpu": 1,
            "mship_diffusers_cpu": 1,
            "mship_llama_server_gpu": 1,
            "mship_llama_server_cpu": 1,
            "mship_stable_diffusion_cpp_cpu": 1,
            "mship_whispercpp_cpu": 1,
            "mship_sherpa_onnx_cpu": 1,
        }

    def test_thin_node_runs_nothing(self):
        assert _node("thin") == {}

    def test_env_override_replaces_the_table(self):
        override = json.dumps({"mship_nemo_gpu": 1})
        with patch.dict(os.environ, {"MSHIP_NODE_CAPABILITIES": override}, clear=True):
            assert node_capability_resources() == {"mship_nemo_gpu": 1.0}

    def test_refuses_without_a_variant(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(VariantError, match="MSHIP_VARIANT is not set"):
            node_capability_resources()


class TestDeploymentCapabilityResources:
    @pytest.mark.parametrize(
        ("loader", "num_gpus", "resource"),
        [
            ("vllm", 1, "mship_vllm_gpu"),
            ("vllm", 0.5, "mship_vllm_gpu"),
            ("vllm", 0, "mship_vllm_cpu"),
            ("whispercpp", 1, "mship_whispercpp_gpu"),
            ("sherpa_onnx", 1, "mship_sherpa_onnx_cpu"),
        ],
    )
    def test_requests_its_loader_on_its_device(self, loader, num_gpus, resource):
        assert deployment_capability_resources(loader, num_gpus) == {resource: 0.001}

    def test_a_cpu_vllm_deploy_lands_only_on_a_cpu_node(self):
        (resource,) = deployment_capability_resources("vllm", 0)
        assert resource not in _node("cuda")
        assert resource in _node("cpu")

    def test_a_gpu_whispercpp_deploy_lands_only_on_a_metal_node(self):
        (resource,) = deployment_capability_resources("whispercpp", 1)
        assert [v for v in VARIANTS if resource in _node(v, "Darwin" if v == "metal" else "Linux")] == ["metal"]

    @pytest.mark.parametrize(("loader", "expected"), [("sherpa_onnx", False), ("whispercpp", True), ("vllm", True)])
    def test_runs_on_gpu(self, loader, expected):
        assert runs_on_gpu(loader) is expected


class TestCapabilitiesModuleIsRayFree:
    def test_import_does_not_pull_in_ray(self):
        result = subprocess.run(
            [sys.executable, "-c", "import modelship.deploy.capabilities, sys; assert 'ray' not in sys.modules"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
