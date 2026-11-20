import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from modelship.deploy.capabilities import (
    GPU,
    LINUX_ONLY_LOADERS,
    VARIANT_LOADERS,
    capable_variants,
    deployment_label_selector,
    node_labels,
    runs_on,
    runs_on_gpu,
)
from modelship.utils.config_schema import ModelLoader
from modelship.utils.variant import VARIANTS, VariantError

_BOOTSTRAP_DIR = Path(__file__).resolve().parents[1] / "bootstrap"
if str(_BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_DIR))

from mship_bootstrap import sd_cpp  # noqa: E402

from modelship.infer.stable_diffusion_cpp.stable_diffusion_cpp_infer import _GPU_DEVICES  # noqa: E402

_PINS = _BOOTSTRAP_DIR / "mship_bootstrap" / "pins"
_LOADER_PACKAGES = {
    "vllm": "vllm",
    "diffusers": "diffusers",
    "whispercpp": "pywhispercpp",
    "sherpa_onnx": "sherpa-onnx",
}


def _pins(variant: str) -> dict[str, str]:
    pinned = {}
    for line in (_PINS / f"{variant}.txt").read_text().splitlines():
        if match := re.match(r"([A-Za-z0-9_.-]+)==\S+(?:\s*;\s*([^\\]+?))?\s*\\?$", line):
            pinned[match.group(1).lower()] = match.group(2) or ""
    return pinned


def _labels(env: dict[str, str], hostname: str = "box", system: str = "Linux") -> dict[str, str]:
    with (
        patch.dict(os.environ, env, clear=True),
        patch("platform.system", return_value=system),
        patch("socket.gethostname", return_value=hostname),
    ):
        return node_labels()


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
    def test_a_linux_only_pin_is_a_linux_only_loader(self, variant, loader):
        if "sys_platform == 'linux'" in _pins(variant).get(_LOADER_PACKAGES[loader], ""):
            assert loader in LINUX_ONLY_LOADERS

    @pytest.mark.parametrize("loader", list(ModelLoader))
    @pytest.mark.parametrize("num_gpus", [0, 1])
    def test_every_loader_runs_somewhere_on_either_device(self, loader, num_gpus):
        assert capable_variants(loader.value, num_gpus)


class TestSdServerLoader:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_listed_where_sd_server_is_built(self, variant):
        built = {name for name, _, _ in sd_cpp._BUILDS}
        assert ("stable_diffusion_cpp" in VARIANT_LOADERS[variant]) == (variant in built)

    def test_on_a_gpu_where_the_loader_names_one(self):
        gpu_variants = {v for v, loaders in VARIANT_LOADERS.items() if GPU in loaders.get("stable_diffusion_cpp", ())}
        assert gpu_variants == set(_GPU_DEVICES)


class TestNodeLabels:
    def test_variant_os_and_hostname(self):
        assert _labels({"MSHIP_VARIANT": "cuda"}, hostname="node-a.lan") == {
            "modelship/variant": "cuda",
            "modelship/os": "linux",
            "modelship/hostname": "node-a",
        }

    def test_hostname_is_made_a_valid_label_value(self):
        labels = _labels({"MSHIP_VARIANT": "cpu"}, hostname="_My Box!" + "x" * 80)
        assert labels["modelship/hostname"] == "My-Box-" + "x" * 55

    def test_no_hostname_label_when_nothing_is_left(self):
        assert "modelship/hostname" not in _labels({"MSHIP_VARIANT": "cpu"}, hostname="--")

    def test_operator_labels_are_added(self):
        env = {"MSHIP_VARIANT": "vulkan", "MSHIP_NODE_LABELS": " gpu=fast, example.com/rack = r1 ,"}
        labels = _labels(env)
        assert labels["gpu"] == "fast"
        assert labels["example.com/rack"] == "r1"
        assert labels["modelship/variant"] == "vulkan"

    @pytest.mark.parametrize(("value", "message"), [("gpu", "is not key=value"), ("=x", "is not key=value")])
    def test_refuses_an_operator_label_without_a_key_and_value(self, value, message):
        with pytest.raises(ValueError, match=message):
            _labels({"MSHIP_VARIANT": "cpu", "MSHIP_NODE_LABELS": value})

    def test_refuses_the_reserved_prefix(self):
        with pytest.raises(ValueError, match="reserved prefix modelship/"):
            _labels({"MSHIP_VARIANT": "cpu", "MSHIP_NODE_LABELS": "modelship/variant=cuda"})

    def test_refuses_without_a_variant(self):
        with pytest.raises(VariantError, match="MSHIP_VARIANT is not set"):
            _labels({})


class TestDeploymentLabelSelector:
    @pytest.mark.parametrize(
        ("loader", "num_gpus", "variants"),
        [
            ("vllm", 1, "in(cuda)"),
            ("vllm", 0.5, "in(cuda)"),
            ("vllm", 0, "in(cpu,vulkan)"),
            ("llama_server", 1, "in(cuda,vulkan,metal)"),
            ("llama_server", 0, "in(cuda,cpu,vulkan,metal)"),
            ("whispercpp", 1, "in(metal)"),
            ("sherpa_onnx", 1, "in(cuda,cpu,vulkan,metal)"),
        ],
    )
    def test_selects_the_variants_that_run_its_loader_on_its_device(self, loader, num_gpus, variants):
        assert deployment_label_selector(loader, num_gpus, {})["modelship/variant"] == variants

    def test_a_linux_only_loader_selects_linux(self):
        assert deployment_label_selector("vllm", 0, {})["modelship/os"] == "linux"
        assert "modelship/os" not in deployment_label_selector("llama_server", 0, {})

    def test_node_selector_is_added_and_replaces_a_computed_key(self):
        selector = deployment_label_selector("llama_server", 1, {"modelship/variant": "vulkan", "gpu": "fast"})
        assert selector == {"modelship/variant": "vulkan", "gpu": "fast"}

    @pytest.mark.parametrize(("loader", "expected"), [("sherpa_onnx", False), ("whispercpp", True), ("vllm", True)])
    def test_runs_on_gpu(self, loader, expected):
        assert runs_on_gpu(loader) is expected


class TestRunsOn:
    def test_a_cpu_vllm_deploy_runs_only_on_a_linux_cpu_or_vulkan_node(self):
        assert [v for v in VARIANTS if runs_on("vllm", 0, v, "Linux")] == ["cpu", "vulkan"]
        assert not runs_on("vllm", 0, "cpu", "Darwin")

    def test_a_gpu_whispercpp_deploy_runs_only_on_a_metal_node(self):
        assert [v for v in VARIANTS if runs_on("whispercpp", 1, v, "Darwin")] == ["metal"]

    def test_a_thin_node_runs_nothing(self):
        assert not any(runs_on(loader.value, n, "thin", "Linux") for loader in ModelLoader for n in (0, 1))


class TestCapabilitiesModuleIsRayFree:
    def test_import_does_not_pull_in_ray(self):
        result = subprocess.run(
            [sys.executable, "-c", "import modelship.deploy.capabilities, sys; assert 'ray' not in sys.modules"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
