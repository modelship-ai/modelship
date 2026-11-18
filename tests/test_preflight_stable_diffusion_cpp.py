"""Tests for the StableDiffusionCppPreflight estimator."""

from __future__ import annotations

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    StableDiffusionCppConfig,
)
from modelship.preflight import HardwareProfile
from modelship.preflight.base import GPUInfo
from modelship.preflight.stable_diffusion_cpp import (
    _VAE_TILING_RAM_THRESHOLD_BYTES,
    StableDiffusionCppPreflight,
)

_GiB = 1024**3


def _make_config(**sdcpp_kwargs) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="test-image",
        model="org/test-sd-gguf",
        usecase=ModelUsecase.image,
        loader=ModelLoader.stable_diffusion_cpp,
        stable_diffusion_cpp_config=StableDiffusionCppConfig(**sdcpp_kwargs),
    )


def test_no_ram_returns_empty():
    rec = StableDiffusionCppPreflight().recommend(_make_config(), HardwareProfile(ram_bytes=0))
    assert rec == {}


def test_low_ram_recommends_vae_tiling():
    hw = HardwareProfile(ram_bytes=_VAE_TILING_RAM_THRESHOLD_BYTES - _GiB)
    rec = StableDiffusionCppPreflight().recommend(_make_config(), hw)
    assert rec == {"vae_tiling": True}


def test_high_ram_no_recommendation():
    hw = HardwareProfile(ram_bytes=_VAE_TILING_RAM_THRESHOLD_BYTES + _GiB)
    rec = StableDiffusionCppPreflight().recommend(_make_config(), hw)
    assert rec == {}


def _gpu(total_gib: float = 16, free_gib: float | None = None) -> HardwareProfile:
    total = int(total_gib * _GiB)
    free = total if free_gib is None else int(free_gib * _GiB)
    return HardwareProfile(gpus=[GPUInfo(index=0, available_bytes=free, name="gpu", total_bytes=total)])


def _model_file(tmp_path, name: str, gib: float) -> str:
    path = tmp_path / name
    with open(path, "wb") as f:
        f.truncate(int(gib * _GiB))
    return str(path)


def _share_config(tmp_path, num_gpus: float, weights_gib: float, **sdcpp_kwargs) -> ModelshipModelConfig:
    config = _make_config(**sdcpp_kwargs)
    config.num_gpus = num_gpus
    config._resolved_path = _model_file(tmp_path, "model.gguf", weights_gib)
    return config


@pytest.fixture
def variant(monkeypatch):
    def set_variant(name: str) -> None:
        monkeypatch.setenv("MSHIP_VARIANT", name)

    set_variant("cuda")
    return set_variant


class TestGpuShare:
    def test_a_whole_gpu_keeps_sd_cpps_fitting(self, tmp_path, variant):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 1, 3.67), _gpu())
        assert rec == {}

    def test_weights_that_fit_stay_on_the_gpu(self, tmp_path, variant):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.5, 3.67), _gpu(16))
        # 8 GiB share less CUDA's 1 GiB sd.cpp doesn't count.
        assert rec == {"max_vram": 7.0, "auto_fit": False}

    def test_weights_that_dont_fit_spill_to_ram(self, tmp_path, variant, caplog):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.25, 3.67), _gpu(16))
        # 4 - 1 = 3 GiB < 3.67 + 0.5.
        assert rec == {"max_vram": 3.0, "auto_fit": True}
        assert "places what doesn't fit in RAM" in caplog.text

    def test_compute_headroom_counts_toward_fitting(self, tmp_path, variant):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.25, 2.6), _gpu(16))
        assert rec == {"max_vram": 3.0, "auto_fit": True}
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.25, 2.5), _gpu(16))
        assert rec == {"max_vram": 3.0, "auto_fit": False}

    @pytest.mark.parametrize("name", ["vulkan", "metal"])
    def test_vulkan_and_metal_leave_a_quarter_gib(self, tmp_path, variant, name):
        variant(name)
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.25, 2.05), _gpu(16))
        assert rec == {"max_vram": 3.75, "auto_fit": False}

    def test_the_budget_rounds_down(self, tmp_path, variant):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 1 / 3, 2.0), _gpu(16))
        assert rec["max_vram"] == 4.33

    def test_a_tiny_share_still_gets_a_budget(self, tmp_path, variant, caplog):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.05, 2.0), _gpu(16))
        assert rec == {"max_vram": 0.25, "auto_fit": True}
        assert "budgeting 0.25 GiB" in caplog.text

    def test_split_files_count_as_weights(self, tmp_path, variant):
        config = _share_config(
            tmp_path,
            0.5,
            5.0,
            t5xxl_path=_model_file(tmp_path, "t5.gguf", 1.5),
            vae_path=_model_file(tmp_path, "ae.gguf", 0.2),
        )
        # 6.7 + 0.5 > 7.0.
        assert StableDiffusionCppPreflight().recommend(config, _gpu(16)) == {"max_vram": 7.0, "auto_fit": True}

    def test_an_unreadable_model_leaves_auto_fit_on(self, tmp_path, variant, caplog):
        config = _share_config(tmp_path, 0.5, 1.0)
        config._resolved_path = str(tmp_path / "missing.gguf")
        assert StableDiffusionCppPreflight().recommend(config, _gpu(16)) == {"max_vram": 7.0, "auto_fit": True}
        assert "model size unreadable" in caplog.text

    def test_less_free_than_the_share_warns(self, tmp_path, variant, caplog):
        rec = StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.5, 1.0), _gpu(16, free_gib=5))
        assert rec == {"max_vram": 7.0, "auto_fit": False}
        assert "less than its 8.00 GiB share" in caplog.text

    def test_no_visible_gpu_recommends_nothing(self, tmp_path, variant):
        assert StableDiffusionCppPreflight().recommend(_share_config(tmp_path, 0.5, 1.0), HardwareProfile()) == {}
