"""ModelDeployment.__init__: a ModelDownloadError must never be
reported to the coordinator as fatal, so it's retried next pass instead of
evicted from the effective config."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.infer.infer_config import ModelLoader
from modelship.infer.model_deployment import (
    ModelDeployment,
    _reject_unsupported_accelerator,
    _reject_unsupported_darwin_loader,
)
from modelship.infer.sources import ModelDownloadError

# Bypass the @serve.deployment wrapper (see test_model_deployment_metrics.py).
_ModelDeployment = ModelDeployment.func_or_class


def _make_config():
    config = MagicMock()
    config.name = "test-model"
    config.loader.value = "vllm"
    config.num_gpus = 0
    return config


class TestRejectUnsupportedDarwinLoader:
    @pytest.mark.parametrize("loader", [ModelLoader.vllm, ModelLoader.diffusers])
    def test_darwin_rejects_unsupported_loaders(self, loader):
        config = MagicMock(loader=loader, name="test-model")
        with (
            patch("modelship.infer.model_deployment.platform.system", return_value="Darwin"),
            pytest.raises(RuntimeError, match=f"loader={loader.value}"),
        ):
            _reject_unsupported_darwin_loader(config)

    def test_darwin_allows_llama_server(self):
        config = MagicMock(loader=ModelLoader.llama_server, name="test-model")
        with patch("modelship.infer.model_deployment.platform.system", return_value="Darwin"):
            _reject_unsupported_darwin_loader(config)  # no raise

    def test_darwin_allows_stable_diffusion_cpp(self):
        # ggml's runtime backend registry picks up Metal automatically.
        config = MagicMock(loader=ModelLoader.stable_diffusion_cpp, name="test-model")
        with patch("modelship.infer.model_deployment.platform.system", return_value="Darwin"):
            _reject_unsupported_darwin_loader(config)  # no raise

    def test_darwin_allows_whispercpp(self):
        config = MagicMock(loader=ModelLoader.whispercpp, name="test-model")
        with patch("modelship.infer.model_deployment.platform.system", return_value="Darwin"):
            _reject_unsupported_darwin_loader(config)  # no raise

    def test_linux_allows_everything(self):
        config = MagicMock(loader=ModelLoader.vllm, name="test-model")
        with patch("modelship.infer.model_deployment.platform.system", return_value="Linux"):
            _reject_unsupported_darwin_loader(config)  # no raise


class TestRejectUnsupportedAccelerator:
    @pytest.mark.parametrize(("accel", "vendor"), [("rocm", "AMD"), ("xpu", "Intel")])
    def test_rejects_amd_and_intel_when_gpus_requested(self, accel, vendor):
        config = MagicMock(num_gpus=1, name="test-model")
        with (
            patch("modelship.infer.model_deployment.detect_accelerator", return_value=accel),
            pytest.raises(RuntimeError, match=vendor),
        ):
            _reject_unsupported_accelerator(config)

    def test_allows_amd_when_no_gpus_requested(self):
        # A CPU-only deploy never touches the GPU, so the vendor is irrelevant.
        config = MagicMock(num_gpus=0, name="test-model")
        with patch("modelship.infer.model_deployment.detect_accelerator", return_value="rocm"):
            _reject_unsupported_accelerator(config)  # no raise

    def test_rejects_a_fractional_gpu_share(self):
        config = MagicMock(num_gpus=0.5, name="test-model")
        with (
            patch("modelship.infer.model_deployment.detect_accelerator", return_value="rocm"),
            pytest.raises(RuntimeError),
        ):
            _reject_unsupported_accelerator(config)

    @pytest.mark.parametrize("accel", ["cuda", "metal", "cpu"])
    def test_allows_supported_accelerators(self, accel):
        config = MagicMock(num_gpus=1, name="test-model")
        with patch("modelship.infer.model_deployment.detect_accelerator", return_value=accel):
            _reject_unsupported_accelerator(config)  # no raise


def _patch_init_globals(**kwargs):
    # @serve.deployment cloudpickles the class, so the unwrapped __init__ carries
    # a reconstructed globals dict; patching the module attribute won't reach it.
    return patch.dict(_ModelDeployment.__init__.__globals__, kwargs)


@pytest.mark.asyncio
async def test_download_error_does_not_report_fatal():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ModelDownloadError("network blip"))

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            _spawn_orphan_reaper=MagicMock(return_value=None),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
            MODEL_LOAD_FAILURES_TOTAL=MagicMock(),
            MODEL_LOAD_DURATION_SECONDS=MagicMock(),
        ),
        patch("modelship.infer.deploy_coordinator.get_or_create_coordinator") as mock_get_coordinator,
        pytest.raises(ModelDownloadError),
    ):
        await _ModelDeployment.__init__(inst, config)

    # No coordinator lookup/report happens on this path.
    mock_get_coordinator.assert_not_called()


@pytest.mark.asyncio
async def test_generic_init_failure_reports_fatal():
    """Control case: a non-download init failure is unchanged — still
    reported fatal, still wrapped in RuntimeError."""
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ValueError("bad config"))

    coordinator = MagicMock()
    coordinator.report_fatal_error.remote = AsyncMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            _spawn_orphan_reaper=MagicMock(return_value=None),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
            MODEL_LOAD_FAILURES_TOTAL=MagicMock(),
            MODEL_LOAD_DURATION_SECONDS=MagicMock(),
            serve=MagicMock(get_replica_context=MagicMock(return_value=MagicMock(app_name="app"))),
        ),
        patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=coordinator),
        pytest.raises(RuntimeError),
    ):
        await _ModelDeployment.__init__(inst, config)

    coordinator.report_fatal_error.remote.assert_called_once()
