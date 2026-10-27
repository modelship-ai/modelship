"""ModelDeployment.__init__: which failures are fatal, and the order of a successful load."""

import logging
import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from modelship.infer.infer_config import ModelLoader
from modelship.infer.model_deployment import (
    ModelDeployment,
    _reject_unsupported_accelerator,
    _reject_unsupported_darwin_loader,
)
from modelship.infer.sources import ModelDownloadError
from modelship.preflight import ModelNotSizedError

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
async def test_a_download_error_is_reported_as_not_fatal():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ModelDownloadError("network blip"))

    coordinator = MagicMock()
    coordinator.report_load_failure.remote = AsyncMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
        ),
        patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=coordinator),
        patch("modelship.infer.model_deployment.serve.get_replica_context", return_value=MagicMock(app_name="app")),
        pytest.raises(ModelDownloadError),
    ):
        await _ModelDeployment.__init__(inst, config)

    coordinator.report_load_failure.remote.assert_called_once_with("app", "vllm", "network blip", False)


@pytest.mark.asyncio
async def test_generic_init_failure_reports_fatal():
    """Control case: a non-download init failure is unchanged — still
    reported fatal, still wrapped in RuntimeError."""
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ValueError("bad config"))

    coordinator = MagicMock()
    coordinator.report_load_failure.remote = AsyncMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
        ),
        patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=coordinator),
        patch("modelship.infer.model_deployment.serve.get_replica_context", return_value=MagicMock(app_name="app")),
        pytest.raises(RuntimeError),
    ):
        await _ModelDeployment.__init__(inst, config)

    coordinator.report_load_failure.remote.assert_called_once_with("app", "vllm", ANY, True)


@pytest.mark.asyncio
async def test_a_model_that_could_not_be_sized_is_reported_without_a_traceback():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ModelNotSizedError("could not be sized for this node"))

    coordinator = MagicMock()
    coordinator.report_load_failure.remote = AsyncMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
        ),
        patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=coordinator),
        patch("modelship.infer.model_deployment.serve.get_replica_context", return_value=MagicMock(app_name="app")),
        pytest.raises(RuntimeError, match=r"^model 'test-model' could not be sized for this node$"),
    ):
        await _ModelDeployment.__init__(inst, config)

    coordinator.report_load_failure.remote.assert_called_once_with(
        "app", "vllm", "could not be sized for this node", True
    )


@pytest.mark.asyncio
async def test_an_init_failure_without_a_deploy_coordinator_still_raises(caplog):
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ValueError("bad config"))

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=mock_base_infer,
        ),
        patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=None),
        pytest.raises(RuntimeError, match="bad config"),
    ):
        await _ModelDeployment.__init__(inst, config)

    assert f"No deploy coordinator to report the load failure of '{config.name}' to" in caplog.messages


@pytest.mark.asyncio
async def test_loads_once_the_lease_is_granted():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()
    config.loader = ModelLoader.llama_server
    events = []

    base_infer = MagicMock()
    base_infer.ensure_downloaded = AsyncMock(side_effect=lambda c: events.append("download"))

    async def lease(model_name):
        events.append("lease")

    infer = MagicMock()
    infer.start = AsyncMock()
    infer.warmup = AsyncMock(side_effect=lambda: events.append("warmup"))
    loader_module = MagicMock()
    loader_module.LlamaServerInfer = MagicMock(side_effect=lambda c: events.append("load") or infer)

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            _reject_unsupported_darwin_loader=MagicMock(),
            _reject_unsupported_accelerator=MagicMock(),
            BaseInfer=base_infer,
            wait_for_deploy_lease=lease,
        ),
        patch.dict(sys.modules, {"modelship.infer.llama_server.llama_server_infer": loader_module}),
    ):
        await _ModelDeployment.__init__(inst, config)

    assert events == ["download", "lease", "load", "warmup"]


@pytest.mark.asyncio
async def test_a_load_records_its_phases(caplog):
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()
    config.loader = ModelLoader.llama_server
    infer = MagicMock(start=AsyncMock(), warmup=AsyncMock())
    loader_module = MagicMock(LlamaServerInfer=MagicMock(return_value=infer))
    download_seconds = MagicMock()
    engine_seconds = MagicMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            _reject_unsupported_darwin_loader=MagicMock(),
            _reject_unsupported_accelerator=MagicMock(),
            BaseInfer=MagicMock(ensure_downloaded=AsyncMock()),
            wait_for_deploy_lease=AsyncMock(),
            time=MagicMock(monotonic=MagicMock(side_effect=[0.0, 40.0, 43.0, 63.0])),
            MODEL_DOWNLOAD_DURATION_SECONDS=download_seconds,
            MODEL_ENGINE_LOAD_DURATION_SECONDS=engine_seconds,
        ),
        patch.dict(sys.modules, {"modelship.infer.llama_server.llama_server_infer": loader_module}),
        caplog.at_level(logging.INFO, logger="modelship.infer.deployment"),
    ):
        await _ModelDeployment.__init__(inst, config)

    tags = {"model": "test-model", "loader": "llama_server"}
    download_seconds.observe.assert_called_once_with(40.0, tags=tags)
    engine_seconds.observe.assert_called_once_with(20.0, tags=tags)
    assert "Loaded 'test-model' in 63.0s: download 40.0s, lease wait 3.0s, engine 20.0s" in caplog.messages


@pytest.mark.asyncio
async def test_a_failed_load_records_no_phases():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()
    config.loader = ModelLoader.llama_server
    infer = MagicMock(start=AsyncMock(side_effect=RuntimeError("engine died")), warmup=AsyncMock())
    loader_module = MagicMock(LlamaServerInfer=MagicMock(return_value=infer))
    download_seconds = MagicMock()
    engine_seconds = MagicMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            _reject_unsupported_darwin_loader=MagicMock(),
            _reject_unsupported_accelerator=MagicMock(),
            BaseInfer=MagicMock(ensure_downloaded=AsyncMock()),
            wait_for_deploy_lease=AsyncMock(),
            MODEL_DOWNLOAD_DURATION_SECONDS=download_seconds,
            MODEL_ENGINE_LOAD_DURATION_SECONDS=engine_seconds,
        ),
        patch.dict(sys.modules, {"modelship.infer.llama_server.llama_server_infer": loader_module}),
        patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=None),
        pytest.raises(RuntimeError, match="engine died"),
    ):
        await _ModelDeployment.__init__(inst, config)

    download_seconds.observe.assert_not_called()
    engine_seconds.observe.assert_not_called()
