"""ModelDeployment.__init__: which failures are fatal, and the order of a successful load."""

import os
import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from modelship.infer.infer_config import ModelLoader
from modelship.infer.model_deployment import ModelDeployment
from modelship.preflight import ModelNotSizedError
from modelship.utils.sources import ModelDownloadError

# Bypass the @serve.deployment wrapper (see test_model_deployment_metrics.py).
_ModelDeployment = ModelDeployment.func_or_class


def _make_config():
    config = MagicMock()
    config.name = "test-model"
    config.loader.value = "vllm"
    config.num_gpus = 0
    return config


@pytest.fixture(autouse=True)
def _cpu_node(monkeypatch):
    monkeypatch.setenv("MSHIP_VARIANT", "cpu")


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
        patch("modelship.deploy.deploy_coordinator.find_coordinator", return_value=coordinator),
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
        patch("modelship.deploy.deploy_coordinator.find_coordinator", return_value=coordinator),
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
        patch("modelship.deploy.deploy_coordinator.find_coordinator", return_value=coordinator),
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
        patch("modelship.deploy.deploy_coordinator.find_coordinator", return_value=None),
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
            BaseInfer=base_infer,
            wait_for_deploy_lease=lease,
        ),
        patch.dict(sys.modules, {"modelship.infer.llama_server.llama_server_infer": loader_module}),
    ):
        await _ModelDeployment.__init__(inst, config)

    assert events == ["download", "lease", "load", "warmup"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("variant", "expected"), [("vulkan", "2"), ("cpu", None), ("cuda", None), ("metal", None)])
async def test_a_vulkan_replica_hands_its_gpu_slots_to_ggml(monkeypatch, variant, expected):
    monkeypatch.setenv("MSHIP_VARIANT", variant)
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()
    config.loader = ModelLoader.llama_server

    ray_module = MagicMock()
    ray_module.get_runtime_context.return_value.get_accelerator_ids.return_value = {"GPU": ["0"]}
    probe = MagicMock()
    probe.ggml_visible_devices.side_effect = lambda slots: "2" if slots == ["0"] else "?"
    base_infer = MagicMock()
    base_infer.ensure_downloaded = AsyncMock()
    infer = MagicMock()
    infer.start = AsyncMock()
    infer.warmup = AsyncMock()
    loader_module = MagicMock()
    loader_module.LlamaServerInfer = MagicMock(return_value=infer)

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            reject_unset_cache_roots=MagicMock(),
            BaseInfer=base_infer,
            wait_for_deploy_lease=AsyncMock(),
            ray=ray_module,
            vulkan_probe=probe,
        ),
        patch.dict(sys.modules, {"modelship.infer.llama_server.llama_server_infer": loader_module}),
    ):
        await _ModelDeployment.__init__(inst, config)

    assert os.environ.get("GGML_VK_VISIBLE_DEVICES") == expected
