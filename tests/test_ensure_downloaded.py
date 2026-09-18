"""BaseInfer.ensure_downloaded — the actor-side model download hook."""

from unittest.mock import AsyncMock, patch

import pytest

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.sources import HfSource, LocalSource, ModelDownloadError, ModelSourceError

_PIN = HfSource(
    repo="org/repo",
    revision="deadbeef",
    filename="model.safetensors",
    patterns=None,
    first_shard=None,
    total_bytes=None,
)


def _vllm_config() -> ModelshipModelConfig:
    return ModelshipModelConfig(name="m", model="org/repo", usecase=ModelUsecase.generate, loader=ModelLoader.vllm)


@pytest.mark.asyncio
class TestEnsureDownloadedModel:
    async def test_noop_without_a_pinned_source(self):
        # e.g. loader=custom: resolve_all_model_sources never sets _pinned_source.
        config = _vllm_config()
        with patch("modelship.infer.base_infer.locked_download", new_callable=AsyncMock) as mock_download:
            await BaseInfer.ensure_downloaded(config)
        mock_download.assert_not_awaited()
        assert config._resolved_path is None

    async def test_downloads_and_sets_resolved_path(self):
        config = _vllm_config()
        config._pinned_source = _PIN
        with patch(
            "modelship.infer.base_infer.locked_download",
            new_callable=AsyncMock,
            return_value="/cache/model.safetensors",
        ) as mock_d:
            await BaseInfer.ensure_downloaded(config)
        mock_d.assert_awaited_once_with(_PIN, "m")
        assert config._resolved_path == "/cache/model.safetensors"

    async def test_idempotent_once_resolved_path_is_set(self):
        config = _vllm_config()
        config._pinned_source = _PIN
        config._resolved_path = "/cache/already-there"
        with patch("modelship.infer.base_infer.locked_download", new_callable=AsyncMock) as mock_d:
            await BaseInfer.ensure_downloaded(config)
        mock_d.assert_not_awaited()
        assert config._resolved_path == "/cache/already-there"

    async def test_source_error_is_not_wrapped_as_a_download_error(self, tmp_path):
        config = _vllm_config()
        config._pinned_source = LocalSource(str(tmp_path / "gone"))
        with pytest.raises(ModelSourceError, match="Local path not found"):
            await BaseInfer.ensure_downloaded(config)
        assert config._resolved_path is None

    async def test_download_failure_wrapped_and_path_not_set(self):
        config = _vllm_config()
        config._pinned_source = _PIN
        with (
            patch(
                "modelship.infer.base_infer.locked_download",
                new_callable=AsyncMock,
                side_effect=OSError("network blip"),
            ),
            pytest.raises(ModelDownloadError, match="network blip"),
        ):
            await BaseInfer.ensure_downloaded(config)
        assert config._resolved_path is None


@pytest.mark.asyncio
class TestEnsureDownloadedMmproj:
    def _llama_config(self) -> ModelshipModelConfig:
        config = ModelshipModelConfig(
            name="m",
            model="org/repo:model.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            llama_server_config=LlamaServerConfig(mmproj="org/repo:mmproj.gguf"),
        )
        assert config.llama_server_config is not None
        config.llama_server_config._pinned_mmproj = _PIN
        return config

    async def test_noop_without_a_pinned_mmproj(self):
        config = ModelshipModelConfig(
            name="m",
            model="org/repo:model.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
        )
        with patch("modelship.infer.base_infer.locked_download", new_callable=AsyncMock) as mock_d:
            await BaseInfer.ensure_downloaded(config)
        mock_d.assert_not_awaited()

    async def test_downloads_and_overwrites_mmproj_field(self):
        config = self._llama_config()
        assert config.llama_server_config is not None
        with patch(
            "modelship.infer.base_infer.locked_download", new_callable=AsyncMock, return_value="/cache/mmproj.gguf"
        ) as mock_d:
            await BaseInfer.ensure_downloaded(config)
        mock_d.assert_awaited_once_with(_PIN, "m")
        assert config.llama_server_config.mmproj == "/cache/mmproj.gguf"
        assert config.llama_server_config._pinned_mmproj is None

    async def test_mmproj_failure_wrapped(self):
        config = self._llama_config()
        with (
            patch(
                "modelship.infer.base_infer.locked_download", new_callable=AsyncMock, side_effect=OSError("disk full")
            ),
            pytest.raises(ModelDownloadError, match="mmproj"),
        ):
            await BaseInfer.ensure_downloaded(config)

    async def test_mmproj_source_error_is_not_wrapped(self):
        config = self._llama_config()
        with (
            patch(
                "modelship.infer.base_infer.locked_download",
                new_callable=AsyncMock,
                side_effect=ModelSourceError("stale"),
            ),
            pytest.raises(ModelSourceError, match="stale"),
        ):
            await BaseInfer.ensure_downloaded(config)
