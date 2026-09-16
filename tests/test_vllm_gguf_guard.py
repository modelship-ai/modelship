"""The vllm loader rejects GGUF models at driver preflight (0.24 dropped in-tree GGUF)."""

from unittest.mock import patch

import pytest

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.sources import HfSource


def _make_cfg(**overrides) -> ModelshipModelConfig:
    base = {
        "name": "m",
        "model": "some/repo-GGUF:*Q4_K_M.gguf",
        "usecase": ModelUsecase.generate,
        "loader": ModelLoader.vllm,
    }
    base.update(overrides)
    return ModelshipModelConfig(**base)


# A single resolved .gguf file — driver knows the filename from the repo
# listing alone, no download needed for the guard to fire.
_GGUF_PIN = HfSource(
    repo="some/repo-GGUF",
    revision="deadbeef",
    filename="model-Q4_K_M.gguf",
    patterns=None,
    first_shard=None,
    total_bytes=None,
)
_SNAPSHOT_PIN = HfSource(
    repo="some/fp8-repo",
    revision="deadbeef",
    filename=None,
    patterns=["*.safetensors"],
    first_shard=None,
    total_bytes=None,
)


class TestVllmGgufGuard:
    def test_vllm_gguf_rejected(self):
        cfg = _make_cfg(loader=ModelLoader.vllm)
        with (
            patch("modelship.infer.sources.check_model_source", return_value=_GGUF_PIN),
            pytest.raises(ValueError, match="GGUF"),
        ):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))

    def test_llama_server_gguf_allowed(self):
        cfg = _make_cfg(loader=ModelLoader.llama_server, num_gpus=0)
        with patch("modelship.infer.sources.check_model_source", return_value=_GGUF_PIN):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))
        assert cfg._pinned_source == _GGUF_PIN
        assert _GGUF_PIN.resolves_to_gguf

    def test_vllm_non_gguf_allowed(self):
        cfg = _make_cfg(loader=ModelLoader.vllm, model="some/fp8-repo")
        with patch("modelship.infer.sources.check_model_source", return_value=_SNAPSHOT_PIN):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))
        assert cfg._pinned_source == _SNAPSHOT_PIN
        assert not _SNAPSHOT_PIN.resolves_to_gguf
