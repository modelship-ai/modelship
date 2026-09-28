"""Driver preflight GGUF rules: the vllm loader rejects GGUF (0.24 dropped in-tree GGUF), llama_server requires it."""

import re
from unittest.mock import patch

import pytest

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.sources import HfSource, LocalSource


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


class TestLlamaServerGgufRule:
    @pytest.mark.parametrize(
        "pinned",
        [
            _SNAPSHOT_PIN,
            _GGUF_PIN._replace(filename="README.md"),
            LocalSource("/models/snapshot"),
        ],
        ids=["snapshot", "non-gguf-file", "local-directory"],
    )
    def test_a_source_that_is_not_a_gguf_file_is_rejected(self, pinned):
        cfg = _make_cfg(loader=ModelLoader.llama_server, model="some/repo-GGUF:README.md", num_gpus=0)
        with (
            patch("modelship.infer.sources.check_model_source", return_value=pinned),
            pytest.raises(ValueError, match=re.escape("'some/repo-GGUF:README.md' does not resolve to a GGUF file")),
        ):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))

    @pytest.mark.parametrize(
        "pinned",
        [
            _GGUF_PIN._replace(filename=None, patterns=["*Q4_K_M*.gguf"], first_shard="model-00001-of-00002.gguf"),
            LocalSource("/models/x.gguf"),
        ],
        ids=["sharded", "local-file"],
    )
    def test_a_gguf_source_is_allowed(self, pinned):
        cfg = _make_cfg(loader=ModelLoader.llama_server, num_gpus=0)
        with patch("modelship.infer.sources.check_model_source", return_value=pinned):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))
        assert cfg._pinned_source == pinned
