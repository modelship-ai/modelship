"""Driver preflight pins a registry name as an ArchiveSource (HEAD-checked) and
a local bundle directory as a LocalSource, validated against its registry entry."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelshipModelConfig, ModelUsecase
from modelship.infer.sherpa_onnx.bundle import bundle_paths
from modelship.infer.sherpa_onnx.registry import REGISTRY
from modelship.infer.sources import ArchiveSource, LocalSource

_HEAD = "modelship.infer.sources.archive.requests.head"


def _cfg(model: str) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="tts", model=model, usecase=ModelUsecase.tts, loader=ModelLoader.sherpa_onnx, num_gpus=0
    )


def _write_bundle(root) -> None:
    root.mkdir()
    for name in ("model.onnx", "tokens.txt", "voices.bin"):
        (root / name).write_bytes(b"x")
    (root / "espeak-ng-data").mkdir()


def test_registry_name_pins_a_hash_named_archive():
    cfg = _cfg("kokoro-en-v0_19")
    fingerprint = cfg.fingerprint()
    entry = REGISTRY["kokoro-en-v0_19"]
    with patch(_HEAD, return_value=MagicMock(headers={"content-length": "319625534"})) as head:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))

    head.assert_called_once()
    assert cfg._pinned_source == ArchiveSource(
        entry.tarball_url,
        entry.sha256,
        f"sherpa_onnx/kokoro-en-v0_19-{entry.sha256[:8]}",
        bundle_paths(entry),
        319625534,
    )
    assert cfg.fingerprint() == fingerprint


def test_unreachable_tarball_fails_preflight():
    cfg = _cfg("kokoro-en-v0_19")
    with (
        patch(_HEAD, side_effect=requests.ConnectionError("offline")),
        pytest.raises(RuntimeError, match="Failed to reach"),
    ):
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))


def test_local_dir_pins_without_network(tmp_path):
    bundle = tmp_path / "kokoro-en-v0_19"
    _write_bundle(bundle)
    cfg = _cfg(str(bundle))
    with patch(_HEAD) as head:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    head.assert_not_called()
    assert cfg._pinned_source == LocalSource(str(bundle), bundle_paths(REGISTRY["kokoro-en-v0_19"]))


def test_local_dir_is_validated_at_preflight(tmp_path):
    bundle = tmp_path / "kokoro-en-v0_19"
    bundle.mkdir()
    cfg = _cfg(str(bundle))
    with pytest.raises(ValueError, match="missing"):
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))


def test_local_dir_basename_must_match_a_registry_name(tmp_path):
    # Config validation (not preflight) is what catches this — a mismatched
    # basename never resolves to a registry entry in the first place.
    bad_dir = tmp_path / "not-a-registered-model"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="not a supported registry name"):
        _cfg(str(bad_dir))
