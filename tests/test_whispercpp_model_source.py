"""Driver preflight maps bare whisper.cpp model names to their HF repo file, without
importing pywhispercpp (a thin coordinator has no pywhispercpp)."""

import sys
from unittest.mock import patch

import pytest

from modelship.deploy.config import WHISPERCPP_REPO, resolve_all_model_sources
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.sources import HfSource
from modelship.infer.whispercpp.whispercpp_infer import WhispercppInfer

_GGML_PIN = HfSource(
    repo="ggerganov/whisper.cpp",
    revision="deadbeef",
    filename="ggml-base.en.bin",
    patterns=None,
    first_shard=None,
    total_bytes=None,
)


def _cfg(model: str) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="stt",
        model=model,
        usecase=ModelUsecase.transcription,
        loader=ModelLoader.whispercpp,
        num_gpus=0,
    )


@pytest.fixture
def no_pywhispercpp(monkeypatch):
    # None in sys.modules makes `import pywhispercpp...` raise, as on the thin image.
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.constants", None)


@pytest.mark.parametrize("model", ["base.en", "large-v3-turbo-q5_0"])
def test_bare_name_checks_the_repo_file(no_pywhispercpp, model):
    cfg = _cfg(model)
    fingerprint = cfg.fingerprint()
    with patch("modelship.infer.sources.check_model_source", return_value=_GGML_PIN) as check:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    check.assert_called_once_with(f"ggerganov/whisper.cpp:ggml-{model}.bin", trust_remote_code=False)
    assert cfg._pinned_source == _GGML_PIN
    assert cfg.model == model
    assert cfg.fingerprint() == fingerprint


def test_unknown_bare_name_names_the_model():
    cfg = _cfg("bse.en")
    err = FileNotFoundError("Selector 'ggml-bse.en.bin' matched no files in HF repo 'ggerganov/whisper.cpp'")
    with (
        patch("modelship.infer.sources.check_model_source", side_effect=err),
        pytest.raises(FileNotFoundError, match=r"'bse\.en' is not a whisper\.cpp model name"),
    ):
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))


def test_repo_ref_passes_through():
    cfg = _cfg("ggerganov/whisper.cpp:ggml-base.en.bin")
    with patch("modelship.infer.sources.check_model_source", return_value=_GGML_PIN) as check:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    check.assert_called_once_with("ggerganov/whisper.cpp:ggml-base.en.bin", trust_remote_code=False)
    assert cfg._pinned_source == _GGML_PIN


def test_missing_repo_file_error_is_not_rewritten():
    cfg = _cfg("ggerganov/whisper.cpp:ggml-nope.bin")
    err = FileNotFoundError("Selector 'ggml-nope.bin' matched no files")
    with (
        patch("modelship.infer.sources.check_model_source", side_effect=err),
        pytest.raises(FileNotFoundError, match=r"^Selector"),
    ):
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))


@pytest.mark.parametrize("relative", [False, True])
def test_local_file_passes_through(tmp_path, monkeypatch, relative):
    path = tmp_path / "ggml-base.en.bin"
    path.write_bytes(b"\x00" * 4)
    if relative:
        monkeypatch.chdir(tmp_path)
    model = path.name if relative else str(path)
    cfg = _cfg(model)
    with patch("modelship.infer.sources.check_model_source", return_value=_GGML_PIN) as check:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    check.assert_called_once_with(model, trust_remote_code=False)


def test_actor_rejects_a_config_without_a_resolved_path():
    pytest.importorskip("pywhispercpp.model")
    infer = WhispercppInfer(_cfg("base.en"))
    with pytest.raises(ValueError, match="no resolved model file"):
        infer._load()


def test_repo_matches_pywhispercpps_download_url():
    # Fails loudly if pywhispercpp moves its built-in models to another repo.
    constants = pytest.importorskip("pywhispercpp.constants")
    expected = f"https://huggingface.co/{WHISPERCPP_REPO}"
    assert expected == constants.MODELS_BASE_URL
