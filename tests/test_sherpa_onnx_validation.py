"""A registry entry's bundle_paths() checked against synthetic bundle trees — the
failure modes a real tarball extraction or a hand-placed local directory can hit."""

import pytest

from modelship.infer.sherpa_onnx.bundle import bundle_paths
from modelship.infer.sherpa_onnx.registry import SherpaOnnxRegistryEntry
from modelship.utils.sources import check_required


def _entry(**overrides) -> SherpaOnnxRegistryEntry:
    defaults = dict(
        tarball_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/fake.tar.bz2",
        sha256="0" * 64,
        family="kokoro",
        usecase="tts",
        files={"model": "model.onnx", "tokens": "tokens.txt", "voices": "voices.bin"},
        dirs={},
        chinese_lexicon=None,
        voice_names=("af_bella",),
    )
    return SherpaOnnxRegistryEntry(**{**defaults, **overrides})


def _write_valid_bundle(root):
    (root / "model.onnx").write_bytes(b"x" * 8)
    (root / "tokens.txt").write_bytes(b"hello")
    (root / "voices.bin").write_bytes(b"x" * 4)


def test_bundle_paths_lists_files_then_lexicon():
    entry = _entry(chinese_lexicon="lexicon-zh.txt")
    assert bundle_paths(entry) == ("model.onnx", "tokens.txt", "voices.bin", "lexicon-zh.txt")


def test_bundle_paths_mark_dirs_with_a_trailing_slash():
    entry = _entry(dirs={"data_dir": "espeak-ng-data"})
    assert bundle_paths(entry) == ("model.onnx", "tokens.txt", "voices.bin", "espeak-ng-data/")


def test_valid_bundle_passes(tmp_path):
    _write_valid_bundle(tmp_path)
    check_required(str(tmp_path), bundle_paths(_entry()))  # no raise


def test_missing_file(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "tokens.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        check_required(str(tmp_path), bundle_paths(_entry()))


def test_missing_lexicon_file(tmp_path):
    _write_valid_bundle(tmp_path)
    entry = _entry(chinese_lexicon="lexicon-zh.txt")
    with pytest.raises(ValueError, match="lexicon"):
        check_required(str(tmp_path), bundle_paths(entry))


def test_missing_bundle_dir(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        check_required(str(tmp_path / "does-not-exist"), bundle_paths(_entry()))
