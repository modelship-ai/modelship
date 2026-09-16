"""Local and HF model sources: ref parsing, pinning, and download dispatch."""

import inspect
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub._snapshot_download
import pytest
from huggingface_hub.utils.tqdm import _create_progress_bar

from modelship.infer.sources import (
    HfSource,
    LocalSource,
    ModelDownloadError,
    ModelSourceError,
    check_model_source,
    download_model_source,
    hf,
    resolve_model_source,
)
from modelship.infer.sources.hf import _TRANSFER_BAR, _DownloadProgressLogger, _select_patterns
from modelship.utils.model_ref import parse_model_ref


def _model_info(files: list[str], sha: str = "deadbeef"):
    return MagicMock(sha=sha, siblings=[MagicMock(rfilename=f, size=None) for f in files])


class TestParseModelRef:
    def test_hf_repo_no_selector(self):
        result = parse_model_ref("Qwen/Qwen3-7B")
        assert result.source == "Qwen/Qwen3-7B"
        assert result.selector is None
        assert result.is_local is False

    def test_hf_repo_with_selector(self):
        result = parse_model_ref("lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf")
        assert result.source == "lmstudio-community/Qwen2.5-7B-Instruct-GGUF"
        assert result.selector == "*Q4_K_M.gguf"
        assert result.is_local is False

    def test_hf_repo_with_exact_filename(self):
        result = parse_model_ref("nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf")
        assert result.source == "nomic-ai/nomic-embed-text-v1.5-GGUF"
        assert result.selector == "nomic-embed-text-v1.5.Q4_K_M.gguf"

    def test_absolute_path_existing(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = parse_model_ref(str(f))
        assert result.source == str(f)
        assert result.selector is None
        assert result.is_local is True

    def test_absolute_path_with_colon_in_name_treated_as_local(self, tmp_path: Path):
        # Absolute paths starting with `/` are not split on `:`, even if the file
        # contains a colon (rare but legal).
        d = tmp_path / "weird_dir"
        d.mkdir()
        result = parse_model_ref(str(d))
        assert result.is_local is True
        assert result.selector is None

    def test_multiple_colons_split_on_first(self):
        result = parse_model_ref("org/repo:path/to/file.gguf")
        assert result.source == "org/repo"
        assert result.selector == "path/to/file.gguf"

    def test_pathy_missing_absolute_path_is_still_local(self, tmp_path: Path):
        # A pathy string (starts with /, ./, ~) is local by syntax alone, not by
        # existence — a typo'd path must fail clearly, not get misread as an HF repo id.
        missing = tmp_path / "does-not-exist"
        result = parse_model_ref(str(missing))
        assert result.source == str(missing)
        assert result.is_local is True

    def test_pathy_missing_path_with_selector_is_still_local(self, tmp_path: Path):
        missing_dir = tmp_path / "does-not-exist"
        result = parse_model_ref(f"{missing_dir}:*.gguf")
        assert result.source == str(missing_dir)
        assert result.selector == "*.gguf"
        assert result.is_local is True

    def test_non_pathy_missing_string_is_not_local(self):
        # No leading /, ./, or ~ — read as an HF repo id.
        result = parse_model_ref("definitely-not-a-real/repo-id")
        assert result.is_local is False

    def test_tilde_path_is_expanded(self, monkeypatch, tmp_path: Path):
        # Path.resolve() never expands `~`, so it must happen here or
        # check_model_source's Path(source).resolve() 404s on a real ~/... ref.
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = parse_model_ref("~/model.gguf")
        assert result.source == str(f)
        assert result.is_local is True

    def test_tilde_path_with_selector_is_expanded(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        d = tmp_path / "models"
        d.mkdir()
        result = parse_model_ref("~/models:*.gguf")
        assert result.source == str(d)
        assert result.selector == "*.gguf"
        assert result.is_local is True


class TestSelectPatterns:
    def test_safetensors_excludes_bin(self):
        files = ["model.safetensors", "pytorch_model.bin", "config.json"]
        patterns = _select_patterns(files)
        assert "*.safetensors" in patterns
        assert "*.bin" not in patterns
        assert "*.bin.index.json" not in patterns

    def test_no_safetensors_falls_back_to_bin(self):
        files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        patterns = _select_patterns(files)
        assert "*.bin" in patterns
        assert "*.safetensors" not in patterns

    def test_sharded_safetensors_index(self):
        files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "config.json",
        ]
        patterns = _select_patterns(files)
        assert "*.safetensors" in patterns
        assert "*.safetensors.index.json" in patterns

    def test_trust_remote_code_includes_py(self):
        files = ["model.safetensors", "config.json", "modeling_custom.py"]
        patterns = _select_patterns(files, trust_remote_code=True)
        assert "*.py" in patterns
        assert "**/*.py" in patterns

    def test_trust_remote_code_default_excludes_py(self):
        files = ["model.safetensors", "config.json"]
        patterns = _select_patterns(files, trust_remote_code=False)
        assert "*.py" not in patterns

    def test_always_includes_tokenizer_and_config(self):
        files = ["model.safetensors", "tokenizer.json", "config.json"]
        patterns = _select_patterns(files)
        assert "tokenizer*" in patterns
        assert "*.json" in patterns
        assert "preprocessor_config.json" in patterns


class TestResolveLocalPath:
    """Local paths never touch HF — check_model_source resolves them fully,
    so resolve_model_source (check + download) needs no HF mocking."""

    def test_local_file(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = resolve_model_source(str(f))
        assert result == str(f.absolute())

    def test_local_dir(self, tmp_path: Path):
        d = tmp_path / "model_snapshot"
        d.mkdir()
        (d / "config.json").write_text("{}")
        result = resolve_model_source(str(d))
        assert result == str(d.absolute())

    def test_local_dir_with_selector_single_match(self, tmp_path: Path):
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-Q4_K_M.gguf").write_text("dummy")
        (d / "model-Q8_0.gguf").write_text("dummy")
        result = resolve_model_source(f"{d}:*Q4_K_M.gguf")
        assert result.endswith("model-Q4_K_M.gguf")

    def test_local_dir_with_selector_no_match(self, tmp_path: Path):
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-Q8_0.gguf").write_text("dummy")
        with pytest.raises(FileNotFoundError, match="matched no files"):
            resolve_model_source(f"{d}:*Q4_K_M.gguf")

    def test_local_dir_with_selector_multiple_matches_returns_first(self, tmp_path: Path):
        # Sharded GGUF case: selector matches several shards; return the first
        # alphabetically so llama.cpp can auto-load the rest.
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-00002-of-00003.gguf").write_text("dummy")
        (d / "model-00001-of-00003.gguf").write_text("dummy")
        (d / "model-00003-of-00003.gguf").write_text("dummy")
        result = resolve_model_source(f"{d}:model-*.gguf")
        assert result.endswith("model-00001-of-00003.gguf")

    def test_local_path_missing(self, tmp_path: Path):
        # A pathy string is local by syntax alone, so a missing path fails clearly
        # from the local branch, not a confusing HF-repo error.
        with pytest.raises(FileNotFoundError, match="Local path not found"):
            resolve_model_source(str(tmp_path / "does-not-exist"))

    def test_download_is_noop_for_local(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        pinned = check_model_source(str(f))
        assert pinned == LocalSource(str(f.absolute()))
        assert download_model_source(pinned) == pinned.path

    def test_download_rejects_a_path_missing_on_the_replica_node(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        pinned = check_model_source(str(f))
        f.unlink()
        with pytest.raises(ModelSourceError, match="Local path not found"):
            download_model_source(pinned)

    def test_download_rechecks_required_paths(self, tmp_path: Path):
        (tmp_path / "model.onnx").write_text("m")
        (tmp_path / "data").mkdir()
        pinned = LocalSource(str(tmp_path), ("model.onnx", "data/"))
        assert download_model_source(pinned) == str(tmp_path)

        (tmp_path / "model.onnx").unlink()
        with pytest.raises(ModelSourceError, match=r"missing 'model\.onnx'"):
            download_model_source(pinned)

    def test_required_dir_must_be_a_directory(self, tmp_path: Path):
        (tmp_path / "data").write_text("not a dir")
        with pytest.raises(ModelSourceError, match="missing 'data/'"):
            download_model_source(LocalSource(str(tmp_path), ("data/",)))


class TestCheckHfRepoDoesNoDownload:
    """check_model_source must never fetch weight bytes — file listing and
    revision both come from a single model_info call."""

    def test_no_download_calls(self):
        files = ["model.safetensors", "config.json", "tokenizer.json"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)) as mock_info,
            patch("modelship.infer.sources.hf.hf_hub_download") as mock_dl,
            patch("modelship.infer.sources.hf.snapshot_download") as mock_snap,
        ):
            check_model_source("Qwen/Qwen3-7B")
            mock_info.assert_called_once_with("Qwen/Qwen3-7B", files_metadata=True)
            mock_dl.assert_not_called()
            mock_snap.assert_not_called()

    def test_pins_commit_sha(self):
        files = ["model.safetensors", "config.json"]
        with patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files, "abc123")):
            pinned = check_model_source("Qwen/Qwen3-7B")
            assert isinstance(pinned, HfSource)
            assert pinned.revision == "abc123"
            assert pinned.repo == "Qwen/Qwen3-7B"

    def test_info_lookup_failure_wrapped(self):
        with (
            patch("modelship.infer.sources.hf.model_info", side_effect=Exception("boom")),
            pytest.raises(RuntimeError, match="Failed to fetch info"),
        ):
            check_model_source("Qwen/Qwen3-7B")


class TestResolveHfRepo:
    """resolve_model_source (check + download) end to end."""

    def test_full_snapshot_calls_universal_filter(self):
        files = ["model.safetensors", "config.json", "tokenizer.json"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            patch("modelship.infer.sources.hf.snapshot_download") as mock_snap,
        ):
            mock_snap.return_value = "/cache/snapshot"
            result = resolve_model_source("Qwen/Qwen3-7B")
            assert result == "/cache/snapshot"
            mock_snap.assert_called_once()
            kwargs = mock_snap.call_args.kwargs
            assert "*.safetensors" in kwargs["allow_patterns"]
            assert "*.bin" not in kwargs["allow_patterns"]
            assert kwargs["revision"] == "deadbeef"

    def test_selector_single_file_uses_hf_hub_download(self):
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            patch("modelship.infer.sources.hf.hf_hub_download") as mock_dl,
        ):
            mock_dl.return_value = "/cache/model-Q4_K_M.gguf"
            result = resolve_model_source("org/repo:*Q4_K_M.gguf")
            assert result == "/cache/model-Q4_K_M.gguf"
            mock_dl.assert_called_once_with(
                "org/repo", "model-Q4_K_M.gguf", revision="deadbeef", tqdm_class=_DownloadProgressLogger
            )

    def test_selector_multiple_matches_returns_first_shard_path(self):
        # Sharded GGUF: download all shards, then return the first shard's full
        # path (not the snapshot dir) since file-path loaders like llama.cpp need it.
        files = ["model-00002-of-00002.gguf", "model-00001-of-00002.gguf"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            patch("modelship.infer.sources.hf.snapshot_download") as mock_snap,
        ):
            mock_snap.return_value = "/cache/snapshot"
            result = resolve_model_source("org/repo:*.gguf")
            assert result == "/cache/snapshot/model-00001-of-00002.gguf"
            mock_snap.assert_called_once_with(
                "org/repo", revision="deadbeef", allow_patterns=["*.gguf"], tqdm_class=_DownloadProgressLogger
            )

    def test_selector_no_match_raises(self):
        files = ["model-Q4_K_M.gguf"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            pytest.raises(FileNotFoundError, match="matched no files"),
        ):
            resolve_model_source("org/repo:*Q8_0.gguf")

    def test_multi_variant_gguf_without_selector_raises(self):
        files = [
            "model-Q2_K.gguf",
            "model-Q4_K_M.gguf",
            "model-Q5_K_M.gguf",
            "model-Q8_0.gguf",
        ]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            pytest.raises(ValueError, match="contains 4 GGUF variants"),
        ):
            resolve_model_source("lmstudio-community/Qwen2.5-7B-Instruct-GGUF")

    def test_single_gguf_without_selector_returns_file_path(self):
        # Single-GGUF repo: resolver must return the file path (not a snapshot
        # dir), because llama_server requires a file path.
        files = ["model.gguf", "config.json"]
        with (
            patch("modelship.infer.sources.hf.model_info", return_value=_model_info(files)),
            patch("modelship.infer.sources.hf.hf_hub_download") as mock_dl,
            patch("modelship.infer.sources.hf.snapshot_download") as mock_snap,
        ):
            mock_dl.return_value = "/cache/model.gguf"
            result = resolve_model_source("org/single-gguf-repo")
            assert result == "/cache/model.gguf"
            mock_dl.assert_called_once_with(
                "org/single-gguf-repo", "model.gguf", revision="deadbeef", tqdm_class=_DownloadProgressLogger
            )
            mock_snap.assert_not_called()

    def test_model_info_failure_wrapped(self):
        with (
            patch("modelship.infer.sources.hf.model_info", side_effect=Exception("auth failure")),
            pytest.raises(RuntimeError, match="Failed to fetch info"),
        ):
            resolve_model_source("private/repo")


class TestDownloadProgressLogger:
    @pytest.fixture
    def active(self, monkeypatch):
        progress = MagicMock()
        monkeypatch.setattr(hf, "_active_download", progress)
        return progress

    def _bar(self, **kwargs):
        # HF's own factory: it passes `name` only to subclasses of its tqdm
        return _create_progress_bar(cls=_DownloadProgressLogger, log_level=logging.INFO, **kwargs)

    @pytest.mark.parametrize("name", ["huggingface_hub.http_get", "huggingface_hub.snapshot_download"])
    def test_byte_bars_are_counted(self, active, name):
        self._bar(name=name, unit="B", total=10).update(4)
        active.add.assert_called_once_with(4)

    def test_snapshot_transfer_mirror_is_skipped(self, active):
        self._bar(name=_TRANSFER_BAR, unit="B", total=10).update(4)
        active.add.assert_not_called()

    def test_file_count_bar_is_skipped(self, active):
        self._bar(total=3).update(1)
        active.add.assert_not_called()

    def test_rollback_is_counted(self, active):
        self._bar(name="huggingface_hub.http_get", unit="B", total=10).update(-4)
        active.add.assert_called_once_with(-4)

    def test_transfer_bar_name_matches_huggingface_hub(self):
        # fails loudly if HF renames the bar, which would double-count snapshot bytes again
        assert f'name="{_TRANSFER_BAR}"' in inspect.getsource(huggingface_hub._snapshot_download)


class TestResolvesToGguf:
    def test_local_file_gguf(self):
        assert LocalSource("/models/x.gguf").resolves_to_gguf

    def test_local_dir_not_gguf(self):
        assert not LocalSource("/models/snapshot").resolves_to_gguf

    def test_hf_single_file_download_gguf(self):
        pinned = HfSource("org/repo", "sha", "model.gguf", None, None, None)
        assert pinned.resolves_to_gguf

    def test_hf_shard_gguf(self):
        pinned = HfSource("org/repo", "sha", None, ["*.gguf"], "model-00001-of-00002.gguf", None)
        assert pinned.resolves_to_gguf

    def test_hf_full_snapshot_not_gguf(self):
        pinned = HfSource("org/repo", "sha", None, ["*.safetensors"], None, None)
        assert not pinned.resolves_to_gguf


class TestDownloadErrorClassification:
    def test_download_failure_is_not_wrapped_by_download_model_source(self):
        # download_model_source raises whatever hf raises; wrapping into
        # ModelDownloadError is BaseInfer.ensure_downloaded's job (it needs the model name).
        pinned = HfSource("org/repo", "sha", "model.safetensors", None, None, None)
        with (
            patch("modelship.infer.sources.hf.hf_hub_download", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            download_model_source(pinned)

    def test_model_download_error_is_a_plain_exception(self):
        # Deliberately not a subclass of a "permanent" error type —
        # ModelDeployment.__init__ special-cases it to skip reporting fatal.
        assert issubclass(ModelDownloadError, Exception)
