"""FileSource: HEAD pinning, fetch-once publication into a hash-named file, and
never publishing a file that failed its sha256 check."""

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelship.utils.sources import (
    FileSource,
    check_file_source,
    download_model_source,
    file,
    is_cached,
    remove_leftovers,
)

_DATA = b"silero-bytes"
_DIGEST = hashlib.sha256(_DATA).hexdigest()


def _fake_download(data: bytes = _DATA):
    def fake(url, dest, on_chunk=None):
        with open(dest, "wb") as out:
            out.write(data)
        if on_chunk is not None:
            on_chunk(len(data))

    return fake


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(file, "cache_dir", lambda: str(root))
    return root


def _source(sha256: str = _DIGEST) -> FileSource:
    return FileSource("http://x/silero.onnx", sha256, "vad/silero-12345678.onnx", None)


class TestCheckFileSource:
    def test_records_the_size_the_server_reports(self):
        response = MagicMock(headers={"content-length": "2313101"})
        with patch("modelship.utils.sources.http.requests.head", return_value=response) as head:
            pinned = check_file_source("http://x/a.onnx", "ab" * 32, "vad/a-abababab.onnx")
        assert head.call_args.kwargs["allow_redirects"] is True
        assert pinned == FileSource("http://x/a.onnx", "ab" * 32, "vad/a-abababab.onnx", 2313101)


class TestDownloadFileSource:
    def test_fetches_into_dest(self, cache_root):
        with patch.object(file, "download", side_effect=_fake_download()):
            path = download_model_source(_source())
        assert path == str(cache_root / "vad" / "silero-12345678.onnx")
        assert Path(path).read_bytes() == _DATA
        assert [p.name for p in (cache_root / "vad").iterdir()] == ["silero-12345678.onnx"]

    def test_a_published_file_is_reused_without_fetching(self, cache_root):
        (cache_root / "vad").mkdir(parents=True)
        (cache_root / "vad" / "silero-12345678.onnx").write_bytes(_DATA)
        with patch.object(file, "download") as download:
            path = download_model_source(_source())
        download.assert_not_called()
        assert path == str(cache_root / "vad" / "silero-12345678.onnx")

    def test_a_sha256_mismatch_publishes_nothing(self, cache_root):
        with (
            patch.object(file, "download", side_effect=_fake_download(b"tampered")),
            pytest.raises(ValueError, match="failed sha256 verification"),
        ):
            download_model_source(_source())
        assert list((cache_root / "vad").iterdir()) == []

    def test_a_failed_fetch_leaves_no_download(self, cache_root):
        with patch.object(file, "download", side_effect=OSError("connection reset")), pytest.raises(OSError):
            download_model_source(_source())
        assert list((cache_root / "vad").iterdir()) == []


class TestCacheState:
    def test_is_cached_once_published(self, cache_root):
        assert not is_cached(_source())
        with patch.object(file, "download", side_effect=_fake_download()):
            download_model_source(_source())
        assert is_cached(_source())

    def test_leftovers_are_removed_and_the_published_file_kept(self, cache_root):
        vad = cache_root / "vad"
        vad.mkdir(parents=True)
        (vad / "silero-12345678.onnx").write_bytes(_DATA)
        (vad / ".silero-12345678.onnx.abc.download").write_bytes(b"x")
        (vad / ".silero-12345678.onnx.def.download.ghi.tmp").write_bytes(b"x")
        assert remove_leftovers(_source()) == 2
        assert sorted(os.listdir(vad)) == ["silero-12345678.onnx"]
