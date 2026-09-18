"""ArchiveSource: HEAD pinning, fetch-once publication into a hash-named dir,
and never deleting a published dir."""

import hashlib
import os
import tarfile
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from modelship.infer.sources import (
    ArchiveSource,
    ModelSourceError,
    archive,
    check_archive_source,
    download_model_source,
    is_cached,
    remove_leftovers,
)
from modelship.utils import verify_sha256

_REQUIRED = ("model.onnx", "tokens.txt", "data/")


def _write_tree(root) -> None:
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    for rel in ("model.onnx", "tokens.txt", "data/a"):
        with open(os.path.join(root, rel), "wb") as f:
            f.write(b"x")


def _make_archive(tmp_path) -> tuple[str, str]:
    src_dir = tmp_path / "src" / "bundle"
    _write_tree(src_dir)
    path = tmp_path / "src_archive.tar.bz2"
    with tarfile.open(path, "w:bz2") as tar:
        tar.add(src_dir, arcname="bundle")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_download(src_archive: str):
    def fake(url, dest, on_chunk=None):
        with open(src_archive, "rb") as f, open(dest, "wb") as out:
            data = f.read()
            out.write(data)
        if on_chunk is not None:
            on_chunk(len(data))

    return fake


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(archive, "cache_dir", lambda: str(root))
    return root


def _source(sha256: str = "0" * 64) -> ArchiveSource:
    return ArchiveSource("http://x/bundle.tar.bz2", sha256, "bundles/bundle-12345678", _REQUIRED, None)


class TestCheckArchiveSource:
    def test_follows_redirects_and_records_size(self):
        response = MagicMock(headers={"content-length": "319625534"})
        with patch("modelship.infer.sources.archive.requests.head", return_value=response) as head:
            pinned = check_archive_source("http://x/a.tar.bz2", "ab" * 32, "d/a-abababab", _REQUIRED)
        assert head.call_args.kwargs["allow_redirects"] is True
        assert pinned == ArchiveSource("http://x/a.tar.bz2", "ab" * 32, "d/a-abababab", _REQUIRED, 319625534)

    @pytest.mark.parametrize("headers", [{}, {"content-length": "0"}])
    def test_unknown_size_is_none(self, headers):
        with patch("modelship.infer.sources.archive.requests.head", return_value=MagicMock(headers=headers)):
            pinned = check_archive_source("http://x/a.tar.bz2", "ab" * 32, "d/a", _REQUIRED)
        assert pinned.total_bytes is None

    @pytest.mark.parametrize(
        "error", [requests.HTTPError("404 Client Error"), requests.ConnectionError("name resolution failed")]
    )
    def test_unreachable_url_is_fatal(self, error):
        response = MagicMock()
        response.raise_for_status.side_effect = error
        with (
            patch("modelship.infer.sources.archive.requests.head", return_value=response),
            pytest.raises(RuntimeError, match=r"Failed to reach http://x/a\.tar\.bz2"),
        ):
            check_archive_source("http://x/a.tar.bz2", "ab" * 32, "d/a", _REQUIRED)


class TestDownloadArchiveSource:
    def test_fetches_into_dest_and_removes_the_archive(self, tmp_path, cache_root):
        src_archive, digest = _make_archive(tmp_path)
        with patch.object(archive, "download", side_effect=_fake_download(src_archive)):
            path = download_model_source(_source(digest))

        assert path == str(cache_root / "bundles" / "bundle-12345678")
        assert os.path.isfile(os.path.join(path, "model.onnx"))
        assert [p.name for p in (cache_root / "bundles").iterdir()] == ["bundle-12345678"]

    def test_published_dest_is_reused_without_fetching(self, cache_root):
        _write_tree(cache_root / "bundles" / "bundle-12345678")
        with patch.object(archive, "download") as download:
            path = download_model_source(_source())
        download.assert_not_called()
        assert path == str(cache_root / "bundles" / "bundle-12345678")

    def test_incomplete_published_dest_is_reported_not_deleted(self, cache_root):
        dest = cache_root / "bundles" / "bundle-12345678"
        _write_tree(dest)
        (dest / "tokens.txt").unlink()
        with (
            patch.object(archive, "download") as download,
            pytest.raises(ModelSourceError, match=r"missing 'tokens\.txt'; delete it to fetch again"),
        ):
            download_model_source(_source())
        download.assert_not_called()
        assert (dest / "model.onnx").is_file()

    def test_extraction_that_publishes_nothing_stays_retryable(self, tmp_path, cache_root):
        src_archive, digest = _make_archive(tmp_path)
        with (
            patch.object(archive, "download", side_effect=_fake_download(src_archive)),
            patch("modelship.utils.os.replace", side_effect=PermissionError("denied")),
            pytest.raises(OSError, match="did not publish") as exc_info,
        ):
            download_model_source(_source(digest))
        assert not isinstance(exc_info.value, ModelSourceError)

    @pytest.mark.parametrize("pinned_digest", [True, False], ids=["extract-fails", "sha256-mismatch"])
    def test_failed_fetch_leaves_no_archive(self, tmp_path, cache_root, pinned_digest):
        garbage = tmp_path / "not-a-tar.bin"
        garbage.write_bytes(b"not a tarball")
        digest = hashlib.sha256(garbage.read_bytes()).hexdigest() if pinned_digest else "0" * 64
        with (
            patch.object(archive, "download", side_effect=_fake_download(str(garbage))),
            pytest.raises((tarfile.ReadError, ValueError)),
        ):
            download_model_source(_source(digest))
        assert list((cache_root / "bundles").iterdir()) == []

    def test_concurrent_fetchers_each_use_their_own_archive(self, tmp_path, cache_root):
        src_archive, digest = _make_archive(tmp_path)

        # b has downloaded but not verified when a publishes and removes its archive
        b_downloaded = threading.Event()
        a_returned = threading.Event()

        def fake_download(url, dest, on_chunk=None):
            _fake_download(src_archive)(url, dest)
            if threading.current_thread().name == "b":
                b_downloaded.set()

        def gated_verify(path, expected):
            if threading.current_thread().name == "b":
                a_returned.wait(5)
            verify_sha256(path, expected)

        errors: list[Exception] = []
        b_results: list[str] = []

        def run_b():
            try:
                b_results.append(download_model_source(_source(digest)))
            except Exception as e:
                errors.append(e)

        with (
            patch.object(archive, "download", side_effect=fake_download),
            patch("modelship.utils.verify_sha256", side_effect=gated_verify),
        ):
            b = threading.Thread(target=run_b, name="b")
            b.start()
            assert b_downloaded.wait(5)
            path = download_model_source(_source(digest))
            a_returned.set()
            b.join(5)

        assert errors == []
        assert b_results == [path]
        assert os.path.isfile(os.path.join(path, "model.onnx"))
        assert list((cache_root / "bundles").glob("*.archive")) == []


class TestIsArchiveCached:
    def test_published_dest(self, cache_root):
        _write_tree(cache_root / "bundles" / "bundle-12345678")
        assert is_cached(_source())

    def test_leftovers_alone_are_not_cached(self, cache_root):
        (cache_root / "bundles" / "bundle-12345678.abc.tmp").mkdir(parents=True)
        assert not is_cached(_source())


class TestRemoveArchiveLeftovers:
    def test_removes_temps_and_keeps_the_published_dest(self, cache_root):
        bundles = cache_root / "bundles"
        _write_tree(bundles / "bundle-12345678")
        (bundles / ".bundle-12345678.abc.archive").write_bytes(b"x")
        (bundles / ".bundle-12345678.abc.archive.def.tmp").write_bytes(b"x")
        (bundles / "bundle-12345678.ghi.tmp" / "data").mkdir(parents=True)
        (bundles / "bundle-99999999.jkl.tmp").mkdir()

        assert remove_leftovers(_source()) == 3
        assert sorted(p.name for p in bundles.iterdir()) == ["bundle-12345678", "bundle-99999999.jkl.tmp"]
        assert (bundles / "bundle-12345678" / "model.onnx").is_file()

    def test_nothing_to_remove(self, cache_root):
        assert remove_leftovers(_source()) == 0
