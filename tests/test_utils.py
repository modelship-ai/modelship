import hashlib
import socket
import tarfile
import threading
import time
from unittest.mock import patch

import pytest
import requests

from modelship import utils
from modelship.utils import download, fetch_and_extract_archive, verify_sha256


@pytest.fixture
def stalling_url():
    """Serves the headers and a few body bytes, then goes silent."""
    server = socket.create_server(("127.0.0.1", 0))
    done = threading.Event()

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial")
            done.wait(10)

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}/a.tar.bz2"
    done.set()
    server.close()


class TestDownload:
    def test_a_stalled_transfer_times_out(self, stalling_url, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "_DOWNLOAD_TIMEOUT_SECONDS", 0.2)
        started = time.monotonic()
        with pytest.raises(requests.RequestException):
            download(stalling_url, str(tmp_path / "a.tar.bz2"))
        assert time.monotonic() - started < 5
        assert list(tmp_path.iterdir()) == []


class TestVerifySha256:
    def test_passes_when_hash_matches(self, tmp_path):
        path = tmp_path / "file.bin"
        path.write_bytes(b"hello")
        verify_sha256(str(path), hashlib.sha256(b"hello").hexdigest())  # no raise

    def test_raises_on_mismatch(self, tmp_path):
        path = tmp_path / "file.bin"
        path.write_bytes(b"hello")
        with pytest.raises(ValueError, match="sha256 verification"):
            verify_sha256(str(path), "0" * 64)


def _make_archive(tmp_path, members: dict[str, bytes]) -> tuple[str, str]:
    archive = tmp_path / "src_archive.tar.gz"
    src_dir = tmp_path / "src"
    with tarfile.open(archive, "w:gz") as tar:
        for arcname, contents in members.items():
            src = src_dir / arcname
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(contents)
            tar.add(src, arcname=arcname)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return str(archive), digest


def _fake_download(src_archive: str):
    def fake_download(url, dest):
        with open(src_archive, "rb") as f, open(dest, "wb") as out:
            out.write(f.read())

    return fake_download


class TestFetchAndExtractArchive:
    def test_flatten_discards_nested_dirs(self, tmp_path):
        src_archive, digest = _make_archive(tmp_path, {"nested/dir/file.txt": b"contents"})
        extract_dir = tmp_path / "extracted"
        with patch("modelship.utils.download", side_effect=_fake_download(src_archive)):
            fetch_and_extract_archive(
                "http://x", digest, str(tmp_path / "archive.tar.gz"), str(extract_dir), flatten=True
            )

        assert (extract_dir / "file.txt").read_bytes() == b"contents"
        assert not (extract_dir / "nested").exists()

    def test_preserves_structure_and_strips_single_top_level_dir(self, tmp_path):
        src_archive, digest = _make_archive(
            tmp_path, {"bundle/model.onnx": b"weights", "bundle/sub/tokens.txt": b"tokens"}
        )
        extract_dir = tmp_path / "extracted"
        with patch("modelship.utils.download", side_effect=_fake_download(src_archive)):
            fetch_and_extract_archive("http://x", digest, str(tmp_path / "archive.tar.gz"), str(extract_dir))

        assert (extract_dir / "model.onnx").read_bytes() == b"weights"
        assert (extract_dir / "sub" / "tokens.txt").read_bytes() == b"tokens"
        assert not (extract_dir / "bundle").exists()

    def test_deletes_archive_by_default(self, tmp_path):
        src_archive, digest = _make_archive(tmp_path, {"file.txt": b"x"})
        archive_path = tmp_path / "archive.tar.gz"
        with patch("modelship.utils.download", side_effect=_fake_download(src_archive)):
            fetch_and_extract_archive("http://x", digest, str(archive_path), str(tmp_path / "extracted"))
        assert not archive_path.exists()

    def test_keep_archive_preserves_it(self, tmp_path):
        src_archive, digest = _make_archive(tmp_path, {"file.txt": b"x"})
        archive_path = tmp_path / "archive.tar.gz"
        with patch("modelship.utils.download", side_effect=_fake_download(src_archive)):
            fetch_and_extract_archive(
                "http://x", digest, str(archive_path), str(tmp_path / "extracted"), keep_archive=True
            )
        assert archive_path.exists()

    def test_sha256_mismatch_deletes_archive_and_raises(self, tmp_path):
        src_archive, _digest = _make_archive(tmp_path, {"file.txt": b"x"})
        archive_path = tmp_path / "archive.tar.gz"
        with (
            patch("modelship.utils.download", side_effect=_fake_download(src_archive)),
            pytest.raises(ValueError, match="sha256 verification"),
        ):
            fetch_and_extract_archive("http://x", "0" * 64, str(archive_path), str(tmp_path / "extracted"))
        assert not archive_path.exists()
