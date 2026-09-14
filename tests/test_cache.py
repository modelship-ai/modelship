import os
from unittest import mock

import pytest
import requests

from modelship.deploy.actor_options import build_cache_env_vars
from modelship.utils import cache_dir, download
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root


def test_build_cache_env_vars_defaults():
    env = {"MSHIP_CACHE_DIR": "/.cache", "MSHIP_NODE_CACHE_DIR": "/opt/mship/node-cache"}
    with mock.patch.dict(os.environ, env, clear=True):
        env_vars = build_cache_env_vars()
        assert env_vars["HF_HOME"] == "/.cache/huggingface"
        assert env_vars["VLLM_CACHE_ROOT"] == "/opt/mship/node-cache/vllm"
        assert env_vars["FLASHINFER_WORKSPACE_BASE"] == "/opt/mship/node-cache/flashinfer"
        assert env_vars["TRITON_CACHE_DIR"] == "/opt/mship/node-cache/triton"
        assert env_vars["VLLM_CONFIG_ROOT"] == "/opt/mship/node-cache/vllm-config"
        # flashinfer never reads it from env
        assert "FLASHINFER_CACHE_DIR" not in env_vars
        assert "HF_TOKEN" not in env_vars
        assert "HF_HUB_OFFLINE" not in env_vars


def test_build_cache_env_vars_forwards_hf_token_and_offline():
    env = {"MSHIP_CACHE_DIR": "/.cache", "HF_TOKEN": "hf_secret", "HF_HUB_OFFLINE": "1"}
    with mock.patch.dict(os.environ, env, clear=True):
        env_vars = build_cache_env_vars()
        assert env_vars["HF_TOKEN"] == "hf_secret"
        assert env_vars["HF_HUB_OFFLINE"] == "1"


def test_build_cache_env_vars_custom_dirs():
    shared, node = "/mnt/shared", "/scratch/node-cache"
    with mock.patch.dict(os.environ, {"MSHIP_CACHE_DIR": shared, "MSHIP_NODE_CACHE_DIR": node}, clear=True):
        env_vars = build_cache_env_vars()
        assert env_vars["HF_HOME"] == f"{shared}/huggingface"
        assert env_vars["VLLM_CACHE_ROOT"] == f"{node}/vllm"
        assert env_vars["FLASHINFER_WORKSPACE_BASE"] == f"{node}/flashinfer"
        assert env_vars["TRITON_CACHE_DIR"] == f"{node}/triton"
        assert env_vars["VLLM_CONFIG_ROOT"] == f"{node}/vllm-config"


def test_utils_cache_dir_default():
    # os.makedirs is mocked to avoid creating real directories in the test environment.
    with mock.patch.dict(os.environ, {"MSHIP_CACHE_DIR": "/.cache"}, clear=True), mock.patch("os.makedirs"):
        assert cache_dir() == "/.cache"


class TestResolveCacheRoot:
    def test_explicit_env_var_wins(self):
        with mock.patch.dict(os.environ, {"MSHIP_CACHE_DIR": "/custom/cache"}, clear=True):
            assert resolve_cache_root() == "/custom/cache"

    def test_writable_container_cache_dir_used(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.isdir", return_value=True),
            mock.patch("os.access", return_value=True),
        ):
            assert resolve_cache_root() == "/.cache"

    def test_present_but_unwritable_container_cache_dir_falls_back(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.isdir", return_value=True),
            mock.patch("os.access", return_value=False),
            mock.patch("os.path.expanduser", return_value="/home/user/.modelship/cache"),
            mock.patch("os.makedirs") as mock_makedirs,
        ):
            assert resolve_cache_root() == "/home/user/.modelship/cache"
        mock_makedirs.assert_called_once_with("/home/user/.modelship/cache", exist_ok=True)

    def test_absent_container_cache_dir_falls_back(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.isdir", return_value=False),
            mock.patch("os.path.expanduser", return_value="/home/user/.modelship/cache"),
            mock.patch("os.makedirs") as mock_makedirs,
        ):
            assert resolve_cache_root() == "/home/user/.modelship/cache"
        mock_makedirs.assert_called_once_with("/home/user/.modelship/cache", exist_ok=True)


class TestResolveNodeCacheRoot:
    def test_explicit_env_var_wins(self):
        env = {"MSHIP_NODE_CACHE_DIR": "/scratch/node-cache", "MSHIP_HOME": "/opt/mship"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert resolve_node_cache_root() == "/scratch/node-cache"

    def test_follows_mship_home_not_the_shared_root(self):
        with mock.patch.dict(os.environ, {"MSHIP_HOME": "/opt/mship", "MSHIP_CACHE_DIR": "/mnt/shared"}, clear=True):
            assert resolve_node_cache_root() == "/opt/mship/node-cache"

    def test_defaults_under_home_dir(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.expanduser", return_value="/home/user/.modelship") as mock_expand,
        ):
            assert resolve_node_cache_root() == "/home/user/.modelship/node-cache"
        mock_expand.assert_called_once_with("~/.modelship")


class _FakeResponse:
    def __init__(self, chunks, status_error=None):
        self._chunks = chunks
        self._status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size=1024):
        yield from self._chunks


def test_download_writes_file(tmp_path):
    dest = tmp_path / "model.onnx"
    with mock.patch("modelship.utils.requests.get", return_value=_FakeResponse([b"abc", b"def"])):
        download("http://x/model.onnx", str(dest))
    assert dest.read_bytes() == b"abcdef"


def test_download_skips_when_present(tmp_path):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"existing")
    with mock.patch("modelship.utils.requests.get") as get:
        download("http://x/model.onnx", str(dest))
    get.assert_not_called()
    assert dest.read_bytes() == b"existing"


def test_download_overwrite_refetches(tmp_path):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"old")
    with mock.patch("modelship.utils.requests.get", return_value=_FakeResponse([b"new"])):
        download("http://x/model.onnx", str(dest), overwrite=True)
    assert dest.read_bytes() == b"new"


def test_interrupted_download_leaves_no_corrupt_file(tmp_path):
    dest = tmp_path / "model.onnx"

    def boom(chunk_size=1024):
        yield b"partial"
        raise ConnectionError("dropped mid-stream")

    resp = _FakeResponse([])
    resp.iter_content = boom
    with mock.patch("modelship.utils.requests.get", return_value=resp), pytest.raises(ConnectionError):
        download("http://x/model.onnx", str(dest))

    # Neither the final path nor any temp file is left behind — next run re-downloads.
    assert not dest.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_download_does_not_save_error_body(tmp_path):
    dest = tmp_path / "model.onnx"
    err = requests.HTTPError("404")
    resp = _FakeResponse([b"<html>404</html>"], status_error=err)
    with mock.patch("modelship.utils.requests.get", return_value=resp), pytest.raises(requests.HTTPError):
        download("http://x/model.onnx", str(dest))
    assert not dest.exists()
