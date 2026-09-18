import os
import threading
from unittest import mock

import pytest
import requests

from modelship.deploy.actor_options import build_cache_env_vars
from modelship.utils import cache_dir, download
from modelship.utils.cache import (
    reject_unset_cache_roots,
    resolve_cache_root,
    resolve_node_cache_root,
    shared_cache_id,
)


def test_build_cache_env_vars_ignores_the_drivers_paths():
    env = {"MSHIP_CACHE_DIR": "/driver/cache", "MSHIP_NODE_CACHE_DIR": "/driver/node", "HF_HOME": "/driver/hf"}
    with mock.patch.dict(os.environ, env, clear=True):
        env_vars = build_cache_env_vars()
    assert env_vars["HF_HOME"] == "${MSHIP_CACHE_DIR}/huggingface"
    assert env_vars["VLLM_CACHE_ROOT"] == "${MSHIP_NODE_CACHE_DIR}/vllm"
    assert env_vars["FLASHINFER_WORKSPACE_BASE"] == "${MSHIP_NODE_CACHE_DIR}/flashinfer"
    assert env_vars["TRITON_CACHE_DIR"] == "${MSHIP_NODE_CACHE_DIR}/triton"
    assert env_vars["VLLM_CONFIG_ROOT"] == "${MSHIP_NODE_CACHE_DIR}/vllm-config"
    # the roots come from each node's own env
    assert "MSHIP_CACHE_DIR" not in env_vars
    assert "MSHIP_NODE_CACHE_DIR" not in env_vars
    # flashinfer never reads it from env
    assert "FLASHINFER_CACHE_DIR" not in env_vars
    assert "HF_TOKEN" not in env_vars
    assert "HF_HUB_OFFLINE" not in env_vars


def test_build_cache_env_vars_expand_with_rays_update_envs():
    # Private Ray API; fails loudly if a Ray bump changes placeholder expansion.
    from ray._private.utils import update_envs

    node_env = {"MSHIP_CACHE_DIR": "/node/cache", "MSHIP_NODE_CACHE_DIR": "/node/node-cache"}
    with mock.patch.dict(os.environ, node_env, clear=True):
        update_envs(build_cache_env_vars())
        assert os.environ["HF_HOME"] == "/node/cache/huggingface"
        assert os.environ["VLLM_CACHE_ROOT"] == "/node/node-cache/vllm"
        assert os.environ["FLASHINFER_WORKSPACE_BASE"] == "/node/node-cache/flashinfer"
        assert os.environ["TRITON_CACHE_DIR"] == "/node/node-cache/triton"
        assert os.environ["VLLM_CONFIG_ROOT"] == "/node/node-cache/vllm-config"


def test_build_cache_env_vars_ignores_the_drivers_hf_settings():
    env = {"MSHIP_CACHE_DIR": "/.cache", "HF_TOKEN": "hf_secret", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "0"}
    with mock.patch.dict(os.environ, env, clear=True):
        env_vars = build_cache_env_vars()
    assert "HF_TOKEN" not in env_vars
    assert "hf_secret" not in env_vars.values()
    assert "HF_HUB_OFFLINE" not in env_vars
    assert env_vars["HF_HUB_DISABLE_XET"] == "1"


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


class TestRejectUnsetCacheRoots:
    @pytest.mark.parametrize(
        "env", [{}, {"MSHIP_CACHE_DIR": "/.cache"}, {"MSHIP_NODE_CACHE_DIR": "/opt/mship/node-cache"}]
    )
    def test_rejects_a_missing_root(self, env):
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError, match="not set on this node"):
            reject_unset_cache_roots()

    def test_allows_both_roots(self):
        env = {"MSHIP_CACHE_DIR": "/.cache", "MSHIP_NODE_CACHE_DIR": "/opt/mship/node-cache"}
        with mock.patch.dict(os.environ, env, clear=True):
            reject_unset_cache_roots()  # no raise


class TestSharedCacheId:
    def test_stable_across_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_CACHE_DIR", str(tmp_path))
        assert shared_cache_id() == shared_cache_id() == (tmp_path / ".mship-cache-id").read_text()

    def test_separate_roots_get_separate_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_CACHE_DIR", str(tmp_path / "a"))
        a = shared_cache_id()
        monkeypatch.setenv("MSHIP_CACHE_DIR", str(tmp_path / "b"))
        assert shared_cache_id() != a

    def test_racing_creators_agree(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_CACHE_DIR", str(tmp_path))
        barrier = threading.Barrier(8)
        ids: list[str] = []

        def create():
            barrier.wait()
            ids.append(shared_cache_id())

        threads = [threading.Thread(target=create) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(ids)) == 1
        assert [p.name for p in tmp_path.iterdir()] == [".mship-cache-id"]


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


def test_download_reports_each_chunk(tmp_path):
    sizes: list[int] = []
    with mock.patch("modelship.utils.requests.get", return_value=_FakeResponse([b"abc", b"", b"de"])):
        download("http://x/model.onnx", str(tmp_path / "model.onnx"), on_chunk=sizes.append)
    assert sizes == [3, 2]


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
