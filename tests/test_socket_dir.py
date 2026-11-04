import os
import stat
from types import SimpleNamespace

import pytest

from modelship.infer import socket_dir


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "mship"
    context = SimpleNamespace(get_session_name=lambda: "session_2026-11-04_09-18-22_768639_1454594")
    monkeypatch.setattr(socket_dir, "socket_root", lambda: str(path))
    monkeypatch.setattr(socket_dir.ray, "get_runtime_context", lambda: context)
    return path


def test_creates_a_private_folder_per_session(root):
    session = root / "session_2026-11-04_09-18-22_768639_1454594"
    assert socket_dir.session_socket_dir() == str(session)
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(session).st_mode) == 0o700


def test_reuses_an_existing_private_root(root):
    root.mkdir(mode=0o700)
    socket_dir.session_socket_dir()
    socket_dir.session_socket_dir()


def test_refuses_a_root_others_can_write(root):
    root.mkdir()
    root.chmod(0o777)
    with pytest.raises(RuntimeError, match="mode 0700"):
        socket_dir.session_socket_dir()


def test_refuses_a_symlinked_root(root, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    root.symlink_to(target)
    with pytest.raises(RuntimeError, match="mode 0700"):
        socket_dir.session_socket_dir()
    assert os.listdir(target) == []
