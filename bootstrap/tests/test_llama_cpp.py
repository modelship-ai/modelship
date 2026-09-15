import os
import stat

import pytest

from mship_bootstrap import llama_cpp, paths
from mship_bootstrap.variants import VARIANTS

_CPU = VARIANTS["cpu"]


@pytest.fixture
def tag_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    return os.path.join(paths.builds_dir("cpu"), "llama.cpp", llama_cpp._LLAMA_CPP_TAG)


def _install_build(tag_dir: str) -> None:
    extract_dir = os.path.join(tag_dir, "extracted")
    os.makedirs(extract_dir)
    open(os.path.join(extract_dir, "llama"), "w").close()


def test_provision_writes_a_wrapper_any_uid_can_run(tag_dir):
    _install_build(tag_dir)
    wrapper = llama_cpp.provision(_CPU)
    assert wrapper == os.path.join(tag_dir, "llama-server.sh")
    assert stat.S_IMODE(os.stat(wrapper).st_mode) == 0o755


def test_locate_leaves_the_wrapper_untouched(tag_dir):
    _install_build(tag_dir)
    wrapper = os.path.join(tag_dir, "llama-server.sh")
    with open(wrapper, "w") as f:
        f.write("bootstrap-written\n")
    assert llama_cpp.locate(_CPU) == wrapper
    with open(wrapper) as f:
        assert f.read() == "bootstrap-written\n"


def test_locate_without_wrapper_returns_none(tag_dir, capsys):
    _install_build(tag_dir)
    assert llama_cpp.locate(_CPU) is None
    assert "mship bootstrap --cpu" in capsys.readouterr().err


def test_locate_never_fetches(tag_dir, monkeypatch):
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", lambda *a, **k: pytest.fail("fetched"))
    assert llama_cpp.locate(_CPU) is None
