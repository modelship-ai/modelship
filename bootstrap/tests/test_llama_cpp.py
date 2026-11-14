import os
import stat

import pytest

from mship_bootstrap import llama_cpp, paths
from mship_bootstrap.variants import VARIANTS

_CPU = VARIANTS["cpu"]


@pytest.fixture
def release_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    return os.path.join(paths.builds_dir("cpu"), "llama.cpp", llama_cpp._LLAMA_CPP_RELEASE)


def _install_build(release_dir: str, mode: int = 0o755) -> str:
    extract_dir = os.path.join(release_dir, "extracted")
    os.makedirs(extract_dir)
    binary = os.path.join(extract_dir, "llama")
    open(binary, "w").close()
    os.chmod(binary, mode)
    return binary


def test_provision_writes_a_wrapper_any_uid_can_run(release_dir):
    _install_build(release_dir)
    wrapper = llama_cpp.provision(_CPU)
    assert wrapper == os.path.join(release_dir, "llama-server.sh")
    assert stat.S_IMODE(os.stat(wrapper).st_mode) == 0o755


def test_provision_fetches_a_rebuild_of_an_installed_tag(release_dir, monkeypatch):
    _install_build(release_dir)
    rebuild_dir = f"{release_dir}-r2"
    fetched = []

    def fetch(url, sha256, archive_path, extract_dir, **kwargs):
        fetched.append(extract_dir)
        _install_build(rebuild_dir)

    monkeypatch.setattr(llama_cpp, "_LLAMA_CPP_RELEASE", f"{llama_cpp._LLAMA_CPP_RELEASE}-r2")
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", fetch)
    assert llama_cpp.provision(_CPU) == os.path.join(rebuild_dir, "llama-server.sh")
    assert fetched == [os.path.join(rebuild_dir, "extracted")]


def test_provision_makes_an_existing_binary_executable(release_dir):
    binary = _install_build(release_dir, mode=0o644)
    assert llama_cpp.provision(_CPU) is not None
    assert stat.S_IMODE(os.stat(binary).st_mode) == 0o755


def test_locate_leaves_the_wrapper_untouched(release_dir):
    _install_build(release_dir)
    wrapper = os.path.join(release_dir, "llama-server.sh")
    with open(wrapper, "w") as f:
        f.write("bootstrap-written\n")
    os.chmod(wrapper, 0o755)
    assert llama_cpp.locate(_CPU) == wrapper
    with open(wrapper) as f:
        assert f.read() == "bootstrap-written\n"


def test_locate_rejects_a_wrapper_this_user_cannot_execute(release_dir, capsys):
    _install_build(release_dir)
    wrapper = os.path.join(release_dir, "llama-server.sh")
    open(wrapper, "w").close()
    os.chmod(wrapper, 0o644)
    assert llama_cpp.locate(_CPU) is None
    assert "not executable" in capsys.readouterr().err


def test_locate_rejects_a_binary_this_user_cannot_execute(release_dir, capsys):
    binary = _install_build(release_dir, mode=0o644)
    wrapper = os.path.join(release_dir, "llama-server.sh")
    open(wrapper, "w").close()
    os.chmod(wrapper, 0o755)
    assert llama_cpp.locate(_CPU) is None
    assert f"{binary} is not executable" in capsys.readouterr().err


def test_locate_without_wrapper_returns_none(release_dir, capsys):
    _install_build(release_dir)
    assert llama_cpp.locate(_CPU) is None
    assert "mship bootstrap --cpu" in capsys.readouterr().err


def test_locate_never_fetches(release_dir, monkeypatch):
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", lambda *a, **k: pytest.fail("fetched"))
    assert llama_cpp.locate(_CPU) is None


def test_provision_fails_when_the_download_fails(release_dir, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("HTTP 404")

    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", fail)
    with pytest.raises(llama_cpp.LlamaCppError, match="HTTP 404"):
        llama_cpp.provision(_CPU)


def _fake_fetch(fetched: list[str]):
    """Each archive extracts to the one file its consumer looks for."""

    def fetch(url, sha256, archive_path, extract_dir, **kwargs):
        name = url.rsplit("/", 1)[1]
        fetched.append(name)
        os.makedirs(extract_dir)
        target = "libggml-vulkan.so" if name.startswith("libggml-vulkan") else "llama"
        open(os.path.join(extract_dir, target), "w").close()

    return fetch


@pytest.mark.parametrize(("machine", "platform_name"), [("x86_64", "linux-x64"), ("aarch64", "linux-arm64")])
def test_vulkan_layers_its_backend_over_the_server_build(release_dir, monkeypatch, machine, platform_name):
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: machine)
    fetched: list[str] = []
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))

    wrapper = llama_cpp.provision(VARIANTS["vulkan"])

    tag = llama_cpp._LLAMA_CPP_TAG
    assert fetched == [f"llama-server-{tag}-{platform_name}.tar.gz", f"libggml-vulkan-{tag}-{platform_name}.tar.gz"]
    assert os.path.isfile(os.path.join(os.path.dirname(wrapper), "extracted", "libggml-vulkan.so"))


def test_vulkan_build_without_its_backend_is_fetched_again(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    _install_build(os.path.join(paths.builds_dir("vulkan"), "llama.cpp", llama_cpp._LLAMA_CPP_RELEASE))
    assert llama_cpp.locate(VARIANTS["vulkan"]) is None

    fetched: list[str] = []
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))
    llama_cpp.provision(VARIANTS["vulkan"])
    assert len(fetched) == 2


def test_cpu_fetches_no_backend(release_dir, monkeypatch):
    fetched: list[str] = []
    monkeypatch.setattr(llama_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))
    llama_cpp.provision(_CPU)
    assert fetched == [f"llama-server-{llama_cpp._LLAMA_CPP_TAG}-linux-x64.tar.gz"]


def test_provision_fails_without_a_build_for_the_platform(release_dir, monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "riscv64")
    with pytest.raises(llama_cpp.LlamaCppError, match="no llama.cpp build for Linux/riscv64"):
        llama_cpp.provision(_CPU)
