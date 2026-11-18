import os
import stat

import pytest

from mship_bootstrap import paths, sd_cpp
from mship_bootstrap.variants import VARIANTS

_CPU = VARIANTS["cpu"]


@pytest.fixture
def release_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_SD_SERVER_BIN", raising=False)
    monkeypatch.setattr(sd_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: "x86_64")
    return os.path.join(paths.builds_dir("cpu"), "stable-diffusion.cpp", sd_cpp._SD_CPP_RELEASE)


def _install_build(release_dir: str, mode: int = 0o755) -> str:
    extract_dir = os.path.join(release_dir, "extracted")
    os.makedirs(extract_dir)
    binary = os.path.join(extract_dir, "sd-server")
    open(binary, "w").close()
    os.chmod(binary, mode)
    return binary


def _fake_fetch(fetched: list[str]):
    def fetch(url, sha256, archive_path, extract_dir, **kwargs):
        fetched.append(url.rsplit("/", 1)[1])
        os.makedirs(extract_dir)
        open(os.path.join(extract_dir, "sd-server"), "w").close()

    return fetch


def test_explicit_binary_wins(release_dir, monkeypatch):
    monkeypatch.setenv("MSHIP_SD_SERVER_BIN", "/opt/sd-server")
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", lambda *a, **k: pytest.fail("fetched"))
    assert sd_cpp.provision(_CPU) == "/opt/sd-server"
    assert sd_cpp.locate(_CPU) == "/opt/sd-server"


def test_provision_writes_a_wrapper_any_uid_can_run(release_dir):
    _install_build(release_dir)
    wrapper = sd_cpp.provision(_CPU)
    assert wrapper == os.path.join(release_dir, "sd-server.sh")
    assert stat.S_IMODE(os.stat(wrapper).st_mode) == 0o755
    with open(wrapper) as f:
        content = f.read()
    extract_dir = os.path.join(release_dir, "extracted")
    assert f'export LD_LIBRARY_PATH="{extract_dir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"' in content
    assert f'exec "{extract_dir}/sd-server" "$@"' in content


def test_provision_makes_an_existing_binary_executable(release_dir):
    binary = _install_build(release_dir, mode=0o644)
    assert sd_cpp.provision(_CPU) is not None
    assert stat.S_IMODE(os.stat(binary).st_mode) == 0o755


@pytest.mark.parametrize(
    ("variant", "machine", "build"),
    [
        ("cpu", "x86_64", "linux-x64-cpu"),
        ("cpu", "aarch64", "linux-arm64-cpu"),
        ("vulkan", "x86_64", "linux-x64-vulkan"),
        ("vulkan", "aarch64", "linux-arm64-vulkan"),
    ],
)
def test_provision_fetches_the_variants_build(release_dir, monkeypatch, variant, machine, build):
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: machine)
    fetched: list[str] = []
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))
    sd_cpp.provision(VARIANTS[variant])
    assert fetched == [f"sd-server-{sd_cpp._SD_CPP_TAG}-{build}.tar.gz"]


@pytest.mark.parametrize("variant", ["metal", "cpu"])
def test_macos_gets_the_metal_build_and_the_dyld_path(release_dir, monkeypatch, variant):
    monkeypatch.setattr(sd_cpp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: "arm64")
    fetched: list[str] = []
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))
    wrapper = sd_cpp.provision(VARIANTS[variant])
    assert fetched == [f"sd-server-{sd_cpp._SD_CPP_TAG}-macos-arm64-metal.tar.gz"]
    with open(wrapper) as f:
        assert "export DYLD_LIBRARY_PATH=" in f.read()


@pytest.mark.parametrize(("machine", "build"), [("x86_64", "linux-x64-cuda13"), ("aarch64", "linux-arm64-cuda13")])
def test_cuda_puts_torchs_cuda_libraries_on_the_loader_path(release_dir, monkeypatch, machine, build):
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: machine)
    torch_cuda_libs = os.path.join(paths.venv_dir("cuda"), sd_cpp._TORCH_CUDA_LIBS)
    os.makedirs(torch_cuda_libs)
    fetched: list[str] = []
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", _fake_fetch(fetched))

    wrapper = sd_cpp.provision(VARIANTS["cuda"])

    assert fetched == [f"sd-server-{sd_cpp._SD_CPP_TAG}-{build}.tar.gz"]
    with open(wrapper) as f:
        assert f":{torch_cuda_libs}${{LD_LIBRARY_PATH" in f.read()


def test_cuda_without_torchs_cuda_libraries_fails_before_fetching(release_dir, monkeypatch):
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", lambda *a, **k: pytest.fail("fetched"))
    with pytest.raises(sd_cpp.SdCppError, match=r"nvidia/cu13/lib is missing"):
        sd_cpp.provision(VARIANTS["cuda"])


def test_provision_fails_without_a_build_for_the_platform(release_dir, monkeypatch):
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: "riscv64")
    with pytest.raises(sd_cpp.SdCppError, match="no stable-diffusion.cpp build for --cpu on Linux/riscv64"):
        sd_cpp.provision(_CPU)


def test_provision_fails_when_the_download_fails(release_dir, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("HTTP 404")

    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", fail)
    with pytest.raises(sd_cpp.SdCppError, match="HTTP 404"):
        sd_cpp.provision(_CPU)


def test_locate_leaves_the_wrapper_untouched(release_dir):
    _install_build(release_dir)
    wrapper = os.path.join(release_dir, "sd-server.sh")
    with open(wrapper, "w") as f:
        f.write("bootstrap-written\n")
    os.chmod(wrapper, 0o755)
    assert sd_cpp.locate(_CPU) == wrapper
    with open(wrapper) as f:
        assert f.read() == "bootstrap-written\n"


def test_locate_rejects_a_binary_this_user_cannot_execute(release_dir, capsys):
    binary = _install_build(release_dir, mode=0o644)
    wrapper = os.path.join(release_dir, "sd-server.sh")
    open(wrapper, "w").close()
    os.chmod(wrapper, 0o755)
    assert sd_cpp.locate(_CPU) is None
    assert f"{binary} is not executable" in capsys.readouterr().err


def test_locate_without_wrapper_returns_none(release_dir, capsys):
    _install_build(release_dir)
    assert sd_cpp.locate(_CPU) is None
    assert "mship bootstrap --cpu" in capsys.readouterr().err


def test_locate_never_fetches(release_dir, monkeypatch):
    monkeypatch.setattr(sd_cpp, "fetch_and_extract_archive", lambda *a, **k: pytest.fail("fetched"))
    assert sd_cpp.locate(_CPU) is None


def test_locate_without_a_build_for_the_platform_returns_none(release_dir, monkeypatch):
    monkeypatch.setattr(sd_cpp.platform, "machine", lambda: "riscv64")
    assert sd_cpp.locate(_CPU) is None
