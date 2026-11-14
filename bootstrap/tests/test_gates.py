from unittest.mock import MagicMock, patch

import pytest

from mship_bootstrap import gates
from mship_bootstrap.variants import VARIANTS


class TestCheckPlatform:
    def test_windows_is_refused(self):
        with patch("platform.system", return_value="Windows"), pytest.raises(gates.GateError, match="Windows"):
            gates.check_platform(VARIANTS["cpu"])

    def test_musl_is_refused_by_libc_ver(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("musl", "1.2.5")),
            pytest.raises(gates.GateError, match="musllinux"),
        ):
            gates.check_platform(VARIANTS["cpu"])

    def test_musl_is_refused_by_loader_probe(self):
        """A glibc-built uv-managed CPython reports glibc even on Alpine."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.39")),
            patch("os.listdir", return_value=["ld-musl-x86_64.so.1"]),
            pytest.raises(gates.GateError, match="Alpine"),
        ):
            gates.check_platform(VARIANTS["cpu"])

    def test_glibc_linux_passes(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.39")),
            patch("os.listdir", return_value=["libc.so.6"]),
            patch("os.confstr", return_value="glibc 2.39"),
        ):
            gates.check_platform(VARIANTS["cpu"])

    def test_vulkan_on_glibc_linux_passes(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.39")),
            patch("os.listdir", return_value=["libc.so.6"]),
            patch("os.confstr", return_value="glibc 2.39"),
        ):
            gates.check_platform(VARIANTS["vulkan"])

    def test_vulkan_off_linux_is_refused(self):
        with (
            patch("platform.system", return_value="Darwin"),
            pytest.raises(gates.GateError, match="--vulkan needs Linux, found Darwin.\nUse --metal on Apple Silicon."),
        ):
            gates.check_platform(VARIANTS["vulkan"])

    @pytest.mark.parametrize("name", ["cpu", "cuda", "vulkan"])
    def test_glibc_older_than_2_39_is_refused(self, name):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.38")),
            patch("os.listdir", return_value=["libc.so.6"]),
            patch("os.confstr", return_value="glibc 2.38"),
            pytest.raises(gates.GateError, match=f"--{name} needs glibc 2.39 or newer; this host has 2.38"),
        ):
            gates.check_platform(VARIANTS[name])

    def test_thin_runs_on_an_older_glibc(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.35")),
            patch("os.listdir", return_value=["libc.so.6"]),
            patch("os.confstr", return_value="glibc 2.35"),
        ):
            gates.check_platform(VARIANTS["thin"])

    def test_macos_passes(self):
        with patch("platform.system", return_value="Darwin"):
            gates.check_platform(VARIANTS["metal"])


def _nvml(count: int = 1, cuda: int = 13020, init: int = 0) -> MagicMock:
    nvml = MagicMock()
    nvml.nvmlInit_v2.return_value = init
    nvml.nvmlDeviceGetCount_v2.side_effect = lambda ref: setattr(ref._obj, "value", count) or 0
    nvml.nvmlSystemGetCudaDriverVersion_v2.side_effect = lambda ref: setattr(ref._obj, "value", cuda) or 0
    return nvml


class TestNvidiaDriver:
    def test_reads_the_gpu_count_and_cuda_version(self):
        nvml = _nvml(count=2, cuda=13020)
        with patch("ctypes.CDLL", return_value=nvml):
            assert gates._nvidia_driver() == (2, 13020)
        nvml.nvmlShutdown.assert_called_once()

    def test_no_nvml_library(self):
        with patch("ctypes.CDLL", side_effect=OSError("libnvidia-ml.so.1: cannot open shared object file")):
            assert gates._nvidia_driver() is None

    def test_nvml_that_fails_to_initialize(self):
        with patch("ctypes.CDLL", return_value=_nvml(init=9)):
            assert gates._nvidia_driver() is None


class TestCheckHardware:
    @pytest.mark.parametrize("required", [None, "cpu", "thin"])
    def test_a_variant_without_an_accelerator_always_passes(self, required):
        with patch.object(gates, "_nvidia_driver") as driver:
            gates.check_hardware(required)
        driver.assert_not_called()

    @pytest.mark.parametrize("driver", [None, (0, 13020), (0, 12080)])
    def test_cuda_without_a_gpu_is_left_to_the_engine(self, driver):
        """The engine refuses it, unless MSHIP_NODE_NUM_GPUS=0 runs it CPU-only."""
        with patch.object(gates, "_nvidia_driver", return_value=driver):
            gates.check_hardware("cuda")

    def test_cuda_with_a_driver_older_than_cuda_13_is_refused(self):
        with (
            patch.object(gates, "_nvidia_driver", return_value=(1, 12080)),
            pytest.raises(gates.GateError, match="supports CUDA 12.8"),
        ):
            gates.check_hardware("cuda")

    def test_cuda_with_a_current_driver_passes(self):
        with patch.object(gates, "_nvidia_driver", return_value=(1, 13000)):
            gates.check_hardware("cuda")

    def test_metal_on_linux_is_refused(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            pytest.raises(gates.GateError, match="Apple Silicon"),
        ):
            gates.check_hardware("metal")

    def test_vulkan_leaves_the_device_check_to_the_engine(self):
        """The engine's probe refuses a node with no usable GPU; `info` prints its error."""
        with patch.object(gates, "_nvidia_driver") as driver, patch("platform.system", return_value="Linux"):
            gates.check_hardware("vulkan")
        driver.assert_not_called()

    def test_metal_on_apple_silicon_passes(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            gates.check_hardware("metal")

    def test_the_toolchain_is_a_separate_check(self):
        with (
            patch.object(gates, "_nvidia_driver", return_value=(1, 13020)),
            patch.object(gates, "cuda_toolkit_gaps", return_value=["ninja"]),
        ):
            gates.check_hardware("cuda")


class TestCheckToolchain:
    def test_cuda_without_nvcc_is_refused(self):
        with (
            patch.object(gates, "cuda_toolkit_gaps", return_value=["nvcc (CUDA toolkit)"]),
            pytest.raises(gates.GateError, match="nvcc"),
        ):
            gates.check_toolchain("cuda")

    def test_both_missing_are_named(self):
        with (
            patch.object(gates, "cuda_toolkit_gaps", return_value=["nvcc (CUDA toolkit)", "ninja"]),
            pytest.raises(gates.GateError, match="nvcc \\(CUDA toolkit\\) and ninja"),
        ):
            gates.check_toolchain("cuda")

    def test_cuda_with_the_toolkit_passes(self):
        with patch.object(gates, "cuda_toolkit_gaps", return_value=[]):
            gates.check_toolchain("cuda")

    def test_other_variants_are_unchecked(self):
        with patch.object(gates, "cuda_toolkit_gaps", return_value=["ninja"]):
            for name in ("cpu", "metal", "thin"):
                gates.check_toolchain(name)


class TestFindNvcc:
    def test_env_var_prefix_wins(self, tmp_path):
        nvcc = tmp_path / "bin" / "nvcc"
        nvcc.parent.mkdir()
        nvcc.touch()
        with patch.dict("os.environ", {"CUDA_HOME": str(tmp_path)}):
            assert gates.find_nvcc() == str(nvcc)

    def test_env_var_pointing_nowhere_falls_through_to_path(self):
        with (
            patch.dict("os.environ", {"CUDA_HOME": "/nonexistent"}),
            patch("shutil.which", return_value="/usr/bin/nvcc"),
        ):
            assert gates.find_nvcc() == "/usr/bin/nvcc"

    def test_conventional_symlink_when_off_path(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", lambda p: p == "/usr/local/cuda/bin/nvcc"),
        ):
            assert gates.find_nvcc() == "/usr/local/cuda/bin/nvcc"

    def test_none_when_absent_everywhere(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
        ):
            assert gates.find_nvcc() is None


class TestCudaToolkitGaps:
    def test_no_gaps_when_both_present(self):
        with (
            patch.object(gates, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
            patch("shutil.which", return_value="/usr/bin/ninja"),
        ):
            assert gates.cuda_toolkit_gaps() == []

    def test_reports_each_missing_piece(self):
        with (
            patch.object(gates, "find_nvcc", return_value=None),
            patch("shutil.which", return_value=None),
        ):
            assert gates.cuda_toolkit_gaps() == ["nvcc (CUDA toolkit)", "ninja"]

    def test_ninja_alone(self):
        with (
            patch.object(gates, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
            patch("shutil.which", return_value=None),
        ):
            assert gates.cuda_toolkit_gaps() == ["ninja"]
