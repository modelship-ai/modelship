"""The stable_diffusion_cpp loader's sd-server subprocess: launch flags, lifecycle (a fake executable, no real
sd-server in CI) and crash handling."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, StableDiffusionCppConfig
from modelship.infer.stable_diffusion_cpp import stable_diffusion_cpp_infer
from modelship.infer.stable_diffusion_cpp.stable_diffusion_cpp_infer import StableDiffusionCppInfer
from modelship.preflight import HardwareProfile
from modelship.utils import socket_dir

_FAKE_SERVER = textwrap.dedent(
    """
    import http.server
    import os
    import sys

    with open(os.environ["FAKE_SD_CWD_FILE"], "w") as f:
        f.write(os.getcwd())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, fmt, *args):
            pass

    port = int(sys.argv[sys.argv.index("--listen-port") + 1])
    http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)

# sd-server exits 0 when it can't listen.
_FAKE_EXIT_0 = "import sys\nprint('listening on http://127.0.0.1:1234')\nsys.exit(0)\n"


def _write_fake_executable(tmp_path, source: str) -> str:
    script = tmp_path / "fake-sd-server"
    script.write_text(f"#!{sys.executable}\n{source}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _make_config(num_gpus: float = 0, **sd_kwargs) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-image",
        model="org/test-sd-gguf",
        usecase=ModelUsecase.image,
        loader=ModelLoader.stable_diffusion_cpp,
        num_gpus=num_gpus,
        stable_diffusion_cpp_config=StableDiffusionCppConfig(**sd_kwargs),
    )
    cfg._resolved_path = "/fake/model.gguf"
    return cfg


def _args(num_gpus: float = 0, **sd_kwargs) -> list[str]:
    return StableDiffusionCppInfer(_make_config(num_gpus, **sd_kwargs))._args(
        "/bin/sd-server", "/fake/model.gguf", "http://127.0.0.1:4321"
    )


def _flag(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


@pytest.fixture(autouse=True)
def cpu_variant(monkeypatch):
    monkeypatch.setenv("MSHIP_VARIANT", "cpu")
    monkeypatch.setattr(stable_diffusion_cpp_infer, "discover_hardware", HardwareProfile)
    monkeypatch.setattr(stable_diffusion_cpp_infer, "run_preflight", lambda config, hw: {})


@pytest.fixture(autouse=True)
def socket_root(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = os.path.join(temp_dir, "mship")
        context = SimpleNamespace(get_session_name=lambda: "session")
        monkeypatch.setattr(socket_dir, "socket_root", lambda: root)
        monkeypatch.setattr(socket_dir.ray, "get_runtime_context", lambda: context)
        yield root


class TestLaunchFlags:
    def test_defaults(self):
        args = _args()
        assert args[0] == "/bin/sd-server"
        assert _flag(args, "--listen-ip") == "127.0.0.1"
        assert _flag(args, "--listen-port") == "4321"
        assert _flag(args, "-m") == "/fake/model.gguf"
        assert _flag(args, "--backend") == "cpu"
        assert _flag(args, "--steps") == "20"
        assert _flag(args, "--cfg-scale") == "7.0"
        assert "--disable-image-metadata" in args
        for absent in (
            "--guidance",
            "--sampling-method",
            "--scheduler",
            "--type",
            "--threads",
            "--vae-tiling",
            "--flash-attn",
            "--offload-to-cpu",
            "--max-vram",
            "--auto-fit",
            "--diffusion-model",
            "--verbose",
        ):
            assert absent not in args

    def test_every_field_maps_to_its_flag(self):
        args = _args(
            sample_steps=4,
            cfg_scale=1.0,
            guidance=3.5,
            sample_method="euler",
            scheduler="karras",
            wtype="q8_0",
            n_threads=6,
            vae_tiling=True,
            flash_attn="auto,vae=off",
            offload_to_cpu=True,
            max_vram=3.75,
            auto_fit=False,
            diffusion_model_path="/m/dit.gguf",
            clip_l_path="/m/clip_l.gguf",
            clip_g_path="/m/clip_g.gguf",
            t5xxl_path="/m/t5.gguf",
            vae_path="/m/ae.gguf",
        )
        assert _flag(args, "--steps") == "4"
        assert _flag(args, "--cfg-scale") == "1.0"
        assert _flag(args, "--guidance") == "3.5"
        assert _flag(args, "--sampling-method") == "euler"
        assert _flag(args, "--scheduler") == "karras"
        assert _flag(args, "--type") == "q8_0"
        assert _flag(args, "--threads") == "6"
        assert "--vae-tiling" in args
        assert _flag(args, "--flash-attn") == "auto,vae=off"
        assert "--offload-to-cpu" in args
        assert _flag(args, "--max-vram") == "3.75"
        assert _flag(args, "--auto-fit") == "off"
        assert _flag(args, "--diffusion-model") == "/m/dit.gguf"
        assert _flag(args, "--clip_l") == "/m/clip_l.gguf"
        assert _flag(args, "--clip_g") == "/m/clip_g.gguf"
        assert _flag(args, "--t5xxl") == "/m/t5.gguf"
        assert _flag(args, "--vae") == "/m/ae.gguf"

    def test_a_yaml_boolean_flash_attn_is_a_word(self):
        assert _flag(_args(flash_attn=True), "--flash-attn") == "on"
        assert _flag(_args(flash_attn=False), "--flash-attn") == "off"

    def test_auto_fit_on(self):
        assert _flag(_args(auto_fit=True), "--auto-fit") == "on"

    def test_trace_logging_makes_sd_server_verbose(self, monkeypatch):
        monkeypatch.setenv("MSHIP_LOG_LEVEL", "TRACE")
        assert "--verbose" in _args()


class TestBackend:
    @pytest.mark.parametrize(
        ("variant", "num_gpus", "expected"),
        [
            ("cpu", 0, "cpu"),
            ("cuda", 0, "cpu"),
            ("vulkan", 0, "cpu"),
            ("metal", 0, "cpu"),
            ("cuda", 1, "cuda0"),
            ("cuda", 0.5, "cuda0"),
            ("vulkan", 1, "vulkan0"),
            ("metal", 0.25, "mtl0"),
        ],
    )
    def test_the_device_is_always_named(self, monkeypatch, variant, num_gpus, expected):
        monkeypatch.setenv("MSHIP_VARIANT", variant)
        assert _flag(_args(num_gpus), "--backend") == expected

    def test_modules_moved_to_the_cpu_come_before_the_device(self, monkeypatch):
        monkeypatch.setenv("MSHIP_VARIANT", "vulkan")
        assert _flag(_args(1, clip_on_cpu=True, vae_on_cpu=True), "--backend") == "te=cpu,vae=cpu,vulkan0"
        assert _flag(_args(1, vae_on_cpu=True), "--backend") == "vae=cpu,vulkan0"

    def test_a_cpu_deploy_ignores_the_module_placement(self):
        assert _flag(_args(0, clip_on_cpu=True, vae_on_cpu=True), "--backend") == "cpu"

    def test_a_gpu_deploy_on_a_variant_without_one_fails(self):
        with pytest.raises(ValueError, match="no GPU on the cpu variant"):
            _args(1)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_waits_for_health_and_runs_in_the_session_dir(self, tmp_path, monkeypatch, socket_root):
        monkeypatch.setenv("MSHIP_SD_SERVER_BIN", _write_fake_executable(tmp_path, _FAKE_SERVER))
        cwd_file = tmp_path / "cwd"
        monkeypatch.setenv("FAKE_SD_CWD_FILE", str(cwd_file))

        infer = StableDiffusionCppInfer(_make_config())
        await infer.start()
        try:
            assert infer._server is not None
            proc = infer._server.proc
            assert proc is not None and proc.poll() is None
            assert infer.serving_image is not None
            assert cwd_file.read_text() == os.path.join(socket_root, "session")
        finally:
            infer.shutdown()

        assert infer._server is None
        assert infer.serving_image is None
        assert proc.wait(timeout=5) is not None

    @pytest.mark.asyncio
    async def test_an_exit_0_during_startup_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_SD_SERVER_BIN", _write_fake_executable(tmp_path, _FAKE_EXIT_0))
        infer = StableDiffusionCppInfer(_make_config())
        with pytest.raises(RuntimeError, match=r"sd-server for 'test-image' exited \(rc=0\) during startup"):
            await infer.start()
        assert infer._server is None

    @pytest.mark.asyncio
    async def test_an_exit_after_startup_kills_the_replica(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_SD_SERVER_BIN", _write_fake_executable(tmp_path, _FAKE_SERVER))
        monkeypatch.setenv("FAKE_SD_CWD_FILE", str(tmp_path / "cwd"))
        infer = StableDiffusionCppInfer(_make_config())
        died = asyncio.Event()
        reasons: list[str] = []
        loop = asyncio.get_running_loop()

        def backend_died(reason: str) -> None:
            reasons.append(reason)
            loop.call_soon_threadsafe(died.set)

        with patch.object(infer, "backend_died", backend_died):
            await infer.start()
        try:
            assert infer._server is not None and infer._server.proc is not None
            infer._server.proc.kill()
            await asyncio.wait_for(died.wait(), timeout=5)
            assert reasons == ["sd-server exited (rc=-9)"]
        finally:
            infer.shutdown()

    @pytest.mark.asyncio
    async def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.delenv("MSHIP_SD_SERVER_BIN", raising=False)
        with pytest.raises(ValueError, match="requires MSHIP_SD_SERVER_BIN"):
            await StableDiffusionCppInfer(_make_config()).start()

    @pytest.mark.asyncio
    async def test_missing_resolved_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_SD_SERVER_BIN", _write_fake_executable(tmp_path, _FAKE_SERVER))
        cfg = _make_config()
        cfg._resolved_path = None
        with pytest.raises(ValueError, match="missing a resolved model path"):
            await StableDiffusionCppInfer(cfg).start()
