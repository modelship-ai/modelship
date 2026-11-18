import contextlib
import os
import socket

import httpx

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy, StableDiffusionCppConfig
from modelship.infer.server_process import ServerProcess, close_client
from modelship.infer.stable_diffusion_cpp.openai.serving_image import OpenAIServingImage
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
)
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight
from modelship.utils.runtime_env import env_setting
from modelship.utils.socket_dir import session_socket_dir
from modelship.utils.variant import current_variant

logger = get_logger("infer.stable_diffusion_cpp")

# Covers a model load from disk and Metal's shader compile on a machine's first launch.
_STARTUP_TIMEOUT_S = float(os.environ.get("MSHIP_SD_SERVER_STARTUP_TIMEOUT", "900"))
_HOST = "127.0.0.1"
# sd-server's device name for the replica's first GPU; Ray's device hiding makes it the assigned one.
_GPU_DEVICES = {"cuda": "cuda0", "vulkan": "vulkan0", "metal": "mtl0"}


class StableDiffusionCppInfer(BaseInfer):
    """Image-generation loader backed by an `sd-server` subprocess (stable-diffusion.cpp), driven over its native
    job API. Loads GGUF-quantized diffusion checkpoints (SD1.5/SDXL/SD-Turbo, Flux) and serves the OpenAI images
    endpoints."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        user_config = model_config.stable_diffusion_cpp_config or StableDiffusionCppConfig()
        user_overrides = user_config.model_dump(exclude_unset=True)

        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        self.config = user_config.model_copy(update=merged)

        self._server: ServerProcess | None = None
        self._client: httpx.AsyncClient | None = None
        self.serving_image: OpenAIServingImage | None = None

    def shutdown(self) -> None:
        self.serving_image = None
        if self._server is not None:
            self._server.stop()
            self._server = None
        if self._client is not None:
            client, self._client = self._client, None
            close_client(client)

    def __del__(self):
        with contextlib.suppress(BaseException):
            self.shutdown()

    async def start(self) -> None:
        binary = os.environ.get("MSHIP_SD_SERVER_BIN")
        if not binary or not os.path.isfile(binary):
            raise ValueError(
                f"stable_diffusion_cpp deployment '{self.model_config.name}' requires MSHIP_SD_SERVER_BIN to point at "
                f"an sd-server executable; got {binary!r}. The node's start log says why the bootstrapper set none, "
                f"and `mship bootstrap --{current_variant()}` provisions one. From source, see docs/development.md."
            )
        model_path = self.model_config._resolved_path
        if not model_path:
            raise ValueError(
                f"StableDiffusionCpp deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        logger.info("Starting sd-server for model: %s", self.model_config.name)
        base_url = f"http://{_HOST}:{_free_port()}"
        self._server = ServerProcess("sd-server", self.model_config.name, self.backend_died)
        try:
            # ggml loads backends from the cwd too, and sd-server scans `./models`.
            await self._server.launch(self._args(binary, model_path, base_url), cwd=session_socket_dir())
            async with httpx.AsyncClient(base_url=base_url) as probe:
                await self._server.wait_healthy(probe, "/v1/models", _STARTUP_TIMEOUT_S)
        except Exception:
            self.shutdown()
            raise
        logger.info("sd-server healthy for '%s' on %s", self.model_config.name, base_url)

        self._server.watch()
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout=None, connect=10.0))
        self.serving_image = OpenAIServingImage(self._client, self.run_cancellable)

    def _args(self, binary: str, model_path: str, base_url: str) -> list[str]:
        c = self.config
        args = [
            binary,
            "--listen-ip",
            _HOST,
            "--listen-port",
            base_url.rsplit(":", 1)[1],
            "-m",
            model_path,
            "--backend",
            self._backend(),
            "--steps",
            str(c.sample_steps),
            "--cfg-scale",
            str(c.cfg_scale),
            "--disable-image-metadata",
        ]
        if c.guidance is not None:
            args += ["--guidance", str(c.guidance)]
        if c.sample_method != "default":
            args += ["--sampling-method", c.sample_method]
        if c.scheduler != "default":
            args += ["--scheduler", c.scheduler]
        if c.wtype != "default":
            args += ["--type", c.wtype]
        if c.n_threads > 0:
            args += ["--threads", str(c.n_threads)]
        if c.vae_tiling:
            args += ["--vae-tiling"]
        if c.flash_attn is not None:
            args += ["--flash-attn", c.flash_attn]
        if c.offload_to_cpu:
            args += ["--offload-to-cpu"]
        if c.max_vram is not None:
            args += ["--max-vram", str(c.max_vram)]
        if c.auto_fit is not None:
            args += ["--auto-fit", "on" if c.auto_fit else "off"]
        for flag, path in (
            ("--diffusion-model", c.diffusion_model_path),
            ("--clip_l", c.clip_l_path),
            ("--clip_g", c.clip_g_path),
            ("--t5xxl", c.t5xxl_path),
            ("--vae", c.vae_path),
        ):
            if path:
                args += [flag, path]
        if env_setting("MSHIP_LOG_LEVEL").upper() == "TRACE":
            args += ["--verbose"]
        return args

    def _backend(self) -> str:
        """Always named: an unnamed GPU that didn't load would leave sd-server silently on the CPU."""
        if self.model_config.num_gpus <= 0:
            return "cpu"
        variant = current_variant()
        if (device := _GPU_DEVICES.get(variant)) is None:
            raise ValueError(f"stable_diffusion_cpp has no GPU on the {variant} variant")
        assignments = []
        if self.config.clip_on_cpu:
            assignments.append("te=cpu")
        if self.config.vae_on_cpu:
            assignments.append("vae=cpu")
        return ",".join([*assignments, device])

    async def warmup(self) -> None:
        if self.serving_image is None:
            return
        logger.info("Warming up stable-diffusion.cpp model: %s", self.model_config.name)
        request = ImageGenerationRequest(model=self.model_config.name, prompt="warmup", n=1, size="64x64")
        result = await self.create_image_generation(request, RawRequestProxy(None, {}))
        if isinstance(result, ErrorResponse):
            logger.warning("warmup image generation failed for '%s': %s", self.model_config.name, result.error.message)
        else:
            logger.info("Warmup image generation done for %s", self.model_config.name)

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return await super().create_image_generation(request, raw_request)
        return await self.serving_image.create_image_generation(request, raw_request)

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return await super().create_image_edit(image_data, mask_data, request, raw_request)
        return await self.serving_image.create_image_edit(image_data, mask_data, request, raw_request)

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return await super().create_image_variation(image_data, request, raw_request)
        return await self.serving_image.create_image_variation(image_data, request, raw_request)


def _free_port() -> int:
    """A port free now; another process can take it before sd-server binds it, which fails the launch."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return sock.getsockname()[1]
