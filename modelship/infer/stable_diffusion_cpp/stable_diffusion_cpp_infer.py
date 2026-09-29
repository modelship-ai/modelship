import asyncio

from stable_diffusion_cpp import StableDiffusion

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy, StableDiffusionCppConfig
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
from modelship.utils.runtime_env import cluster_env_value

logger = get_logger("infer.stable_diffusion_cpp")


class StableDiffusionCppInfer(BaseInfer):
    """CPU-only image-generation loader backed by stable-diffusion.cpp (via the
    stable-diffusion-cpp-python bindings). Loads GGUF-quantized single-file
    diffusion checkpoints (SD1.5/SDXL/SD-Turbo, all-in-one Flux) and serves the
    OpenAI images endpoints."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        user_config = model_config.stable_diffusion_cpp_config or StableDiffusionCppConfig()
        user_overrides = user_config.model_dump(exclude_unset=True)

        # Preflight: hardware-aware safe defaults the user can override; user
        # values always win and divergences are logged.
        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        self.config = user_config.model_copy(update=merged)

        # Verbose native logging when MSHIP_LOG_LEVEL is TRACE.
        mship_log_level = cluster_env_value("MSHIP_LOG_LEVEL").upper()
        self._verbose = mship_log_level == "TRACE"

        # CPU-only in v1: the actor is given num_gpus=0 in actor_options, so warn
        # if the config asked for GPUs (the request is ignored).
        if model_config.num_gpus and model_config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': stable_diffusion_cpp currently only supports CPU.",
                model_config.num_gpus,
                model_config.name,
            )

        self.sd: StableDiffusion | None = None
        self.serving_image: OpenAIServingImage | None = None
        logger.info(
            "initialising stable-diffusion.cpp engine (verbose=%s) with config: %s",
            self._verbose,
            self.config.model_dump(),
        )

    def shutdown(self) -> None:
        if self.sd is not None:
            logger.info("Shutting down stable-diffusion.cpp engine for %s", self.model_config.name)
        self.serving_image = None
        self.sd = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        logger.info("Start stable-diffusion.cpp infer for model: %s", self.model_config.name)
        model_path = self.model_config._resolved_path
        if not model_path:
            raise ValueError(
                f"StableDiffusionCpp deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        loop = asyncio.get_running_loop()
        sd = await loop.run_in_executor(None, self._load, model_path)
        self.sd = sd
        self.serving_image = OpenAIServingImage(sd, self.config)

    def _load(self, model_path: str) -> StableDiffusion:
        c = self.config
        # vae_decode_only=False so the VAE encoder is available for img2img
        # (edits / variations), not just txt2img decode. Optional aux paths back
        # split checkpoints; empty string means "unused" to the binding.
        return StableDiffusion(
            model_path=model_path,
            diffusion_model_path=c.diffusion_model_path or "",
            clip_l_path=c.clip_l_path or "",
            clip_g_path=c.clip_g_path or "",
            t5xxl_path=c.t5xxl_path or "",
            vae_path=c.vae_path or "",
            n_threads=c.n_threads,
            wtype=c.wtype,
            vae_decode_only=False,
            verbose=self._verbose,
            **c.model_kwargs,
        )

    async def warmup(self) -> None:
        if self.serving_image is None:
            return
        logger.info("Warming up stable-diffusion.cpp model: %s", self.model_config.name)
        proxy = RawRequestProxy(None, {})
        request = ImageGenerationRequest(model=self.model_config.name, prompt="warmup", n=1, size="64x64")
        await self.create_image_generation(request, proxy)
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
