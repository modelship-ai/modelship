import torch

from modelship.infer.base_infer import BaseInfer
from modelship.infer.diffusers.openai.serving_image import OpenAIServingImage
from modelship.infer.infer_config import DiffusersConfig, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
)

logger = get_logger("infer.diffusers")


def _dummy_png(width: int, height: int) -> bytes:
    """A solid-grey PNG used to exercise the edit/variation paths during warmup."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class DiffusersInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        mem_frac = self._get_memory_fraction()
        if torch.cuda.is_available() and mem_frac is not None:
            torch.cuda.set_per_process_memory_fraction(mem_frac)

    def shutdown(self) -> None:
        # The img2img / inpaint pipelines and the serving wrapper all hold
        # references to the text2img pipeline's shared components (VAE / UNet /
        # text-encoder). Every holder must be dropped before empty_cache() can
        # actually reclaim the GPU memory. Idempotent — the getattr guards make
        # repeat calls (e.g. graceful shutdown then __del__) safe.
        for attr in ("serving_image", "_img2img", "_inpaint", "_pipeline"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            from modelship.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "diffusers_pipeline"})

    async def start(self):
        from diffusers.pipelines.auto_pipeline import (
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
        )

        config = self.model_config.diffusers_config or DiffusersConfig()
        dtype = _TORCH_DTYPES[config.dtype]

        if not self.model_config._resolved_path:
            raise ValueError(
                f"Diffusers deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        logger.info(
            "Loading diffusers pipeline: %s (dtype=%s, device=%s)",
            self.model_config._resolved_path,
            config.dtype,
            self.device,
        )
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_config._resolved_path,
            dtype=dtype,
        ).to(device=self.device, dtype=dtype)

        # img2img / inpaint back /v1/images/edits and /v1/images/variations.
        # from_pipe can fail for models AutoPipeline can't map; degrade
        # gracefully (those endpoints return a clear error) rather than crash
        # the whole deployment, which still serves text2img generation.
        try:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self._pipeline)
        except Exception as e:
            self._img2img = None
            logger.warning(
                "img2img pipeline unavailable for %s (edits/variations disabled): %s", self.model_config.name, e
            )
        try:
            self._inpaint = AutoPipelineForInpainting.from_pipe(self._pipeline)
        except Exception as e:
            self._inpaint = None
            logger.warning("inpaint pipeline unavailable for %s (masked edits disabled): %s", self.model_config.name, e)

        self.serving_image: OpenAIServingImage | None = (
            OpenAIServingImage(
                pipeline=self._pipeline,
                config=config,
                run_blocking=self.run_blocking,
                img2img_pipeline=self._img2img,
                inpaint_pipeline=self._inpaint,
            )
            if self.model_config.usecase is ModelUsecase.image
            else None
        )

    async def warmup(self) -> None:
        if self.serving_image is None:
            return
        logger.info("Warming up diffusers model: %s", self.model_config.name)
        proxy = RawRequestProxy(None, {})
        gen_request = ImageGenerationRequest(
            model=self.model_config.name,
            prompt="warmup",
            n=1,
            size="64x64",
        )
        await self.create_image_generation(gen_request, proxy)

        dummy_png = _dummy_png(64, 64)
        edit_request = ImageEditRequest.model_construct(
            model=self.model_config.name, prompt="warmup", n=1, size="64x64", strength=None
        )
        await self.create_image_edit(dummy_png, None, edit_request, proxy)
        variation_request = ImageVariationRequest.model_construct(
            model=self.model_config.name, n=1, size="64x64", strength=None
        )
        await self.create_image_variation(dummy_png, variation_request, proxy)
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
