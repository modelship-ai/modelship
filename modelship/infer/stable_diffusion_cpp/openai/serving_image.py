from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from modelship.infer.image_serving_common import (
    alpha_mask,
    build_response,
    decode_image,
    load_image,
    load_mask,
    parse_size,
)
from modelship.infer.infer_config import RawRequestProxy, StableDiffusionCppConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
    create_error_response,
)
from modelship.utils import base_request_id

if TYPE_CHECKING:
    from stable_diffusion_cpp import StableDiffusion

logger = get_logger("infer.stable_diffusion_cpp.image")

# Default img2img strength when the request omits it. Edits stay closer to the
# source (lower strength) since the prompt guides the change; variations have no
# prompt, so they need a higher strength to diverge from the input. Mirrors the
# diffusers adapter.
_DEFAULT_EDIT_STRENGTH = 0.7
_DEFAULT_VARIATION_STRENGTH = 0.8

# sd.cpp treats seed=-1 as random. Use it so repeated generations (and especially
# variations, which share one input) don't return byte-identical images.
_RANDOM_SEED = -1


class OpenAIServingImage:
    """OpenAI images adapter over a stable-diffusion.cpp handle. The native
    `generate_image` call serves all three endpoints — txt2img (no init image),
    img2img edits (init image, optional mask), and variations (init image, empty
    prompt)."""

    request_id_prefix = "img"

    def __init__(
        self,
        sd: StableDiffusion,
        config: StableDiffusionCppConfig,
        run_blocking: Callable[[Callable[[threading.Event], Any]], Awaitable[Any]],
    ):
        self.sd = sd
        self.config = config
        self._run_blocking = run_blocking

    def _generate(self, **kwargs):
        """Invoke sd.cpp with the per-model defaults applied. Callers supply the
        per-request bits (prompt, width/height, init_image, mask, strength, n)."""
        return self.sd.generate_image(
            cfg_scale=self.config.cfg_scale,
            sample_method=self.config.sample_method,
            scheduler=self.config.scheduler,
            sample_steps=self.config.sample_steps,
            vae_tiling=self.config.vae_tiling,
            seed=_RANDOM_SEED,
            **kwargs,
        )

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info(
            "image generation request %s: prompt=%r, n=%d, size=%s", request_id, request.prompt, request.n, request.size
        )

        try:
            width, height = parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        def run(_stop: threading.Event) -> ImageGenerationResponse:
            images = self._generate(prompt=request.prompt, width=width, height=height, batch_count=request.n)
            return build_response(images, revised_prompt=request.prompt)

        response: ImageGenerationResponse = await self._run_blocking(run)
        logger.log(TRACE, "image response %s: num_images=%d", request_id, len(response.data))
        return response

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        raw_request: RawRequestProxy,
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"

        try:
            width, height = parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        strength = request.strength if request.strength is not None else _DEFAULT_EDIT_STRENGTH

        # Decode / resize / mask work and inference are all CPU-bound; run the
        # whole chain in the executor so the event loop is never blocked.
        def run(_stop: threading.Event) -> ImageGenerationResponse | ErrorResponse:
            try:
                src = decode_image(image_data)
                image = src.convert("RGB").resize((width, height))
                # An explicit `mask` upload wins; otherwise, per the OpenAI edits
                # spec, the input image's own transparency marks the edit region.
                mask = load_mask(mask_data, width, height) if mask_data is not None else alpha_mask(src, width, height)
            except ValueError as e:
                return create_error_response(str(e))

            inpaint = mask is not None
            logger.info(
                "image edit request %s: prompt=%r, n=%d, size=%s, inpaint=%s",
                request_id,
                request.prompt,
                request.n,
                request.size,
                inpaint,
            )

            kwargs: dict = {
                "prompt": request.prompt,
                "init_image": image,
                "width": width,
                "height": height,
                "strength": strength,
                "batch_count": request.n,
            }
            if inpaint:
                kwargs["mask_image"] = mask
            images = self._generate(**kwargs)
            return build_response(images, revised_prompt=request.prompt)

        response: ImageGenerationResponse | ErrorResponse = await self._run_blocking(run)
        if isinstance(response, ImageGenerationResponse):
            logger.log(TRACE, "image edit response %s: num_images=%d", request_id, len(response.data))
        return response

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("image variation request %s: n=%d, size=%s", request_id, request.n, request.size)

        try:
            width, height = parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        strength = request.strength if request.strength is not None else _DEFAULT_VARIATION_STRENGTH

        def run(_stop: threading.Event) -> ImageGenerationResponse | ErrorResponse:
            try:
                image = load_image(image_data, width, height)
            except ValueError as e:
                return create_error_response(str(e))
            images = self._generate(
                prompt="", init_image=image, width=width, height=height, strength=strength, batch_count=request.n
            )
            return build_response(images, revised_prompt=None)

        response: ImageGenerationResponse | ErrorResponse = await self._run_blocking(run)
        if isinstance(response, ImageGenerationResponse):
            logger.log(TRACE, "image variation response %s: num_images=%d", request_id, len(response.data))
        return response
