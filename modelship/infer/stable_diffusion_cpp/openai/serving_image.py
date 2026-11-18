from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from PIL import Image

from modelship.infer.base_infer import ClientDisconnectedError
from modelship.infer.image_serving_common import (
    alpha_mask,
    build_response,
    decode_image,
    load_image,
    load_mask,
    parse_size,
)
from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ImageVariationRequest,
    create_error_response,
)
from modelship.utils import base_request_id

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

_POLL_INTERVAL_S = 0.1
# sd-server's `max_batch_count`; the OpenAI API allows n up to 10.
_MAX_BATCH_COUNT = 8
_REQUEST_TIMEOUT_S = 10.0

RunCancellable = Callable[[Coroutine[Any, Any, Any], RawRequestProxy], Coroutine[Any, Any, Any]]


class OpenAIServingImage:
    """OpenAI images adapter over sd-server's job API (`/sdcpp/v1/img_gen`). One job serves each endpoint:
    txt2img (no init image), img2img edits (init image, optional mask) and variations (init image, empty prompt).
    sd-server queues jobs and runs one at a time; a disconnect cancels the job, queued or running."""

    request_id_prefix = "img"

    def __init__(self, client: httpx.AsyncClient, run_cancellable: RunCancellable):
        self._client = client
        self._run_cancellable = run_cancellable

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

        body = {"prompt": request.prompt, "width": width, "height": height, "batch_count": request.n}
        return await self._generate(body, request_id, raw_request, request.prompt, request.output_format)

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

        def prepare() -> tuple[str, str | None]:
            src = decode_image(image_data)
            image = src.convert("RGB").resize((width, height))
            # An explicit `mask` upload wins; otherwise, per the OpenAI edits
            # spec, the input image's own transparency marks the edit region.
            mask = load_mask(mask_data, width, height) if mask_data is not None else alpha_mask(src, width, height)
            return _png_b64(image), None if mask is None else _png_b64(mask)

        try:
            image_b64, mask_b64 = await asyncio.to_thread(prepare)
        except ValueError as e:
            return create_error_response(str(e))

        logger.info(
            "image edit request %s: prompt=%r, n=%d, size=%s, inpaint=%s",
            request_id,
            request.prompt,
            request.n,
            request.size,
            mask_b64 is not None,
        )
        body = {
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "batch_count": request.n,
            "init_image": image_b64,
            "strength": request.strength if request.strength is not None else _DEFAULT_EDIT_STRENGTH,
        }
        if mask_b64 is not None:
            body["mask_image"] = mask_b64
        return await self._generate(body, request_id, raw_request, request.prompt, request.output_format)

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("image variation request %s: n=%d, size=%s", request_id, request.n, request.size)
        try:
            width, height = parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        try:
            image_b64 = await asyncio.to_thread(lambda: _png_b64(load_image(image_data, width, height)))
        except ValueError as e:
            return create_error_response(str(e))

        body = {
            "prompt": "",
            "width": width,
            "height": height,
            "batch_count": request.n,
            "init_image": image_b64,
            "strength": request.strength if request.strength is not None else _DEFAULT_VARIATION_STRENGTH,
        }
        return await self._generate(body, request_id, raw_request, None, request.output_format)

    async def _generate(
        self,
        body: dict[str, Any],
        request_id: str,
        raw_request: RawRequestProxy,
        revised_prompt: str | None,
        output_format: str,
    ) -> ImageGenerationResponse | ErrorResponse:
        try:
            images = await self._run_cancellable(
                self._run_jobs({**body, "seed": _RANDOM_SEED}, request_id), raw_request
            )
        except ClientDisconnectedError:
            logger.info("image request %s aborted: client disconnected", request_id)
            return create_error_response("Client disconnected")
        if isinstance(images, ErrorResponse):
            return images

        if output_format == "png":
            data = [ImageObject(b64_json=b64, revised_prompt=revised_prompt) for b64 in images]
            response = ImageGenerationResponse(created=int(time.time()), data=data)
        else:
            response = await asyncio.to_thread(
                lambda: build_response(
                    [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in images], revised_prompt, output_format
                )
            )
        logger.log(TRACE, "image response %s: num_images=%d", request_id, len(response.data))
        return response

    async def _run_jobs(self, body: dict[str, Any], request_id: str) -> list[str] | ErrorResponse:
        """One job per `_MAX_BATCH_COUNT` images, one after another."""
        images: list[str] = []
        remaining = body["batch_count"]
        while remaining > 0:
            count = min(remaining, _MAX_BATCH_COUNT)
            batch = await self._run_job({**body, "batch_count": count}, request_id)
            if isinstance(batch, ErrorResponse):
                return batch
            images += batch
            remaining -= count
        return images

    async def _run_job(self, body: dict[str, Any], request_id: str) -> list[str] | ErrorResponse:
        """Submits a job and polls it to the end; cancelled, it cancels the job too. Returns base64 PNGs."""
        try:
            resp = await self._client.post("/sdcpp/v1/img_gen", json=body, timeout=_REQUEST_TIMEOUT_S)
        except httpx.HTTPError as e:
            logger.warning("image request %s failed: %s", request_id, e)
            return create_error_response(f"sd-server request failed: {e}", status_code=502)
        if resp.status_code != 202:
            detail = _error_detail(resp)
            logger.warning("image request %s refused (%d): %s", request_id, resp.status_code, detail)
            return create_error_response(detail, status_code=resp.status_code if resp.status_code < 500 else 502)
        job_id = resp.json()["id"]
        logger.log(TRACE, "image request %s is sd-server job %s", request_id, job_id)

        try:
            while True:
                try:
                    resp = await self._client.get(f"/sdcpp/v1/jobs/{job_id}", timeout=_REQUEST_TIMEOUT_S)
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    logger.warning("image request %s: polling job %s failed: %s", request_id, job_id, e)
                    return create_error_response(f"sd-server job {job_id} lost: {e}", status_code=502)
                job = resp.json()
                status = job.get("status")
                if status == "completed":
                    images = sorted(job["result"]["images"], key=lambda image: image["index"])
                    return [image["b64_json"] for image in images]
                if status in ("failed", "cancelled"):
                    message = (job.get("error") or {}).get("message") or status
                    logger.warning("image request %s: job %s %s: %s", request_id, job_id, status, message)
                    return create_error_response(f"image generation {status}: {message}", status_code=500)
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            logger.info("image request %s cancelled; cancelling sd-server job %s", request_id, job_id)
            with contextlib.suppress(httpx.HTTPError):
                await asyncio.shield(self._client.post(f"/sdcpp/v1/jobs/{job_id}/cancel", timeout=_REQUEST_TIMEOUT_S))
            raise


def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _error_detail(resp: httpx.Response) -> str:
    with contextlib.suppress(ValueError, AttributeError):
        if isinstance(error := resp.json().get("error"), str):
            return error
    return resp.text or f"sd-server answered {resp.status_code}"
