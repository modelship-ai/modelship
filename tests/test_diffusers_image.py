import asyncio
import base64
import io
import threading
import time

import pytest
from pydantic import ValidationError

pytest.importorskip("diffusers")

from PIL import Image

from modelship.infer.diffusers.openai.serving_image import (
    _DEFAULT_EDIT_STRENGTH,
    _DEFAULT_VARIATION_STRENGTH,
    OpenAIServingImage,
)
from modelship.infer.infer_config import DiffusersConfig, RawRequestProxy
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
)
from tests.test_base_infer_exclusive import ExclusiveInfer, wait_for_turn


class _SteppingPipeline:
    def __init__(self):
        self.started = threading.Event()
        self.steps = 0

    def _run(self, num_inference_steps, on_step):
        self.started.set()
        for i in range(num_inference_steps):
            time.sleep(0.01)
            self.steps += 1
            on_step(i)
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


class _StepEndPipeline(_SteppingPipeline):
    def __call__(self, *, num_inference_steps, callback_on_step_end=None, **_kwargs):
        return self._run(num_inference_steps, lambda i: callback_on_step_end and callback_on_step_end(self, i, i, {}))


class _LegacyCallbackPipeline(_SteppingPipeline):
    def __call__(self, *, num_inference_steps, callback=None, **_kwargs):
        return self._run(num_inference_steps, lambda i: callback and callback(i, i, None))


class _StubPipeline:
    """Records the kwargs it was called with and returns `n` solid PIL images."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs.get("num_images_per_prompt", 1)
        images = [Image.new("RGB", (8, 8), (10, 20, 30)) for _ in range(n)]
        return type("Result", (), {"images": images})()


class _ConcurrencyProbe:
    """Stub pipeline that records the peak number of overlapping calls."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)  # widen the window so unserialized calls would overlap
        with self._lock:
            self.active -= 1
        n = kwargs.get("num_images_per_prompt", 1)
        return type("Result", (), {"images": [Image.new("RGB", (8, 8)) for _ in range(n)]})()


def _png_bytes(w: int = 16, h: int = 16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _rgba_png(w: int = 16, h: int = 16, *, transparent: bool) -> bytes:
    """An RGBA PNG. With ``transparent=True`` a central square has alpha 0
    (the OpenAI 'edit here' region); otherwise the image is fully opaque."""
    img = Image.new("RGBA", (w, h), (200, 100, 50, 255))
    if transparent:
        for x in range(w // 4, w * 3 // 4):
            for y in range(h // 4, h * 3 // 4):
                img.putpixel((x, y), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _serving(img2img=None, inpaint=None, pipeline=None, infer=None) -> OpenAIServingImage:
    return OpenAIServingImage(
        pipeline=pipeline or _StubPipeline(),  # type: ignore[arg-type]
        config=DiffusersConfig(num_inference_steps=5, guidance_scale=3.0),
        run_exclusive=(infer or ExclusiveInfer()).run_exclusive,
        img2img_pipeline=img2img,
        inpaint_pipeline=inpaint,
    )


def _proxy() -> RawRequestProxy:
    return RawRequestProxy(None, {})


def _assert_valid_b64_image(obj) -> None:
    raw = base64.b64decode(obj.b64_json)
    Image.open(io.BytesIO(raw)).verify()


@pytest.mark.asyncio
class TestImageEdit:
    async def test_edit_without_mask_uses_img2img(self):
        img2img = _StubPipeline()
        serving = _serving(img2img=img2img)
        request = ImageEditRequest.model_construct(model="m", prompt="add a hat", n=2, size="32x32", strength=None)

        result = await serving.create_image_edit(_png_bytes(), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 2
        for obj in result.data:
            assert obj.revised_prompt == "add a hat"
            _assert_valid_b64_image(obj)
        assert len(img2img.calls) == 1
        call = img2img.calls[0]
        assert call["prompt"] == "add a hat"
        assert call["strength"] == _DEFAULT_EDIT_STRENGTH
        assert call["num_inference_steps"] == 5
        assert call["guidance_scale"] == 3.0
        assert "mask_image" not in call

    async def test_edit_with_mask_uses_inpaint(self):
        img2img = _StubPipeline()
        inpaint = _StubPipeline()
        serving = _serving(img2img=img2img, inpaint=inpaint)
        request = ImageEditRequest.model_construct(model="m", prompt="erase", n=1, size="16x16", strength=0.5)

        result = await serving.create_image_edit(_png_bytes(), _png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(inpaint.calls) == 1
        assert len(img2img.calls) == 0
        call = inpaint.calls[0]
        assert call["strength"] == 0.5
        assert "mask_image" in call

    async def test_edit_transparent_image_no_mask_uses_inpaint(self):
        # RGBA input with transparency + no separate mask -> alpha is the mask.
        img2img = _StubPipeline()
        inpaint = _StubPipeline()
        serving = _serving(img2img=img2img, inpaint=inpaint)
        request = ImageEditRequest.model_construct(model="m", prompt="fill", n=1, size="16x16", strength=None)

        result = await serving.create_image_edit(_rgba_png(transparent=True), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(inpaint.calls) == 1
        assert len(img2img.calls) == 0
        assert "mask_image" in inpaint.calls[0]

    async def test_edit_opaque_rgba_no_mask_uses_img2img(self):
        img2img = _StubPipeline()
        inpaint = _StubPipeline()
        serving = _serving(img2img=img2img, inpaint=inpaint)
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="16x16", strength=None)

        result = await serving.create_image_edit(_rgba_png(transparent=False), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(img2img.calls) == 1
        assert len(inpaint.calls) == 0

    async def test_edit_rgba_mask_uses_alpha_channel(self):
        # OpenAI masks encode the edit region via alpha (transparent = edit); the
        # derived diffusers mask must be white there and black where opaque.
        inpaint = _StubPipeline()
        serving = _serving(img2img=_StubPipeline(), inpaint=inpaint)
        request = ImageEditRequest.model_construct(model="m", prompt="fill", n=1, size="16x16", strength=None)

        result = await serving.create_image_edit(_png_bytes(), _rgba_png(transparent=True), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        mask = inpaint.calls[0]["mask_image"]
        assert mask.getpixel((8, 8)) == 255  # transparent center -> edit
        assert mask.getpixel((0, 0)) == 0  # opaque corner -> keep

    async def test_edit_mask_has_soft_edges_when_resized(self):
        # Thresholding before the resize lets interpolation soften the binary
        # edge into a gradient when the mask is scaled to a different size.
        inpaint = _StubPipeline()
        serving = _serving(img2img=_StubPipeline(), inpaint=inpaint)
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="64x64", strength=None)

        await serving.create_image_edit(_rgba_png(16, 16, transparent=True), None, request, _proxy())

        mask = inpaint.calls[0]["mask_image"]
        assert mask.size == (64, 64)
        assert any(0 < v < 255 for v in mask.getdata())  # anti-aliased boundary

    async def test_edit_missing_img2img_pipeline_errors(self):
        serving = _serving(img2img=None)
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="16x16", strength=None)
        result = await serving.create_image_edit(_png_bytes(), None, request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_edit_missing_inpaint_pipeline_errors(self):
        serving = _serving(img2img=_StubPipeline(), inpaint=None)
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="16x16", strength=None)
        result = await serving.create_image_edit(_png_bytes(), _png_bytes(), request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_edit_bad_image_bytes_errors(self):
        serving = _serving(img2img=_StubPipeline())
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="16x16", strength=None)
        result = await serving.create_image_edit(b"not an image", None, request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_edit_truncated_image_errors(self):
        # Headers parse but pixel data is cut off — must surface as an error,
        # not an uncaught OSError from a later convert()/resize().
        full = _png_bytes(64, 64)
        serving = _serving(img2img=_StubPipeline())
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="64x64", strength=None)
        result = await serving.create_image_edit(full[: len(full) // 2], None, request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_edit_invalid_size_errors(self):
        serving = _serving(img2img=_StubPipeline())
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="nonsense", strength=None)
        result = await serving.create_image_edit(_png_bytes(), None, request, _proxy())
        assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
class TestImageVariation:
    async def test_variation_uses_img2img_with_empty_prompt(self):
        img2img = _StubPipeline()
        serving = _serving(img2img=img2img)
        request = ImageVariationRequest.model_construct(model="m", n=3, size="16x16", strength=None)

        result = await serving.create_image_variation(_png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 3
        for obj in result.data:
            assert obj.revised_prompt is None
            _assert_valid_b64_image(obj)
        call = img2img.calls[0]
        assert call["prompt"] == ""
        assert call["strength"] == _DEFAULT_VARIATION_STRENGTH

    async def test_variation_respects_request_strength(self):
        img2img = _StubPipeline()
        serving = _serving(img2img=img2img)
        request = ImageVariationRequest.model_construct(model="m", n=1, size="16x16", strength=0.9)
        await serving.create_image_variation(_png_bytes(), request, _proxy())
        assert img2img.calls[0]["strength"] == 0.9

    async def test_variation_missing_img2img_pipeline_errors(self):
        serving = _serving(img2img=None)
        request = ImageVariationRequest.model_construct(model="m", n=1, size="16x16", strength=None)
        result = await serving.create_image_variation(_png_bytes(), request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_concurrent_inference_is_serialized(self):
        # The GPU lock must serialize pipeline forward passes — two concurrent
        # requests should never overlap inside the (non-thread-safe) pipeline.
        probe = _ConcurrencyProbe()
        serving = _serving(img2img=probe)
        request = ImageVariationRequest.model_construct(model="m", n=1, size="16x16", strength=None)
        await asyncio.gather(
            serving.create_image_variation(_png_bytes(), request, _proxy()),
            serving.create_image_variation(_png_bytes(), request, _proxy()),
        )
        assert probe.max_active == 1


@pytest.mark.asyncio
class TestCancellation:
    @pytest.mark.parametrize("pipeline_type", [_StepEndPipeline, _LegacyCallbackPipeline])
    async def test_a_cancelled_generation_stops_between_steps(self, pipeline_type):
        pipeline = pipeline_type()
        infer = ExclusiveInfer()
        serving = _serving(pipeline=pipeline, infer=infer)
        serving.config = DiffusersConfig(num_inference_steps=100)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        task = asyncio.ensure_future(serving.create_image_generation(request, _proxy()))
        await asyncio.to_thread(pipeline.started.wait, 5)
        task.cancel()
        await wait_for_turn(infer)
        assert pipeline.steps < 100


class TestProtocolResponseFormat:
    @staticmethod
    def _upload():
        from fastapi import UploadFile

        return UploadFile(file=io.BytesIO(b"x"), filename="x.png")

    def test_edit_rejects_non_b64_response_format(self):
        with pytest.raises(ValidationError) as exc:
            ImageEditRequest(image=self._upload(), prompt="p", model="m", response_format="url")  # type: ignore[arg-type]
        assert any(e["loc"] == ("response_format",) for e in exc.value.errors())

    def test_variation_rejects_non_b64_response_format(self):
        with pytest.raises(ValidationError) as exc:
            ImageVariationRequest(image=self._upload(), model="m", response_format="url")  # type: ignore[arg-type]
        assert any(e["loc"] == ("response_format",) for e in exc.value.errors())

    def test_strength_bounds_enforced(self):
        with pytest.raises(ValidationError) as exc:
            ImageEditRequest(image=self._upload(), prompt="p", model="m", strength=2.0)
        assert any(e["loc"] == ("strength",) for e in exc.value.errors())
