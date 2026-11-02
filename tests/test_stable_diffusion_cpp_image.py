import asyncio
import base64
import io
import threading
import time

import pytest
from PIL import Image

from modelship.infer.infer_config import RawRequestProxy, StableDiffusionCppConfig
from modelship.infer.stable_diffusion_cpp.openai.serving_image import (
    _DEFAULT_EDIT_STRENGTH,
    _DEFAULT_VARIATION_STRENGTH,
    OpenAIServingImage,
)
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
)
from tests.test_base_infer_exclusive import ExclusiveInfer


class _StubSD:
    """Stand-in for a stable-diffusion.cpp handle. Records the kwargs each
    `generate_image` call received and returns `batch_count` solid PIL images."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs.get("batch_count", 1)
        return [Image.new("RGB", (8, 8), (10, 20, 30)) for _ in range(n)]


class _ConcurrencyProbeSD:
    """Records the peak number of overlapping generate_image calls."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate_image(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)  # widen the window so unserialized calls would overlap
        with self._lock:
            self.active -= 1
        n = kwargs.get("batch_count", 1)
        return [Image.new("RGB", (8, 8)) for _ in range(n)]


def _png_bytes(w: int = 16, h: int = 16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _rgba_png(w: int = 16, h: int = 16, *, transparent: bool) -> bytes:
    img = Image.new("RGBA", (w, h), (200, 100, 50, 255))
    if transparent:
        for x in range(w // 4, w * 3 // 4):
            for y in range(h // 4, h * 3 // 4):
                img.putpixel((x, y), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _serving(sd=None) -> OpenAIServingImage:
    return OpenAIServingImage(
        sd=sd if sd is not None else _StubSD(),  # type: ignore[arg-type]
        config=StableDiffusionCppConfig(
            sample_steps=5, cfg_scale=3.0, sample_method="euler", scheduler="karras", vae_tiling=True
        ),
        run_exclusive=ExclusiveInfer().run_exclusive,
    )


def _proxy() -> RawRequestProxy:
    return RawRequestProxy(None, {})


def _assert_valid_b64_image(obj) -> None:
    raw = base64.b64decode(obj.b64_json)
    Image.open(io.BytesIO(raw)).verify()


@pytest.mark.asyncio
class TestImageGeneration:
    async def test_generation_returns_n_images_and_forwards_config(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageGenerationRequest(model="m", prompt="a red bicycle", n=2, size="64x32")

        result = await serving.create_image_generation(request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 2
        for obj in result.data:
            assert obj.revised_prompt == "a red bicycle"
            _assert_valid_b64_image(obj)
        call = sd.calls[0]
        assert call["prompt"] == "a red bicycle"
        assert call["width"] == 64 and call["height"] == 32
        assert call["batch_count"] == 2
        # Per-model config defaults are applied on every call.
        assert call["cfg_scale"] == 3.0
        assert call["sample_steps"] == 5
        assert call["sample_method"] == "euler"
        assert call["scheduler"] == "karras"
        assert call["vae_tiling"] is True
        assert "init_image" not in call  # txt2img

    async def test_generation_invalid_size_errors(self):
        serving = _serving()
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="not-a-size")
        result = await serving.create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)

    async def test_calls_are_serialized(self):
        probe = _ConcurrencyProbeSD()
        serving = _serving(probe)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        await asyncio.gather(*(serving.create_image_generation(request, _proxy()) for _ in range(4)))
        assert probe.max_active == 1


@pytest.mark.asyncio
class TestImageEdit:
    async def test_edit_without_mask_passes_init_image_no_mask(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageEditRequest.model_construct(model="m", prompt="add a hat", n=2, size="32x32", strength=None)

        result = await serving.create_image_edit(_png_bytes(), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 2
        call = sd.calls[0]
        assert call["prompt"] == "add a hat"
        assert call["init_image"] is not None
        assert call["strength"] == _DEFAULT_EDIT_STRENGTH
        assert "mask_image" not in call

    async def test_edit_with_mask_passes_mask_image(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageEditRequest.model_construct(model="m", prompt="erase", n=1, size="16x16", strength=0.5)

        result = await serving.create_image_edit(_png_bytes(), _png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        call = sd.calls[0]
        assert call["strength"] == 0.5
        assert "mask_image" in call

    async def test_edit_transparent_image_derives_mask_from_alpha(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageEditRequest.model_construct(model="m", prompt="fill", n=1, size="16x16", strength=None)

        result = await serving.create_image_edit(_rgba_png(transparent=True), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        call = sd.calls[0]
        assert "mask_image" in call
        # Transparent center -> repaint (white); opaque corner -> keep (black).
        mask = call["mask_image"]
        assert mask.getpixel((8, 8)) == 255
        assert mask.getpixel((0, 0)) == 0

    async def test_edit_opaque_rgba_no_mask_is_plain_img2img(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="16x16", strength=None)

        result = await serving.create_image_edit(_rgba_png(transparent=False), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert "mask_image" not in sd.calls[0]

    async def test_edit_invalid_size_errors(self):
        serving = _serving()
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="bad", strength=None)
        result = await serving.create_image_edit(_png_bytes(), None, request, _proxy())
        assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
class TestImageVariation:
    async def test_variation_uses_empty_prompt_and_default_strength(self):
        sd = _StubSD()
        serving = _serving(sd)
        request = ImageVariationRequest.model_construct(model="m", n=3, size="16x16", strength=None)

        result = await serving.create_image_variation(_png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 3
        for obj in result.data:
            assert obj.revised_prompt is None
            _assert_valid_b64_image(obj)
        call = sd.calls[0]
        assert call["prompt"] == ""
        assert call["init_image"] is not None
        assert call["strength"] == _DEFAULT_VARIATION_STRENGTH

    async def test_variation_invalid_size_errors(self):
        serving = _serving()
        request = ImageVariationRequest.model_construct(model="m", n=1, size="bad", strength=None)
        result = await serving.create_image_variation(_png_bytes(), request, _proxy())
        assert isinstance(result, ErrorResponse)
