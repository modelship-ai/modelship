import asyncio
import base64
import io
import json
import re
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import RawRequestProxy
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
from tests.test_base_infer_run_blocking import ExclusiveInfer


def _png_b64(w: int = 8, h: int = 8, color=(10, 20, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeSdServer:
    """sd-server's job API: a job is `generating` for `polls` polls, then ends with `outcome`."""

    def __init__(self, *, polls: int = 2, outcome: str = "completed", submit_status: int = 202):
        self.polls = polls
        self.outcome = outcome
        self.submit_status = submit_status
        self.submitted: list[dict] = []
        self.cancelled: list[str] = []
        self._seen: dict[str, int] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sdcpp/v1/img_gen":
            if self.submit_status != 202:
                return httpx.Response(self.submit_status, json={"error": "invalid generation parameters"})
            body = json.loads(request.content)
            self.submitted.append(body)
            job_id = f"job_{len(self.submitted)}"
            self._seen[job_id] = 0
            return httpx.Response(202, json={"id": job_id, "kind": "img_gen", "status": "queued"})
        if match := re.fullmatch(r"/sdcpp/v1/jobs/(\w+)/cancel", request.url.path):
            self.cancelled.append(match.group(1))
            return httpx.Response(200, json={"id": match.group(1), "cancel_requested": True})
        if match := re.fullmatch(r"/sdcpp/v1/jobs/(\w+)", request.url.path):
            job_id = match.group(1)
            self._seen[job_id] += 1
            if self._seen[job_id] <= self.polls:
                return httpx.Response(200, json={"id": job_id, "status": "generating", "result": None})
            if self.outcome == "failed":
                error = {"code": "generation_failed", "message": "generate_image returned empty results"}
                return httpx.Response(200, json={"id": job_id, "status": "failed", "result": None, "error": error})
            n = self.submitted[int(job_id.split("_")[1]) - 1]["batch_count"]
            images = [{"index": i, "b64_json": _png_b64()} for i in reversed(range(n))]
            result = {"output_format": "png", "images": images}
            return httpx.Response(200, json={"id": job_id, "status": "completed", "result": result})
        return httpx.Response(404)


def _serving(server: _FakeSdServer, infer: BaseInfer | None = None) -> OpenAIServingImage:
    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler), base_url="http://sd-server")
    return OpenAIServingImage(client, (infer or ExclusiveInfer()).run_cancellable)


def _png_bytes(w: int = 16, h: int = 16) -> bytes:
    return base64.b64decode(_png_b64(w, h, (200, 100, 50)))


def _rgba_png(w: int = 16, h: int = 16, *, transparent: bool) -> bytes:
    img = Image.new("RGBA", (w, h), (200, 100, 50, 255))
    if transparent:
        for x in range(w // 4, w * 3 // 4):
            for y in range(h // 4, h * 3 // 4):
                img.putpixel((x, y), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _proxy() -> RawRequestProxy:
    return RawRequestProxy(None, {})


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    from modelship.infer.stable_diffusion_cpp.openai import serving_image

    monkeypatch.setattr(serving_image, "_POLL_INTERVAL_S", 0.001)


@pytest.mark.asyncio
class TestImageGeneration:
    async def test_generation_submits_a_job_and_returns_its_images_in_order(self):
        server = _FakeSdServer()
        request = ImageGenerationRequest(model="m", prompt="a red bicycle", n=2, size="64x32")

        result = await _serving(server).create_image_generation(request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert [obj.revised_prompt for obj in result.data] == ["a red bicycle"] * 2
        for obj in result.data:
            assert _decode(obj.b64_json).format == "PNG"
        assert server.submitted == [
            {"prompt": "a red bicycle", "width": 64, "height": 32, "batch_count": 2, "seed": -1}
        ]

    async def test_generation_invalid_size_errors_without_a_job(self):
        server = _FakeSdServer()
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="not-a-size")
        result = await _serving(server).create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert server.submitted == []

    async def test_concurrent_requests_are_all_submitted(self):
        """sd-server queues jobs itself; nothing here holds one back for another."""
        server = _FakeSdServer(polls=5)
        serving = _serving(server)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        results = await asyncio.gather(*(serving.create_image_generation(request, _proxy()) for _ in range(4)))
        assert all(isinstance(r, ImageGenerationResponse) for r in results)
        assert len(server.submitted) == 4

    async def test_a_refused_job_returns_sd_servers_message(self):
        server = _FakeSdServer(submit_status=400)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        result = await _serving(server).create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 400
        assert result.error.message == "invalid generation parameters"

    async def test_a_full_queue_is_a_429(self):
        server = _FakeSdServer(submit_status=429)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        result = await _serving(server).create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 429

    async def test_a_failed_job_is_a_500(self):
        server = _FakeSdServer(outcome="failed")
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        result = await _serving(server).create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 500
        assert "generate_image returned empty results" in result.error.message

    async def test_an_unreachable_server_is_a_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://sd-server")
        serving = OpenAIServingImage(client, ExclusiveInfer().run_cancellable)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        result = await serving.create_image_generation(request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 502


class _DisconnectedRawRequest:
    """A watchable proxy whose client is gone from the first poll."""

    request_id = "req-1"
    is_watchable = True
    received_at = 0.0


async def _all_disconnected(request_ids: list[str]) -> list[str]:
    return request_ids


@pytest.mark.asyncio
class TestCancel:
    async def test_a_disconnect_cancels_the_job(self):
        server = _FakeSdServer(polls=10_000)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        with patch.object(BaseInfer, "_poll_disconnected_ids", staticmethod(_all_disconnected)):
            result = await _serving(server).create_image_generation(request, _DisconnectedRawRequest())  # type: ignore[arg-type]

        assert isinstance(result, ErrorResponse)
        assert "disconnect" in result.error.message.lower()
        assert server.cancelled == ["job_1"]

    async def test_a_cancelled_request_cancels_the_job(self):
        server = _FakeSdServer(polls=10_000)
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16")
        task = asyncio.ensure_future(_serving(server).create_image_generation(request, _proxy()))
        while not server.submitted:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert server.cancelled == ["job_1"]


@pytest.mark.asyncio
class TestImageEdit:
    async def test_edit_without_mask_sends_init_image_no_mask(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="add a hat", n=2, size="32x32", strength=None, output_format="png"
        )

        result = await _serving(server).create_image_edit(_png_bytes(), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 2
        body = server.submitted[0]
        assert body["prompt"] == "add a hat"
        assert _decode(body["init_image"]).size == (32, 32)
        assert body["strength"] == _DEFAULT_EDIT_STRENGTH
        assert "mask_image" not in body

    async def test_edit_with_mask_sends_a_one_channel_mask(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="erase", n=1, size="16x16", strength=0.5, output_format="png"
        )

        result = await _serving(server).create_image_edit(_png_bytes(), _png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        body = server.submitted[0]
        assert body["strength"] == 0.5
        assert _decode(body["mask_image"]).mode == "L"

    async def test_edit_transparent_image_derives_mask_from_alpha(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="fill", n=1, size="16x16", strength=None, output_format="png"
        )

        result = await _serving(server).create_image_edit(_rgba_png(transparent=True), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        # Transparent center -> repaint (white); opaque corner -> keep (black).
        mask = _decode(server.submitted[0]["mask_image"])
        assert mask.getpixel((8, 8)) == 255
        assert mask.getpixel((0, 0)) == 0

    async def test_edit_opaque_rgba_no_mask_is_plain_img2img(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="x", n=1, size="16x16", strength=None, output_format="png"
        )

        result = await _serving(server).create_image_edit(_rgba_png(transparent=False), None, request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert "mask_image" not in server.submitted[0]

    async def test_edit_undecodable_image_errors_without_a_job(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="x", n=1, size="16x16", strength=None, output_format="png"
        )
        result = await _serving(server).create_image_edit(b"not an image", None, request, _proxy())
        assert isinstance(result, ErrorResponse)
        assert server.submitted == []

    async def test_edit_invalid_size_errors(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(model="m", prompt="x", n=1, size="bad", strength=None)
        result = await _serving(server).create_image_edit(_png_bytes(), None, request, _proxy())
        assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
class TestImageVariation:
    async def test_variation_uses_empty_prompt_and_default_strength(self):
        server = _FakeSdServer()
        request = ImageVariationRequest.model_construct(
            model="m", n=3, size="16x16", strength=None, output_format="png"
        )

        result = await _serving(server).create_image_variation(_png_bytes(), request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 3
        assert all(obj.revised_prompt is None for obj in result.data)
        body = server.submitted[0]
        assert body["prompt"] == ""
        assert _decode(body["init_image"]).size == (16, 16)
        assert body["strength"] == _DEFAULT_VARIATION_STRENGTH

    async def test_variation_invalid_size_errors(self):
        server = _FakeSdServer()
        request = ImageVariationRequest.model_construct(model="m", n=1, size="bad", strength=None)
        result = await _serving(server).create_image_variation(_png_bytes(), request, _proxy())
        assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
class TestOutputFormat:
    @pytest.mark.parametrize(("output_format", "pil_format"), [("webp", "WEBP"), ("jpeg", "JPEG")])
    async def test_generation_reencodes_sd_servers_png(self, output_format, pil_format):
        server = _FakeSdServer()
        request = ImageGenerationRequest(model="m", prompt="x", n=1, size="16x16", output_format=output_format)

        result = await _serving(server).create_image_generation(request, _proxy())

        assert isinstance(result, ImageGenerationResponse)
        assert _decode(result.data[0].b64_json).format == pil_format

    async def test_edit_honors_output_format(self):
        server = _FakeSdServer()
        request = ImageEditRequest.model_construct(
            model="m", prompt="x", n=1, size="16x16", strength=None, output_format="webp"
        )
        result = await _serving(server).create_image_edit(_png_bytes(), None, request, _proxy())
        assert isinstance(result, ImageGenerationResponse)
        assert _decode(result.data[0].b64_json).format == "WEBP"

    async def test_variation_honors_output_format(self):
        server = _FakeSdServer()
        request = ImageVariationRequest.model_construct(
            model="m", n=1, size="16x16", strength=None, output_format="webp"
        )
        result = await _serving(server).create_image_variation(_png_bytes(), request, _proxy())
        assert isinstance(result, ImageGenerationResponse)
        assert _decode(result.data[0].b64_json).format == "WEBP"
