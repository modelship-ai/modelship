"""End-to-end coverage for the stable_diffusion_cpp loader: image generation, edits (with and without a mask)
and variations on the CPU and on a GPU share, a dropped request stopping sd-server, concurrent requests, and a
model sd-server can't load."""

import base64
import contextlib
import io
import json
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx
import psutil
import pytest

from tests.conftest import OPENAI_API_BASE, serve_apps
from tests.test_cluster_join import _poll
from tests.test_dropped_request_integration import _replica

# A text model's GGUF: passes the source check, sd-server exits loading it.
_NOT_A_DIFFUSION_MODEL = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf"


def _sd_server(model: str) -> psutil.Process:
    _, pid = _replica(model)
    return next(c for c in psutil.Process(pid).children(recursive=True) if "--listen-port" in c.cmdline())


def _cpu_percent(proc: psutil.Process) -> float:
    return proc.cpu_percent(interval=1.0)


@contextlib.contextmanager
def _held_image_request(body: dict):
    """A generation request on a raw socket, dropped on exit without reading the response."""
    url = urlsplit(f"{OPENAI_API_BASE}/images/generations")
    payload = json.dumps(body).encode()
    head = (
        f"POST {url.path} HTTP/1.1\r\nHost: {url.netloc}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n"
    )
    with socket.create_connection((url.hostname, url.port)) as sock:
        sock.sendall(head.encode() + payload)
        yield


def _image(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64)))


@pytest.mark.integration
@pytest.mark.stable_diffusion_cpp
class TestImageStableDiffusionCpp:
    """One single-file SD2.1 GGUF deployment backs all three endpoints; the
    generated image is reused as input for the edit and variation calls."""

    SIZE = "256x256"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("image-cpu-model")

    @staticmethod
    def _decode(b64: str) -> bytes:
        return base64.b64decode(b64)

    @staticmethod
    def _assert_png(data: bytes, *, expect_size: tuple[int, int] | None = None) -> None:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.verify()
        assert img.format == "PNG"
        if expect_size is not None:
            assert img.size == expect_size

    def test_image_generation(self, client):
        response = client.images.generate(
            model="image-cpu-model", prompt="a red apple on a table", size=self.SIZE, response_format="b64_json"
        )
        assert response.data[0].b64_json
        self._assert_png(self._decode(response.data[0].b64_json), expect_size=(256, 256))

    def test_image_edit(self, client, tmp_path):
        source = client.images.generate(
            model="image-cpu-model", prompt="a plain blue sky", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            edited = client.images.edit(
                model="image-cpu-model",
                image=f,
                prompt="a blue sky with a bright sun",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(256, 256))

    def test_image_edit_with_mask(self, client, tmp_path):
        from PIL import Image

        source = client.images.generate(
            model="image-cpu-model", prompt="a grassy field", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        # White rectangle marks the region to repaint; the rest is preserved.
        mask = Image.new("RGB", (256, 256), (0, 0, 0))
        for x in range(64, 192):
            for y in range(64, 192):
                mask.putpixel((x, y), (255, 255, 255))
        mask_path = tmp_path / "mask.png"
        mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            edited = client.images.edit(
                model="image-cpu-model",
                image=img_f,
                mask=mask_f,
                prompt="a grassy field with a small red flower",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(256, 256))

    def test_a_dropped_request_stops_sd_server(self):
        """Uncancelled, eight 768² images keep sd-server busy for minutes; cancelled, it stops within a step."""
        sd_server = _sd_server("image-cpu-model")
        body = {"model": "image-cpu-model", "prompt": "a busy city street", "size": "768x768", "n": 8}
        with _held_image_request(body):
            assert _poll(lambda: _cpu_percent(sd_server) > 50, deadline_s=30), "the request never reached sd-server"
        assert _poll(lambda: _cpu_percent(sd_server) < 10, deadline_s=45), "sd-server kept generating"

    def test_concurrent_requests_all_complete(self):
        """sd-server queues them; ten images take two of its jobs (eight at most each)."""
        counts = [1, 2, 10]

        def generate(n: int) -> httpx.Response:
            body = {"model": "image-cpu-model", "prompt": "a small boat", "size": "128x128", "n": n}
            return httpx.post(f"{OPENAI_API_BASE}/images/generations", json=body, timeout=600)

        with ThreadPoolExecutor(len(counts)) as pool:
            responses = list(pool.map(generate, counts))
        for n, resp in zip(counts, responses, strict=True):
            assert resp.status_code == 200, resp.text
            assert len(resp.json()["data"]) == n

    def test_output_format_webp(self):
        body = {"model": "image-cpu-model", "prompt": "a red apple", "size": self.SIZE, "output_format": "webp"}
        resp = httpx.post(f"{OPENAI_API_BASE}/images/generations", json=body, timeout=600)
        assert resp.status_code == 200, resp.text
        img = _image(resp.json()["data"][0]["b64_json"])
        assert (img.format, img.size) == ("WEBP", (256, 256))

    def test_image_variation(self, client, tmp_path):
        source = client.images.generate(
            model="image-cpu-model", prompt="a yellow lemon", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            variation = client.images.create_variation(
                model="image-cpu-model", image=f, size=self.SIZE, response_format="b64_json"
            )
        assert variation.data[0].b64_json
        self._assert_png(self._decode(variation.data[0].b64_json), expect_size=(256, 256))


@pytest.mark.integration
@pytest.mark.stable_diffusion_cpp
@pytest.mark.cuda
@pytest.mark.vulkan
@pytest.mark.metal
class TestImageStableDiffusionCppGpu:
    """A quarter of one GPU: preflight's `--max-vram` budget and the pinned GPU backend."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("image-gpu-sdcpp-model")

    def test_image_generation(self, client):
        response = client.images.generate(
            model="image-gpu-sdcpp-model", prompt="a red apple on a table", size="512x512", response_format="b64_json"
        )
        img = _image(response.data[0].b64_json)
        assert (img.format, img.size) == ("PNG", (512, 512))

    def test_image_edit_with_mask(self, client, tmp_path):
        """img2img runs the VAE encoder on the GPU too."""
        from PIL import Image

        source = client.images.generate(
            model="image-gpu-sdcpp-model", prompt="a grassy field", size="512x512", response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(base64.b64decode(source.data[0].b64_json))
        mask = Image.new("RGB", (512, 512), (0, 0, 0))
        mask.paste((255, 255, 255), (128, 128, 384, 384))
        mask_path = tmp_path / "mask.png"
        mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            edited = client.images.edit(
                model="image-gpu-sdcpp-model",
                image=img_f,
                mask=mask_f,
                prompt="a grassy field with a small red flower",
                size="512x512",
                response_format="b64_json",
            )
        img = _image(edited.data[0].b64_json)
        assert (img.format, img.size) == ("PNG", (512, 512))


@pytest.mark.integration
@pytest.mark.stable_diffusion_cpp
class TestAModelSdServerCannotLoad:
    def test_fails_the_deploy_with_sd_servers_exit(self, model_deployer):
        log = model_deployer.run(
            "--name",
            "sd-broken",
            "--model",
            _NOT_A_DIFFUSION_MODEL,
            "--loader",
            "stable_diffusion_cpp",
            "--num-cpus",
            "1",
            log_name="sd-broken",
            expect_code=1,
        )
        assert re.search(r"sd-server for 'sd-broken' exited \(rc=-?\d+\) during startup", log), log
        assert not [name for name in serve_apps() if name.startswith("modelship.sd-broken-")]
