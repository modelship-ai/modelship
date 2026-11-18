"""End-to-end coverage for the stable_diffusion_cpp loader: CPU image
generation, edits (with and without a mask), and variations."""

import base64
import io

import pytest


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
        from PIL import Image

        response = client.images.generate(
            model="image-gpu-sdcpp-model", prompt="a red apple on a table", size="512x512", response_format="b64_json"
        )
        img = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        assert (img.format, img.size) == ("PNG", (512, 512))
