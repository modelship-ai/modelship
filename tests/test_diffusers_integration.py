"""End-to-end coverage for the diffusers loader: image generation, edits
(with and without a mask), and variations."""

import base64
import io

import pytest


@pytest.mark.integration
@pytest.mark.diffusers
class TestImage:
    """One sd-turbo deployment backs all three endpoints (weight-shared via
    from_pipe); the generated image is reused as input for edit/variation calls."""

    SIZE = "512x512"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("image-model")

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
            model="image-model", prompt="a red apple on a table", size=self.SIZE, response_format="b64_json"
        )
        assert response.data[0].b64_json
        self._assert_png(self._decode(response.data[0].b64_json), expect_size=(512, 512))

    def test_image_edit(self, client, tmp_path):
        source = client.images.generate(
            model="image-model", prompt="a plain blue sky", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            edited = client.images.edit(
                model="image-model",
                image=f,
                prompt="a blue sky with a bright sun",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(512, 512))

    def test_image_edit_with_mask(self, client, tmp_path):
        from PIL import Image

        source = client.images.generate(
            model="image-model", prompt="a grassy field", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        # White rectangle marks the region to repaint; the rest is preserved.
        mask = Image.new("RGB", (512, 512), (0, 0, 0))
        for x in range(128, 384):
            for y in range(128, 384):
                mask.putpixel((x, y), (255, 255, 255))
        mask_path = tmp_path / "mask.png"
        mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            edited = client.images.edit(
                model="image-model",
                image=img_f,
                mask=mask_f,
                prompt="a grassy field with a small red flower",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(512, 512))

    def test_image_variation(self, client, tmp_path):
        source = client.images.generate(
            model="image-model", prompt="a yellow lemon", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            variation = client.images.create_variation(
                model="image-model", image=f, size=self.SIZE, response_format="b64_json"
            )
        assert variation.data[0].b64_json
        self._assert_png(self._decode(variation.data[0].b64_json), expect_size=(512, 512))
