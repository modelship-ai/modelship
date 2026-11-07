"""Image generation, edit, and variation schemas."""

from typing import Literal

from fastapi import UploadFile
from pydantic import Field, model_validator

from modelship.openai.protocol.base import OpenAIBaseModel


class ImageGenerationRequest(OpenAIBaseModel):
    model: str = Field(..., description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    n: int = Field(default=1, ge=1, le=10, description="The number of images to generate.")
    size: str = Field(default="512x512", description="The size of the generated images in WxH format.")
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned.",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="The image codec the returned base64 data is encoded in.",
    )


# Type for the OpenAI `image[]` array field: a single upload, a list of uploads
# (the spec allows several), or absent. Declared as an explicit aliased field on
# the request models below rather than relying on extra-field forwarding.
_ImageArray = UploadFile | list[UploadFile] | None


def _coalesce_image_array(image: UploadFile | None, image_array: _ImageArray) -> UploadFile:
    """Fold the OpenAI `image[]` array form onto the singular `image`.

    OpenAI's gpt-image-1 edits/variations accept the upload as an array under
    `image[]` (multiple input images), while the older DALL·E 2 form uses the
    singular `image`. Clients such as Open WebUI send `image[]`. Prefer an
    explicit `image`; otherwise take the first `image[]` entry (the diffusers
    img2img path uses a single image). Raise if neither was supplied."""
    if image is not None:
        return image
    if isinstance(image_array, list):
        image_array = image_array[0] if image_array else None
    if image_array is None:
        raise ValueError("Field required: provide 'image' (or 'image[]')")
    return image_array


class ImageEditRequest(OpenAIBaseModel):
    image: UploadFile | None = Field(default=None, description="The image to edit.")
    # OpenAI gpt-image-1 sends the upload as the array field `image[]`; declare it
    # explicitly (aliased) so FastAPI's form decomposition extracts it without
    # depending on extra-field forwarding. exclude=True keeps the UploadFile out
    # of model_dump() so it never crosses the Ray process boundary; the validator
    # folds it into `image`.
    image_array: _ImageArray = Field(default=None, alias="image[]", exclude=True)
    prompt: str = Field(..., description="A text description of the desired edit.")
    mask: UploadFile | None = Field(
        default=None,
        description="An optional mask; fully transparent areas indicate where the image should be edited (inpainting).",
    )
    model: str = Field(..., description="The model to use for image editing.")
    n: int = Field(default=1, ge=1, le=10, description="The number of edited images to generate.")
    size: str = Field(default="512x512", description="The size of the generated images in WxH format.")
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned.",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="The image codec the returned base64 data is encoded in.",
    )
    # modelship extension (not in OpenAI spec) — controls how far the output may
    # diverge from the input image (0.0 keeps it, 1.0 ignores it). Defaults are
    # applied serving-side. Documented in docs/extensions.md.
    strength: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _accept_image_array(self) -> "ImageEditRequest":
        self.image = _coalesce_image_array(self.image, self.image_array)
        self.image_array = None
        return self


class ImageVariationRequest(OpenAIBaseModel):
    image: UploadFile | None = Field(default=None, description="The image to use as the basis for the variation(s).")
    image_array: _ImageArray = Field(default=None, alias="image[]", exclude=True)
    model: str = Field(..., description="The model to use for image variations.")
    n: int = Field(default=1, ge=1, le=10, description="The number of variations to generate.")
    size: str = Field(default="512x512", description="The size of the generated images in WxH format.")
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned.",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="The image codec the returned base64 data is encoded in.",
    )
    # modelship extension (not in OpenAI spec) — controls how far each variation
    # diverges from the input image (0.0 keeps it, 1.0 ignores it). Defaults are
    # applied serving-side. Documented in docs/extensions.md.
    strength: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _accept_image_array(self) -> "ImageVariationRequest":
        self.image = _coalesce_image_array(self.image, self.image_array)
        self.image_array = None
        return self


class ImageObject(OpenAIBaseModel):
    b64_json: str = Field(..., description="The base64-encoded JSON of the generated image.")
    revised_prompt: str | None = Field(default=None, description="The prompt that was used to generate the image.")


class ImageGenerationResponse(OpenAIBaseModel):
    created: int = Field(..., description="The Unix timestamp of when the response was created.")
    data: list[ImageObject] = Field(..., description="The list of generated images.")


__all__ = [
    "ImageEditRequest",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageObject",
    "ImageVariationRequest",
]
