"""Loader-agnostic helpers shared by the image serving adapters (`diffusers`,
`stable_diffusion_cpp`). These deal only in PIL images and the OpenAI images
protocol — no backend (diffusers / sd.cpp) imports — so both loaders reuse them
without pulling in each other's dependencies."""

import base64
import io
import time

from PIL import Image

from modelship.openai.protocol import ImageGenerationResponse, ImageObject

_PIL_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


def build_response(images: list, revised_prompt: str | None, output_format: str = "png") -> ImageGenerationResponse:
    fmt = _PIL_FORMAT[output_format]
    data = []
    for img in images:
        if fmt == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data.append(ImageObject(b64_json=b64, revised_prompt=revised_prompt))
    return ImageGenerationResponse(created=int(time.time()), data=data)


def parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format '{size}', expected WxH (e.g. '512x512')")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive, got {w}x{h}")
    if w % 8 != 0 or h % 8 != 0:
        raise ValueError(f"Width and height must be multiples of 8, got {w}x{h}")
    return w, h


def decode_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        # Image.open is lazy (headers only); force decoding now so truncated /
        # corrupt pixel data raises here and is wrapped, rather than later
        # during convert()/resize() where it would escape as an OSError.
        img.load()
        return img
    except Exception as e:
        raise ValueError(f"Could not decode input image: {e}") from e


def load_image(data: bytes, width: int, height: int, mode: str = "RGB") -> Image.Image:
    return decode_image(data).convert(mode).resize((width, height))


def load_mask(data: bytes, width: int, height: int) -> Image.Image:
    """Load an uploaded edit mask into an inpaint mask (white = edit region).
    Per the OpenAI spec a mask encodes the edit region in its alpha channel
    (transparent = edit), so prefer that; fall back to luminance for an
    opaque / grayscale mask (white = edit)."""
    img = decode_image(data)
    mask = alpha_mask(img, width, height)
    if mask is not None:
        return mask
    return img.convert("L").resize((width, height))


def alpha_mask(img: Image.Image, width: int, height: int) -> Image.Image | None:
    """Derive an inpaint mask from an already-decoded image's alpha channel,
    per the OpenAI edits spec: when no separate mask is supplied, transparent
    areas of the image mark the region to edit. Returns a mask (white = edit,
    black = keep) sized to (width, height), or None when the image has no alpha
    channel or is fully opaque (so the caller falls back to plain img2img)."""
    if "A" not in img.mode and "transparency" not in img.info:
        return None
    alpha = img.convert("RGBA").getchannel("A")
    extrema = alpha.getextrema()  # single-band "L" -> (min, max)
    if not extrema or not isinstance(extrema[0], int | float) or extrema[0] >= 255:
        return None  # empty band or fully opaque — nothing marked for editing
    # Transparent (alpha 0) -> white (repaint); opaque -> black (keep). Threshold
    # at the source resolution, then resize: the interpolation softens the
    # binary edge into a gradient, which blends better in inpainting than a hard,
    # aliased boundary.
    lut = [255 if a < 128 else 0 for a in range(256)]
    return alpha.point(lut).resize((width, height))
