from __future__ import annotations

import os

VARIANTS = ("cuda", "cpu", "vulkan", "metal", "thin")


class VariantError(RuntimeError):
    pass


def current_variant() -> str:
    variant = os.environ.get("MSHIP_VARIANT", "").strip()
    if variant not in VARIANTS:
        found = f"is {variant!r}" if variant else "is not set"
        raise VariantError(
            f"MSHIP_VARIANT {found}; expected one of {', '.join(VARIANTS)}. "
            "The mship bootstrapper sets it; set it yourself to run the engine from source."
        )
    return variant
