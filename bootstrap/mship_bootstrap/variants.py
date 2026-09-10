"""Variant definitions and resolution. Chosen at runtime, not at install time;
no default and no auto-detection."""

from __future__ import annotations

import os
from typing import NamedTuple

from . import paths

_PYPI_INDEX = "https://pypi.org/simple"
_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PYTORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
# Embeds the vLLM version; bump alongside the `vllm` pin in the engine's pyproject.
_VLLM_CPU_INDEX = "https://wheels.vllm.ai/0.28.0/cpu"


def _index_args(*accelerator_indexes: str) -> tuple[str, ...]:
    """PyPI outranks the accelerator indexes, which also serve triton and torchcodec
    at the locked versions but as different artifacts — the lock's hashes are PyPI's,
    so a wrong-index pick fails `uv pip sync` outright. `unsafe-first-match`
    (not `first-index`) so a package PyPI carries at another version still falls
    through; the pinned hashes are what make it safe."""
    indexes = (_PYPI_INDEX, *accelerator_indexes)
    return (*(arg for index in indexes for arg in ("--index", index)), "--index-strategy", "unsafe-first-match")


class Variant(NamedTuple):
    name: str
    extras: tuple[str, ...]
    index_args: tuple[str, ...]
    # None means runs anywhere.
    requires_accelerator: str | None
    serves_models: bool
    summary: str


VARIANTS: dict[str, Variant] = {
    "cuda": Variant(
        name="cuda",
        extras=("cuda",),
        index_args=_index_args(_PYTORCH_CU130_INDEX),
        requires_accelerator="cuda",
        serves_models=True,
        summary="NVIDIA GPU node (vLLM, Diffusers, llama.cpp GPU offload)",
    ),
    "cpu": Variant(
        name="cpu",
        extras=("cpu", "vllm-cpu"),
        index_args=_index_args(_PYTORCH_CPU_INDEX, _VLLM_CPU_INDEX),
        requires_accelerator=None,
        serves_models=True,
        summary="CPU node (vLLM CPU, llama.cpp, whisper.cpp, sherpa-onnx, SD)",
    ),
    "metal": Variant(
        name="metal",
        extras=("metal",),
        index_args=(),
        requires_accelerator="metal",
        serves_models=True,
        summary="Apple Silicon (Metal offload)",
    ),
    "thin": Variant(
        name="thin",
        extras=("thin",),
        index_args=(),
        serves_models=False,
        requires_accelerator=None,
        summary="coordinator/head only — joins capacity from other nodes",
    ),
}

VARIANT_ORDER = ("cuda", "cpu", "metal", "thin")

NO_VARIANT_ERROR = (
    "error: no variant selected\n\n"
    + "".join(f"  mship bootstrap --{n:<8} {VARIANTS[n].summary}\n" for n in VARIANT_ORDER)
    + (
        "\nBootstrapping provisions a pinned Python 3.12.10 environment for the variant\n"
        "(several GB for --cuda). `mship deploy` then needs no variant flag.\n"
    )
)

_RECORDED_HEADER = "# Written by mship bootstrap — do not edit; run 'mship bootstrap --<variant>' instead.\n"


class VariantError(Exception):
    pass


def split_variant_flag(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pull the variant flag out of argv, leaving everything else for the engine."""
    found: list[str] = []
    rest: list[str] = []
    for arg in argv:
        name = arg[2:] if arg.startswith("--") else None
        if name in VARIANTS:
            found.append(name)
        else:
            rest.append(arg)
    if len(found) > 1:
        raise VariantError(f"error: pick one variant, got {', '.join('--' + f for f in sorted(set(found)))}")
    return (found[0] if found else None), rest


def read_recorded(path: str) -> str | None:
    """MSHIP_VARIANT out of an env file, or None if there is none to read. No other
    key is consumed and the file never reaches os.environ."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "MSHIP_VARIANT":
            return value.strip().strip("'\"").strip() or None
    return None


def write_recorded(path: str, name: str) -> None:
    """Last bootstrap wins. Read-only so an editor warns first."""
    tmp = f"{path}.partial"
    with open(tmp, "w") as f:
        f.write(f"{_RECORDED_HEADER}MSHIP_VARIANT={name}\n")
    os.chmod(tmp, 0o444)
    os.replace(tmp, path)


def resolve(flag: str | None, env: dict[str, str] | None = None, recorded: str | None = None) -> Variant:
    """Flag wins over MSHIP_VARIANT; disagreement between those two is an error.
    `recorded` is only a default, so either overrides it silently."""
    env = os.environ if env is None else env
    from_env = (env.get("MSHIP_VARIANT") or "").strip() or None
    recorded = (recorded or "").strip() or None

    if from_env is not None and from_env not in VARIANTS:
        raise VariantError(
            f"error: MSHIP_VARIANT={from_env!r} is not a variant; expected one of {', '.join(VARIANT_ORDER)}"
        )
    if flag is not None and from_env is not None and flag != from_env:
        raise VariantError(f"error: --{flag} conflicts with MSHIP_VARIANT={from_env}")

    chosen = flag or from_env
    if chosen is None and recorded is not None:
        if recorded not in VARIANTS:
            raise VariantError(
                f"error: MSHIP_VARIANT={recorded!r} in {paths.env_file()} is not a variant; "
                f"expected one of {', '.join(VARIANT_ORDER)}"
            )
        chosen = recorded
    if chosen is None:
        raise VariantError(NO_VARIANT_ERROR)
    return VARIANTS[chosen]


def engine_requirement(variant: Variant, version: str) -> str:
    return f"mship-engine[{','.join(variant.extras)}]=={version}"
