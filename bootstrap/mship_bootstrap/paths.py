"""Filesystem layout owned by the bootstrapper, rooted at MSHIP_HOME.

MSHIP_HOME must stay node-local; MSHIP_CACHE_DIR is engine-owned, may point at
shared storage, and is not read here.
"""

from __future__ import annotations

import os

_DEFAULT_HOME = "~/.modelship"


def home() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("MSHIP_HOME") or _DEFAULT_HOME))


def bin_dir() -> str:
    return os.path.join(home(), "bin")


def env_dir(variant: str) -> str:
    return os.path.join(home(), "envs", variant)


def venv_dir(variant: str) -> str:
    return os.path.join(env_dir(variant), ".venv")


def venv_python(variant: str) -> str:
    return os.path.join(venv_dir(variant), "bin", "python")


def env_file() -> str:
    """Where `mship bootstrap` records the variant it built."""
    return os.path.join(home(), "env")


def pins_copy(variant: str) -> str:
    """Copy of the pins that built this environment; written only after a
    successful sync."""
    return os.path.join(env_dir(variant), "pins.txt")


def pins_staging(variant: str) -> str:
    return os.path.join(env_dir(variant), "pins.txt.partial")


def builds_dir(variant: str) -> str:
    """Per-variant: a ggml backend mutates the extracted tree and the wrapper bakes
    in a venv-specific library path."""
    return os.path.join(home(), "builds", variant)
