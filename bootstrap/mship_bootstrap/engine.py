"""Provisioning the engine environment: one per variant, no version in the path.

`uv pip sync` is declarative, so upgrades converge in place and a partial
environment repairs itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources

from . import gates, paths
from .variants import VARIANTS, Variant, engine_requirement, read_recorded

PYTHON_VERSION = "3.12.10"


class EngineError(RuntimeError):
    pass


def read_pins(variant: Variant) -> str:
    try:
        return resources.files(f"{__package__}.pins").joinpath(f"{variant.name}.txt").read_text()
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise EngineError(f"error: no pins file shipped for variant {variant.name!r}") from e


def provision(variant: Variant, uv: str, version: str) -> str:
    """Returns the interpreter to exec. Safe to call on every invocation."""
    env_dir = paths.env_dir(variant.name)
    venv = paths.venv_dir(variant.name)
    os.makedirs(env_dir, exist_ok=True)

    requirements = _write_requirements(variant, version)

    if not os.path.isfile(paths.venv_python(variant.name)):
        _run([uv, "venv", "--python", PYTHON_VERSION, venv], "create the engine environment")

    _run(
        [uv, "pip", "sync", "--python", venv, *variant.index_args, requirements],
        f"install the {variant.name} engine environment",
    )
    # Only now — a failed sync must not leave the stamp is_current reads.
    os.replace(requirements, paths.pins_copy(variant.name))
    return paths.venv_python(variant.name)


def is_current(variant: Variant, version: str) -> bool:
    """Whether this environment was built by this mship, for this variant. pins.txt's
    last line is the engine requirement, carrying both the extras and the version."""
    if not os.path.isfile(paths.venv_python(variant.name)):
        return False
    return _recorded_engine_requirement(variant) == engine_requirement(variant, version)


def _recorded_engine_requirement(variant: Variant) -> str | None:
    try:
        with open(paths.pins_copy(variant.name)) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    return lines[-1].strip() if lines else None


def describe_staleness(variant: Variant, version: str) -> str:
    """Why `is_current` said no, and how to fix it."""
    fix = f"\n\nRun: mship bootstrap --{variant.name}"
    if not os.path.isfile(paths.venv_python(variant.name)):
        others = [n for n in provisioned_variants() if n != variant.name]
        also = f"\nProvisioned: {', '.join(others)}" if others else ""
        return f"error: the {variant.name} environment has not been bootstrapped.{also}{fix}"
    recorded = _recorded_engine_requirement(variant) or ""
    built_for = recorded.partition("==")[2] or "an unknown version"
    return f"error: the {variant.name} environment was built for mship {built_for}, but this is {version}.{fix}"


def _write_requirements(variant: Variant, version: str) -> str:
    """The shipped pins plus the engine line, which `uv export --no-emit-project`
    omits. Staged: promoted to pins.txt once the sync succeeds."""
    body = read_pins(variant)
    target = paths.pins_staging(variant.name)
    with open(target, "w") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
        f.write(f"{engine_requirement(variant, version)}\n")
    return target


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise EngineError(f"error: {cmd[0]} not found") from e
    except subprocess.CalledProcessError as e:
        raise EngineError(f"error: failed to {what} (uv exited {e.returncode})") from e


def provisioned_variants() -> list[str]:
    """Variants with a usable interpreter on disk. Cheap — no tree walk."""
    root = os.path.join(paths.home(), "envs")
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(n for n in names if n in VARIANTS and os.path.isfile(paths.venv_python(n)))


def describe_envs() -> list[tuple[str, str]]:
    """(variant, human-readable size) for every provisioned environment. Sizing
    walks the whole tree, so this stays confined to `mship info`."""
    return [(name, _human_size(_tree_size(paths.env_dir(name)))) for name in provisioned_variants()]


def _tree_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def print_info(version: str) -> None:
    print(f"mship (bootstrapper): {version}")
    print(f"python:               {sys.version.split()[0]} ({sys.executable})")
    print(f"MSHIP_HOME:           {paths.home()}")
    print(f"uv:                   {shutil.which('uv') or os.path.join(paths.bin_dir(), 'uv')}")
    print(f"variant:              {read_recorded(paths.env_file()) or 'none recorded'}")
    envs = describe_envs()
    if not envs:
        print("environments:         none provisioned")
        return
    print("environments:")
    for name, size in envs:
        stale = "" if is_current(VARIANTS[name], version) else "  (stale — re-run bootstrap)"
        print(f"  {name:<6} {size:>8}  {paths.env_dir(name)}{stale}")
    if any(name == "cuda" for name, _ in envs):
        print(f"cuda toolkit:         {'ok' if gates.find_nvcc() else 'missing nvcc'}")
