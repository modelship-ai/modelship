"""Entry point. `bootstrap` provisions a variant's environment; everything else
resolves one, checks it is current, and execs the engine inside it."""

from __future__ import annotations

import os
import sys

from . import __version__, engine, gates, llama_cpp, paths, uv_binary, variants
from .variants import VariantError, split_variant_flag
from .variants import resolve as resolve_variant

_COMMANDS = ("bootstrap", "start", "join", "deploy", "info")

_USAGE = """usage: mship {bootstrap,start,join,deploy,info} [--cuda|--cpu|--metal|--thin] [args]

  bootstrap   install the engine environment for a variant (run once)
  start       start a cluster on this machine and serve models; stays running
  join        add this machine to a running cluster as a worker; stays running
  deploy      change the models of the cluster running on this machine, then exit
  info        report bootstrapper state, or the engine's own report with a variant
"""


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in _COMMANDS:
        sys.stderr.write(_USAGE)
        sys.exit(2)

    command, rest = argv[0], argv[1:]
    try:
        flag, rest = split_variant_flag(rest)
    except VariantError as e:
        sys.exit(str(e))

    # Answerable before any variant exists.
    if command == "info" and flag is None and not (os.environ.get("MSHIP_VARIANT") or "").strip():
        engine.print_info(__version__)
        return

    try:
        variant = resolve_variant(flag, recorded=variants.read_recorded(paths.env_file()))
        gates.check_platform()
        if command == "bootstrap":
            _bootstrap(variant, rest)
            return
        gates.check_hardware(variant.requires_accelerator)
        if not engine.is_current(variant, __version__):
            sys.exit(engine.describe_staleness(variant, __version__))
    except (VariantError, gates.GateError, uv_binary.UvError, engine.EngineError) as e:
        sys.exit(str(e))

    python = paths.venv_python(variant.name)
    env = _engine_env(variant)
    os.execve(python, [python, "-m", "modelship.launcher", command, *rest], env)


def _bootstrap(variant, rest: list[str]) -> None:
    # Nothing here is forwarded to the engine.
    if rest:
        sys.exit(f"error: mship bootstrap takes no arguments besides the variant, got {' '.join(rest)}")

    gates.check_toolchain(variant.name)

    uv = uv_binary.ensure_uv()
    engine.provision(variant, uv, __version__)
    if variant.serves_models:
        llama_cpp.provision(variant)
    variants.write_recorded(paths.env_file(), variant.name)

    print(f"\nmship {__version__} bootstrapped for --{variant.name} in {paths.env_dir(variant.name)}")
    print(f"Recorded in {paths.env_file()}; `mship start` now needs no variant flag.")


def _engine_env(variant) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MSHIP_CACHE_DIR", os.path.join(paths.home(), "cache"))
    env.setdefault("MSHIP_NODE_CACHE_DIR", os.path.join(paths.home(), "node-cache"))

    if variant.serves_models and (wrapper := llama_cpp.locate(variant)):
        env["MSHIP_LLAMA_SERVER_BIN"] = wrapper
        if variant.name == "cuda":
            llama_cpp.warn_if_no_cuda_device(wrapper)

    if variant.name == "thin":
        # Same contract as the thin image.
        env.setdefault("MSHIP_NODE_NUM_CPUS", "0")
        env.setdefault("MSHIP_NODE_NUM_GPUS", "0")

    return env
