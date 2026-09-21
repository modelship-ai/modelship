"""Engine entry point: `python -m modelship.launcher`, no console script.

Ray-free until it hands off to modelship.driver.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import sys
from typing import TYPE_CHECKING

import yaml

from modelship.deploy.capabilities import LOADER_MODULES
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root

if TYPE_CHECKING:
    from pydantic import ValidationError

    from modelship.utils.config_schema import ModelshipConfig

_REQUIRED_PYTHON = (3, 12, 10)


_COMMANDS = ("start", "join", "deploy", "info")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        print(f"usage: python -m modelship.launcher {{{','.join(_COMMANDS)}}} [args]", file=sys.stderr)
        sys.exit(2)

    command, rest = argv[0], argv[1:]
    if command == "info":
        _cmd_info()
    else:
        _cmd_run(command, rest)


def _cmd_run(command: str, argv: list[str]) -> None:
    from modelship.utils.cli import apply_args_to_env, parse_args

    args = parse_args(command, argv)
    apply_args_to_env(args)
    _guard_python_version()

    if command != "join":
        config = _validate_config(args)
        # Only a start with capacity of its own runs models in this environment.
        if config is not None and command == "start" and not _advertises_no_capacity():
            _check_loader_capabilities({m.loader.value for m in config.models})

    from modelship.driver import run

    run(command, argv)


def _cmd_info() -> None:
    accelerator = detect_accelerator()
    print(f"accelerator: {accelerator}")
    print(f"python: {platform.python_version()}")
    print(f"cache: {resolve_cache_root()}")
    print(f"node cache: {resolve_node_cache_root()}")
    try:
        import ray

        print(f"ray: {ray.__version__}")
    except Exception:
        print("ray: not installed")

    print(f"llama-server: {os.environ.get('MSHIP_LLAMA_SERVER_BIN') or 'unset'}")


def _guard_python_version() -> None:
    if sys.version_info[:3] != _REQUIRED_PYTHON:
        got = ".".join(map(str, sys.version_info[:3]))
        print(f"mship requires Python {'.'.join(map(str, _REQUIRED_PYTHON))} exactly, found {got}.", file=sys.stderr)
        sys.exit(1)


def _advertises_no_capacity() -> bool:
    """A thin coordinator reserves 0 CPUs and 0 GPUs, so every model in its config is
    bound for a node that joins later."""
    reserved = []
    for var in ("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"):
        try:
            reserved.append(float(os.environ[var]))
        except (KeyError, ValueError):
            return False
    return not any(reserved)


def _validate_config(args: argparse.Namespace) -> ModelshipConfig | None:
    """Validate the requested models before the driver imports ray. None when
    the invocation asks for none."""
    from pydantic import ValidationError

    from modelship.deploy.config import resolve_input_models, validate_models

    try:
        raw_models = resolve_input_models(args)
        if raw_models is None:
            return None
        return validate_models(raw_models)
    except ValidationError as e:
        print(f"error: invalid config: {_flagged(e) if args.model else e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        sys.exit(1)


def _flagged(error: ValidationError) -> str:
    """Rewrite pydantic's `models.0.<block>.<field>` locations as
    `--<block>.<field>` so a single-model deploy reports its own flags, not a
    models.yaml shape."""
    return re.sub(
        r"\bmodels\.\d+\.([a-z_]+)(?:\.([a-z_]+))?",
        lambda m: "--" + ".".join(part for part in m.groups() if part).replace("_", "-"),
        str(error),
    )


def _check_loader_capabilities(loaders: set[str]) -> None:
    for loader in loaders:
        module = LOADER_MODULES.get(loader)
        if module and importlib.util.find_spec(module) is None:
            print(
                f"error: models.yaml uses loader: {loader}, but '{module}' isn't installed in this "
                "environment. Use loader: llama_server for GGUF models on this hardware.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
