"""Engine entry point: `python -m modelship.launcher`, no console script.

Ray-free until it hands off to modelship.driver.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from typing import TYPE_CHECKING

import yaml

from modelship.deploy.capabilities import deployment_capability_resources, node_capability_resources
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root
from modelship.utils.variant import VariantError, current_variant

if TYPE_CHECKING:
    from pydantic import ValidationError

    from modelship.utils.config_schema import ModelshipConfig, ModelshipModelConfig

_REQUIRED_PYTHON = (3, 12, 10)


_COMMANDS = ("start", "join", "deploy", "stop", "info")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        print(f"usage: python -m modelship.launcher {{{','.join(_COMMANDS)}}} [args]", file=sys.stderr)
        sys.exit(2)

    command, rest = argv[0], argv[1:]
    if command == "info":
        _cmd_info()
    elif command == "stop":
        _cmd_stop(rest)
    else:
        _cmd_run(command, rest)


def _cmd_stop(argv: list[str]) -> None:
    from modelship.deploy.local_nodes import stop_local_nodes
    from modelship.logging import configure_logging
    from modelship.utils.cli import parse_args

    args = parse_args("stop", argv)
    configure_logging()
    sys.exit(stop_local_nodes(args.timeout))


def _cmd_run(command: str, argv: list[str]) -> None:
    from modelship.utils.cli import apply_args_to_env, parse_args

    args = parse_args(command, argv)
    apply_args_to_env(args)
    _guard_python_version()
    if command in ("start", "join"):
        try:
            current_variant()
        except VariantError as e:
            sys.exit(f"error: {e}")

    remote = command == "deploy" and bool(os.environ.get("MSHIP_RAY_DASHBOARD_URL"))
    if command in ("start", "deploy") and getattr(args, "cancel", None) is None and not remote:
        config = _validate_config(args)
        # Only a start with capacity of its own runs models in this environment.
        if config is not None and command == "start" and not _advertises_no_capacity():
            _check_capabilities(config.models)

    from modelship.driver import run

    run(command, argv)


def _cmd_info() -> None:
    print(f"variant: {os.environ.get('MSHIP_VARIANT') or 'unset'}")
    print(f"python: {platform.python_version()}")
    print(f"cache: {resolve_cache_root()}")
    print(f"node cache: {resolve_node_cache_root()}")
    try:
        import ray

        print(f"ray: {ray.__version__}")
    except Exception:
        print("ray: not installed")

    print(f"llama-server: {os.environ.get('MSHIP_LLAMA_SERVER_BIN') or 'unset'}")
    variant = os.environ.get("MSHIP_VARIANT")
    if variant == "cuda":
        _print_cuda_devices()
    elif variant == "vulkan":
        _print_vulkan_devices()
    elif variant == "metal":
        _print_metal_devices()


def _print_cuda_devices() -> None:
    from modelship.preflight.base import nvml_probe

    result = nvml_probe()
    if (version := result.cuda_version) is not None:
        too_old = " (mship needs 13.0 or newer)" if version < 13000 else ""
        print(f"cuda driver: supports CUDA {version // 1000}.{version % 1000 // 10}{too_old}")
    if not (visible := result.visible()):
        print(f"cuda: {result.why_none_visible()}")
        return
    for slot, gpu in enumerate(visible):
        size = f"{gpu.total_bytes / 1024**3:.2f} GiB, {gpu.available_bytes / 1024**3:.2f} GiB free"
        print(f"cuda GPU {slot}: {gpu.name} ({size})")


def _print_metal_devices() -> None:
    try:
        from ray._private.accelerators import AppleGPUAcceleratorManager

        count = AppleGPUAcceleratorManager.get_current_node_num_accelerators()
    except Exception as e:
        print(f"metal: Ray's Apple GPU count failed: {e}")
        return
    print(f"metal: Ray finds {count} Apple GPU(s)")


def _print_vulkan_devices() -> None:
    from modelship.preflight.base import detect_available_ram_bytes
    from modelship.utils import vulkan_probe

    result = vulkan_probe.probe()
    if result.error is not None:
        print(f"vulkan: {result.error.message}")
        return
    available_ram = detect_available_ram_bytes()
    for slot, device in enumerate(result.devices):
        try:
            memory = vulkan_probe.device_memory(device, available_ram)
            size = f"{memory.total_bytes / 1024**3:.2f} GiB, {memory.free_bytes / 1024**3:.2f} GiB free"
        except vulkan_probe.MemoryUnreadableError as e:
            size = f"memory unknown: {e}"
        print(f"vulkan GPU {slot}: {device.name} ({device.type}, Vulkan index {device.index}, {size})")


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


def _check_capabilities(models: list[ModelshipModelConfig]) -> None:
    node = node_capability_resources()
    for model in models:
        if not deployment_capability_resources(model.loader.value, model.num_gpus).keys() <= node.keys():
            sys.exit(
                f"error: model '{model.name}' uses loader: {model.loader.value} with num_gpus: {model.num_gpus:g}, "
                f"which a {current_variant()} node doesn't run."
            )


if __name__ == "__main__":
    main()
