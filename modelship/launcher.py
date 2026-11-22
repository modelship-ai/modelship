"""Engine entry point: `python -m modelship.launcher`, no console script.

Ray-free until it hands off to modelship.driver.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import platform
import sys
from typing import TYPE_CHECKING

import yaml

from modelship.deploy.capabilities import node_labels, operator_node_labels, runs_on
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
            operator_node_labels()
        except (VariantError, ValueError) as e:
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
    from modelship.utils.ray_auth import resolve_ray_auth_env

    os.environ.setdefault("RAY_num_grpc_internal_threads", "2")
    # Before `import ray`: RAY_AUTH_MODE latches at import.
    resolve_ray_auth_env()
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
    print(f"sd-server: {os.environ.get('MSHIP_SD_SERVER_BIN') or 'unset'}")
    variant = os.environ.get("MSHIP_VARIANT")
    if variant in ("cuda", "vulkan", "metal"):
        _print_gpus(variant)
    _print_node_labels()
    _print_clusters()


def _print_node_labels() -> None:
    """The labels start and join give this node."""
    try:
        labels = node_labels()
    except VariantError:
        return
    except ValueError as e:
        print(f"node labels: {e}")
        return
    print(f"node labels: {_format_labels(labels)}")


def _print_clusters() -> None:
    """Each cluster a node on this machine belongs to, with its live nodes."""
    try:
        import ray
        from ray._private.services import find_gcs_addresses
    except ImportError:
        return
    addresses = sorted(find_gcs_addresses())
    if not addresses:
        print("cluster: no node running on this machine")
    # Seconds ray.init retries the GCS, once a second, logging each failure to stderr.
    os.environ.setdefault("RAY_gcs_server_port_wait_time_s", "5")
    for address in addresses:
        try:
            with _stderr_silenced():
                ray.init(address=address, logging_level=logging.ERROR, log_to_driver=False)
            nodes = [node for node in ray.nodes() if node["Alive"]]
        except Exception as e:
            print(f"cluster {address}: can't connect: {_connect_error(e)}")
            continue
        finally:
            ray.shutdown()
        print(f"cluster {address}: {len(nodes)} node(s)")
        for node in sorted(nodes, key=lambda n: "node:__internal_head__" not in n["Resources"]):
            print(f"  {_describe_node(node)}")


@contextlib.contextmanager
def _stderr_silenced():
    """Points fd 2 at /dev/null: Ray's C++ client logs there directly."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _connect_error(error: Exception) -> str:
    chain: list[BaseException] = []
    cause: BaseException | None = error
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    if any("InvalidAuthToken" in str(e) or type(e).__name__ == "AuthenticationError" for e in chain):
        return "it requires its Ray auth token: set MSHIP_RAY_AUTH_TOKEN."
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


def _describe_node(node: dict) -> str:
    resources = node["Resources"]
    head = " (head)" if "node:__internal_head__" in resources else ""
    labels = {k: v for k, v in sorted(node["Labels"].items()) if k != "ray.io/node-id"}
    return (
        f"{node['NodeManagerHostname']}{head}: {resources.get('CPU', 0):g} CPU(s), {resources.get('GPU', 0):g} GPU(s)"
        f"  {_format_labels(labels)}"
    )


def _format_labels(labels: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in labels.items())


def _print_gpus(variant: str) -> None:
    """count_gpus()'s count, the one start and join advertise, then each GPU."""
    from modelship.preflight.base import count_gpus, nvml_probe

    if variant == "cuda" and (version := nvml_probe().cuda_version) is not None:
        too_old = " (mship needs 13.0 or newer)" if version < 13000 else ""
        print(f"cuda driver: supports CUDA {version // 1000}.{version % 1000 // 10}{too_old}")
    found = count_gpus()
    if not found.count:
        print(f"{variant}: {found.why_none}")
        return
    print(f"{variant}: {found.count} GPU(s)")
    if variant == "vulkan":
        _print_vulkan_devices()
    else:
        _print_gpu_memory(variant)


def _print_gpu_memory(variant: str) -> None:
    from modelship.preflight import detect_gpus

    for gpu in detect_gpus():
        size = f"{gpu.total_bytes / 1024**3:.2f} GiB, {gpu.available_bytes / 1024**3:.2f} GiB free"
        print(f"{variant} GPU {gpu.index}: {gpu.name} ({size})")


def _print_vulkan_devices() -> None:
    from modelship.preflight.base import shared_gpu_free_bytes
    from modelship.utils import vulkan_probe

    available_ram = shared_gpu_free_bytes()
    for slot, device in enumerate(vulkan_probe.probe().devices):
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
    """Validate the requested models and pipelines before the driver imports ray.
    None when the invocation asks for no models."""
    from pydantic import ValidationError

    from modelship.deploy.config import resolve_input_models, resolve_input_pipelines, validate_models

    try:
        raw_models = resolve_input_models(args)
        raw_pipelines = resolve_input_pipelines(args)
        if raw_models is None and raw_pipelines is None:
            return None
        config = validate_models(raw_models or [], raw_pipelines)
    except ValidationError as e:
        print(f"error: invalid config: {_flagged(e, args) if _from_cli(args) else e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        sys.exit(1)
    if config.pipelines:
        names = ", ".join(pipeline.name for pipeline in config.pipelines)
        print(f"error: pipelines can't be deployed yet ({names}); remove them to deploy the models.", file=sys.stderr)
        sys.exit(1)
    return config if raw_models is not None else None


def _from_cli(args: argparse.Namespace) -> bool:
    return getattr(args, "models", None) is not None or getattr(args, "pipelines", None) is not None


def _flagged(error: ValidationError, args: argparse.Namespace) -> str:
    """Pydantic's `models.<i>.<path>` and `pipelines.<i>.<path>` locations as the
    `--model.<path>`/`--pipeline.<path>` flags a CLI deploy passed, not a models.yaml shape."""
    from modelship.utils.model_flags import flag_locations

    counts = {"models": len(args.models or []), "pipelines": len(args.pipelines or [])}
    return flag_locations(str(error), counts)


def _check_capabilities(models: list[ModelshipModelConfig]) -> None:
    variant = current_variant()
    for model in models:
        if not runs_on(model.loader.value, model.num_gpus, variant, platform.system()):
            sys.exit(
                f"error: model '{model.name}' uses loader: {model.loader.value} with num_gpus: {model.num_gpus:g}, "
                f"which a {variant} node doesn't run."
            )


if __name__ == "__main__":
    main()
