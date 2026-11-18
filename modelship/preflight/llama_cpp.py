from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
from typing import Any

from modelship.infer.infer_config import LlamaServerConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight.base import (
    HardwareProfile,
    ModelNotSizedError,
    gpu_share_bytes,
    ram_reserve_bytes,
    shared_gpu_free_bytes,
)
from modelship.utils.variant import current_variant

logger = get_logger("preflight.llama_cpp")

# n_ctx alignment; llama.cpp has no hard requirement, powers of 256 are
# convention.
_NCTX_ALIGNMENT = 256

# A fit below this n_ctx per slot is not used. Doubles as fit-params' own
# `-fitc` floor, so it never solves below what we'd accept.
_MIN_NCTX = 512

_FIT_TIMEOUT_S = 30

# Per-device MiB left free by `-fitt`. Broadcasts to every device on a whole-GPU
# deploy; a fractional deploy and a CPU deploy each compute their own margin.
_FIT_MARGIN_MIB = 1024

# The flags of `llama fit-params`' one stdout line, e.g. `-c N -ngl M -ts a,b -ot "pattern=CPU,..."`.
_FIT_FLAGS = {"-c", "-ngl", "-ts", "-ot"}

# `-fitp on` prints one `<device> <model> <context> <compute>` row in MiB per device, `Host` for RAM.
_FIT_ROW_RE = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)

# llama.cpp's split-file naming: `<prefix>-00001-of-00003.gguf`.
_SPLIT_GGUF_RE = re.compile(r"^(.*)-\d{5}-of-(\d{5})\.gguf$")

# The `<elapsed> <level> ` prefix llama.cpp puts on each stderr line.
_LOG_PREFIX_RE = re.compile(r"^[\d.]+\s+[A-Z]\s+")


class _FitError(Exception):
    """fit-params gave no usable answer; the message says why."""


class LlamaServerPreflight:
    """Sizes the `llama_server` loader's launch args via `llama fit-params`,
    which builds the real KV cache and compute buffers without loading weights."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Thread alignment is independent of context/offload sizing — recommend
        # it even when the fit below declines.
        threads_rec = _recommend_threads(config)

        model_path = config._resolved_path
        if not model_path or not model_path.endswith(".gguf") or not os.path.isfile(model_path):
            logger.info("preflight '%s': skipping — resolved path is not a GGUF file: %s", config.name, model_path)
            return threads_rec

        binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
        if not binary:
            logger.info("preflight '%s': skipping — MSHIP_LLAMA_SERVER_BIN not set", config.name)
            return threads_rec
        if not os.path.isfile(binary):
            logger.info("preflight '%s': skipping — MSHIP_LLAMA_SERVER_BIN=%s does not exist", config.name, binary)
            return threads_rec

        server_config = config.llama_server_config or LlamaServerConfig()
        fields_set = server_config.model_fields_set
        pinned_ctx = "n_ctx" in fields_set
        pinned_ngl = "n_gpu_layers" in fields_set
        pinned_ts = "tensor_split" in fields_set

        budget_mib = _cpu_ram_budget_mib(hw) if config.num_gpus == 0 else None
        # With a RAM budget the fit still runs: it is the check that the set n_ctx fits.
        all_pinned = pinned_ctx and pinned_ngl and pinned_ts
        if all_pinned and budget_mib is None and config.num_gpus == 0:
            logger.info("preflight '%s': n_ctx, n_gpu_layers and tensor_split all pinned — nothing to fit", config.name)
            return threads_rec

        base_args = [
            binary,
            "fit-params",
            "-m",
            model_path,
            "--parallel",
            str(server_config.parallel),
            "-b",
            str(server_config.n_batch),
            "-ub",
            str(server_config.ubatch_size),
            "-fa",
            server_config.flash_attn,
            "-ctk",
            server_config.cache_type_k,
            "-ctv",
            server_config.cache_type_v,
        ]
        base_args += device_args(config.num_gpus)
        args = list(base_args)
        if pinned_ctx:
            args += ["-c", str(server_config.n_ctx * server_config.parallel)]
        if pinned_ngl:
            args += ["-ngl", str(server_config.n_gpu_layers)]
        if pinned_ts and server_config.tensor_split:
            args += ["-ts", ",".join(str(v) for v in server_config.tensor_split)]
        if server_config.override_tensor and config.num_gpus > 0:
            args += ["-ot", ",".join(server_config.override_tensor)]

        rec: dict[str, Any] = {}
        if all_pinned and config.num_gpus > 0:
            logger.info("preflight '%s': n_ctx, n_gpu_layers and tensor_split all pinned — nothing to fit", config.name)
        else:
            try:
                if budget_mib is None:
                    margin_mib = _fit_margin_mib(config, hw)
                else:
                    margin_mib = _cpu_fit_margin_mib(config, args, model_path, budget_mib)
                rec = _run_fit(
                    config, [*args, "-fitc", str(_MIN_NCTX), "-fitt", str(margin_mib)], server_config.parallel
                )
            except _FitError as e:
                if budget_mib is None:
                    logger.warning("preflight '%s': %s", config.name, e)
                    return threads_rec
                raise ModelNotSizedError(
                    f"could not be sized for this node, where {budget_mib} MiB of RAM is available to it: {e}. "
                    "Give the node more free memory or lower n_ctx on its models, "
                    "or pass --no-preflight to load it without this check."
                ) from None

        if config.num_gpus > 0:
            launch = _launch_flags(server_config.model_copy(update=rec), config.num_gpus)
            _check_ram_outside_the_gpu(config, hw, [*base_args, *launch], model_path)
        return {**threads_rec, **rec}


def _fit_margin_mib(config: ModelshipModelConfig, hw: HardwareProfile) -> int:
    if not hw.gpus:
        return _FIT_MARGIN_MIB
    gpu = hw.gpus[0] if len(hw.gpus) == 1 else min(hw.gpus, key=lambda g: g.available_bytes)
    budget = gpu.available_bytes
    if 0 < config.num_gpus < 1:
        budget = min(budget, int(gpu_share_bytes(config, gpu)))
    return max(_FIT_MARGIN_MIB, int((gpu.engine_free - budget) / 1024**2 + _FIT_MARGIN_MIB))


# The dlopen'd backends: naming one fails the launch when it didn't load. Metal is linked in.
_DEVICE_PREFIXES = {"cuda": "CUDA", "vulkan": "Vulkan"}


def device_args(num_gpus: float) -> list[str]:
    if num_gpus <= 0:
        return ["--device", "none"]
    if prefix := _DEVICE_PREFIXES.get(current_variant()):
        return ["--device", ",".join(f"{prefix}{i}" for i in range(math.ceil(num_gpus)))]
    return []


def _cpu_ram_budget_mib(hw: HardwareProfile) -> int | None:
    """Free RAM less the reserve, in MiB. None when the RAM probe read nothing."""
    free_bytes = hw.sizing_ram_bytes
    if not free_bytes:
        return None
    reserve_bytes = ram_reserve_bytes(hw.ram_bytes)
    return max(0, int((free_bytes - reserve_bytes) / 1024**2))


def _cpu_fit_margin_mib(config: ModelshipModelConfig, args: list[str], model_path: str, budget_mib: int) -> int:
    """`-fitt` for a CPU deploy. fit-params fits against all physical RAM, so the
    margin is the RAM outside `budget_mib` plus the weights its estimate leaves out."""
    uncounted_mib = max(0, _weights_mib(model_path) - _estimated_host_weights_mib(args))
    logger.info(
        "preflight llama_server '%s': RAM budget %d MiB, %d MiB of weights outside the fit-params estimate",
        config.name,
        budget_mib,
        uncounted_mib,
    )
    return max(_FIT_MARGIN_MIB, _physical_ram_mib() - budget_mib + uncounted_mib)


def _physical_ram_mib() -> int:
    """The figure llama.cpp's CPU device reports as free: all physical RAM."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 1024**2
    except (ValueError, OSError):
        return 0


def _weights_mib(model_path: str) -> int:
    """Size on disk in MiB, summed over every part of a split GGUF."""
    match = _SPLIT_GGUF_RE.match(model_path)
    if match is None:
        paths = [model_path]
    else:
        prefix, count = match.groups()
        paths = [f"{prefix}-{part:05d}-of-{count}.gguf" for part in range(1, int(count) + 1)]
    return sum(os.path.getsize(path) for path in paths if os.path.isfile(path)) // 1024**2


def _estimated_host_weights_mib(args: list[str]) -> int:
    """Host weights in fit-params' own memory estimate, in MiB."""
    return _estimate_mib(args)["Host"][0]


def _estimate_mib(args: list[str]) -> dict[str, tuple[int, int, int]]:
    """fit-params' memory estimate for exactly `args`: device → (model, context, compute) MiB, `Host` for RAM."""
    rows = _FIT_ROW_RE.findall(_invoke([*args, "-fit", "off", "-fitp", "on"]).stdout)
    estimate = {device: (int(model), int(context), int(compute)) for device, model, context, compute in rows}
    if "Host" not in estimate:
        raise _FitError("could not read fit-params memory estimate")
    return estimate


def _launch_flags(server_config: LlamaServerConfig, num_gpus: float) -> list[str]:
    """The sizing flags llama_server_infer launches a GPU deploy with."""
    flags = ["-c", str(server_config.n_ctx * server_config.parallel), "-ngl", str(server_config.n_gpu_layers)]
    if server_config.tensor_split:
        flags += ["-ts", ",".join(str(v) for v in server_config.tensor_split)]
    if server_config.override_tensor and num_gpus > 0:
        flags += ["-ot", ",".join(server_config.override_tensor)]
    return flags


def _check_ram_outside_the_gpu(config: ModelshipModelConfig, hw: HardwareProfile, args: list[str], model_path: str):
    """fit-params fits a GPU deploy to its GPUs only; what it leaves in RAM is checked here. A GPU that uses
    system RAM takes its part from the same free memory, which a container's limit doesn't cover."""
    budget_mib = _cpu_ram_budget_mib(hw)
    if budget_mib is None:
        logger.warning("preflight '%s': free RAM unreadable; the part left in RAM is not checked", config.name)
        return
    try:
        estimate = _estimate_mib(args)
    except _FitError as e:
        logger.warning("preflight '%s': the part left in RAM is not checked: %s", config.name, e)
        return
    uncounted_mib = max(0, _weights_mib(model_path) - sum(model for model, _, _ in estimate.values()))
    ram_mib = sum(estimate["Host"]) + uncounted_mib
    gpu_mib = sum(sum(row) for device, row in estimate.items() if device != "Host")
    logger.info(
        "preflight llama_server '%s': %d MiB in RAM (%d MiB of weights outside the estimate), %d MiB on the GPU; "
        "%d MiB of RAM available",
        config.name,
        ram_mib,
        uncounted_mib,
        gpu_mib,
        budget_mib,
    )
    if ram_mib > budget_mib:
        raise ModelNotSizedError(
            f"needs {ram_mib} MiB of RAM for the part that doesn't fit on its GPU, and {budget_mib} MiB is "
            "available to it. Give it more of the GPU (num_gpus), use a smaller quant or free RAM on the node, "
            "or pass --no-preflight to load it without this check."
        )
    if any(gpu.shared_memory for gpu in hw.gpus) and (free := shared_gpu_free_bytes()) is not None:
        shared_mib = free // 1024**2
        if ram_mib + gpu_mib > shared_mib:
            raise ModelNotSizedError(
                f"needs {ram_mib + gpu_mib} MiB of system RAM, which its GPU shares, and {shared_mib} MiB is "
                "available. Use a smaller quant or lower n_ctx, free RAM on the node, "
                "or pass --no-preflight to load it without this check."
            )


def _invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs fit-params. `_FitError` when it can't be run, times out or exits non-zero."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_FIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise _FitError(f"fit-params invocation failed: {e}") from None
    if result.returncode != 0:
        lines = [_LOG_PREFIX_RE.sub("", line.strip()) for line in result.stderr.strip().splitlines()]
        # An argument error is followed by usage text.
        line = next((line for line in reversed(lines) if line.startswith("error")), lines[-1] if lines else "")
        detail = " ".join(line.split()).rstrip(".")
        raise _FitError(f"fit-params exited {result.returncode}" + (f": {detail}" if detail else ""))
    return result


def _run_fit(config: ModelshipModelConfig, args: list[str], parallel: int) -> dict[str, Any]:
    stdout = _invoke(args).stdout.strip()
    line = stdout.splitlines()[-1] if stdout else ""
    try:
        tokens = shlex.split(line)
        flags = dict(zip(tokens[::2], tokens[1::2], strict=True))
        if not {"-c", "-ngl"} <= flags.keys() <= _FIT_FLAGS:
            raise ValueError
        ctx_total = int(flags["-c"])
        ngl = int(flags["-ngl"])
        tensor_split = [float(v) for v in flags["-ts"].split(",")] if "-ts" in flags else None
    except ValueError:
        raise _FitError(f"could not parse fit-params output: {line!r}") from None

    per_slot = (ctx_total // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
    if per_slot < _MIN_NCTX:
        raise _FitError(
            f"fit-params context {ctx_total} across parallel={parallel} yields n_ctx={per_slot} (< {_MIN_NCTX})"
        )
    rec: dict[str, Any] = {"n_ctx": per_slot}
    if ngl >= 0:
        rec["n_gpu_layers"] = ngl
    if tensor_split:
        rec["tensor_split"] = tensor_split
    if "-ot" in flags:
        rec["override_tensor"] = flags["-ot"].split(",")

    logger.info(
        "preflight llama_server '%s': fit-params -> n_ctx=%s n_gpu_layers=%s tensor_split=%s override_tensor=%s",
        config.name,
        rec.get("n_ctx"),
        rec.get("n_gpu_layers"),
        rec.get("tensor_split"),
        rec.get("override_tensor"),
    )
    return rec


def _recommend_threads(config: ModelshipModelConfig) -> dict[str, Any]:
    """Aligns llama-server's threads to `config.num_cpus` (>= 1 only; the 0.1
    default isn't a real budget). Declines rather than undercut `parallel`."""
    if config.num_cpus < 1:
        return {}
    threads = int(config.num_cpus)
    parallel = config.llama_server_config.parallel if config.llama_server_config else 1
    if threads < parallel:
        logger.info(
            "preflight '%s': skipping thread alignment — num_cpus=%d would undercut parallel=%d slots",
            config.name,
            threads,
            parallel,
        )
        return {}
    logger.info("preflight '%s': aligning llama-server threads to num_cpus=%d", config.name, threads)
    return {"threads": threads}
