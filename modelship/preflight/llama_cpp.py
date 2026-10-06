from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from modelship.infer.infer_config import LlamaServerConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile, ModelNotSizedError, gpu_share_bytes

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

# Share of the node's RAM a CPU deploy's budget leaves free, on top of `_FIT_MARGIN_MIB`.
_RAM_RESERVE_FRACTION = 0.05

# `llama fit-params`' one stdout line: `-c N -ngl M [-ts a,b,...]`.
_FIT_ARGS_RE = re.compile(r"^-c\s+(-?\d+)\s+-ngl\s+(-?\d+)(?:\s+-ts\s+([\d.,]+))?\s*$")

# `-fitp on` prints one `<device> <model> <context> <compute>` row in MiB per device.
_FIT_HOST_ROW_RE = re.compile(r"^Host\s+(\d+)\s+\d+\s+\d+\s*$", re.MULTILINE)

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

        if pinned_ctx and pinned_ngl and pinned_ts:
            logger.info("preflight '%s': n_ctx, n_gpu_layers and tensor_split all pinned — nothing to fit", config.name)
            return threads_rec

        args = [
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
        if config.num_gpus == 0:
            args += ["-dev", "none"]
        if pinned_ctx:
            args += ["-c", str(server_config.n_ctx * server_config.parallel)]
        if pinned_ngl:
            args += ["-ngl", str(server_config.n_gpu_layers)]
        if pinned_ts and server_config.tensor_split:
            args += ["-ts", ",".join(str(v) for v in server_config.tensor_split)]

        budget_mib = _cpu_ram_budget_mib(hw) if config.num_gpus == 0 else None
        try:
            if budget_mib is None:
                margin_mib = _fit_margin_mib(config, hw)
            else:
                margin_mib = _cpu_fit_margin_mib(config, args, model_path, budget_mib)
            args += ["-fitc", str(_MIN_NCTX), "-fitt", str(margin_mib)]
            rec = _run_fit(config, args, server_config.parallel)
        except _FitError as e:
            if budget_mib is None:
                logger.warning("preflight '%s': %s", config.name, e)
                return threads_rec
            raise ModelNotSizedError(
                f"could not be sized for this node, where {budget_mib} MiB of RAM is available to it: {e}. "
                "Pass --no-preflight to load it without this check."
            ) from None
        return {**threads_rec, **rec}


def _fit_margin_mib(config: ModelshipModelConfig, hw: HardwareProfile) -> int:
    """`-fitt` reads from *free* VRAM; convert a fractional num_gpus' declared
    share of *total* capacity into a margin against that free figure."""
    if not (0 < config.num_gpus < 1) or not hw.gpus:
        return _FIT_MARGIN_MIB
    gpu = hw.gpus[0] if len(hw.gpus) == 1 else min(hw.gpus, key=lambda g: g.available_bytes)
    share_mib = gpu_share_bytes(config, gpu) / 1024**2
    free_mib = gpu.available_bytes / 1024**2
    return max(_FIT_MARGIN_MIB, int(free_mib - share_mib + _FIT_MARGIN_MIB))


def _cpu_ram_budget_mib(hw: HardwareProfile) -> int | None:
    """Free RAM less the reserve, in MiB. None when the RAM probe read nothing."""
    free_bytes = hw.sizing_ram_bytes
    if not free_bytes:
        return None
    reserve_bytes = hw.ram_bytes * _RAM_RESERVE_FRACTION + _FIT_MARGIN_MIB * 1024**2
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
    match = _FIT_HOST_ROW_RE.search(_invoke([*args, "-fitp", "on"]).stdout)
    if match is None:
        raise _FitError("could not read fit-params memory estimate")
    return int(match[1])


def _invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs fit-params. `_FitError` when it can't be run, times out or exits non-zero."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_FIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise _FitError(f"fit-params invocation failed: {e}") from None
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        detail = " ".join(_LOG_PREFIX_RE.sub("", lines[-1].strip()).split()).rstrip(".") if lines else ""
        raise _FitError(f"fit-params exited {result.returncode}" + (f": {detail}" if detail else ""))
    return result


def _run_fit(config: ModelshipModelConfig, args: list[str], parallel: int) -> dict[str, Any]:
    stdout = _invoke(args).stdout.strip()
    line = stdout.splitlines()[-1] if stdout else ""
    match = _FIT_ARGS_RE.match(line)
    if match is None:
        raise _FitError(f"could not parse fit-params output: {line!r}")

    ctx_total_raw, ngl_raw, ts_raw = match.groups()
    ctx_total = int(ctx_total_raw)
    ngl = int(ngl_raw)

    rec: dict[str, Any] = {}
    if ctx_total == 0:
        # 0 means "model's own maximum, unconstrained"; round-trips as-is since
        # `_launch` sends `n_ctx * parallel` and llama-server resolves 0 itself.
        rec["n_ctx"] = 0
    else:
        per_slot = (ctx_total // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if per_slot < _MIN_NCTX:
            raise _FitError(
                f"fit-params context {ctx_total} across parallel={parallel} yields n_ctx={per_slot} (< {_MIN_NCTX})"
            )
        rec["n_ctx"] = per_slot
    if ngl >= 0:
        rec["n_gpu_layers"] = ngl
    if ts_raw:
        rec["tensor_split"] = [float(v) for v in ts_raw.split(",")]

    logger.info(
        "preflight llama_server '%s': fit-params -> n_ctx=%s n_gpu_layers=%s tensor_split=%s",
        config.name,
        rec.get("n_ctx"),
        rec.get("n_gpu_layers"),
        rec.get("tensor_split"),
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
