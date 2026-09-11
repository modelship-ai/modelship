"""Runs `llama-server` directly against the same models.yaml modelship reads,
bypassing Ray, to A/B against the llama_server loader with an identical
launch command. Mirrors `LlamaServerInfer._launch` — keep both in sync."""

from __future__ import annotations

import os

# Must precede any huggingface_hub import — HF_HOME latches at its import time.
# The same set the driver hands each replica via runtime_env, so both arms
# resolve weights into the mounted cache.
from modelship.deploy.actor_options import build_cache_env_vars

for _key, _value in build_cache_env_vars().items():
    os.environ.setdefault(_key, _value)

import math  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from modelship.infer.infer_config import (  # noqa: E402
    LlamaServerConfig,
    ModelLoader,
    ModelshipConfig,
    ModelUsecase,
)
from modelship.infer.model_resolver import resolve_model_source  # noqa: E402
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight  # noqa: E402

CONFIG_PATH = Path(os.environ.get("MSHIP_CONFIG", "/modelship/config/models.yaml"))


def _pinned(name: str, derived):
    """The harness's replay of what the modelship arm launched with, if set."""
    value = os.environ.get(f"BENCH_PIN_{name}")
    if value is None:
        return derived
    print(f"rawllama pinned {name.lower()}={value} (derived {derived})", flush=True)
    return value


def main() -> int:
    binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
    if not binary or not os.path.isfile(binary):
        print(f"MSHIP_LLAMA_SERVER_BIN must point at a llama-server executable; got {binary!r}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = ModelshipConfig.model_validate(raw)
    llama_models = [m for m in cfg.models if m.loader == ModelLoader.llama_server]
    if len(llama_models) != 1:
        print(
            f"bench expects exactly one llama_server model in {CONFIG_PATH}, got {len(llama_models)}", file=sys.stderr
        )
        return 2

    m = llama_models[0]
    user_config = m.llama_server_config or LlamaServerConfig()

    # Preflight reads the GGUF's own metadata, so resolve the path first. The
    # driver does this for the actor.
    model_path = resolve_model_source(m.model)
    m._resolved_path = model_path
    print(f"rawllama resolved model -> {model_path}", flush=True)

    # The same recommendation/override merge the actor runs; MSHIP_PREFLIGHT
    # applies to both arms.
    recommendation = run_preflight(m, discover_hardware(read_free_memory=True))
    print(f"rawllama preflight recommendation: {recommendation or 'none'}", flush=True)
    merged = merge_with_user_overrides(recommendation, user_config.model_dump(exclude_unset=True), model_name=m.name)
    k = user_config.model_copy(update=merged)

    # Preflight reads free RAM/VRAM, which moves between the two phases. The
    # harness pins what the modelship arm launched with.
    n_ctx_total = int(_pinned("N_CTX_TOTAL", k.n_ctx * k.parallel))
    n_gpu_layers = int(_pinned("N_GPU_LAYERS", k.n_gpu_layers))

    args = [
        binary,
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "-m",
        model_path,
        "-c",
        str(n_ctx_total),
        "-b",
        str(k.n_batch),
        "-ub",
        str(k.ubatch_size),
        "-fa",
        k.flash_attn,
        "-ctk",
        k.cache_type_k,
        "-ctv",
        k.cache_type_v,
        "--parallel",
        str(k.parallel),
        "--jinja",
        "--reasoning-format",
        "auto",
        "--no-webui",
        # /v1/models reports this as "id"; no --api-key, matching vanilla llama-server.
        "--alias",
        m.name,
    ]
    # Ray only sets CUDA_VISIBLE_DEVICES for GPU-reserving actors, so a
    # num_gpus=0 deploy may still see every GPU — force no offload.
    if m.num_gpus > 0:
        args += ["-ngl", str(n_gpu_layers)]
        if k.tensor_split:
            args += ["-ts", ",".join(str(v) for v in k.tensor_split)]
        # Bypasses Ray's own CUDA_VISIBLE_DEVICES restriction — set it explicitly.
        # A fractional num_gpus rounds up to one device.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, math.ceil(m.num_gpus))))
    else:
        args += ["-ngl", "0"]
    if k.threads is not None:
        args += ["--threads", str(k.threads)]
    if k.chat_template:
        flag = "--chat-template-file" if os.path.isfile(k.chat_template) else "--chat-template"
        args += [flag, k.chat_template]
    if k.mmproj:
        mmproj_path = resolve_model_source(k.mmproj)
        args += ["--mmproj", mmproj_path]
    if m.usecase == ModelUsecase.embed:
        args += ["--embedding"]
    if k.cache_reuse > 0:
        args += ["--cache-reuse", str(k.cache_reuse)]
    if k.context_shift:
        args += ["--context-shift"]
    if k.cache_ram_mib is not None:
        args += ["--cache-ram", str(k.cache_ram_mib)]

    # shlex.join, not " ".join: an inline --chat-template has to survive the
    # parity checker's shlex.split.
    print("rawllama exec:", shlex.join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
