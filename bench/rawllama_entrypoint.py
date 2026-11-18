"""Runs `llama-server` directly against the same models.yaml modelship reads,
bypassing Ray, to A/B against the llama_server loader with an identical
launch command: the flags come from the loader's own `serve_args`."""

from __future__ import annotations

import os

from modelship.deploy.actor_options import build_cache_env_vars
from modelship.utils.cache import resolve_cache_root, resolve_node_cache_root

# Before any huggingface_hub import: HF_HOME latches there. Ray expands the ${root} placeholders for replicas; we do it here.
os.environ.setdefault("MSHIP_CACHE_DIR", resolve_cache_root())
os.environ.setdefault("MSHIP_NODE_CACHE_DIR", resolve_node_cache_root())
for _key, _value in build_cache_env_vars().items():
    os.environ.setdefault(_key, os.path.expandvars(_value))

import math  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from modelship.infer.infer_config import LlamaServerConfig, ModelLoader, ModelshipConfig  # noqa: E402
from modelship.infer.llama_server.llama_server_infer import serve_args  # noqa: E402
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight  # noqa: E402
from modelship.utils.sources import resolve_model_source  # noqa: E402

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
    recommendation = run_preflight(m, discover_hardware())
    print(f"rawllama preflight recommendation: {recommendation or 'none'}", flush=True)
    merged = merge_with_user_overrides(recommendation, user_config.model_dump(exclude_unset=True), model_name=m.name)
    k = user_config.model_copy(update=merged)

    # Preflight reads free RAM/VRAM, which moves between the two phases. The
    # harness pins what the modelship arm launched with.
    n_ctx_total = int(_pinned("N_CTX_TOTAL", k.n_ctx * k.parallel))
    n_gpu_layers = int(_pinned("N_GPU_LAYERS", k.n_gpu_layers))
    tensor_split = _pinned("TENSOR_SPLIT", ",".join(str(v) for v in k.tensor_split) if k.tensor_split else "")
    override_tensor = _pinned("OVERRIDE_TENSOR", ",".join(k.override_tensor) if k.override_tensor else "")
    k = k.model_copy(
        update={
            "n_ctx": n_ctx_total // k.parallel,
            "n_gpu_layers": n_gpu_layers,
            "tensor_split": [float(v) for v in tensor_split.split(",")] if tensor_split else None,
            "override_tensor": override_tensor.split(",") if override_tensor else None,
            "mmproj": resolve_model_source(k.mmproj) if k.mmproj else None,
        }
    )

    args = [
        binary,
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        *serve_args(model_path, m, k),
        # /v1/models reports this as "id"; no --api-key, matching vanilla llama-server.
        "--alias",
        m.name,
    ]
    if m.num_gpus > 0:
        # Bypasses Ray's own CUDA_VISIBLE_DEVICES restriction — set it explicitly.
        # A fractional num_gpus rounds up to one device.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, math.ceil(m.num_gpus))))

    # shlex.join, not " ".join: an inline --chat-template has to survive the
    # parity checker's shlex.split.
    print("rawllama exec:", shlex.join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
