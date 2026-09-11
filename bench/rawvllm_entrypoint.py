"""Runs `vllm serve` directly against the same models.yaml modelship reads,
bypassing Ray, to A/B against the vllm loader with identical engine kwargs.
Mirrors `VllmInfer.__init__` — keep both in sync."""

from __future__ import annotations

import os

# Must precede any huggingface_hub import — HF_HOME latches at its import time.
# The same set the driver hands each replica via runtime_env, so both arms
# resolve weights into the mounted cache.
from modelship.deploy.actor_options import build_cache_env_vars

for _key, _value in build_cache_env_vars().items():
    os.environ.setdefault(_key, _value)

import json  # noqa: E402
import math  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from modelship.infer.infer_config import (  # noqa: E402
    ModelLoader,
    ModelshipConfig,
    resolve_gpu_memory_utilization,
)
from modelship.infer.model_resolver import resolve_model_source  # noqa: E402
from modelship.infer.vllm.vllm_infer import resolve_max_model_len  # noqa: E402
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight  # noqa: E402
from modelship.utils.config_schema import VllmEngineConfig  # noqa: E402

CONFIG_PATH = Path(os.environ.get("MSHIP_CONFIG", "/modelship/config/models.yaml"))


def _pinned(name: str, derived):
    """The harness's replay of what the modelship arm launched with, if set."""
    value = os.environ.get(f"BENCH_PIN_{name}")
    if value is None:
        return derived
    print(f"rawvllm pinned {name.lower()}={value} (derived {derived})", flush=True)
    return value


def main() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = ModelshipConfig.model_validate(raw)
    vllm_models = [m for m in cfg.models if m.loader == ModelLoader.vllm]
    if len(vllm_models) != 1:
        print(f"bench expects exactly one vllm model in {CONFIG_PATH}, got {len(vllm_models)}", file=sys.stderr)
        return 2

    m = vllm_models[0]

    # Ray gives the modelship actor exactly num_gpus device(s); a bare
    # subprocess would otherwise inherit every GPU --gpus exposed.
    if m.num_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, math.ceil(m.num_gpus))))

    # Preflight reads the checkpoint's config.json, so resolve the path first.
    # The driver does this for the actor.
    m._resolved_path = resolve_model_source(m.model, trust_remote_code=m.vllm_engine_kwargs.trust_remote_code)
    print(f"rawvllm resolved model -> {m._resolved_path}", flush=True)

    # The same recommendation/override merge the actor runs; MSHIP_PREFLIGHT
    # applies to both arms.
    recommendation = run_preflight(m, discover_hardware())
    print(f"rawvllm preflight recommendation: {recommendation or 'none'}", flush=True)
    merged = merge_with_user_overrides(
        recommendation, m.vllm_engine_kwargs.model_dump(exclude_unset=True), model_name=m.name
    )
    gpu_memory_utilization = resolve_gpu_memory_utilization(m, merged.pop("gpu_memory_utilization", None))
    k = VllmEngineConfig(**merged)
    max_model_len = resolve_max_model_len(k)

    # Preflight reads free RAM/VRAM, which moves between the two phases. The
    # harness pins what the modelship arm launched with.
    gpu_memory_utilization = float(_pinned("GPU_MEMORY_UTILIZATION", gpu_memory_utilization))
    max_model_len = int(_pinned("MAX_MODEL_LEN", max_model_len))

    args = ["vllm", "serve", m._resolved_path, "--host", "0.0.0.0", "--port", "8000", "--served-model-name", m.name]
    args += ["--tensor-parallel-size", str(k.tensor_parallel_size)]
    args += ["--pipeline-parallel-size", str(k.pipeline_parallel_size)]
    args += ["--dtype", k.dtype]
    # Both derived, not config keys; the modelship arm calls the same two.
    args += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    args += ["--max-model-len", str(max_model_len)]
    args += ["--kv-cache-dtype", k.kv_cache_dtype or "auto"]
    if k.tokenizer:
        args += ["--tokenizer", k.tokenizer]
    if k.trust_remote_code:
        args += ["--trust-remote-code"]
    if k.quantization:
        args += ["--quantization", k.quantization]
    if k.enable_auto_tool_choice:
        args += ["--enable-auto-tool-choice"]
    if k.tool_call_parser:
        args += ["--tool-call-parser", k.tool_call_parser]
    if k.enforce_eager:
        args += ["--enforce-eager"]
    if k.enable_prefix_caching is not None:
        args += ["--enable-prefix-caching" if k.enable_prefix_caching else "--no-enable-prefix-caching"]
    if k.max_num_batched_tokens is not None:
        args += ["--max-num-batched-tokens", str(k.max_num_batched_tokens)]
    if k.max_num_seqs is not None:
        args += ["--max-num-seqs", str(k.max_num_seqs)]
    # None means False to VllmInfer, so both are emitted explicitly rather than
    # left to vLLM's own default.
    args += ["--enable-log-requests" if k.enable_log_requests else "--no-enable-log-requests"]
    # store_true, with no --no- form: absent is False.
    if k.disable_log_stats:
        args += ["--disable-log-stats"]
    args += ["--chat-template-content-format", k.chat_template_content_format]
    # Forwarded only when set, so vLLM's own defaults apply otherwise — same as
    # VllmInfer's mm_kwargs.
    if k.limit_mm_per_prompt is not None:
        args += ["--limit-mm-per-prompt", json.dumps(k.limit_mm_per_prompt, sort_keys=True, separators=(",", ":"))]
    if k.mm_processor_kwargs is not None:
        args += ["--mm-processor-kwargs", json.dumps(k.mm_processor_kwargs, sort_keys=True, separators=(",", ":"))]
    # distributed_executor_backend is derived internally by modelship (not a
    # config field), so the raw phase relies on vLLM's own default executor here.

    # shlex.join, not " ".join: the JSON-valued flags have to survive the
    # parity checker's shlex.split.
    print("rawvllm exec:", shlex.join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
