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
from vllm.tokenizers import get_tokenizer  # noqa: E402

from modelship.infer.infer_config import (  # noqa: E402
    ModelLoader,
    ModelshipConfig,
    ModelUsecase,
    resolve_gpu_memory_utilization,
)
from modelship.infer.model_resolver import resolve_model_source  # noqa: E402
from modelship.infer.vllm.parsing.detect import resolve_reasoning_parser, resolve_tool_parser  # noqa: E402
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

    # Ray gives the modelship actor exactly the devices it reserved; a bare
    # subprocess would otherwise inherit every GPU --gpus exposed. A multi-slot
    # deploy reserves one whole-GPU bundle per slot and config validation
    # collapses num_gpus to 1.0, so tp x pp carries that count instead.
    world_size = m.vllm_engine_kwargs.tensor_parallel_size * m.vllm_engine_kwargs.pipeline_parallel_size
    gpus_reserved = max(math.ceil(m.num_gpus), world_size if world_size > 1 else 0)
    if gpus_reserved:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus_reserved))
        print(f"rawvllm CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

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

    # Mirrors init_serving_chat: same resolvers, same usecase gate. Reads the
    # template off the configured tokenizer, which vLLM defaults to the model.
    tool_parser = reasoning_parser = None
    if m.usecase == ModelUsecase.generate:
        tokenizer = get_tokenizer(k.tokenizer or m._resolved_path, trust_remote_code=k.trust_remote_code)
        try:
            template = tokenizer.get_chat_template()
        except ValueError:
            # A base model carries no template; the actor leaves both unset too.
            template = None
        tool_parser = resolve_tool_parser(m, template)
        reasoning_parser = resolve_reasoning_parser(m, template)
    print(f"rawvllm parsers: tool={tool_parser} reasoning={reasoning_parser}", flush=True)

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
    # Resolved above, not read off k: an unset config field still auto-detects.
    if tool_parser:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    if reasoning_parser:
        args += ["--reasoning-parser", reasoning_parser]
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
    # Derived internally by modelship, not a config field: vLLM's own default
    # for a multi-slot deploy is mp, so leaving it out would compare two
    # executors rather than two wrappers around one.
    if world_size > 1:
        args += ["--distributed-executor-backend", "ray"]

    # shlex.join, not " ".join: the JSON-valued flags have to survive the
    # parity checker's shlex.split.
    print("rawvllm exec:", shlex.join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
