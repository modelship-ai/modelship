from __future__ import annotations

import contextlib
import json
import logging
import math
import os
from typing import Any, NamedTuple, cast

from modelship.infer.infer_config import (
    ModelshipModelConfig,
    default_gpu_memory_utilization,
    resolve_gpu_memory_utilization,
)
from modelship.logging import get_logger
from modelship.preflight._mla import MLAInfo
from modelship.preflight._mla import kv_bytes_per_token as mla_kv_bytes_per_token
from modelship.preflight._sliding_window import SlidingWindowInfo, fit_len_with_sliding, seq_kv_bytes
from modelship.preflight.base import HardwareProfile

logger = get_logger("preflight.vllm")

# vLLM default; KV cache is allocated in pages of `block_size` tokens.
_DEFAULT_BLOCK_SIZE = 16

# Concurrency floor for hybrid/SSM models: vLLM parks a fixed recurrent-state
# buffer per sequence slot (sized by max_num_seqs, not max_model_len).
_MIN_MAX_NUM_SEQS = 8

# Overhead on top of weight bytes: AWQ/Marlin transposed packs, quant scales,
# embedding tables, torch.compile artifacts not in safetensors `total_size`.
_OVERHEAD_WEIGHT_FRACTION = 0.14

# torch_dtype string -> bytes per element. KV cache uses the compute dtype,
# not storage dtype, so AWQ/GPTQ models are still 2 bytes per element.
_DTYPE_BYTES = {
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "float32": 4,
    "float": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}

# Floor for `max_num_batched_tokens` on multimodal models — vLLM refuses to
# start if one MM item tokenizes to more than this.
_MULTIMODAL_BATCHED_TOKENS_FLOOR = 8192

# CPU backend: vLLM's worker reserves `gpu_memory_utilization * total_memory`
# and hard-raises if it exceeds available RAM.
_CPU_RAM_UTILIZATION = 0.8
_CPU_OVERHEAD_FIXED_BYTES = 2 * 1024**3
# Clamp the auto-picked KV budget to ~4 full-length sequences, so a large RAM
# box doesn't reserve an oversized fraction for a small-context model.
_CPU_KV_SEQUENCES = 4
# Context-length cap used when the model config doesn't declare
# max_position_embeddings.
_UNKNOWN_CONTEXT_LENGTH_CAP = 32768

# MLA's chunked-prefill decompression workspace row cap, mirrored from vLLM's
# MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size.
_MLA_WORKSPACE_ROW_CAP = 65536

# vLLM's SchedulerConfig.DEFAULT_MAX_NUM_SEQS, used when the config leaves it unset.
_VLLM_DEFAULT_MAX_NUM_SEQS = 128

# Safety margin on the MLA workspace's gpu_util reduction, covering PyTorch
# CUDA allocator rounding/fragmentation on top of the raw tensor size.
_MLA_WORKSPACE_SAFETY_MARGIN = 1.1


class VllmPreflight:
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Branch on the reservation, not hardware discoverability —
        # discover_hardware()'s pynvml fallback can report GPUs Ray didn't
        # assign to this num_gpus=0 deploy.
        if config.num_gpus == 0:
            return self._recommend_cpu(config, hw)
        return self._recommend_gpu(config, hw)

    def _recommend_gpu(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        shared_util = self._shared_memory_gpu_memory_utilization(config, hw)
        if shared_util is not None:
            rec["gpu_memory_utilization"] = shared_util
        model_path = config._resolved_path
        if not model_path:
            logger.info("preflight '%s': no resolved model path; auto-fit only", config.name)
            return rec

        model_cfg = _load_model_config_json(model_path)
        if model_cfg is None:
            logger.info(
                "preflight '%s': config.json not found or unreadable at %s; auto-fit only",
                config.name,
                model_path,
            )
            return rec

        text_cfg = _resolve_text_config(model_cfg)

        # vLLM has no auto-fit for max_num_seqs, and its default overruns the
        # per-slot mamba block pool.
        if _resolve_mamba_state(config, model_path) is not None:
            rec["max_num_seqs"] = _MIN_MAX_NUM_SEQS
            logger.info("preflight vllm '%s': hybrid/SSM → max_num_seqs=%d", config.name, _MIN_MAX_NUM_SEQS)

        # Multimodal: bump `max_num_batched_tokens` to fit one image/audio item
        # per batch.
        if _is_multimodal(model_cfg):
            mm_tokens_per_item = _estimate_mm_tokens_per_item(model_cfg)
            mnbt = _recommended_mm_batched_tokens(mm_tokens_per_item)
            rec["max_num_batched_tokens"] = mnbt
            logger.info(
                "preflight vllm '%s': multimodal detected → suggested max_num_batched_tokens=%d "
                "(mm_tokens_per_item≈%s)",
                config.name,
                mnbt,
                mm_tokens_per_item if mm_tokens_per_item is not None else "unknown",
            )

        gpu_util = self._mla_gpu_memory_utilization(config, hw, text_cfg, model_cfg, shared_util)
        if gpu_util is not None:
            rec["gpu_memory_utilization"] = gpu_util

        return rec

    def _shared_memory_gpu_memory_utilization(self, config: ModelshipModelConfig, hw: HardwareProfile) -> float | None:
        """On a GPU that shares the node's RAM, the share of its total that is free after the
        reserve, capped at the default. None on a dedicated-memory GPU and for a fractional num_gpus."""
        shared = [g for g in hw.gpus if g.shared_memory and g.total_bytes]
        if not shared or 0 < config.num_gpus < 1:
            return None
        free_share = min(g.available_bytes / g.total_bytes for g in shared)
        gpu_util = max(math.floor(min(default_gpu_memory_utilization(config), free_share) * 100) / 100, 0.01)
        logger.info(
            "preflight vllm '%s': the GPU shares the node's memory, %.0f%% of it free after the reserve "
            "→ gpu_memory_utilization=%.2f",
            config.name,
            free_share * 100,
            gpu_util,
        )
        return gpu_util

    def _mla_gpu_memory_utilization(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        text_cfg: dict,
        model_cfg: dict,
        shared_util: float | None = None,
    ) -> float | None:
        """MLA's chunked-prefill workspace sits outside the KV pool, and gpu_util
        is the only lever that leaves room for it. None for every other model."""
        mla = _resolve_mla(text_cfg)
        if mla is None or config.vllm_engine_kwargs.enforce_eager or not hw.gpus:
            return None

        # TP requires homogeneous GPUs, so take the smallest. gmu multiplies the
        # device total, so the haircut divides by that same basis.
        gpu_basis = min(g.sizing_total_bytes for g in hw.gpus)
        tp_size = max(config.vllm_engine_kwargs.tensor_parallel_size, 1)
        dtype_bytes = _resolve_compute_dtype_bytes(text_cfg, model_cfg)
        workspace_bytes = _mla_chunked_prefill_workspace_bytes(text_cfg, config, dtype_bytes, mla, tp_size)

        # A fractional num_gpus is the fraction, then a shared-memory GPU's free share, else the default.
        gpu_util = resolve_gpu_memory_utilization(config, shared_util)
        gpu_util = max(gpu_util - _MLA_WORKSPACE_SAFETY_MARGIN * workspace_bytes / gpu_basis, 0.01)
        logger.info(
            "preflight vllm '%s': MLA workspace %.2f GiB on a %.2f GiB GPU → gpu_memory_utilization=%.4f",
            config.name,
            workspace_bytes / 1024**3,
            gpu_basis / 1024**3,
            gpu_util,
        )
        return round(gpu_util, 4)

    def _recommend_cpu(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        model_path = config._resolved_path
        if not model_path:
            logger.info("preflight '%s': skipping — no resolved model path", config.name)
            return {}

        model_cfg = _load_model_config_json(model_path)
        if model_cfg is None:
            logger.info(
                "preflight '%s': skipping — config.json not found or unreadable at %s",
                config.name,
                model_path,
            )
            return {}

        text_cfg = _resolve_text_config(model_cfg)
        kv_per_token, max_position_embeddings = _kv_bytes_per_token(text_cfg, model_cfg, config)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — config.json missing KV-cache geometry "
                "(num_hidden_layers/num_key_value_heads/head_dim). Top-level keys=%s, "
                "architectures=%s",
                config.name,
                sorted(model_cfg.keys()),
                model_cfg.get("architectures"),
            )
            return {}

        weight_bytes = _estimate_weight_footprint(model_path)
        weight_overhead = _OVERHEAD_WEIGHT_FRACTION * weight_bytes
        ctx_cap = max_position_embeddings or _UNKNOWN_CONTEXT_LENGTH_CAP
        # vLLM's CPU worker multiplies gpu_memory_utilization by a total that
        # is clamped to the cgroup limit; hw.ram_bytes is the same quantity.
        denom_ram = hw.ram_bytes
        if denom_ram <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        # Hybrid/SSM state accounting is device-agnostic; the CPU worker draws
        # mamba state from the same gmu*RAM pool, only the pool source differs.
        mamba = _resolve_mamba_state(config, model_path)
        if mamba is not None:
            kv_per_token = _correct_kv_for_hybrid(kv_per_token, mamba)
        sliding = _resolve_sliding_window(text_cfg) if mamba is None else None

        return self._recommend_cpu_auto_gmu(
            config, hw, kv_per_token, weight_bytes, weight_overhead, ctx_cap, denom_ram, mamba, sliding
        )

    def _recommend_cpu_auto_gmu(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        kv_per_token: float,
        weight_bytes: int,
        weight_overhead: float,
        ctx_cap: int,
        denom_ram: int,
        mamba: MambaStateInfo | None,
        sliding: SlidingWindowInfo | None,
    ) -> dict[str, Any]:
        """Sizes both max_model_len and gpu_memory_utilization, targeting up to
        `_CPU_RAM_UTILIZATION` of free RAM after weights/overhead."""
        kv_budget = (
            hw.sizing_ram_bytes * _CPU_RAM_UTILIZATION - weight_bytes - weight_overhead - _CPU_OVERHEAD_FIXED_BYTES
        )
        if kv_budget <= 0:
            logger.warning(
                "preflight '%s': no KV-cache budget on CPU (available=%.2f GiB, est. weights=%.2f GiB); "
                "model likely won't fit; deploy will be attempted anyway.",
                config.name,
                hw.sizing_ram_bytes / 1024**3,
                (weight_bytes + weight_overhead) / 1024**3,
            )
            return {}

        if mamba is not None:
            return self._recommend_cpu_auto_gmu_hybrid(
                config, kv_budget, kv_per_token, ctx_cap, denom_ram, weight_bytes, weight_overhead, mamba
            )

        if sliding is not None:
            max_tokens = _fit_len_with_sliding(kv_budget, kv_per_token, sliding, ctx_cap)
        else:
            max_tokens = int(kv_budget // kv_per_token)
        suggested = min((max_tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE, ctx_cap)
        if suggested < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': CPU budget yields max_model_len=%d (< block_size); skipping recommendation",
                config.name,
                suggested,
            )
            return {}

        clamped_kv_bytes = min(kv_budget, _CPU_KV_SEQUENCES * _seq_kv_bytes(kv_per_token, sliding, suggested))
        # kv_budget took the weights out; add them back so the fraction still
        # leaves clamped_kv_bytes once vLLM subtracts RSS.
        reservation = weight_bytes + weight_overhead + _CPU_OVERHEAD_FIXED_BYTES + clamped_kv_bytes
        recommended_gmu = _cpu_gmu(reservation, denom_ram)

        logger.info(
            "preflight vllm cpu '%s': sizing_ram=%.2f GiB weights≈%.2f GiB kv/token=%d B "
            "→ suggested max_model_len=%d gpu_memory_utilization=%.3f",
            config.name,
            hw.sizing_ram_bytes / 1024**3,
            (weight_bytes + weight_overhead) / 1024**3,
            int(kv_per_token),
            suggested,
            recommended_gmu,
        )
        return {"gpu_memory_utilization": recommended_gmu}

    def _recommend_cpu_auto_gmu_hybrid(
        self,
        config: ModelshipModelConfig,
        kv_budget: float,
        kv_per_token: float,
        ctx_cap: int,
        denom_ram: int,
        weight_bytes: int,
        weight_overhead: float,
        mamba: MambaStateInfo,
    ) -> dict[str, Any]:
        """Hybrid on the auto-gmu path: splits kv_budget between mamba state and
        attention KV, then back-computes a gmu covering weights + state + KV
        (vLLM sizes CPU KV budget as gmu*RAM - RSS)."""
        target_len = config.vllm_engine_kwargs.max_model_len or ctx_cap
        rec = _apply_hybrid_fit(
            config.name,
            kv_budget,
            mamba.per_seq_state_bytes,
            kv_per_token,
            target_len,
            config.vllm_engine_kwargs.max_num_seqs,
            mamba.default_max_num_seqs,
        )
        if not rec:
            return {}

        chosen_len = rec.pop("max_model_len")
        chosen_seqs = rec.get("max_num_seqs", config.vllm_engine_kwargs.max_num_seqs or _MIN_MAX_NUM_SEQS)
        state_bytes = mamba.per_seq_state_bytes * chosen_seqs
        # Clamp attention KV to ~a few full-length sequences (kv_budget already
        # had weights set aside, so its remainder after state is the KV room).
        attn_kv = min(kv_budget - state_bytes, _CPU_KV_SEQUENCES * kv_per_token * chosen_len)
        reservation = weight_bytes + weight_overhead + _CPU_OVERHEAD_FIXED_BYTES + state_bytes + max(attn_kv, 0)
        recommended_gmu = _cpu_gmu(reservation, denom_ram)
        rec["gpu_memory_utilization"] = recommended_gmu

        logger.info(
            "preflight vllm cpu '%s': hybrid state %.1f MiB/seq x %d seqs = %.2f GiB + weights + attn KV -> "
            "gpu_memory_utilization=%.3f (max_model_len=%d)",
            config.name,
            mamba.per_seq_state_bytes / 1024**2,
            chosen_seqs,
            state_bytes / 1024**3,
            recommended_gmu,
            chosen_len,
        )
        return rec


def _cpu_gmu(reservation: float, denom_ram: int) -> float:
    """Fraction of RAM to reserve. vLLM's CPU worker sizes the KV pool as
    `gmu * total - process RSS`, so the fraction has to span the weights too,
    not just the KV bytes."""
    return min(max(round(reservation / denom_ram, 3), 0.01), 0.9)


def _load_model_config_json(model_path: str) -> dict | None:
    """Read the standard transformers-layout `config.json` from a model
    directory."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        logger.debug("preflight: failed to parse %s", cfg_path, exc_info=True)
        return None


def _resolve_text_config(model_cfg: dict) -> dict:
    """Multimodal models nest language-model geometry in a sub-config; try
    common nesting keys if num_hidden_layers isn't top-level."""
    if model_cfg.get("num_hidden_layers") or model_cfg.get("num_layers"):
        return model_cfg
    for key in ("text_config", "language_config", "llm_config", "language_model_config"):
        sub = model_cfg.get(key)
        if isinstance(sub, dict) and (sub.get("num_hidden_layers") or sub.get("num_layers")):
            return sub
    return model_cfg


def _resolve_mla(text_cfg: dict) -> MLAInfo | None:
    """None for ordinary MHA/GQA models."""
    kv_lora_rank = text_cfg.get("kv_lora_rank")
    qk_rope_head_dim = text_cfg.get("qk_rope_head_dim")
    num_heads = text_cfg.get("num_attention_heads")
    if not (kv_lora_rank and qk_rope_head_dim and num_heads):
        return None
    qk_nope_head_dim = text_cfg.get("qk_nope_head_dim") or 0
    v_head_dim = text_cfg.get("v_head_dim") or qk_nope_head_dim
    return MLAInfo(kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, num_heads)


def _kv_bytes_per_token(text_cfg: dict, model_cfg: dict, config: ModelshipModelConfig) -> tuple[int | None, int | None]:
    """Returns (bytes-per-token-across-all-TP-ranks, max_position_embeddings).
    Reads geometry from `text_cfg`, falling back to `model_cfg`."""
    num_layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers")
    if not num_layers:
        return None, None
    kv_dtype_bytes = _resolve_kv_dtype_bytes(text_cfg, model_cfg, config)
    max_position_embeddings = text_cfg.get("max_position_embeddings") or model_cfg.get("max_position_embeddings")
    max_position_embeddings = int(max_position_embeddings) if max_position_embeddings else None

    mla = _resolve_mla(text_cfg)
    if mla is not None:
        return mla_kv_bytes_per_token(mla, kv_dtype_bytes, num_layers), max_position_embeddings

    num_attention_heads = text_cfg.get("num_attention_heads")
    num_kv_heads = text_cfg.get("num_key_value_heads") or num_attention_heads
    hidden_size = text_cfg.get("hidden_size")
    head_dim = text_cfg.get("head_dim")
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    if not (num_kv_heads and head_dim):
        return None, None

    # Each token stores both K and V (factor of 2) for every layer.
    per_token = 2 * num_kv_heads * head_dim * kv_dtype_bytes * num_layers
    return int(per_token), max_position_embeddings


def _resolve_kv_dtype_bytes(text_cfg: dict, model_cfg: dict, config: ModelshipModelConfig) -> int:
    user_kv = (config.vllm_engine_kwargs.kv_cache_dtype or "auto").lower()
    if user_kv.startswith("fp8"):
        return 1
    return _resolve_compute_dtype_bytes(text_cfg, model_cfg)


def _resolve_compute_dtype_bytes(text_cfg: dict, model_cfg: dict) -> int:
    """The model's forward-pass dtype, used for activations and CUDA-graph
    workspace regardless of any kv_cache_dtype override."""
    torch_dtype = (text_cfg.get("torch_dtype") or model_cfg.get("torch_dtype") or "float16").lower()
    return _DTYPE_BYTES.get(torch_dtype, 2)


def _recommended_mm_batched_tokens(mm_tokens_per_item: int | None) -> int:
    """Floor for `max_num_batched_tokens` on multimodal models — enough to fit
    one image/audio item in one batch with headroom for text tokens."""
    floor = _MULTIMODAL_BATCHED_TOKENS_FLOOR
    if mm_tokens_per_item is not None:
        floor = max(floor, mm_tokens_per_item * 2)
    return floor


def _mla_chunked_prefill_workspace_bytes(
    text_cfg: dict, config: ModelshipModelConfig, dtype_bytes: int, mla: MLAInfo, tp_size: int
) -> int:
    """MLA's decompressed-K/V scratch buffer, sized by context length —
    mirrors vLLM's `determine_chunked_prefill_workspace_size`."""
    max_model_len = (
        config.vllm_engine_kwargs.max_model_len
        or text_cfg.get("max_position_embeddings")
        or _UNKNOWN_CONTEXT_LENGTH_CAP
    )
    # Larger of 8 full-length requests and 4 pages per slot, capped, then
    # re-floored at one page per slot — that floor is applied after the cap.
    pages = (config.vllm_engine_kwargs.max_num_seqs or _VLLM_DEFAULT_MAX_NUM_SEQS) * _DEFAULT_BLOCK_SIZE
    rows = max(min(max(8 * max_model_len, 4 * pages), _MLA_WORKSPACE_ROW_CAP), pages)
    # The up-projection runs on this rank's shard of the heads, so unlike the
    # replicated latent cache this buffer does shrink with TP.
    local_heads = max(mla.num_heads // max(tp_size, 1), 1)
    return int(rows * local_heads * (mla.qk_nope_head_dim + mla.v_head_dim) * dtype_bytes)


# Re-exported for existing call sites/tests that import these from here.
_seq_kv_bytes = seq_kv_bytes
_fit_len_with_sliding = fit_len_with_sliding


def _resolve_sliding_window(text_cfg: dict) -> SlidingWindowInfo | None:
    """Split layers by `layer_types`, else `sliding_window_pattern` (every Nth
    layer full), else a bare `sliding_window` (all layers slide). None if uniform."""
    window = text_cfg.get("sliding_window")
    if not window or text_cfg.get("use_sliding_window") is False:
        return None
    n_total = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers")
    if not n_total:
        return None

    layer_types = text_cfg.get("layer_types")
    pattern = text_cfg.get("sliding_window_pattern")
    if layer_types:
        n_sliding = sum(1 for t in layer_types if isinstance(t, str) and "sliding" in t)
    elif isinstance(pattern, int) and pattern > 0:
        n_sliding = n_total - n_total // pattern
    else:
        n_sliding = n_total

    if n_sliding <= 0:
        return None
    return SlidingWindowInfo(n_total - n_sliding, n_sliding, int(n_total), int(window))


def _is_multimodal(model_cfg: dict) -> bool:
    """Multimodal models carry a sub-config for the non-text modality or
    advertise a conditional-generation architecture."""
    for key in ("vision_config", "audio_config", "video_config", "mm_processor_kwargs"):
        if model_cfg.get(key) is not None:
            return True
    architectures = model_cfg.get("architectures") or []
    arch_blob = " ".join(architectures).lower()
    return any(marker in arch_blob for marker in ("forconditionalgeneration", "vlm", "multimodal", "vision", "audio"))


def _estimate_mm_tokens_per_item(model_cfg: dict) -> int | None:
    """Lower-bound tokens-per-image estimate from the vision encoder's patch
    grid: (image_size / patch_size)². Ignores token mergers, erring toward
    over-estimate."""
    vision = model_cfg.get("vision_config") or {}
    image_size = vision.get("image_size")
    patch_size = vision.get("patch_size")
    if not (image_size and patch_size):
        return None
    try:
        patches_per_side = int(image_size) // int(patch_size)
    except (TypeError, ValueError):
        return None
    if patches_per_side <= 0:
        return None
    return patches_per_side * patches_per_side


def _estimate_weight_footprint(model_path: str) -> int:
    """Estimate on-disk weight footprint: safetensors preferred, falling back
    to `.bin`/`.pt`. Takes the max of the index's `total_size` and summed
    `*.safetensors` file sizes, since some checkpoints ship files the index
    doesn't reference."""
    try:
        names = os.listdir(model_path)
    except OSError:
        return 0

    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    index_total = _read_index_total_size(safetensors_index) if os.path.isfile(safetensors_index) else 0
    directory_total = sum(os.path.getsize(os.path.join(model_path, n)) for n in names if n.endswith(".safetensors"))
    if index_total or directory_total:
        return max(index_total, directory_total)

    pytorch_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.isfile(pytorch_index):
        total = _read_index_total_size(pytorch_index)
        if total:
            return total

    return sum(os.path.getsize(os.path.join(model_path, n)) for n in names if n.endswith((".bin", ".pt")))


def _read_index_total_size(index_path: str) -> int:
    try:
        with open(index_path) as f:
            idx = json.load(f)
    except Exception:
        logger.debug("preflight: failed to read weight index %s", index_path, exc_info=True)
        return 0
    total = idx.get("metadata", {}).get("total_size")
    return int(total) if total else 0


class MambaStateInfo(NamedTuple):
    """Recurrent-state accounting for a hybrid/SSM model. `per_seq_state_bytes`
    is one concurrent sequence slot's mamba state across all state layers
    (per worker, PP-folded)."""

    per_seq_state_bytes: int
    n_state_layers: int
    n_full_attention_layers: int
    n_total_layers: int
    default_max_num_seqs: int


@contextlib.contextmanager
def _quiet_vllm_logging():
    """Silences vLLM's INFO/WARNING noise from building a throwaway engine
    config; restores levels afterward."""
    names = ("vllm", "vllm.config", "vllm.engine", "vllm.transformers_utils")
    prev = {n: logging.getLogger(n).level for n in names}
    for n in names:
        logging.getLogger(n).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for n, lvl in prev.items():
            logging.getLogger(n).setLevel(lvl)


def _resolve_mamba_state(config: ModelshipModelConfig, model_path: str) -> MambaStateInfo | None:
    """Recurrent-state accounting for hybrid/SSM models, via a throwaway
    offline vLLM engine config and vLLM's own `is_hybrid`/
    `get_mamba_state_shape_from_config` primitives. None on any failure or
    for ordinary transformers."""
    tp = max(config.vllm_engine_kwargs.tensor_parallel_size, 1)
    pp = max(config.vllm_engine_kwargs.pipeline_parallel_size, 1)
    prev_offline = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.interfaces import is_attention_free, is_hybrid

        # The model is already local; force offline so config building never
        # reaches the network.
        for k in prev_offline:
            os.environ[k] = "1"

        with _quiet_vllm_logging():
            engine_args = EngineArgs(
                model=model_path,
                load_format="dummy",
                enforce_eager=True,
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                # ParallelConfig validates world_size against visible GPUs; this
                # runs in the 0-GPU PG bundle.
                distributed_executor_backend="ray" if tp * pp > 1 else None,
                dtype=cast("Any", config.vllm_engine_kwargs.dtype or "auto"),
                trust_remote_code=config.vllm_engine_kwargs.trust_remote_code,
            )
            vllm_config = engine_args.create_engine_config()

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        archs = model_config.hf_config.architectures
        if not archs:
            return None
        cls, _arch = ModelRegistry.resolve_model_cls(archs, model_config)

        if not (is_hybrid(cls) or is_attention_free(cls)):
            return None

        # Per-slot bytes for one state layer, summed over conv + temporal caches.
        # `cls` exposes these dynamic classmethods that nn.Module's typing hides.
        state_cls: Any = cls
        shapes = state_cls.get_mamba_state_shape_from_config(vllm_config)
        dtypes = state_cls.get_mamba_state_dtype_from_config(vllm_config)
        per_slot = sum(math.prod(shape) * dt.itemsize for shape, dt in zip(shapes, dtypes, strict=True))

        # Authoritative layer split (per PP stage). "Not attention" == "has
        # recurrent state"; over-counting exotic MLP-only layers errs safe.
        n_full_attention = model_config.get_num_layers_by_block_type(parallel_config, "attention")
        n_total = model_config.get_num_layers(parallel_config)
        n_state = n_total - n_full_attention
        if n_state <= 0:
            return None

        return MambaStateInfo(
            per_seq_state_bytes=int(per_slot * n_state),
            n_state_layers=n_state,
            n_full_attention_layers=n_full_attention,
            n_total_layers=n_total,
            default_max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
        )
    except Exception:
        # Not debug: this drops both the state reservation and the KV correction.
        logger.warning(
            "preflight '%s': mamba-state resolution failed; hybrid sizing will be skipped", config.name, exc_info=True
        )
        return None
    finally:
        for k, v in prev_offline.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _correct_kv_for_hybrid(kv_per_token: float, mamba: MambaStateInfo) -> float:
    """`_kv_bytes_per_token` counts all layers; on a hybrid only the
    full-attention layers hold a token-growing KV cache, so scale it down."""
    if mamba.n_total_layers <= 0:
        return kv_per_token
    return kv_per_token * mamba.n_full_attention_layers / mamba.n_total_layers


def _apply_hybrid_fit(
    config_name: str,
    kv_pool: float,
    per_seq_state: int,
    kv_per_token: float,
    target_len: int | None,
    user_max_num_seqs: int | None,
    default_max_num_seqs: int,
) -> dict[str, Any]:
    """Device-agnostic fit ladder for hybrid models. `kv_pool` is bytes
    available for KV cache + mamba state; protects max_model_len, uses
    max_num_seqs as the shock absorber."""

    def fit_len(budget: float) -> int:
        tokens = int(budget // kv_per_token) if kv_per_token > 0 else 0
        aligned = (tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE
        if target_len:
            aligned = min(aligned, target_len)
        return aligned

    # User pinned max_num_seqs: honor it, size context around the resulting
    # mandatory state reservation.
    if user_max_num_seqs is not None:
        budget = kv_pool - per_seq_state * user_max_num_seqs
        suggested = fit_len(budget) if budget > 0 else 0
        if suggested < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': hybrid state at max_num_seqs=%d (%.2f GiB) leaves no room for a "
                "minimum context in the %.2f GiB KV pool; deploy will be attempted anyway.",
                config_name,
                user_max_num_seqs,
                per_seq_state * user_max_num_seqs / 1024**3,
                kv_pool / 1024**3,
            )
            return {}
        logger.info(
            "preflight '%s': hybrid, user max_num_seqs=%d (state %.2f GiB) → max_model_len=%d",
            config_name,
            user_max_num_seqs,
            per_seq_state * user_max_num_seqs / 1024**3,
            suggested,
        )
        return {"max_model_len": suggested}

    # Auto: floor concurrency so state memory is minimal, then protect context.
    budget_at_floor = kv_pool - per_seq_state * _MIN_MAX_NUM_SEQS
    if budget_at_floor <= 0:
        logger.warning(
            "preflight '%s': hybrid state at the floor of %d seqs (%.2f GiB) exceeds the %.2f GiB "
            "KV pool; deploy will be attempted anyway.",
            config_name,
            _MIN_MAX_NUM_SEQS,
            per_seq_state * _MIN_MAX_NUM_SEQS / 1024**3,
            kv_pool / 1024**3,
        )
        return {}

    if target_len and budget_at_floor >= target_len * kv_per_token:
        chosen_len = target_len  # full capability preserved
    else:
        chosen_len = fit_len(budget_at_floor)
        if chosen_len < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': hybrid budget yields max_model_len=%d (< block_size); "
                "deploy will be attempted anyway.",
                config_name,
                chosen_len,
            )
            return {}
        logger.info(
            "preflight '%s': hybrid, trimming context to max_model_len=%d to fit the %.2f GiB KV pool "
            "at floor concurrency (reduces the max-context contract).",
            config_name,
            chosen_len,
            kv_pool / 1024**3,
        )

    # Spend leftover budget on concurrency, capped at vLLM's own default.
    leftover = budget_at_floor - chosen_len * kv_per_token
    extra = int(leftover // per_seq_state) if per_seq_state > 0 else 0
    seqs = min(default_max_num_seqs, _MIN_MAX_NUM_SEQS + max(extra, 0))
    logger.info(
        "preflight '%s': hybrid → max_num_seqs=%d (floor %d + %d from %.2f GiB surplus; per-seq state %.1f MiB)",
        config_name,
        seqs,
        _MIN_MAX_NUM_SEQS,
        seqs - _MIN_MAX_NUM_SEQS,
        leftover / 1024**3,
        per_seq_state / 1024**2,
    )
    return {"max_model_len": chosen_len, "max_num_seqs": seqs}
