"""Pydantic schemas for models.yaml. Must stay ray-free — ray latches
RAY_AUTH_MODE at import, before resolve_ray_auth_env() runs.
"""

import hashlib
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from modelship.logging import get_logger

if TYPE_CHECKING:
    # Annotation only; a real import pulls huggingface_hub.
    from modelship.infer.sources import PinnedSource

_logger = get_logger("config")


# Hex chars of the per-deployment fingerprint suffix; 10 = 40 bits.
FINGERPRINT_LEN = 10

# `{gateway}.{model}-{fingerprint}`; gateway names never contain '.'.
_DEPLOYMENT_NAME_RE = re.compile(rf"([^.]+)\.(.+)-[0-9a-f]{{{FINGERPRINT_LEN}}}")

# Excluded from the fingerprint: `name` is already in the deployment name, and Ray Serve
# updates the replica-count fields in place when serve.run() re-binds an app.
_FINGERPRINT_EXCLUDED_FIELDS = {"name", "num_replicas", "autoscaling_config"}

# vLLM's own default.
_VLLM_GPU_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
# vLLM's CPU backend reads gpu_memory_utilization as a fraction of HOST RAM
# rather than VRAM, so num_gpus == 0 takes a lower one.
_VLLM_CPU_DEFAULT_GPU_MEMORY_UTILIZATION = 0.4

ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


class ModelUsecase(StrEnum):
    generate = "generate"
    embed = "embed"
    transcription = "transcription"
    translation = "translation"
    tts = "tts"
    image = "image"


class ModelLoader(StrEnum):
    vllm = "vllm"
    diffusers = "diffusers"
    llama_server = "llama_server"
    stable_diffusion_cpp = "stable_diffusion_cpp"
    whispercpp = "whispercpp"
    sherpa_onnx = "sherpa_onnx"


# Derived by modelship, so not fields at all; named here for a better message
# than extra="forbid" gives an unknown key.
_VLLM_DERIVED_KEYS = {
    "model": "the engine always loads the resolved top-level `model:` source. Set that instead.",
    "gpu_memory_utilization": (
        "it's always derived from num_gpus, so Ray's schedule and vLLM's actual VRAM "
        "allocation can never disagree. Set num_gpus instead."
    ),
}


class _StrictModel(BaseModel):
    """Base for every models.yaml schema: an unknown key is an error, not a
    silently ignored typo."""

    model_config = ConfigDict(extra="forbid")


class VllmEngineConfig(_StrictModel):
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # Unset means vLLM fits the context to memory; the sentinel for that is derived.
    max_model_len: int | None = Field(default=None, ge=1)
    dtype: str = "auto"
    tokenizer: str | None = None
    trust_remote_code: bool = False
    enable_log_requests: bool | None = False
    disable_log_stats: bool | None = False
    kv_cache_dtype: str | None = None
    quantization: str | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    enable_reasoning: bool | None = None
    reasoning_parser: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    enforce_eager: bool | None = None
    enable_prefix_caching: bool | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    limit_mm_per_prompt: dict[str, int | dict[str, int]] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_derived_keys(cls, data):
        if isinstance(data, dict):
            for key, reason in _VLLM_DERIVED_KEYS.items():
                if key in data:
                    raise ValueError(f"vllm_engine_kwargs.{key} cannot be set: {reason}")
        return data


class DiffusersConfig(_StrictModel):
    torch_dtype: str = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class LlamaServerConfig(_StrictModel):
    """Tunables for the ``llama_server`` loader, which drives a `llama-server`
    subprocess over its native OpenAI-compatible HTTP API."""

    n_ctx: int = 2048
    n_batch: int = 512
    # Preflight recommends a concrete count, so this default rarely launches; any
    # negative value means llama-server's own auto-fit to free device memory.
    n_gpu_layers: int = -1
    # Preflight recommends num_cpus here when the deploy reserves whole CPUs.
    threads: int | None = None
    # The process is launched with `n_ctx * parallel`: llama-server splits its
    # total context across slots.
    parallel: int = Field(default=1, ge=1)
    chat_template: str | None = None
    mmproj: str | None = None
    # Checked by `resolve_all_model_sources`, downloaded by
    # `BaseInfer.ensure_downloaded`, which overwrites `mmproj` above.
    _pinned_mmproj: "PinnedSource | None" = PrivateAttr(default=None)
    cache_reuse: int = Field(default=0, ge=0)
    context_shift: bool = False
    cache_ram_mib: int | None = Field(default=None, ge=-1)
    ubatch_size: int = Field(default=512, ge=1)
    flash_attn: Literal["on", "off", "auto"] = "auto"
    cache_type_k: Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] = "f16"
    cache_type_v: Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] = "f16"
    # Proportional split across GPUs for offloading (`-ts`); None splits evenly.
    tensor_split: list[float] | None = Field(default=None, min_length=1)


class WhispercppConfig(_StrictModel):
    """Tunables for the ``whispercpp`` loader, which runs whisper.cpp in-process
    via `pywhispercpp` bindings (no subprocess)."""

    n_threads: int | None = None
    flash_attn: bool = False


class StableDiffusionCppConfig(_StrictModel):
    """Tunables for the CPU-only `stable_diffusion_cpp` image loader
    (stable-diffusion.cpp via stable-diffusion-cpp-python)."""

    sample_steps: int = 20
    cfg_scale: float = 7.0
    sample_method: str = "default"
    scheduler: str = "default"
    wtype: str = "default"
    n_threads: int = -1
    vae_tiling: bool = False
    # Split checkpoints: pre-placed local paths only, since source resolution
    # handles single-file models.
    diffusion_model_path: str | None = None
    clip_l_path: str | None = None
    clip_g_path: str | None = None
    t5xxl_path: str | None = None
    vae_path: str | None = None
    # Forwarded verbatim to the StableDiffusion constructor.
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class AutoscalingConfig(_StrictModel):
    """The subset of Ray Serve's ``autoscaling_config`` modelship surfaces. When
    set, load drives the replica count between ``min_replicas`` and
    ``max_replicas`` instead of the fixed ``num_replicas``."""

    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=1, ge=1)
    initial_replicas: int | None = Field(default=None, ge=0)
    target_ongoing_requests: float | None = Field(default=None, gt=0)
    upscale_delay_s: float | None = Field(default=None, ge=0)
    downscale_delay_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_bounds(self):
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"autoscaling_config: max_replicas ({self.max_replicas}) must be >= min_replicas ({self.min_replicas})."
            )
        if self.initial_replicas is not None and not (self.min_replicas <= self.initial_replicas <= self.max_replicas):
            raise ValueError(
                f"autoscaling_config: initial_replicas ({self.initial_replicas}) must be "
                f"within [min_replicas={self.min_replicas}, max_replicas={self.max_replicas}]."
            )
        return self

    def to_serve_dict(self) -> dict[str, Any]:
        """The kwargs Ray Serve's ``.options(autoscaling_config=...)`` expects.
        Unset (None) tunables are omitted so Serve applies its own defaults."""
        out: dict[str, Any] = {"min_replicas": self.min_replicas, "max_replicas": self.max_replicas}
        for key in ("initial_replicas", "target_ongoing_requests", "upscale_delay_s", "downscale_delay_s"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


class ModelshipModelConfig(_StrictModel):
    name: str
    model: str | None = None
    usecase: ModelUsecase
    loader: ModelLoader
    # Ray resource reservations: a negative value is meaningless to Ray and, for
    # num_gpus, reads as "not fractional" everywhere downstream.
    num_gpus: float = Field(default=0, ge=0)
    num_cpus: float = Field(default=0.1, ge=0)
    # Serve rejects < 1 deep in deployment; same for max_ongoing_requests.
    num_replicas: int = Field(default=1, ge=1)
    # Load-driven replica scaling; mutually exclusive with the fixed num_replicas.
    autoscaling_config: AutoscalingConfig | None = None
    max_ongoing_requests: int | None = Field(default=None, ge=1)
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    diffusers_config: DiffusersConfig | None = None
    llama_server_config: LlamaServerConfig | None = None
    stable_diffusion_cpp_config: StableDiffusionCppConfig | None = None
    whispercpp_config: WhispercppConfig | None = None
    # Forwarded into the chat-template Jinja render on every text loader; only
    # does something if the model's template branches on the key.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Checked by `resolve_all_model_sources`, downloaded into
    # `_resolved_path` by `BaseInfer.ensure_downloaded`.
    _pinned_source: "PinnedSource | None" = PrivateAttr(default=None)
    _resolved_path: str | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def default_diffusers_usecase(cls, data):
        # The image-only loaders may omit `usecase`; an explicit non-image value
        # is still rejected below.
        image_loaders = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
        if isinstance(data, dict) and data.get("loader") in image_loaders and data.get("usecase") is None:
            data = {**data, "usecase": ModelUsecase.image}
        return data

    @model_validator(mode="after")
    def check_autoscaling_excludes_num_replicas(self):
        # Ray Serve rejects both at once; catch it here rather than deep in
        # serve.run(). An untouched default num_replicas is fine.
        if self.autoscaling_config is not None and "num_replicas" in self.model_fields_set:
            raise ValueError(
                f"model '{self.name}': set either num_replicas or autoscaling_config, not both. "
                f"num_replicas pins a fixed replica count; autoscaling_config scales between "
                f"min_replicas and max_replicas on load."
            )
        return self

    @model_validator(mode="after")
    def validate_whole_gpu_only_loaders_num_gpus(self):
        # sherpa_onnx is exempt: it never touches CUDA.
        whole_gpu_loaders = (ModelLoader.llama_server, ModelLoader.whispercpp)
        if self.loader in whole_gpu_loaders and self.num_gpus >= 1 and self.num_gpus != int(self.num_gpus):
            raise ValueError(
                f"num_gpus={self.num_gpus!r} is not allowed for the {self.loader.value} loader: "
                f"use a fraction < 1 to share one GPU, or a whole integer number of GPUs."
            )
        return self

    @model_validator(mode="after")
    def check_model_required(self):
        if not self.model:
            raise ValueError(f"`model:` is required for loader={self.loader!r}")
        if self.loader in (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp) and (
            self.usecase is not ModelUsecase.image
        ):
            raise ValueError(f"loader={self.loader.value!r} only supports usecase='image', got {self.usecase!r}")
        return self

    @model_validator(mode="after")
    def check_sherpa_onnx_model_and_usecase(self):
        # `model:` must be a registry name, or a local dir named for one.
        if self.loader != ModelLoader.sherpa_onnx:
            return self
        from modelship.infer.sherpa_onnx.registry import registry_name, registry_names

        assert self.model is not None  # enforced by check_model_required above
        name = registry_name(self.model)
        names = registry_names()
        if name not in names:
            raise ValueError(
                f"model '{self.name}': sherpa_onnx model {self.model!r} is not a supported registry name "
                f"(or a local directory whose basename matches one). Supported names: {', '.join(names)}"
            )
        if self.usecase is not ModelUsecase.tts:
            raise ValueError(f"loader='sherpa_onnx' only supports usecase='tts' (v1 scope), got {self.usecase!r}")
        return self

    @model_validator(mode="after")
    def normalize_num_gpus_and_tp(self):
        """Enforce the num_gpus / tensor_parallel semantics for vLLM.

        - num_gpus < 1: one shared GPU, tp=pp=1 only — Ray packs fractional
          placement-group bundles onto the same physical GPU.
        - num_gpus >= 1: whole GPUs only, and tp x pp implies the count when
          either is set; tp is derived from num_gpus when neither is.
        """
        ng = self.num_gpus
        if self.loader != ModelLoader.vllm:
            return self

        if ng <= 0:
            return self

        tp = self.vllm_engine_kwargs.tensor_parallel_size
        pp = self.vllm_engine_kwargs.pipeline_parallel_size
        world_size = tp * pp

        if 0 < ng < 1:
            if world_size > 1:
                raise ValueError(
                    f"num_gpus={ng!r} (fractional) is not compatible with "
                    f"tensor_parallel_size x pipeline_parallel_size > 1 "
                    f"(got {tp} x {pp}). Ray packs fractional placement-group "
                    f"bundles onto the same physical GPU, which breaks tensor "
                    f"parallelism. Use whole GPUs for multi-slot deploys "
                    f"(e.g. num_gpus={world_size}) or drop the parallelism "
                    f"settings to share a single GPU."
                )
            return self

        # ng >= 1: integer-only.
        if ng != int(ng):
            raise ValueError(
                f"num_gpus={ng!r} is not allowed: values >= 1 must be integers. "
                f"Use a fractional value < 1 to share a single GPU, or an integer "
                f"to request that many whole GPUs."
            )
        ng_int = int(ng)

        if world_size > 1:
            if ng_int != world_size:
                raise ValueError(
                    f"num_gpus={ng_int} does not match tensor_parallel_size x "
                    f"pipeline_parallel_size={tp} x {pp}={world_size}. Either drop "
                    f"num_gpus (it's derived from tp x pp) or set num_gpus={world_size}."
                )
            if "num_gpus" in self.model_fields_set:
                _logger.warning(
                    "num_gpus=%d is redundant for model '%s': it matches "
                    "tensor_parallel_size x pipeline_parallel_size=%d, which "
                    "already determines the GPU count. Safe to drop.",
                    ng_int,
                    self.name,
                    world_size,
                )
        else:
            # tp=pp=1: auto-derive tp from num_gpus.
            self.vllm_engine_kwargs.tensor_parallel_size = ng_int

        self.num_gpus = 1.0
        return self

    def fingerprint(self) -> str:
        """Stable hash of the fields that drive placement/runtime, used as the
        deployment-name suffix so reconcile detects drift by name."""
        payload = self.model_dump_json(exclude=_FINGERPRINT_EXCLUDED_FIELDS)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]

    def deployment_name(self, gateway_name: str) -> str:
        return f"{gateway_name}.{self.name}-{self.fingerprint()}"


def parse_deployment_name(app_name: str) -> tuple[str, str] | None:
    """(gateway, model name) of an app named by `deployment_name`, else None."""
    match = _DEPLOYMENT_NAME_RE.fullmatch(app_name)
    return (match[1], match[2]) if match else None


def resolve_gpu_memory_utilization(config: ModelshipModelConfig, recommended: float | None = None) -> float:
    """The VRAM fraction vLLM may claim (host RAM on a CPU deploy). Precedence: a
    fractional num_gpus, the share Ray reserved, then preflight, then the default."""
    if 0 < config.num_gpus < 1:
        return config.num_gpus
    return recommended if recommended is not None else default_gpu_memory_utilization(config)


def default_gpu_memory_utilization(config: ModelshipModelConfig) -> float:
    """Last fallback for gpu_memory_utilization: 0.9 on GPU, 0.4 on a CPU deploy."""
    if config.num_gpus == 0:
        return _VLLM_CPU_DEFAULT_GPU_MEMORY_UTILIZATION
    return _VLLM_GPU_DEFAULT_GPU_MEMORY_UTILIZATION


class ModelshipConfig(_StrictModel):
    models: list[ModelshipModelConfig]

    @model_validator(mode="after")
    def check_unique_names(self):
        """A model name maps to exactly one deployment: reject any name reused
        across entries, whether the repeated config is identical or not."""
        seen: dict[str, str] = {}
        for cfg in self.models:
            fp = cfg.fingerprint()
            prior = seen.get(cfg.name)
            if prior is None:
                seen[cfg.name] = fp
            elif prior == fp:
                raise ValueError(
                    f"Duplicate model entries named {cfg.name!r} with identical config. "
                    f"For multiple identical replicas, use num_replicas instead."
                )
            else:
                raise ValueError(f"duplicate model name {cfg.name!r}: a model name maps to exactly one deployment.")
        return self
