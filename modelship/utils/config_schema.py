"""Pydantic schemas for models.yaml. Must stay ray-free — ray latches
RAY_AUTH_MODE at import, before resolve_ray_auth_env() runs.
"""

import hashlib
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator, model_validator

from modelship.logging import get_logger
from modelship.openai.protocol.realtime import ReasoningEffort

if TYPE_CHECKING:
    # Annotation only; a real import pulls huggingface_hub.
    from modelship.utils.sources import PinnedSource

_logger = get_logger("config")


# Hex chars of the per-deployment fingerprint suffix; 10 = 40 bits.
FINGERPRINT_LEN = 10

# `{gateway}.{model}-{fingerprint}`; gateway names never contain '.'.
_DEPLOYMENT_NAME_RE = re.compile(rf"([^.]+)\.(.+)-[0-9a-f]{{{FINGERPRINT_LEN}}}")
# `{gateway}.{pipeline}-p{fingerprint}`, a pipeline's app; the `p` means neither pattern matches the other's names.
_PIPELINE_APP_NAME_RE = re.compile(rf"([^.]+)\.(.+)-p[0-9a-f]{{{FINGERPRINT_LEN}}}")

# Excluded from the fingerprint: `name` is already in the deployment name, and the
# replica-count fields are applied to the app while it runs.
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
    # an assistant pipeline's; no loader serves it yet
    realtime = "realtime"


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


class ReasoningConfig(_StrictModel):
    reasoning_start_str: str | None = Field(default=None, min_length=1)
    reasoning_end_str: str | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def reject_reasoning_parser(cls, data):
        if isinstance(data, dict) and "reasoning_parser" in data:
            raise ValueError(
                "vllm_engine_kwargs.reasoning_config.reasoning_parser cannot be set: it's always the model's "
                "reasoning parser. Set vllm_engine_kwargs.reasoning_parser instead."
            )
        return data


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
    default_chat_template_kwargs: dict[str, Any] | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
    reasoning_config: ReasoningConfig | None = None
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
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class LlamaServerConfig(_StrictModel):
    """Tunables for the ``llama_server`` loader, which drives a `llama-server`
    subprocess over its native OpenAI-compatible HTTP API."""

    n_ctx: int = Field(default=2048, ge=1)
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
    cache_ram_mib: int = Field(default=0, ge=-1)
    ubatch_size: int = Field(default=512, ge=1)
    flash_attn: Literal["on", "off", "auto"] = "auto"
    cache_type_k: Literal["f32", "f16", "bf16", "f8_e4m3", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] = "f16"
    cache_type_v: Literal["f32", "f16", "bf16", "f8_e4m3", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] = "f16"
    # Proportional split across GPUs for offloading (`-ts`); None splits evenly.
    tensor_split: list[float] | None = Field(default=None, min_length=1)
    # `pattern=buffer` entries for `-ot`, e.g. `exps=CPU`.
    override_tensor: list[str] | None = Field(default=None, min_length=1)
    reasoning: Literal["on", "off", "auto"] | None = None
    reasoning_budget: int | None = Field(default=None, ge=0)
    reasoning_budget_message: str | None = Field(default=None, min_length=1)
    reasoning_preserve: bool | None = None
    # llama-server fetches http(s) media URLs itself; unset, it refuses local and private addresses.
    media_allow_private_urls: bool = False

    @field_validator("flash_attn", "reasoning", mode="before")
    @classmethod
    def words_for_yaml_booleans(cls, value):
        if isinstance(value, bool):
            return "on" if value else "off"
        return value


class WhispercppConfig(_StrictModel):
    """Tunables for the ``whispercpp`` loader, which runs whisper.cpp in-process
    via `pywhispercpp` bindings (no subprocess)."""

    # Unset: the CPUs the deploy reserves, at most one per physical core; below one CPU, whisper.cpp's min(4, cores).
    n_threads: int | None = None
    # whisper.cpp's own default.
    flash_attn: bool = True


class SherpaOnnxConfig(_StrictModel):
    """Tunables for the ``sherpa_onnx`` loader."""

    # Voice for a request that names none or an unknown one; unset, the bundle's first voice.
    default_voice: str | None = None


class StableDiffusionCppConfig(_StrictModel):
    """Tunables for the ``stable_diffusion_cpp`` image loader, which runs an
    ``sd-server`` subprocess (stable-diffusion.cpp). Each field maps to one sd-server flag."""

    sample_steps: int = Field(default=20, ge=1)
    cfg_scale: float = 7.0
    # Distilled guidance (`--guidance`), for models with a guidance input such as Flux; None keeps sd.cpp's.
    guidance: float | None = None
    sample_method: str = "default"
    scheduler: str = "default"
    wtype: str = "default"
    # -1 lets sd.cpp pick.
    n_threads: int = Field(default=-1, ge=-1)
    vae_tiling: bool = False
    # `--flash-attn`: on, off, auto, or module assignments such as `auto,vae=off`; None keeps sd.cpp's `auto`.
    flash_attn: str | None = Field(default=None, min_length=1)
    clip_on_cpu: bool = False
    vae_on_cpu: bool = False
    offload_to_cpu: bool = False
    # GPU budget in GiB (`--max-vram`) and `--auto-fit`; preflight sets both for a fractional `num_gpus`.
    max_vram: float | None = Field(default=None, gt=0)
    auto_fit: bool | None = None
    # Split checkpoints: pre-placed local paths only, since source resolution
    # handles single-file models.
    diffusion_model_path: str | None = None
    clip_l_path: str | None = None
    clip_g_path: str | None = None
    t5xxl_path: str | None = None
    vae_path: str | None = None

    @field_validator("flash_attn", mode="before")
    @classmethod
    def words_for_yaml_booleans(cls, value):
        if isinstance(value, bool):
            return "on" if value else "off"
        return value


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
    # Ray label selector merged over the loader's: {key: value | !value | in(a,b) | !in(a,b)}.
    node_selector: dict[str, str] = Field(default_factory=dict)
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    diffusers_config: DiffusersConfig | None = None
    llama_server_config: LlamaServerConfig | None = None
    stable_diffusion_cpp_config: StableDiffusionCppConfig | None = None
    whispercpp_config: WhispercppConfig | None = None
    sherpa_onnx_config: SherpaOnnxConfig | None = None

    # Checked by `resolve_all_model_sources`, downloaded into
    # `_resolved_path` by `BaseInfer.ensure_downloaded`.
    _pinned_source: "PinnedSource | None" = PrivateAttr(default=None)
    _resolved_path: str | None = PrivateAttr(default=None)
    # Silero VAD, for a sherpa_onnx speech-to-text model that cuts long audio at silences; set like the two above
    _pinned_vad: "PinnedSource | None" = PrivateAttr(default=None)
    _resolved_vad: str | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def default_diffusers_usecase(cls, data):
        # The image-only loaders may omit `usecase`; an explicit non-image value
        # is still rejected below.
        image_loaders = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
        if isinstance(data, dict) and data.get("loader") in image_loaders and data.get("usecase") is None:
            data = {**data, "usecase": ModelUsecase.image}
        return data

    @field_validator("node_selector")
    @classmethod
    def sort_node_selector(cls, value: dict[str, str]) -> dict[str, str]:
        return sorted_node_selector(value)

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
    def check_replicas_share_a_context_length(self):
        # Unset, each replica fits its own context length to the node it lands on.
        if self.num_replicas == 1 and self.autoscaling_config is None:
            return self
        if self.loader == ModelLoader.vllm:
            key = "vllm_engine_kwargs.max_model_len"
            fixed = self.vllm_engine_kwargs.max_model_len is not None
        elif self.loader == ModelLoader.llama_server:
            key = "llama_server_config.n_ctx"
            server = self.llama_server_config
            fixed = server is not None and "n_ctx" in server.model_fields_set
        else:
            return self
        if not fixed:
            raise ValueError(
                f"model '{self.name}': num_replicas above 1 or autoscaling_config needs {key} set, "
                f"so every replica serves the same context length."
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
    def check_stable_diffusion_cpp_uses_one_gpu(self):
        if self.loader == ModelLoader.stable_diffusion_cpp and self.num_gpus > 1:
            raise ValueError(
                f"num_gpus={self.num_gpus!r} is not allowed for the stable_diffusion_cpp loader: "
                f"it runs on one GPU, so use 1, or a fraction < 1 to share it."
            )
        return self

    @model_validator(mode="after")
    def check_name_has_no_hash(self):
        # Serve rejects '#' in a deployment name, which embeds the model name.
        if "#" in self.name:
            raise ValueError(f"name {self.name!r} must not contain '#'")
        return self

    @model_validator(mode="after")
    def check_usecase_has_a_loader(self):
        if self.usecase is ModelUsecase.realtime:
            raise ValueError(
                f"model '{self.name}': no loader serves usecase 'realtime'; an assistant pipeline serves it."
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
        from modelship.infer.sherpa_onnx.registry import REGISTRY, registry_name, registry_names

        assert self.model is not None  # enforced by check_model_required above
        name = registry_name(self.model)
        names = registry_names()
        if name not in names:
            raise ValueError(
                f"model '{self.name}': sherpa_onnx model {self.model!r} is not a supported registry name "
                f"(or a local directory whose basename matches one). Supported names: {', '.join(names)}"
            )
        entry = REGISTRY[name]
        if self.usecase.value != entry.usecase:
            raise ValueError(
                f"model '{self.name}': sherpa_onnx model {name!r} is a {entry.usecase} model; set "
                f"usecase: {entry.usecase} (got {self.usecase.value!r})."
            )
        voice = self.sherpa_onnx_config.default_voice if self.sherpa_onnx_config else None
        voices = getattr(entry, "voice_names", ())
        if voice is not None and not voices:
            raise ValueError(f"model '{self.name}': sherpa_onnx_config.default_voice is for a tts model, not {name!r}.")
        if voice is not None and voice not in voices:
            raise ValueError(
                f"model '{self.name}': sherpa_onnx_config.default_voice {voice!r} is not a voice of {name!r}. "
                f"Voices: {', '.join(voices)}"
            )
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

        tp = self.vllm_engine_kwargs.tensor_parallel_size
        pp = self.vllm_engine_kwargs.pipeline_parallel_size
        world_size = tp * pp

        if ng <= 0:
            if world_size > 1 and "num_gpus" not in self.model_fields_set:
                self.num_gpus = 1.0
            return self

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

    def scaling(self) -> dict:
        return {"num_replicas": self.num_replicas, "autoscaling_config": self.autoscaling_config}


def parse_deployment_name(app_name: str) -> tuple[str, str] | None:
    """(gateway, model name) of an app named by `deployment_name`, else None."""
    match = _DEPLOYMENT_NAME_RE.fullmatch(app_name)
    return (match[1], match[2]) if match else None


def parse_pipeline_app_name(app_name: str) -> tuple[str, str] | None:
    """(gateway, pipeline name) of an app named by `PipelineConfig.deployment_name`, else None."""
    match = _PIPELINE_APP_NAME_RE.fullmatch(app_name)
    return (match[1], match[2]) if match else None


def parse_gateway_app_name(app_name: str) -> tuple[str, str] | None:
    """(gateway, model or pipeline name) of a model app or a pipeline's app, else None."""
    return parse_deployment_name(app_name) or parse_pipeline_app_name(app_name)


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


class SessionReasoning(_StrictModel):
    effort: ReasoningEffort | None = None


class SessionServerVad(_StrictModel):
    """Server VAD, with OpenAI's defaults."""

    type: Literal["server_vad"]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    prefix_padding_ms: int = Field(default=300, ge=0, le=5000)
    silence_duration_ms: int = Field(default=500, ge=0, le=10000)
    create_response: bool = True
    interrupt_response: bool = True


class SessionAudioInput(_StrictModel):
    # null: the client commits each turn; unset: server VAD with its defaults
    turn_detection: SessionServerVad | None = None


class SessionAudio(_StrictModel):
    input: SessionAudioInput | None = None


class SessionDefaults(_StrictModel):
    """A pipeline's defaults for the OpenAI Realtime session fields, in the session's
    own shape; a client's `session.update` overrides any of them."""

    instructions: str | None = None
    # [text] or [audio]
    output_modalities: Annotated[list[Literal["text", "audio"]], Field(min_length=1, max_length=1)] | None = None
    max_output_tokens: Annotated[int, Field(ge=1)] | Literal["inf"] | None = None
    reasoning: SessionReasoning | None = None
    audio: SessionAudio | None = None


def sorted_node_selector(value: dict[str, str]) -> dict[str, str]:
    """A model's or a pipeline's `node_selector`, sorted: key order means nothing to Ray, so it leaves the
    fingerprint alone."""
    return dict(sorted(value.items()))


class PipelineConfig(_StrictModel):
    """What every pipeline type has: a pipeline is its own app, chaining models of its gateway. Experimental."""

    # the usecase the type serves, as a model's `usecase`
    USECASE: ClassVar[ModelUsecase]

    name: str
    num_replicas: int = Field(default=1, ge=1)
    # Ray label selector merged over the pipeline's, as a model's `node_selector`
    node_selector: dict[str, str] = Field(default_factory=dict)

    @field_validator("node_selector")
    @classmethod
    def sort_node_selector(cls, value: dict[str, str]) -> dict[str, str]:
        return sorted_node_selector(value)

    @model_validator(mode="after")
    def check_name_has_no_hash(self):
        # Serve rejects '#' in a deployment name, as for a model's.
        if "#" in self.name:
            raise ValueError(f"name {self.name!r} must not contain '#'")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump_json(exclude={"name", "num_replicas"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]

    def deployment_name(self, gateway_name: str) -> str:
        """Its app's name."""
        return f"{gateway_name}.{self.name}-p{self.fingerprint()}"

    def scaling(self) -> dict:
        return {"num_replicas": self.num_replicas}

    def models(self) -> list[str]:
        """The names of the models it calls."""
        raise NotImplementedError


# the microWakeWord models pymicro-wakeword ships
BUILTIN_WAKE_WORDS = ("okay_nabu", "hey_jarvis", "hey_mycroft", "alexa")


class WakeWord(_StrictModel):
    # a built-in word, or the path of a word's microWakeWord v2 manifest (.json), its .tflite beside it
    model: str
    # the score the word fires above; unset, its manifest's
    cutoff: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("model")
    @classmethod
    def check_builtin_or_manifest(cls, value: str) -> str:
        if value not in BUILTIN_WAKE_WORDS and not value.endswith(".json"):
            builtins = ", ".join(BUILTIN_WAKE_WORDS)
            raise ValueError(f"wake word {value!r} is neither a built-in word ({builtins}) nor a .json manifest")
        return value


class FollowUp(_StrictModel):
    # after an answer, the user talks again without the wake word: never, when the answer asks something, or always
    mode: Literal["off", "question", "always"] = "off"
    # how long after the answer ends a turn may start without the wake word
    listen_s: float = Field(default=10.0, gt=0, le=300)


class Wake(_StrictModel):
    """Wake words: a server VAD turn of a realtime session starts at one, or in a follow-up's window."""

    words: Annotated[list[WakeWord], Field(min_length=1)]
    # how long after the wake word the user has to start talking
    listen_s: float = Field(default=10.0, gt=0, le=300)
    follow_up: FollowUp = Field(default_factory=FollowUp)

    @field_validator("words", mode="before")
    @classmethod
    def accept_names(cls, value: Any) -> Any:
        # a word given by its model alone
        if isinstance(value, list):
            return [{"model": word} if isinstance(word, str) else word for word in value]
        return value


class AssistantConfig(_StrictModel):
    """An assistant's own settings, as a model's loader keeps its settings in `<loader>_config`."""

    # model names, from the gateway's models
    stt: str | None = None
    llm: str | None = None
    tts: str | None = None
    session_defaults: SessionDefaults | None = None
    # full: the user can talk over an answer, as on a device that cancels its own echo; half: the audio a session gets
    # while its answer plays is dropped, so a speaker's echo is never heard as the user
    duplex: Literal["full", "half"] = "full"
    # while an answer plays, how long speech lasts before it starts a turn and cuts the answer off
    min_interruption_ms: int = Field(default=500, ge=0, le=5000)
    wake: Wake | None = None


class AssistantPipelineConfig(PipelineConfig):
    """`type: assistant`, a `/v1/realtime` session over speech to text, an LLM and text to speech, each optional
    but the first two not both."""

    USECASE: ClassVar[ModelUsecase] = ModelUsecase.realtime

    type: Literal["assistant"]
    assistant_config: AssistantConfig

    @model_validator(mode="after")
    def check_stt_or_llm(self):
        if self.assistant_config.stt is None and self.assistant_config.llm is None:
            raise ValueError(
                f"pipeline {self.name!r} names no stt and no llm model in assistant_config: set at least one."
            )
        return self

    @model_validator(mode="after")
    def check_audio_output_has_tts(self):
        assistant = self.assistant_config
        modalities = assistant.session_defaults.output_modalities if assistant.session_defaults is not None else None
        if modalities is not None and "audio" in modalities and assistant.tts is None:
            raise ValueError(
                f"pipeline {self.name!r}: assistant_config.session_defaults.output_modalities asks for audio, but it"
                " names no tts model to speak with."
            )
        return self

    @model_validator(mode="after")
    def check_turn_detection_has_stt(self):
        defaults = self.assistant_config.session_defaults
        given = defaults.audio.input if defaults is not None and defaults.audio is not None else None
        if given is not None and given.turn_detection is not None and self.assistant_config.stt is None:
            raise ValueError(
                f"pipeline {self.name!r}: assistant_config.session_defaults.audio.input.turn_detection needs an stt"
                " model to take audio."
            )
        return self

    @model_validator(mode="after")
    def check_wake_has_stt(self):
        if self.assistant_config.wake is not None and self.assistant_config.stt is None:
            raise ValueError(f"pipeline {self.name!r}: assistant_config.wake needs an stt model to hear the user.")
        return self

    def models(self) -> list[str]:
        assistant = self.assistant_config
        return [model for model in (assistant.stt, assistant.llm, assistant.tts) if model is not None]


# every pipeline type; `type` tells them apart once there is more than one
AnyPipelineConfig = AssistantPipelineConfig
PIPELINE_TYPES: tuple[type[PipelineConfig], ...] = (AssistantPipelineConfig,)
_PIPELINE_ADAPTER: TypeAdapter[AnyPipelineConfig] = TypeAdapter(AnyPipelineConfig)


def parse_pipeline(raw: Any) -> PipelineConfig:
    """The pipeline config of its type that *raw* describes."""
    return _PIPELINE_ADAPTER.validate_python(raw)


class ModelshipConfig(_StrictModel):
    models: list[ModelshipModelConfig]
    pipelines: list[AnyPipelineConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_unique_pipeline_names(self):
        names = [pipeline.name for pipeline in self.pipelines]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"duplicate pipeline name {duplicates[0]!r}: a pipeline name maps to one pipeline.")
        return self

    @model_validator(mode="after")
    def check_pipeline_names_differ_from_models(self):
        # /v1/models lists both
        models = {cfg.name for cfg in self.models}
        if clashes := sorted(pipeline.name for pipeline in self.pipelines if pipeline.name in models):
            raise ValueError(f"pipeline {clashes[0]!r} has the name of a model: give one of them another name.")
        return self

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
