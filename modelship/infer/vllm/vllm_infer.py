import asyncio
import io
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, ClassVar, cast

from fastapi import UploadFile
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response
from vllm.config.model import ModelDType as VllmModelDType
from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs
from vllm.entrypoints.chat_utils import ChatTemplateConfig as VllmChatTemplateConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as VllmErrorResponse,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath as VllmBaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels as VllmOpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest as VllmEmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as VllmServingEmbedding
from vllm.entrypoints.serve.utils.request_logger import RequestLogger as VllmRequestLogger
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest as VllmTranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse as VllmTranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseVerbose as VllmTranscriptionResponseVerbose,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription as VllmOpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationRequest as VllmTranslationRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponse as VllmTranslationResponse,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponseVerbose as VllmTranslationResponseVerbose,
)
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation as VllmOpenAIServingTranslation,
)
from vllm.exceptions import VLLMValidationError as VllmValidationError
from vllm.inputs import EngineInput as VllmEngineInput
from vllm.parser import Parser as VllmParser
from vllm.renderers.online_renderer import OnlineRenderer as VllmOnlineRenderer
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.tokenizers import TokenizerLike as VllmTokenizerLike
from vllm.usage.usage_lib import UsageContext as VllmUsageContext
from vllm.v1.engine.async_llm import AsyncLLM as VllmAsyncLLM

from modelship.infer.base_infer import MINIMAL_WAV, BaseInfer, ClientDisconnectedError
from modelship.infer.infer_config import (
    ModelshipModelConfig,
    ModelUsecase,
    RawRequestProxy,
    VllmEngineConfig,
    resolve_gpu_memory_utilization,
)
from modelship.infer.vllm import engine_ops
from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.parsing.detect import (
    detect_template_toggle_defaults,
    resolve_reasoning_parser,
    resolve_tool_parser,
)
from modelship.logging import TRACE, get_logger
from modelship.metrics import _ENABLED as _METRICS_ENABLED
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionTokenUsageInfo,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    PromptTokenUsageInfo,
    ResponseObject,
    ResponsesRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.responses.adapter import (
    UnsupportedResponsesFeatureError,
    responses_request_to_chat,
)
from modelship.openai.utils.chat import (
    ParsedChatOutput,
    UnsupportedContentError,
    build_from_parsed,
    encode_chat_sse_chunk,
    encode_error_sse,
    normalize_chat_messages,
)
from modelship.openai.utils.responses import build_response_from_parsed, responses_validation_error
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight
from modelship.utils import base_request_id

logger = get_logger("infer.vllm")

# vLLM's sentinel for "fit the context to what memory profiling leaves".
_AUTO_FIT_MAX_MODEL_LEN = -1


def resolve_max_model_len(engine_kwargs: VllmEngineConfig) -> int:
    """Derived, not a config key: the schema takes a real length or nothing."""
    return engine_kwargs.max_model_len or _AUTO_FIT_MAX_MODEL_LEN


def _to_error_response(
    err: VllmValidationError | VllmErrorResponse | Exception | str,
    *,
    err_type: str = "invalid_request_error",
    status_code: HTTPStatus | int = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    """The one place this loader converts any failure into a client-facing
    `ErrorResponse`, funnelling every shape through `create_error_response`.

    """
    if isinstance(err, VllmValidationError):
        base = err.args[0] if err.args else str(err)
        return create_error_response(
            message=base,
            err_type="invalid_request_error",
            status_code=HTTPStatus.BAD_REQUEST,
            param=err.parameter,
        )
    if isinstance(err, VllmErrorResponse):
        info = err.error
        return create_error_response(
            message=info.message,
            err_type=info.type,
            status_code=info.code,
            param=info.param,
        )
    return create_error_response(err, err_type=err_type, status_code=status_code)


def _vllm_stream_error(exc: Exception) -> str | None:
    """`client_error` mapper for `BaseInfer._stream_responses`: a mid-stream
    `VllmValidationError` is client-safe to relay verbatim; anything else falls
    through to the generic "Internal error during generation" message."""
    if isinstance(exc, VllmValidationError):
        base = exc.args[0] if exc.args else str(exc)
        return str(base)
    return None


def _deployment_summary(name: str, vllm_config: Any) -> list[str]:
    """What actually got deployed, read back from the engine-resolved config.
    Every number is vLLM's own post-profiling value, not a preflight estimate.
    The fields filled in during KV-cache init are `None` until then, never
    missing, so each is tested against `None` rather than truthiness."""
    mc, cc, sc, pc = (
        vllm_config.model_config,
        vllm_config.cache_config,
        vllm_config.scheduler_config,
        vllm_config.parallel_config,
    )
    ctx = mc.max_model_len
    # `original_max_model_len` keeps the pre-resolution request, so -1 marks a
    # context vLLM binary-searched against real free memory. Absent means we
    # don't know how it was chosen, so claim neither.
    requested_ctx = getattr(mc, "original_max_model_len", None)
    if requested_ctx == -1:
        origin = " (auto-fit to free memory)"
    elif requested_ctx is not None:
        origin = " (as requested)"
    else:
        origin = ""
    lines = [f"deployed '{name}':", f"  context      {ctx:,} tokens{origin}"]

    # 0.0 is a real answer: the pool cannot hold one full-length request.
    concurrency = getattr(cc, "kv_cache_max_concurrency", None)
    at_full = f" — the KV pool holds {concurrency:.1f} of them at that length" if concurrency is not None else ""
    lines.append(f"  concurrency  up to {sc.max_num_seqs} concurrent requests{at_full}")

    # `kv_cache_size_tokens` is the group-aware capacity. Don't pair it with
    # `num_gpu_blocks x block_size`: those are per-group, so on a model whose
    # requests span several groups (cross-attention, hybrid) the product is
    # unrelated to the capacity — Whisper reports 20,679 tokens against
    # 7,201 x 16.
    kv_tokens = getattr(cc, "kv_cache_size_tokens", None)
    if kv_tokens is not None:
        lines.append(f"  kv cache     {kv_tokens:,} tokens")

    # Two separate facts, both measured: a decode sequence pins one mamba block,
    # so max_num_seqs cannot exceed the count; and raising max_num_seqs grows the
    # cudagraph capture set, which shrinks the count itself.
    if getattr(mc, "is_hybrid", False) and cc.num_gpu_blocks is not None:
        lines.append(
            f"  hybrid       {cc.num_gpu_blocks:,} mamba blocks — max_num_seqs cannot exceed this, "
            "and raising it shrinks the count"
        )

    dtype = str(mc.dtype).removeprefix("torch.")
    quant = f", {mc.quantization}" if mc.quantization else ""
    parallel = f", tp={pc.tensor_parallel_size} pp={pc.pipeline_parallel_size}" if pc.world_size > 1 else ""
    lines.append(f"  execution    {dtype}{quant}, gpu_memory_utilization={cc.gpu_memory_utilization}{parallel}")
    return lines


def _trace_request(
    request_id: str, vllm_request: VllmChatCompletionRequest, sampling_params: VllmSamplingParams
) -> None:
    """TRACE-log the effective request: the generation budget and the
    `chat_template_kwargs` actually in play (e.g. `enable_thinking`), so a
    truncated-mid-reasoning failure can be tied back to what was sent."""
    if not logger.isEnabledFor(TRACE):
        return
    logger.log(
        TRACE,
        "%s request: max_tokens=%s chat_template_kwargs=%s messages=%s tools=%s",
        request_id,
        sampling_params.max_tokens,
        vllm_request.chat_template_kwargs,
        vllm_request.messages,
        vllm_request.tools,
    )


def _trace_parsed_response(
    request_id: str,
    choices: Sequence[ParsedChatOutput],
    finish_reasons: Sequence[str | None],
    usage: UsageInfo,
) -> None:
    """TRACE-log the parsed non-stream result: finish reason, usage, and both
    reasoning and content per choice — so an answer trapped in `reasoning` (empty
    `content`) versus budget exhaustion (`finish=length`) is obvious at a glance."""
    if not logger.isEnabledFor(TRACE):
        return
    for i, parsed in enumerate(choices):
        fr = finish_reasons[i] if i < len(finish_reasons) else None
        logger.log(
            TRACE,
            "%s response choice[%d]: finish=%s reasoning_len=%d content_len=%d tool_calls=%d reasoning=%r content=%r",
            request_id,
            i,
            fr,
            len(parsed.reasoning or ""),
            len(parsed.content or ""),
            len(parsed.tool_calls),
            parsed.reasoning,
            parsed.content,
        )
    logger.log(TRACE, "%s usage: %s", request_id, usage)


async def _trace_chunks(chunks: AsyncGenerator[Any, None], request_id: str) -> AsyncGenerator[Any, None]:
    """Pass streaming chunks through unchanged while buffering reasoning + content
    deltas, TRACE-logging a summary when the stream ends. Only wrapped around a
    stream when TRACE is enabled, so it costs nothing otherwise."""
    reasoning: list[str] = []
    content: list[str] = []
    try:
        async for chunk in chunks:
            for choice in chunk.choices:
                if choice.delta.reasoning:
                    reasoning.append(choice.delta.reasoning)
                if choice.delta.content:
                    content.append(choice.delta.content)
            yield chunk
    finally:
        r, c = "".join(reasoning), "".join(content)
        logger.log(
            TRACE,
            "%s response (stream): reasoning_len=%d content_len=%d reasoning=%r content=%r",
            request_id,
            len(r),
            len(c),
            r,
            c,
        )


@dataclass
class _VllmPrepared:
    """Everything a stream/no-stream seam needs once `_prepare_chat`/`_prepare_responses`
    has rendered the chat template and derived sampling params."""

    vllm_request: VllmChatCompletionRequest
    engine_input: VllmEngineInput
    sampling_params: VllmSamplingParams


class VllmInfer(BaseInfer[_VllmPrepared]):
    _vllm_usecases: ClassVar[list[ModelUsecase]] = [
        ModelUsecase.generate,
        ModelUsecase.embed,
        ModelUsecase.transcription,
        ModelUsecase.translation,
    ]

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)

        if not model_config._resolved_path:
            raise ValueError(
                f"vllm deployment '{model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )
        self._model_path: str = model_config._resolved_path

        user_overrides = model_config.vllm_engine_kwargs.model_dump(exclude_unset=True)

        # Preflight: hardware-aware safe defaults the user can override.
        # User-supplied values always win; divergences are logged so
        # misconfigured deploys are visible without spelunking vLLM logs.
        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        config_engine_kwargs = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        # Derived, not a config key: popped before the kwargs model, which forbids it.
        self._gpu_memory_utilization = resolve_gpu_memory_utilization(
            model_config, config_engine_kwargs.pop("gpu_memory_utilization", None)
        )

        self.vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        # Derived at the engine boundary, so logged beside the kwargs dump
        # rather than inside it, which reads None where the engine gets -1.
        self._max_model_len = resolve_max_model_len(self.vllm_engine_kwargs)
        logger.info(
            "initialising vllm engine with args: %s (model=%s, gpu_memory_utilization=%s, max_model_len=%s)",
            self.vllm_engine_kwargs.model_dump(),
            self._model_path,
            self._gpu_memory_utilization,
            self._max_model_len,
        )

        # Force the ray executor for multi-slot deploys: the outer actor sits
        # in a 0-GPU PG bundle, but vLLM's ParallelConfig validates world_size
        # against the actor's visible GPUs before consulting the backend. The
        # ray backend skips that check (workers claim their own bundles via the
        # inherited placement group).
        world_size = self.vllm_engine_kwargs.tensor_parallel_size * self.vllm_engine_kwargs.pipeline_parallel_size
        distributed_executor_backend = "ray" if world_size > 1 else None

        # Multimodal knobs: only forward when the user set them so we inherit
        # vLLM's own defaults (empty dict / None) otherwise.
        mm_kwargs: dict[str, Any] = {}
        if self.vllm_engine_kwargs.limit_mm_per_prompt is not None:
            mm_kwargs["limit_mm_per_prompt"] = self.vllm_engine_kwargs.limit_mm_per_prompt
        if self.vllm_engine_kwargs.mm_processor_kwargs is not None:
            mm_kwargs["mm_processor_kwargs"] = self.vllm_engine_kwargs.mm_processor_kwargs

        engine_args = VllmAsyncEngineArgs(
            model=self._model_path,
            tensor_parallel_size=self.vllm_engine_kwargs.tensor_parallel_size,
            pipeline_parallel_size=self.vllm_engine_kwargs.pipeline_parallel_size,
            max_model_len=self._max_model_len,
            dtype=cast("VllmModelDType", self.vllm_engine_kwargs.dtype),
            tokenizer=self.vllm_engine_kwargs.tokenizer,
            trust_remote_code=self.vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=self._gpu_memory_utilization,
            distributed_executor_backend=distributed_executor_backend,
            enable_log_requests=self.vllm_engine_kwargs.enable_log_requests
            if self.vllm_engine_kwargs.enable_log_requests is not None
            else False,
            disable_log_stats=self.vllm_engine_kwargs.disable_log_stats
            if self.vllm_engine_kwargs.disable_log_stats is not None
            else False,
            quantization=self.vllm_engine_kwargs.quantization,
            kv_cache_dtype=self.vllm_engine_kwargs.kv_cache_dtype or "auto",  # type: ignore[arg-type]
            enforce_eager=self.vllm_engine_kwargs.enforce_eager or False,
            enable_prefix_caching=self.vllm_engine_kwargs.enable_prefix_caching,
            max_num_batched_tokens=self.vllm_engine_kwargs.max_num_batched_tokens,
            max_num_seqs=self.vllm_engine_kwargs.max_num_seqs,
            **mm_kwargs,
        )

        usage_context = VllmUsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)

        stat_loggers: list | None = None
        if _METRICS_ENABLED:
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger as VllmRayPrometheusStatLogger

            stat_loggers = [VllmRayPrometheusStatLogger]

        self.engine = VllmAsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self) -> None:
        try:
            if engine := getattr(self, "engine", None):
                logger.info("Shutting down vllm engine for %s", self.model_config.name)
                engine.shutdown()
        except Exception:
            from modelship.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "vllm_engine"})
            logger.exception("Failed to shutdown vllm engine for %s", self.model_config.name)

    def __del__(self):
        self.shutdown()

    async def start(self):
        logger.info("Start vllm infer for model: %s", self.model_config)
        self.vllm_config = self.engine.vllm_config
        try:
            for line in _deployment_summary(self.model_config.name, self.vllm_config):
                logger.info("%s", line)
        except Exception:
            logger.debug("could not build deployment summary", exc_info=True)
        self.supported_tasks = await self.engine.get_supported_tasks()
        logger.info("Supported_tasks: %s", self.supported_tasks)

        self._caps = VllmCapabilities.detect(self.vllm_config.model_config)
        logger.info("vllm capabilities for '%s': %s", self.model_config.name, self._caps)

        await self.init_serving_chat()
        self.serving_embedding = await self.init_serving_embedding()
        self.serving_transcription = await self.init_serving_transcription()
        self.serving_translation = await self.init_serving_translation()

        if self.engine.output_handler is not None:
            self.engine.output_handler.add_done_callback(self._on_engine_output_handler_done)
        else:
            logger.warning("vllm engine for '%s' has no output_handler to watch for crashes", self.model_config.name)

    def _on_engine_output_handler_done(self, future: asyncio.Future) -> None:
        """shutdown() cancels output_handler intentionally, so a non-cancelled
        completion here means the engine core died on its own."""
        if future.cancelled():
            return
        exc = future.exception()
        logger.error("vllm engine core for '%s' died", self.model_config.name, exc_info=exc)
        # The handler can also just return, which is still a death but carries no exception.
        self.backend_died(f"vllm engine core died: {exc}" if exc else "vllm engine core exited without raising")

    async def warmup(self) -> None:
        logger.info("Warming up vllm model: %s", self.model_config.name)
        dummy_proxy = RawRequestProxy(None, {})

        if hasattr(self, "openai_serving_render"):
            request = ChatCompletionRequest(
                model=self.model_config.name, messages=[{"role": "user", "content": "warmup"}], max_tokens=1, seed=-1
            )
            result = await self.create_chat_completion(request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup chat completion done for %s", self.model_config.name)

        elif self.serving_embedding is not None:
            request = EmbeddingCompletionRequest(
                model=self.model_config.name,
                input="warmup",
            )
            await self.create_embedding(request, dummy_proxy)
            logger.info("Warmup embedding done for %s", self.model_config.name)

        elif self.serving_transcription is not None:
            request = TranscriptionRequest(
                model=self.model_config.name, file=UploadFile(file=io.BytesIO(MINIMAL_WAV)), seed=-1
            )
            audio_data = MINIMAL_WAV
            result = await self.create_transcription(audio_data, request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup transcription done for %s", self.model_config.name)

        elif self.serving_translation is not None:
            request = TranslationRequest(
                model=self.model_config.name, file=UploadFile(file=io.BytesIO(MINIMAL_WAV)), seed=-1
            )
            audio_data = MINIMAL_WAV
            result = await self.create_translation(audio_data, request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup translation done for %s", self.model_config.name)

    async def init_serving_chat(self) -> None:
        """Sets up the render/parse pipeline `create_chat_completion` drives directly
        (see engine_ops), if the model supports it. Leaves `openai_serving_render`
        unset otherwise — callers gate on `hasattr(self, "openai_serving_render")`."""
        logger.info("init_serving_chat: %s, %s", self.supported_tasks, self.model_config.usecase)
        if not (self.model_config.usecase is ModelUsecase.generate and "generate" in self.supported_tasks):
            return

        # get_chat_template isn't in vLLM's TokenizerLike protocol (it's a plain
        # HF PreTrainedTokenizer method). It raises on a base model, which carries
        # no template — leave the pipeline unset instead of killing the replica.
        try:
            template = cast(Any, self.engine.get_tokenizer()).get_chat_template()
        except ValueError as exc:
            logger.warning(
                "'%s' has no usable chat template (%s) — the model is deployed but has no reachable "
                "endpoint, since modelship serves no /v1/completions route",
                self.model_config.name,
                exc,
            )
            return

        # A reasoning parser reads chat_template_kwargs["enable_thinking"] (and
        # similar toggles) and defaults them True when absent, but a template may
        # default the same toggle False — leaving the parser primed to reason on a
        # model that was never told to. Pin each toggle to the template's own
        # detected default so both sides agree; user/config values still win.
        if template is not None:
            defaults = detect_template_toggle_defaults(template, cast(Any, self.engine.get_tokenizer()))
            applied = {k: v for k, v in defaults.items() if k not in self.model_config.chat_template_kwargs}
            for key, value in applied.items():
                self.model_config.chat_template_kwargs[key] = value
            if applied:
                logger.info("Pinned chat-template toggle defaults for '%s': %s", self.model_config.name, applied)
            overridden = {k: v for k, v in defaults.items() if k not in applied}
            if overridden:
                logger.info(
                    "Chat-template toggle defaults for '%s' overridden by config: %s (detected defaults: %s)",
                    self.model_config.name,
                    {k: self.model_config.chat_template_kwargs[k] for k in overridden},
                    overridden,
                )

        tool_parser_name = resolve_tool_parser(self.model_config, template)
        enable_tools = tool_parser_name is not None
        reasoning_parser_name = resolve_reasoning_parser(self.model_config, template) or ""

        self._enable_auto_tools = enable_tools
        self.openai_serving_render = VllmOnlineRenderer(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            request_logger=VllmRequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=enable_tools,
            tool_parser=tool_parser_name,
            reasoning_parser=reasoning_parser_name,
        )
        tokenizer = self.openai_serving_render.renderer.tokenizer
        assert tokenizer is not None, "vllm renderer has no tokenizer (skip_tokenizer_init=True is unsupported here)"
        self._tokenizer: VllmTokenizerLike = tokenizer

    def _make_parsers(self, vllm_request: VllmChatCompletionRequest, n: int) -> list[VllmParser | None]:
        """One stateful parser instance per choice — the streaming path (see `engine_ops.make_parsers`)."""
        return engine_ops.make_parsers(
            self.openai_serving_render, self._tokenizer, vllm_request, vllm_request.chat_template_kwargs, n
        )

    def _make_parser(self, vllm_request: VllmChatCompletionRequest) -> VllmParser | None:
        """Single parser instance for the non-streaming path, which reuses it
        (stateless `.parse()` call) across every choice — see `engine_ops.build_choices`."""
        return self._make_parsers(vllm_request, n=1)[0]

    async def _render_bundle(self, vllm_request: VllmChatCompletionRequest) -> ErrorResponse | _VllmPrepared:
        """Render the chat template and derive sampling params — the single place
        this happens, so every seam sees a pre-generation failure the same way."""
        try:
            result = await engine_ops.render_and_params(self.openai_serving_render, vllm_request)
        except VllmValidationError as exc:
            logger.info("Validation error rendering chat request: %s", exc)
            return _to_error_response(exc)
        if isinstance(result, VllmErrorResponse):
            logger.info("Validation error rendering chat request: %s", result.error.message)
            return _to_error_response(result)
        engine_input, sampling_params = result
        return _VllmPrepared(vllm_request, engine_input, sampling_params)

    async def init_serving_embedding(self) -> VllmServingEmbedding | None:
        logger.info("init_serving_embedding: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            VllmServingEmbedding(
                engine_client=self.engine,
                models=VllmOpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[VllmBaseModelPath(name=self.model_config.name, model_path=self._model_path)],
                ),
                request_logger=VllmRequestLogger(max_log_len=None),
                chat_template_config=VllmChatTemplateConfig(chat_template=None, chat_template_content_format="auto"),
            )
            if self.model_config.usecase is ModelUsecase.embed
            and any(task in self.supported_tasks for task in ["embed", "embedding"])
            else None
        )

    async def init_serving_transcription(self) -> VllmOpenAIServingTranscription | None:
        logger.info("init_serving_transcription: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            VllmOpenAIServingTranscription(
                engine_client=self.engine,
                models=VllmOpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[VllmBaseModelPath(name=self.model_config.name, model_path=self._model_path)],
                ),
                request_logger=VllmRequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def init_serving_translation(self) -> VllmOpenAIServingTranslation | None:
        logger.info("init_serving_translation: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            VllmOpenAIServingTranslation(
                engine_client=self.engine,
                models=VllmOpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[VllmBaseModelPath(name=self.model_config.name, model_path=self._model_path)],
                ),
                request_logger=VllmRequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def _prepare_chat(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | _VllmPrepared:
        if not hasattr(self, "openai_serving_render"):
            return await super()._prepare_chat(request, raw_request)
        try:
            request.messages = normalize_chat_messages(
                request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as exc:
            return _to_error_response(exc)
        try:
            vllm_request = engine_ops.build_vllm_request(
                request, self.model_config.chat_template_kwargs, cache_salt=raw_request.identity
            )
        except ValidationError as exc:
            # pydantic's .args is () here (str(exc) is the only useful message),
            # matching api.py's _validation_error_from_cause for the same reason.
            logger.info("Validation error building vllm request: %s", exc)
            return create_error_response(
                message=str(exc), err_type="invalid_request_error", status_code=HTTPStatus.BAD_REQUEST
            )
        return await self._render_bundle(vllm_request)

    async def _create_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        prepared: _VllmPrepared,
        raw_request: RawRequestProxy,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat path via `engine_ops`, bypassing `OpenAIServingChat`.
        Rendering already succeeded in `_prepare_chat` — this only drives generation."""
        request_id = f"chatcmpl-{base_request_id(raw_request)}"
        vllm_request = prepared.vllm_request
        _trace_request(request_id, vllm_request, prepared.sampling_params)

        stream = engine_ops.stream_chat_completion(
            self.engine,
            self.openai_serving_render,
            vllm_request,
            prepared.engine_input,
            prepared.sampling_params,
            request_id,
            self.model_config.name,
            self._tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=bool(request.logprobs),
            num_output_top_logprobs=request.top_logprobs,
        )
        chunks = self.run_cancellable_stream(stream, raw_request)
        if logger.isEnabledFor(TRACE):
            chunks = _trace_chunks(chunks, request_id)
        try:
            async for chunk in chunks:
                yield encode_chat_sse_chunk(chunk)
        except ClientDisconnectedError:
            logger.info("chat request %s aborted: client disconnected", request_id)
            return
        except VllmValidationError as exc:
            yield encode_error_sse(_to_error_response(exc))
            yield "data: [DONE]\n\n"
            return
        except Exception:
            logger.exception("chat request %s failed mid-stream", request_id)
            yield encode_error_sse(
                _to_error_response("Internal error during generation", err_type="api_error", status_code=500)
            )
            yield "data: [DONE]\n\n"
            return
        yield "data: [DONE]\n\n"

    async def _create_chat_completion_no_stream(
        self,
        request: ChatCompletionRequest,
        prepared: _VllmPrepared,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ChatCompletionResponse:
        """Non-stream chat path via `engine_ops`, bypassing `OpenAIServingChat`."""
        vllm_request = prepared.vllm_request
        engine_input, sampling_params = prepared.engine_input, prepared.sampling_params

        parser = self._make_parser(vllm_request)
        prompt_token_ids = engine_ops.extract_prompt_token_ids(self.openai_serving_render, engine_input)
        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_request, parser, prompt_token_ids)

        request_id = f"chatcmpl-{base_request_id(raw_request)}"
        _trace_request(request_id, vllm_request, sampling_params)
        try:
            final_res = await self.run_cancellable(
                engine_ops.consume_final_output(
                    self.engine,
                    engine_input,
                    sampling_params,
                    request_id,
                    reasoning_ended=reasoning_ended,
                    parser=parser,
                    chat_template_kwargs=vllm_request.chat_template_kwargs,
                ),
                raw_request,
            )
        except ClientDisconnectedError:
            return _to_error_response("Client disconnected")
        except VllmValidationError as exc:
            return _to_error_response(exc)

        choices, finish_reasons, logprobs_list = engine_ops.build_choices(
            final_res,
            vllm_request,
            parser,
            self._tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=bool(request.logprobs),
            num_output_top_logprobs=request.top_logprobs,
        )

        if final_res.prompt_token_ids is None:
            return _to_error_response("vllm returned no prompt_token_ids for a completed request", status_code=502)
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        reasoning_tokens = engine_ops.total_reasoning_tokens(final_res.outputs, parser)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
            if final_res.num_cached_tokens
            else None,
        )
        _trace_parsed_response(request_id, choices, finish_reasons, usage)
        return build_from_parsed(
            request_id=request_id,
            model_name=self.model_config.name,
            choices=choices,
            usage=usage,
            finish_reasons=finish_reasons,
            logprobs=logprobs_list,
        )

    async def _prepare_responses(
        self, request: ResponsesRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | _VllmPrepared:
        if not hasattr(self, "openai_serving_render"):
            return await super()._prepare_responses(request, raw_request)

        try:
            chat_request = responses_request_to_chat(request)
        except UnsupportedResponsesFeatureError as e:
            return _to_error_response(e)
        except ValidationError as e:
            return responses_validation_error(e)

        try:
            chat_request.messages = normalize_chat_messages(
                chat_request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as exc:
            return _to_error_response(exc)
        chat_request.stream = request.stream or False
        try:
            vllm_request = engine_ops.build_vllm_request(
                chat_request, self.model_config.chat_template_kwargs, cache_salt=raw_request.identity
            )
        except ValidationError as exc:
            # Same 400 as the `responses_request_to_chat` failure above.
            logger.info("Validation error building vllm request: %s", exc)
            return responses_validation_error(exc)
        return await self._render_bundle(vllm_request)

    async def _create_response_no_stream(
        self,
        request: ResponsesRequest,
        prepared: _VllmPrepared,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ResponseObject:
        """Non-stream Responses path via `engine_ops`, shaping items directly from
        `ParsedChatOutput` instead of round-tripping through a `ChatCompletionResponse`."""
        vllm_request = prepared.vllm_request
        engine_input, sampling_params = prepared.engine_input, prepared.sampling_params

        parser = self._make_parser(vllm_request)
        prompt_token_ids = engine_ops.extract_prompt_token_ids(self.openai_serving_render, engine_input)
        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_request, parser, prompt_token_ids)

        request_id = f"resp-{base_request_id(raw_request)}"
        _trace_request(request_id, vllm_request, sampling_params)
        try:
            final_res = await self.run_cancellable(
                engine_ops.consume_final_output(
                    self.engine,
                    engine_input,
                    sampling_params,
                    request_id,
                    reasoning_ended=reasoning_ended,
                    parser=parser,
                    chat_template_kwargs=vllm_request.chat_template_kwargs,
                ),
                raw_request,
            )
        except ClientDisconnectedError:
            return _to_error_response("Client disconnected")
        except VllmValidationError as exc:
            return _to_error_response(exc)

        choices, finish_reasons, _logprobs_list = engine_ops.build_choices(
            final_res,
            vllm_request,
            parser,
            self._tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )

        if final_res.prompt_token_ids is None:
            return _to_error_response("vllm returned no prompt_token_ids for a completed request", status_code=502)
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        reasoning_tokens = engine_ops.total_reasoning_tokens(final_res.outputs, parser)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
            if final_res.num_cached_tokens
            else None,
        )
        _trace_parsed_response(request_id, choices, finish_reasons, usage)
        return build_response_from_parsed(
            choices[0],
            request,
            usage=usage,
            finish_reason=finish_reasons[0],
            model=self.model_config.name,
        )

    async def _create_response_stream(
        self,
        request: ResponsesRequest,
        prepared: _VllmPrepared,
        raw_request: RawRequestProxy,
        *,
        response_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Native streaming Responses path: feeds `BaseInfer._stream_responses` directly
        from `engine_ops.stream_chat_completion`'s typed chunks — no chat SSE text
        round trip. Rendering already succeeded in `_prepare_responses`."""
        request_id = f"resp-{base_request_id(raw_request)}"
        vllm_request = prepared.vllm_request
        _trace_request(request_id, vllm_request, prepared.sampling_params)

        stream = engine_ops.stream_chat_completion(
            self.engine,
            self.openai_serving_render,
            vllm_request,
            prepared.engine_input,
            prepared.sampling_params,
            request_id,
            self.model_config.name,
            self._tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )
        chunks = self.run_cancellable_stream(stream, raw_request)
        if logger.isEnabledFor(TRACE):
            chunks = _trace_chunks(chunks, request_id)
        async for event in self._stream_responses(
            request, chunks, request_id=request_id, client_error=_vllm_stream_error, response_id=response_id
        ):
            yield event

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | Response:
        if self.serving_embedding is None:
            return await super().create_embedding(request, raw_request)
        vllm_request = VllmEmbeddingCompletionRequest(**request.model_dump())
        try:
            result = await self.serving_embedding(vllm_request, cast("Request", raw_request))
        except VllmValidationError as exc:
            return _to_error_response(exc)
        if isinstance(result, VllmErrorResponse):
            return _to_error_response(result)
        return cast("ErrorResponse | Response", result)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_transcription is None:
            return await super().create_transcription(audio_data, request, raw_request)
        # `file` is a required field on vLLM's own request schema but unused by
        # create_transcription (audio_data is passed separately) — model_construct
        # skips validation instead of raising on the field modelship never populates.
        vllm_request = VllmTranscriptionRequest.model_construct(**request.model_dump())
        vllm_request.timestamp_granularities = []
        try:
            result = await self.serving_transcription.create_transcription(
                audio_data, vllm_request, cast("Request", raw_request)
            )
        except VllmValidationError as exc:
            return _to_error_response(exc)
        if isinstance(result, VllmErrorResponse):
            return _to_error_response(result)
        if isinstance(result, VllmTranscriptionResponseVerbose):
            return TranscriptionResponseVerbose.model_validate(result.model_dump())
        if isinstance(result, VllmTranscriptionResponse):
            return TranscriptionResponse.model_validate(result.model_dump())
        if isinstance(result, AsyncGenerator):
            return cast("AsyncGenerator[str, None]", result)
        raise TypeError(f"Unexpected transcription result type: {type(result).__name__}")

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_translation is None:
            return await super().create_translation(audio_data, request, raw_request)
        # `file` is a required field on vLLM's own request schema but unused by
        # create_translation (audio_data is passed separately) — model_construct
        # skips validation instead of raising on the field modelship never populates.
        vllm_request = VllmTranslationRequest.model_construct(**request.model_dump())
        try:
            result = await self.serving_translation.create_translation(
                audio_data, vllm_request, cast("Request", raw_request)
            )
        except VllmValidationError as exc:
            return _to_error_response(exc)
        if isinstance(result, VllmErrorResponse):
            return _to_error_response(result)
        if isinstance(result, VllmTranslationResponseVerbose):
            return TranslationResponseVerbose.model_validate(result.model_dump())
        if isinstance(result, VllmTranslationResponse):
            return TranslationResponse.model_validate(result.model_dump())
        if isinstance(result, AsyncGenerator):
            return cast("AsyncGenerator[str, None]", result)
        raise TypeError(f"Unexpected translation result type: {type(result).__name__}")
