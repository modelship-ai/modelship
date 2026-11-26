import asyncio
import contextlib
import json
import logging
import os
import threading
from collections.abc import AsyncGenerator
from typing import Any, cast

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy, WhispercppConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    SttCapabilities,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    create_error_response,
    encode_error_sse,
)
from modelship.utils import base_request_id
from modelship.utils.audio import decode_audio

logger = get_logger("infer.whispercpp")

_WHISPER_SAMPLE_RATE = 16000
# whisper.cpp writes the ggml magic as a little-endian uint32, so on disk it reads "lmgg".
_GGML_MAGIC = b"lmgg"
_GGUF_MAGIC = b"GGUF"

# whisper.cpp detects the language when it is empty; "auto" does the same but logs an unknown-language error first.
_DETECT_LANGUAGE = ""

# TranslationResponseVerbose.language is documented as always "english" (the output language).
_TRANSLATION_OUTPUT_LANGUAGE = "english"


class WhispercppInfer(BaseInfer):
    """In-process whisper.cpp speech-to-text loader via `pywhispercpp` (no subprocess)."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.config = model_config.whispercpp_config or WhispercppConfig()
        self.model: Any = None
        self._multilingual = True

    def shutdown(self) -> None:
        self.model = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(None, self._load)

    def _load(self) -> Any:
        import _pywhispercpp as _pw
        from pywhispercpp.model import Model

        use_gpu = self.model_config.num_gpus > 0
        context_params: dict[str, Any] = {"use_gpu": use_gpu, "flash_attn": self.config.flash_attn}

        kwargs: dict[str, Any] = {"context_params": context_params}
        n_threads = _threads(self.config, self.model_config.num_cpus)
        if n_threads is not None:
            kwargs["n_threads"] = n_threads

        resolved_path = self.model_config._resolved_path
        if resolved_path is None:
            raise ValueError(
                f"whispercpp deployment '{self.model_config.name}' has no resolved model file. "
                f"Check driver logs for resolution errors."
            )
        _validate_ggml_model_file(resolved_path, self.model_config.name)

        logger.info("loading whisper.cpp model %r (gpu=%s) for '%s'", resolved_path, use_gpu, self.model_config.name)
        # raw stderr writes; redirecting into our own logger would deadlock (fd 2 -> our handlers' pipe)
        kwargs["redirect_whispercpp_logs_to"] = False if logger.isEnabledFor(logging.DEBUG) else None
        model: Any = Model(resolved_path, **kwargs)
        self._multilingual = bool(_pw.whisper_is_multilingual(model._ctx))
        return model

    async def warmup(self) -> None:
        pass

    def _decode_kwargs(self, request: TranscriptionRequest | TranslationRequest, *, translate: bool) -> dict[str, Any]:
        """Every setting a request can change, for one `Model.transcribe` call.

        pywhispercpp keeps the settings of each call for the next one, so each key is
        set on every call, or one request's language or prompt would reach the next."""
        return {
            "translate": translate,
            "temperature": request.temperature,
            # The source language for both tasks; to_language has no whisper.cpp setting and is unread.
            "language": request.language or self._unset_language(),
            "initial_prompt": request.prompt or "",
        }

    def _unset_language(self) -> str:
        """The language for a request that names none: detected, or English on an English-only model."""
        return _DETECT_LANGUAGE if self._multilingual else "en"

    # -- transcription -----------------------------------------------------

    async def stt_capabilities(self) -> SttCapabilities:
        # an English-only model always answers "en"
        return SttCapabilities(streaming=False, reports_language=self._multilingual)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"asr-{base_request_id(raw_request)}"
        logger.info("transcription request %s: stream=%s", request_id, request.stream)
        if request.stream:
            return self._stream(audio_data, request, translate=False, request_id=request_id)
        result = await self._transcribe_once(audio_data, request, translate=False, request_id=request_id)
        return cast("ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose", result)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"translate-{base_request_id(raw_request)}"
        logger.info("translation request %s: stream=%s", request_id, request.stream)
        if request.stream:
            return self._stream(audio_data, request, translate=True, request_id=request_id)
        result = await self._transcribe_once(audio_data, request, translate=True, request_id=request_id)
        return cast("ErrorResponse | TranslationResponse | TranslationResponseVerbose", result)

    async def _transcribe_once(
        self,
        audio_data: bytes,
        request: TranscriptionRequest | TranslationRequest,
        *,
        translate: bool,
        request_id: str,
    ) -> (
        ErrorResponse
        | TranscriptionResponse
        | TranscriptionResponseVerbose
        | TranslationResponse
        | TranslationResponseVerbose
    ):
        try:
            samples, duration_seconds = decode_audio(audio_data, _WHISPER_SAMPLE_RATE)
        except Exception as e:
            logger.warning("%s: failed to decode audio: %s", request_id, e)
            return create_error_response(f"failed to decode audio: {e}")

        decode_kwargs = self._decode_kwargs(request, translate=translate)

        def decode(stop: threading.Event) -> tuple[list[Any], str]:
            segments = self.model.transcribe(samples, abort_callback=stop.is_set, **decode_kwargs)
            return segments, _decoded_language(self.model)

        segments, decoded_language = await self.run_blocking(decode)
        # segments don't always start with a space; joined like the streamed transcript
        text = " ".join(part for part in (s.text.strip() for s in segments) if part)
        logger.log(TRACE, "%s response: text=%r", request_id, text)

        verbose = request.response_format == "verbose_json"
        if not verbose:
            if translate:
                return TranslationResponse(text=text)
            return TranscriptionResponse(text=text, usage=TranscriptionUsageAudio(seconds=duration_seconds))

        detected_language = _TRANSLATION_OUTPUT_LANGUAGE if translate else decoded_language
        response_segments = [
            TranscriptionSegment(id=i, start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text.strip())
            for i, s in enumerate(segments)
        ]
        usage = TranscriptionUsageAudio(seconds=duration_seconds)
        if translate:
            return TranslationResponseVerbose(
                language=detected_language,
                duration=duration_seconds,
                text=text,
                segments=response_segments,
                usage=usage,
            )
        return TranscriptionResponseVerbose(
            language=detected_language,
            duration=duration_seconds,
            text=text,
            segments=response_segments,
            usage=usage,
        )

    async def _stream(
        self,
        audio_data: bytes,
        request: TranscriptionRequest | TranslationRequest,
        *,
        translate: bool,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """One `transcript.text.delta` per decoded segment (no token-level API)."""
        try:
            samples, _duration_seconds = decode_audio(audio_data, _WHISPER_SAMPLE_RATE)
        except Exception as e:
            logger.warning("%s: failed to decode audio: %s", request_id, e)
            yield encode_error_sse(create_error_response(f"failed to decode audio: {e}"))
            return

        decode_kwargs = self._decode_kwargs(request, translate=translate)
        segments: AsyncGenerator[Any, None] = self.stream_blocking(
            lambda emit, stop: self.model.transcribe(
                samples, new_segment_callback=emit, abort_callback=stop.is_set, **decode_kwargs
            )
        )
        text_parts: list[str] = []
        try:
            async with contextlib.aclosing(segments):
                async for segment in segments:
                    segment_text = segment.text.strip()
                    if segment_text:
                        text_parts.append(segment_text)
                        yield _encode_transcript_event({"type": "transcript.text.delta", "delta": segment_text})
            yield _encode_transcript_event({"type": "transcript.text.done", "text": " ".join(text_parts)})
        except Exception as exc:
            logger.warning("%s failed mid-stream: %s", request_id, exc)
            yield encode_error_sse(
                create_error_response(f"whisper.cpp transcription failed: {exc}", err_type="api_error", status_code=502)
            )


def _threads(config: WhispercppConfig, num_cpus: float) -> int | None:
    """The decode's thread count: the configured one, else the CPUs the deploy reserves, at most one
    per physical core. None below one reserved CPU keeps whisper.cpp's own `min(4, cores)`."""
    if config.n_threads is not None:
        return config.n_threads
    if num_cpus < 1:
        return None
    import psutil

    cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    return min(int(num_cpus), cores)


def _decoded_language(model: Any) -> str:
    """The language the last `Model.transcribe` call decoded in: the one it was given, or the one it detected."""
    import _pywhispercpp as _pw

    return _pw.whisper_lang_str(_pw.whisper_full_lang_id(model._ctx))


def _encode_transcript_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _validate_ggml_model_file(path: str, model_name: str) -> None:
    """Rejects GGUF/safetensors and checks the ggml magic bytes — a selector
    match at driver preflight can't always tell the format from the filename alone."""
    if not os.path.isfile(path):
        raise ValueError(
            f"whispercpp deployment '{model_name}' is missing a resolved model file at {path!r}. "
            f"Check driver logs for resolution errors."
        )
    with open(path, "rb") as f:
        magic = f.read(4)
    if path.lower().endswith(".gguf") or magic == _GGUF_MAGIC:
        raise ValueError(
            f"whispercpp deployment '{model_name}': {path!r} is a GGUF file. whisper.cpp needs its own legacy "
            f"ggml/bin format (files named like `ggml-base.en.bin`), not GGUF."
        )
    if path.lower().endswith(".safetensors"):
        raise ValueError(
            f"whispercpp deployment '{model_name}': {path!r} is a safetensors checkpoint. whisper.cpp needs the "
            f"legacy ggml/bin format; use `loader: vllm` for a HF-format Whisper checkpoint instead."
        )
    if magic != _GGML_MAGIC:
        raise ValueError(
            f"whispercpp deployment '{model_name}': {path!r} does not look like a whisper.cpp ggml model file "
            f"(expected the 4-byte magic {_GGML_MAGIC!r}, got {magic!r})."
        )
