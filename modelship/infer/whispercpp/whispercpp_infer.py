import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncGenerator
from typing import Any, cast

from modelship.infer.base_infer import BaseInfer, ClientDisconnectedError
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy, WhispercppConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    create_error_response,
)
from modelship.openai.utils.chat import encode_error_sse
from modelship.utils import base_request_id
from modelship.utils.audio import decode_audio

logger = get_logger("infer.whispercpp")

_WHISPER_SAMPLE_RATE = 16000
# whisper.cpp writes the ggml magic as a little-endian uint32, so on disk it reads "lmgg".
_GGML_MAGIC = b"lmgg"
_GGUF_MAGIC = b"GGUF"

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
        context_params: dict[str, Any] = {"use_gpu": use_gpu}
        if self.config.flash_attn:
            context_params["flash_attn"] = True

        kwargs: dict[str, Any] = {"context_params": context_params}
        if self.config.n_threads is not None:
            kwargs["n_threads"] = self.config.n_threads

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
        kwargs: dict[str, Any] = {"translate": translate, "temperature": request.temperature}
        # Source hint for both tasks; no target-language knob exists, so to_language is unread.
        if request.language:
            kwargs["language"] = request.language
        if request.prompt:
            kwargs["initial_prompt"] = request.prompt
        return kwargs

    async def _detect_language(self, samples: Any) -> str:
        # Detection on an English-only model returns an out-of-range id, which
        # pywhispercpp maps to a wrong language.
        if not self._multilingual:
            return "en"
        (lang, _prob), _all = await asyncio.to_thread(self.model.auto_detect_language, samples)
        return lang

    # -- transcription -----------------------------------------------------

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"asr-{base_request_id(raw_request)}"
        logger.info("transcription request %s: stream=%s", request_id, request.stream)
        if request.stream:
            return self.run_cancellable_stream(
                self._stream(audio_data, request, translate=False, request_id=request_id), raw_request
            )
        try:
            result = await self.run_cancellable(
                self._transcribe_once(audio_data, request, translate=False, request_id=request_id), raw_request
            )
        except ClientDisconnectedError:
            logger.info("transcription request %s aborted: client disconnected", request_id)
            return create_error_response("Client disconnected")
        return cast("ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose", result)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"translate-{base_request_id(raw_request)}"
        logger.info("translation request %s: stream=%s", request_id, request.stream)
        if request.stream:
            return self.run_cancellable_stream(
                self._stream(audio_data, request, translate=True, request_id=request_id), raw_request
            )
        try:
            result = await self.run_cancellable(
                self._transcribe_once(audio_data, request, translate=True, request_id=request_id), raw_request
            )
        except ClientDisconnectedError:
            logger.info("translation request %s aborted: client disconnected", request_id)
            return create_error_response("Client disconnected")
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
        segments = await asyncio.to_thread(self.model.transcribe, samples, **decode_kwargs)
        text = "".join(s.text for s in segments).strip()
        logger.log(TRACE, "%s response: text=%r", request_id, text)

        verbose = request.response_format == "verbose_json"
        if not verbose:
            if translate:
                return TranslationResponse(text=text)
            return TranscriptionResponse(text=text, usage=TranscriptionUsageAudio(seconds=int(duration_seconds)))

        if translate:
            detected_language = _TRANSLATION_OUTPUT_LANGUAGE
        else:
            language = decode_kwargs.get("language")
            detected_language = language if language else await self._detect_language(samples)

        response_segments = [
            TranscriptionSegment(id=i, start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text.strip())
            for i, s in enumerate(segments)
        ]
        usage = TranscriptionUsageAudio(seconds=int(duration_seconds))
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
        """One `transcript.text.delta` per decoded segment (no token-level API).
        `abort_event` mirrors whisper.cpp's `abort_callback`; set in `finally` so
        an early `aclose()` (client disconnect) stops it between segments."""
        try:
            samples, _duration_seconds = decode_audio(audio_data, _WHISPER_SAMPLE_RATE)
        except Exception as e:
            logger.warning("%s: failed to decode audio: %s", request_id, e)
            yield encode_error_sse(create_error_response(f"failed to decode audio: {e}"))
            return

        decode_kwargs = self._decode_kwargs(request, translate=translate)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        abort_event = threading.Event()
        _done = object()

        def on_segment(segment: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, segment)

        def run() -> None:
            try:
                self.model.transcribe(
                    samples,
                    new_segment_callback=on_segment,
                    abort_callback=abort_event.is_set,
                    **decode_kwargs,
                )
            except BaseException as exc:  # forwarded to the consumer below, not swallowed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _done)

        worker = asyncio.create_task(asyncio.to_thread(run))
        text_parts: list[str] = []
        try:
            while True:
                item = await queue.get()
                if item is _done:
                    break
                if isinstance(item, BaseException):
                    raise item
                segment_text = item.text.strip()
                if segment_text:
                    text_parts.append(segment_text)
                    yield _encode_transcript_event({"type": "transcript.text.delta", "delta": segment_text})
            yield _encode_transcript_event({"type": "transcript.text.done", "text": " ".join(text_parts)})
        except Exception as exc:
            logger.warning("%s failed mid-stream: %s", request_id, exc)
            yield encode_error_sse(
                create_error_response(f"whisper.cpp transcription failed: {exc}", err_type="api_error", status_code=502)
            )
        finally:
            abort_event.set()
            await worker


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
