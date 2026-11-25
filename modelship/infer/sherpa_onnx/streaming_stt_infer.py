"""Streaming speech to text on sherpa-onnx: the registry's streaming `AsrEntry` bundles, through
`transcribe_push`/`transcribe_finish`/`transcribe_close` and `/v1/audio/transcriptions`. The replica's
`StreamDecoder` (`stream_decoder.py`) holds the streams and decodes them."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from types import ModuleType
from typing import Any

import numpy as np

from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.sherpa_onnx.family import nemo_streaming_transducer
from modelship.infer.sherpa_onnx.registry import AsrEntry, StreamFeed
from modelship.infer.sherpa_onnx.sherpa_onnx_infer import PROVIDER, SherpaOnnxInfer
from modelship.infer.sherpa_onnx.stream_decoder import StreamDecoder, StreamResult
from modelship.infer.sherpa_onnx.stt_infer import SAMPLE_RATE, segments
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    SttCapabilities,
    SttStreamOptions,
    SttStreamText,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionUsageAudio,
    create_error_response,
    encode_error_sse,
)
from modelship.utils import base_request_id
from modelship.utils.audio import decode_audio, from_pcm16

logger = get_logger("infer.sherpa_onnx")

# `/v1/audio/transcriptions` with stream=true feeds a file this much at a time, one delta each
FILE_STEP_S = 1.0
# verbose_json needs a language; nemotron reports none
_UNKNOWN_LANGUAGE = "unknown"
# `AsrEntry.family` -> its module, which builds the recognizer and reads words from its tokens
_FAMILIES: dict[str, ModuleType] = {"nemo_streaming_transducer": nemo_streaming_transducer}


class SherpaOnnxStreamingSttInfer(SherpaOnnxInfer):
    """sherpa-onnx streaming speech to text."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        assert isinstance(self.registry_entry, AsrEntry) and self.registry_entry.stream is not None
        self.entry: AsrEntry = self.registry_entry
        self.feed: StreamFeed = self.registry_entry.stream
        self.family = _FAMILIES[self.entry.family]
        self.decoder: StreamDecoder | None = None

    def shutdown(self) -> None:
        if self.decoder is not None:
            self.decoder.stop()
            self.decoder = None

    def __del__(self):
        self.shutdown()

    def _load(self) -> None:
        recognizer = self.family.recognizer(self.entry, self.bundle_dir(), self.threads, PROVIDER)
        self.decoder = StreamDecoder(recognizer, self.feed, self.model_config.name)

    async def stt_capabilities(self) -> SttCapabilities:
        return SttCapabilities(streaming=True, reports_language=False, chunk_ms=self.feed.chunk_ms)

    async def _run(self, key: str, samples: np.ndarray, rate: int, language: str | None, finish: bool) -> StreamResult:
        assert self.decoder is not None
        return await asyncio.wrap_future(self.decoder.submit(key, samples, rate, language, finish))

    async def _stream_text(
        self, key: str, pcm: bytes, rate: int, options: SttStreamOptions, finish: bool
    ) -> ErrorResponse | SttStreamText:
        try:
            result = await self._run(key, from_pcm16(pcm), rate, options.language, finish)
        except Exception as e:
            logger.warning("stream %s: %s", key, e)
            return create_error_response(f"streamed transcription failed: {e}", err_type="api_error", status_code=500)
        return SttStreamText(text=result.text, language=result.language, audio_ms=result.audio_ms)

    async def transcribe_push(
        self, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> ErrorResponse | SttStreamText:
        return await self._stream_text(key, pcm, rate, options, finish=False)

    async def transcribe_finish(
        self, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> ErrorResponse | SttStreamText:
        return await self._stream_text(key, pcm, rate, options, finish=True)

    async def transcribe_close(self, key: str) -> None:
        if self.decoder is not None:
            self.decoder.close(key)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"stt-{base_request_id(raw_request)}"
        if request.prompt:
            logger.debug("%s: %s takes no prompt; ignoring it", request_id, self.model_config.name)
        try:
            samples, duration = await self.run_blocking(
                lambda _stop: decode_audio(audio_data, SAMPLE_RATE), exclusive=False
            )
        except Exception as e:
            logger.warning("%s: failed to decode audio: %s", request_id, e)
            return create_error_response(f"failed to decode audio: {e}")
        samples = np.asarray(samples, dtype=np.float32)
        key = f"{request_id}-{uuid.uuid4().hex}"
        logger.info("transcription request %s: %.1f s, stream=%s", request_id, duration, request.stream)
        if request.stream:
            return self._stream(key, samples, request.language, request_id)

        try:
            result = await self._run(key, samples, SAMPLE_RATE, request.language, finish=True)
        except asyncio.CancelledError:
            if self.decoder is not None:
                self.decoder.close(key)
            raise
        except Exception as e:
            logger.warning("%s failed: %s", request_id, e)
            return create_error_response(
                f"sherpa-onnx transcription failed: {e}", err_type="api_error", status_code=502
            )
        logger.log(TRACE, "%s response: text=%r", request_id, result.text)
        usage = TranscriptionUsageAudio(seconds=duration)
        if request.response_format != "verbose_json":
            return TranscriptionResponse(text=result.text, usage=usage)
        found = self.family.words(result.tokens, result.timestamps, 0.0)
        return TranscriptionResponseVerbose(
            language=request.language or _UNKNOWN_LANGUAGE,
            duration=duration,
            text=result.text,
            words=found if "word" in request.timestamp_granularities else [],
            segments=segments(found),
            usage=usage,
        )

    async def _stream(
        self, key: str, samples: np.ndarray, language: str | None, request_id: str
    ) -> AsyncGenerator[str, None]:
        """A `transcript.text.delta` per `FILE_STEP_S` of audio whose text grew, then `transcript.text.done`."""
        step = int(FILE_STEP_S * SAMPLE_RATE)
        sent = text = ""
        finished = False
        try:
            for at in range(0, max(len(samples), 1), step):
                last = at + step >= len(samples)
                result = await self._run(key, samples[at : at + step], SAMPLE_RATE, language, finish=last)
                finished, text = last, result.text
                if text.startswith(sent) and len(text) > len(sent):
                    yield _event({"type": "transcript.text.delta", "delta": text[len(sent) :]})
                    sent = text
            yield _event({"type": "transcript.text.done", "text": text})
        except Exception as exc:
            logger.warning("%s failed mid-stream: %s", request_id, exc)
            yield encode_error_sse(
                create_error_response(f"sherpa-onnx transcription failed: {exc}", err_type="api_error", status_code=502)
            )
        finally:
            if not finished and self.decoder is not None:
                self.decoder.close(key)


def _event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
