"""Offline speech to text on sherpa-onnx: the registry's `AsrEntry` bundles, on `/v1/audio/transcriptions`.

Audio longer than `MAX_SEGMENT_S` is cut at Silero's silences into pieces of at most that length, decoded one
after another. A transducer's memory and time grow faster than its input.
"""

import contextlib
import json
import threading
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np

from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.sherpa_onnx.family import nemo_transducer
from modelship.infer.sherpa_onnx.registry import AsrEntry
from modelship.infer.sherpa_onnx.sherpa_onnx_infer import PROVIDER, SherpaOnnxInfer
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
    TranscriptionWord,
    create_error_response,
    encode_error_sse,
)
from modelship.utils import base_request_id
from modelship.utils.audio import decode_audio
from modelship.utils.vad import SpeechStopped, Vad, VadSettings

logger = get_logger("infer.sherpa_onnx")

SAMPLE_RATE = 16000
MAX_SEGMENT_S = 30.0
# a transcript segment ends at a word ending in one of these
_SENTENCE_ENDS = (".", "?", "!", "。", "？", "！")
# verbose_json needs a language; these models neither take nor report one
_UNKNOWN_LANGUAGE = "unknown"
# `AsrEntry.family` -> its module, which builds the recognizer and reads words from its tokens
_FAMILIES: dict[str, ModuleType] = {"nemo_transducer": nemo_transducer}


@dataclass(frozen=True)
class Piece:
    """A stretch of the input decoded on its own, `offset` seconds into it."""

    offset: float
    samples: np.ndarray


@dataclass(frozen=True)
class Decoded:
    text: str
    # timed from the start of the input
    words: list[TranscriptionWord]


def pieces(samples: np.ndarray, vad_model: str | None) -> list[Piece]:
    """*samples* as one piece, or cut at speech boundaries when longer than `MAX_SEGMENT_S`."""
    if len(samples) <= MAX_SEGMENT_S * SAMPLE_RATE or vad_model is None:
        return [Piece(0.0, samples)]
    vad = Vad(vad_model, VadSettings(), max_speech_s=MAX_SEGMENT_S)
    events = vad.feed(samples) + vad.flush()
    return [Piece(e.audio_start_ms / 1000, e.audio) for e in events if isinstance(e, SpeechStopped)]


def segments(found: list[TranscriptionWord]) -> list[TranscriptionSegment]:
    """Sentences: a segment closes after a word that ends one."""
    result: list[TranscriptionSegment] = []
    current: list[TranscriptionWord] = []
    for word in found:
        current.append(word)
        if word.word.endswith(_SENTENCE_ENDS):
            result.append(_segment(len(result), current))
            current = []
    if current:
        result.append(_segment(len(result), current))
    return result


def _segment(index: int, sentence: list[TranscriptionWord]) -> TranscriptionSegment:
    text = " ".join(word.word for word in sentence)
    return TranscriptionSegment(id=index, start=sentence[0].start, end=sentence[-1].end, text=text)


def _encode_transcript_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class SherpaOnnxSttInfer(SherpaOnnxInfer):
    """sherpa-onnx offline speech to text."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        assert isinstance(self.registry_entry, AsrEntry)
        self.entry: AsrEntry = self.registry_entry
        self.family = _FAMILIES[self.entry.family]
        self.recognizer: Any = None

    def shutdown(self) -> None:
        self.recognizer = None

    def __del__(self):
        self.shutdown()

    def _load(self) -> None:
        self.recognizer = self.family.recognizer(self.entry, self.bundle_dir(), self.threads, PROVIDER)

    def _decode(self, piece: Piece) -> Decoded:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, piece.samples)
        self.recognizer.decode_stream(stream)
        result = stream.result
        found = self.family.words(list(result.tokens), list(result.timestamps), list(result.durations), piece.offset)
        return Decoded(result.text.strip(), found)

    def _decode_all(self, parts: list[Piece], stop: threading.Event, emit: Callable[[Decoded], None]) -> None:
        for part in parts:
            if stop.is_set():
                return
            emit(self._decode(part))

    def _prepare(self, audio_data: bytes) -> tuple[list[Piece], float]:
        samples, duration = decode_audio(audio_data, SAMPLE_RATE)
        return pieces(np.asarray(samples, dtype=np.float32), self.model_config._resolved_vad), duration

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        request_id = f"stt-{base_request_id(raw_request)}"
        if request.prompt or request.language:
            logger.debug("%s: %s takes no prompt or language; ignoring them", request_id, self.model_config.name)
        try:
            parts, duration = await self.run_blocking(lambda _stop: self._prepare(audio_data), exclusive=False)
        except Exception as e:
            logger.warning("%s: failed to decode audio: %s", request_id, e)
            return create_error_response(f"failed to decode audio: {e}")
        logger.info(
            "transcription request %s: %.1f s in %d piece(s), stream=%s",
            request_id,
            duration,
            len(parts),
            request.stream,
        )
        if request.stream:
            return self._stream(parts, request_id)

        decoded: list[Decoded] = []
        await self.run_blocking(lambda stop: self._decode_all(parts, stop, decoded.append), exclusive=False)
        text = " ".join(piece.text for piece in decoded if piece.text)
        logger.log(TRACE, "%s response: text=%r", request_id, text)
        usage = TranscriptionUsageAudio(seconds=duration)
        if request.response_format != "verbose_json":
            return TranscriptionResponse(text=text, usage=usage)
        found = [word for piece in decoded for word in piece.words]
        return TranscriptionResponseVerbose(
            language=request.language or _UNKNOWN_LANGUAGE,
            duration=duration,
            text=text,
            words=found if "word" in request.timestamp_granularities else [],
            segments=segments(found),
            usage=usage,
        )

    async def _stream(self, parts: list[Piece], request_id: str) -> AsyncGenerator[str, None]:
        """One `transcript.text.delta` per decoded piece."""
        decoded = cast(
            AsyncGenerator[Decoded, None],
            self.stream_blocking(lambda emit, stop: self._decode_all(parts, stop, emit), exclusive=False),
        )
        texts: list[str] = []
        try:
            async with contextlib.aclosing(decoded):
                async for piece in decoded:
                    if piece.text:
                        texts.append(piece.text)
                        yield _encode_transcript_event({"type": "transcript.text.delta", "delta": piece.text})
            yield _encode_transcript_event({"type": "transcript.text.done", "text": " ".join(texts)})
        except Exception as exc:
            logger.warning("%s failed mid-stream: %s", request_id, exc)
            yield encode_error_sse(
                create_error_response(f"sherpa-onnx transcription failed: {exc}", err_type="api_error", status_code=502)
            )
