"""Streaming speech to text on sherpa-onnx: the registry's streaming `AsrEntry` bundles, through
`transcribe_push`/`transcribe_finish`/`transcribe_close` and `/v1/audio/transcriptions`.

One thread per replica holds every stream. It takes the jobs the event loop queues, feeds their audio, and decodes
the streams that have a chunk ready together with `decode_streams`, one chunk each per round, taking new jobs
between rounds. A job is answered once its stream has no chunk left to decode. A stream opens with
`StreamFeed.lead_ms` of zeros; a finished one takes `StreamFeed.tail_ms` of zeros, less the quiet audio it ends
in. A stream no job reached for `IDLE_S` is dropped.
"""

import asyncio
import concurrent.futures
import contextlib
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np

from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.sherpa_onnx.family import nemo_streaming_transducer
from modelship.infer.sherpa_onnx.registry import AsrEntry, StreamFeed
from modelship.infer.sherpa_onnx.sherpa_onnx_infer import PROVIDER, SherpaOnnxInfer
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

# a stream no job reached for this long is dropped
IDLE_S = 60.0
# how often the decode thread looks for idle streams while it has nothing to do
SWEEP_S = 10.0
# `/v1/audio/transcriptions` with stream=true feeds a file this much at a time, one delta each
FILE_STEP_S = 1.0
# a 10 ms frame whose peak stays below this is quiet
QUIET_PEAK = 0.01
# verbose_json needs a language; nemotron reports none
_UNKNOWN_LANGUAGE = "unknown"
# `AsrEntry.family` -> its module, which builds the recognizer and reads words from its tokens
_FAMILIES: dict[str, ModuleType] = {"nemo_streaming_transducer": nemo_streaming_transducer}


@dataclass
class _Stream:
    handle: Any
    rate: int
    language: str | None
    # the caller's samples, at `rate`
    received: int = 0
    # the quiet audio the stream ends in
    quiet_ms: float = 0.0
    used: float = field(default_factory=time.monotonic)


@dataclass
class _Job:
    key: str
    samples: np.ndarray
    rate: int
    language: str | None
    finish: bool
    future: concurrent.futures.Future


@dataclass(frozen=True)
class _Close:
    key: str


@dataclass(frozen=True)
class _Result:
    text: str
    language: str | None
    audio_ms: float
    tokens: list[str]
    # seconds into the caller's audio
    timestamps: list[float]


def quiet_tail_ms(samples: np.ndarray, rate: int) -> float | None:
    """How much quiet audio *samples* end in, in 10 ms frames; None when every frame is quiet."""
    step = rate // 100
    count = len(samples) // step
    if count == 0:
        return None
    peaks = np.abs(samples[len(samples) - count * step :].reshape(count, step)).max(axis=1)
    loud = np.nonzero(peaks >= QUIET_PEAK)[0]
    if len(loud) == 0:
        return None
    return (count - 1 - loud[-1]) * 10.0


class StreamDecoder:
    """The decode thread and the streams it holds. Thread-safe to submit to; every recognizer call runs on its
    thread."""

    def __init__(self, recognizer: Any, feed: StreamFeed, name: str):
        self._recognizer = recognizer
        self._feed = feed
        self._jobs: queue.SimpleQueue[_Job | _Close | None] = queue.SimpleQueue()
        self._streams: dict[str, _Stream] = {}
        self._pending: dict[str, list[_Job]] = {}
        self._swept = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"stt-decode-{name}", daemon=True)
        self._thread.start()

    def submit(
        self, key: str, samples: np.ndarray, rate: int, language: str | None, finish: bool
    ) -> "concurrent.futures.Future[_Result]":
        future: concurrent.futures.Future[_Result] = concurrent.futures.Future()
        self._jobs.put(_Job(key, samples, rate, language, finish, future))
        return future

    def close(self, key: str) -> None:
        self._jobs.put(_Close(key))

    def stop(self) -> None:
        self._jobs.put(None)

    @property
    def streams(self) -> int:
        return len(self._streams)

    def _run(self) -> None:
        while True:
            try:
                taken = self._take()
            except _StoppedError as stopped:
                error = RuntimeError("the speech-to-text model shut down")
                for job in stopped.queued:
                    _settle(job.future, error=error)
                self._fail_all(error)
                return
            for item in taken:
                if isinstance(item, _Close):
                    self._close(item.key)
                else:
                    self._apply(item)
            try:
                self._decode_round()
            except Exception as e:
                logger.exception("decoding failed")
                self._fail_all(e)
            self._sweep()

    def _take(self) -> list[_Job | _Close]:
        """The queued jobs; waits for one when no stream has a job waiting."""
        taken: list[_Job | _Close | None] = []
        if not self._pending:
            with contextlib.suppress(queue.Empty):
                taken.append(self._jobs.get(timeout=SWEEP_S))
        with contextlib.suppress(queue.Empty):
            while True:
                taken.append(self._jobs.get_nowait())
        if None in taken:
            raise _StoppedError([item for item in taken if isinstance(item, _Job)])
        return [item for item in taken if item is not None]

    def _close(self, key: str) -> None:
        """Drops stream *key*; the jobs waiting on it fail."""
        self._streams.pop(key, None)
        for job in self._pending.pop(key, []):
            _settle(job.future, error=KeyError(f"stream {key!r} was closed"))

    def _apply(self, job: _Job) -> None:
        stream = self._streams.get(job.key)
        try:
            if stream is not None and stream.rate != job.rate:
                raise ValueError(f"stream {job.key!r} takes {stream.rate} Hz audio, not {job.rate} Hz")
            if stream is None:
                stream = self._open(job)
            if len(job.samples):
                stream.handle.accept_waveform(job.rate, job.samples)
                stream.received += len(job.samples)
                quiet = quiet_tail_ms(job.samples, job.rate)
                stream.quiet_ms = stream.quiet_ms + len(job.samples) / job.rate * 1000 if quiet is None else quiet
            if job.finish:
                pad = max(0.0, self._feed.tail_ms - stream.quiet_ms)
                stream.handle.accept_waveform(job.rate, np.zeros(int(job.rate * pad / 1000), dtype=np.float32))
                stream.handle.input_finished()
        except Exception as e:
            self._streams.pop(job.key, None)
            _settle(job.future, error=e)
            return
        stream.used = time.monotonic()
        self._pending.setdefault(job.key, []).append(job)

    def _open(self, job: _Job) -> _Stream:
        handle = self._recognizer.create_stream()
        if job.language:
            handle.set_option("language", job.language)
        handle.accept_waveform(job.rate, np.zeros(job.rate * self._feed.lead_ms // 1000, dtype=np.float32))
        stream = _Stream(handle, job.rate, job.language)
        self._streams[job.key] = stream
        return stream

    def _decode_round(self) -> None:
        """Decodes one chunk of every stream with a job waiting that has one ready, then answers the jobs whose
        stream has none left."""
        recognizer = self._recognizer
        ready = [
            stream.handle
            for key in self._pending
            if (stream := self._streams.get(key)) is not None and recognizer.is_ready(stream.handle)
        ]
        if ready:
            recognizer.decode_streams(ready)
        for key in list(self._pending):
            stream = self._streams.get(key)
            if stream is not None and recognizer.is_ready(stream.handle):
                continue
            jobs = self._pending.pop(key)
            if stream is None:
                for job in jobs:
                    _settle(job.future, error=KeyError(f"stream {key!r} was closed"))
                continue
            result = self._result(stream)
            if any(job.finish for job in jobs):
                del self._streams[key]
            for job in jobs:
                _settle(job.future, result)

    def _result(self, stream: _Stream) -> _Result:
        found = self._recognizer.get_result_all(stream.handle)
        lead = self._feed.lead_ms / 1000
        return _Result(
            text=found.text.strip(),
            language=stream.language,
            audio_ms=stream.received / stream.rate * 1000,
            tokens=list(found.tokens),
            timestamps=[max(0.0, t - lead) for t in found.timestamps],
        )

    def _sweep(self) -> None:
        now = time.monotonic()
        if now - self._swept < SWEEP_S:
            return
        self._swept = now
        idle = [k for k, s in self._streams.items() if k not in self._pending and now - s.used >= IDLE_S]
        for key in idle:
            del self._streams[key]
        if idle:
            logger.info("dropped %d stream(s) idle for %.0f s", len(idle), IDLE_S)

    def _fail_all(self, error: Exception) -> None:
        for key, jobs in self._pending.items():
            self._streams.pop(key, None)
            for job in jobs:
                _settle(job.future, error=error)
        self._pending.clear()


class _StoppedError(Exception):
    def __init__(self, queued: list[_Job]):
        super().__init__("stopped")
        # the jobs taken with the stop
        self.queued = queued


def _settle(future: concurrent.futures.Future, result: Any = None, error: Exception | None = None) -> None:
    """Answers *future* unless its caller cancelled it."""
    with contextlib.suppress(concurrent.futures.InvalidStateError):
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)


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
        return SttCapabilities(streaming=True, reports_language=False)

    async def _run(self, key: str, samples: np.ndarray, rate: int, language: str | None, finish: bool) -> _Result:
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
