"""The streams of a streaming speech-to-text replica, and the one thread that decodes them.

The thread takes the jobs callers queue, feeds their audio, and decodes the streams that have a chunk ready together
with the recognizer's `decode_streams`, one chunk each per round, taking new jobs between rounds. A job is answered
once its stream has no chunk left to decode. A stream opens with `StreamFeed.lead_ms` of zeros; a finished one takes
`StreamFeed.tail_ms` of zeros, less the quiet audio it ends in. A stream no job reached for `IDLE_S` is dropped.
"""

import concurrent.futures
import contextlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modelship.infer.sherpa_onnx.registry import StreamFeed
from modelship.logging import get_logger

logger = get_logger("infer.sherpa_onnx")

# a stream no job reached for this long is dropped
IDLE_S = 60.0
# how often the decode thread looks for idle streams while it has nothing to do
SWEEP_S = 10.0
# a 10 ms frame whose peak stays below this is quiet
QUIET_PEAK = 0.01


@dataclass(frozen=True)
class StreamResult:
    text: str
    language: str | None
    audio_ms: float
    tokens: list[str]
    # seconds into the caller's audio
    timestamps: list[float]


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
    ) -> "concurrent.futures.Future[StreamResult]":
        """Appends *samples* (float32 at *rate*) to stream *key*, opening it with *language* when there is none, and
        ends it when *finish*; the stream's text once it has no chunk left to decode."""
        future: concurrent.futures.Future[StreamResult] = concurrent.futures.Future()
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

    def _result(self, stream: _Stream) -> StreamResult:
        found = self._recognizer.get_result_all(stream.handle)
        lead = self._feed.lead_ms / 1000
        return StreamResult(
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
