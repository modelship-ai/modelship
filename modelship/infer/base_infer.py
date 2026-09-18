import asyncio
import contextlib
import os
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, NoReturn, TypeVar

from ray import serve
from ray.exceptions import RayActorError

from modelship.infer import infer_config
from modelship.infer.downloads import locked_download
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.sources import ModelDownloadError, ModelSourceError
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
    RawSpeechResponse,
    ResponseObject,
    ResponsesRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from modelship.openai.protocol.responses.streaming import ResponsesStreamTranslator

logger = get_logger("infer")

_NOT_SUPPORTED = ErrorResponse(
    error=ErrorInfo(message="model does not support this action", type="invalid_request_error")
)
_NOT_SUPPORTED._http_status = 404

# 44-byte WAV header + 2 bytes of silence (one 16-bit sample at 16 kHz mono)
_MINIMAL_WAV_HEADER = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    36 + 2,
    b"WAVE",  # RIFF chunk
    b"fmt ",
    16,
    1,
    1,
    16000,
    32000,
    2,
    16,  # fmt sub-chunk: PCM, mono, 16 kHz, 16-bit
    b"data",
    2,  # data sub-chunk: 2 bytes
)
MINIMAL_WAV = _MINIMAL_WAV_HEADER + b"\x00\x00"

_DISCONNECT_POLL_INTERVAL_S = 0.1

T = TypeVar("T")


class ClientDisconnectedError(Exception):
    """Raised by `BaseInfer.run_cancellable` when the client disconnects before
    the guarded work finishes."""


class BaseInfer[Prepared](ABC):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_config = model_config
        # request_id -> local event, set by the shared disconnect pump below.
        # One pump per replica (this instance) amortizes disconnect polling
        # across every request the replica is currently serving, instead of
        # each request polling the DisconnectRegistry actor independently.
        self._watched: dict[str, asyncio.Event] = {}
        self._pump_task: asyncio.Task[None] | None = None
        # DeployCoordinator handle, resolved on first use.
        self._coordinator: Any = None

    def _get_memory_fraction(self) -> float | None:
        """Return the GPU memory fraction if explicitly set and < 1.0, otherwise None."""
        if self.model_config.num_gpus > 0 and self.model_config.num_gpus < 1.0:
            return self.model_config.num_gpus
        return None

    @staticmethod
    async def ensure_downloaded(model_config: ModelshipModelConfig) -> None:
        """Actor-side hook: download (or confirm already-cached) this
        deployment's model weights, then stamp the final path(s) onto
        `model_config`. Called before the loader is constructed, since
        preflight needs the file on disk during the loader's own `__init__`.

        Downloads hold a cluster-wide lease (see `locked_download`).
        Failures are wrapped in `ModelDownloadError` so they classify
        as transient (retried next pass), except `ModelSourceError`, which
        stays fatal. Idempotent once `_resolved_path` is set."""
        if model_config._pinned_source is not None and model_config._resolved_path is None:
            try:
                model_config._resolved_path = await locked_download(model_config._pinned_source, model_config.name)
            except ModelSourceError:
                raise
            except Exception as e:
                raise ModelDownloadError(f"Failed to download model for '{model_config.name}': {e}") from e
            logger.info("Downloaded '%s' -> %s", model_config.name, model_config._resolved_path)

        llama_cfg = model_config.llama_server_config
        if llama_cfg is not None and llama_cfg._pinned_mmproj is not None:
            # Overwrites the public `mmproj` field with the final path.
            # Clearing the pin makes a second call a no-op.
            try:
                llama_cfg.mmproj = await locked_download(llama_cfg._pinned_mmproj, model_config.name)
            except ModelSourceError:
                raise
            except Exception as e:
                raise ModelDownloadError(f"Failed to download mmproj for '{model_config.name}': {e}") from e
            llama_cfg._pinned_mmproj = None
            logger.info("Downloaded mmproj for '%s' -> %s", model_config.name, llama_cfg.mmproj)

    async def run_cancellable(self, work: Coroutine[Any, Any, T], raw_request: RawRequestProxy) -> T:
        """Run `work` to completion, or cancel it and raise `ClientDisconnectedError`
        if the client disconnects first.

        A non-streaming Ray Serve call has no socket to watch: unlike streaming
        (where Starlette's own `StreamingResponse` races disconnect against the
        body iterator and cancellation propagates down through the whole chain
        automatically), a single-shot non-stream call would otherwise run to
        completion for a client that's already gone. This polls
        `RawRequestProxy.is_disconnected()` (the same cross-process
        DisconnectRegistry signal the streaming path's disconnect ultimately
        traces back to) alongside `work` and cancels whichever loses.

        Cancelling the task is often sufficient by itself — e.g. vLLM's
        `AsyncLLM.generate()` aborts its own engine-side request when its
        consuming task is cancelled, needing no extra cleanup here. Loaders
        whose engine needs cleanup beyond task cancellation (freeing a
        connection/slot, etc.) should override `on_generation_aborted`.
        """
        event = self._watch_disconnect(raw_request)
        task = asyncio.ensure_future(work)
        watch = asyncio.ensure_future(event.wait())
        try:
            done, _pending = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await self.on_generation_aborted()
            raise ClientDisconnectedError
        finally:
            # Unconditional, regardless of how the try exits — including this
            # coroutine's own task being cancelled from outside (e.g. replica
            # shutdown) while suspended in asyncio.wait above, which otherwise
            # leaves `task` (the actual inference work) running unobserved.
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._unwatch_disconnect(raw_request)

    async def run_cancellable_stream(
        self, work: AsyncGenerator[T, None], raw_request: RawRequestProxy
    ) -> AsyncGenerator[T, None]:
        """Streaming counterpart of `run_cancellable`.

        Races each pulled item against the same disconnect signal: cancelling
        the in-flight `__anext__()` call delivers `CancelledError` straight into
        `work`'s currently-suspended frame (and transitively into whatever it's
        awaiting, e.g. an engine's own generator), the same way cancelling a
        plain task does for `run_cancellable`. The `finally` block's `aclose()`
        is what actually guarantees `work` is closed on every exit path —
        including the consumer closing *this* generator early (`GeneratorExit`
        propagating out of the `yield`), which the disconnect branch's own
        `aclose()` above doesn't cover. It's a defensive no-op wherever `work`
        already self-terminated.

        `next_item` is tracked outside the loop so `finally` can reach it: if
        this generator is torn down (cancelled or `aclose()`d) while suspended
        in the `asyncio.wait` below rather than at the `yield`, `next_item`'s
        `__anext__()` call is still in flight and still owns `work`'s frame —
        calling `work.aclose()` before that settles raises `RuntimeError:
        aclose(): asynchronous generator is already running`. It must be
        cancelled and awaited first, same as the disconnect branch above does.
        """
        event = self._watch_disconnect(raw_request)
        watch = asyncio.ensure_future(event.wait())
        next_item: asyncio.Task[T] | None = None
        try:
            while True:
                next_item = asyncio.ensure_future(work.__anext__())
                done, _pending = await asyncio.wait({next_item, watch}, return_when=asyncio.FIRST_COMPLETED)
                if next_item in done:
                    try:
                        item = next_item.result()
                    except StopAsyncIteration:
                        return
                    yield item
                    continue
                next_item.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item
                await work.aclose()
                await self.on_generation_aborted()
                raise ClientDisconnectedError
        finally:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            self._unwatch_disconnect(raw_request)
            if next_item is not None and not next_item.done():
                next_item.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item
            await work.aclose()

    async def _stream_responses(
        self,
        request: ResponsesRequest,
        chunks: AsyncGenerator[ChatCompletionStreamResponse, None],
        *,
        request_id: str,
        client_error: Callable[[Exception], str | None] = lambda _exc: None,
        response_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Drive a Responses event stream: start() -> process() per chunk ->
        finish()/fail(). Yields plain event dicts; SSE/WS framing happens at the
        transport edge. `chunks.aclose()` in `finally` is required since closing
        this generator early doesn't propagate into `chunks` otherwise. `response_id`,
        when given, pins the id instead of the translator minting its own.
        """
        translator = ResponsesStreamTranslator(request, response_id=response_id)
        try:
            for event in translator.start():
                yield event
            try:
                async for chunk in chunks:
                    for event in translator.process(chunk):
                        yield event
            except ClientDisconnectedError:
                logger.info("responses request %s aborted: client disconnected", request_id)
                return
            except Exception as exc:
                message = client_error(exc)
                if message is None:
                    logger.exception("responses request %s failed mid-stream", request_id)
                    message = "Internal error during generation"
                for event in translator.fail(message):
                    yield event
                return
            for event in translator.finish():
                yield event
        finally:
            await chunks.aclose()

    async def on_generation_aborted(self) -> None:
        """Hook for loaders whose engine needs cleanup beyond task cancellation
        when `run_cancellable` aborts a request on client disconnect. No-op by
        default — most engines are naturally cleaned up by cancellation alone
        (or, like a blocking call already running in a thread pool, can't be
        interrupted early regardless of what happens here)."""
        return None

    def _watch_disconnect(self, raw_request: RawRequestProxy) -> asyncio.Event:
        """Register `raw_request` with the shared per-replica disconnect pump and
        return a local event that fires once it disconnects. Unwatchable proxies
        (no registry/id — e.g. an internal warmup request) get an event that
        simply never fires, matching `is_disconnected()`'s "always connected"
        behavior for them.
        """
        event = asyncio.Event()
        if not raw_request.is_watchable:
            return event
        assert raw_request.request_id is not None
        self._watched[raw_request.request_id] = event
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.ensure_future(self._disconnect_pump())
        return event

    def _unwatch_disconnect(self, raw_request: RawRequestProxy) -> None:
        """Drop `raw_request` from the pump's watch set. Stops the pump once
        nothing is left to poll, rather than leaving it spinning idle. Mirrors
        `_watch_disconnect`'s `is_watchable` guard — an unwatchable proxy was
        never registered, so there's nothing to remove."""
        if not raw_request.is_watchable:
            return
        assert raw_request.request_id is not None
        self._watched.pop(raw_request.request_id, None)
        if not self._watched and self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    async def _disconnect_pump(self) -> None:
        """One background poller per replica, shared by every in-flight request's
        `run_cancellable`/`run_cancellable_stream` call. Batches what would
        otherwise be one DisconnectRegistry RPC per request per poll interval
        into a single `is_set_many` RPC per interval, fanning results out to
        each request's local event.
        """
        while self._watched:
            disconnected = await self._poll_disconnected_ids(list(self._watched))
            for request_id in disconnected:
                event = self._watched.get(request_id)
                if event is not None:
                    event.set()
            await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)

    @staticmethod
    async def _poll_disconnected_ids(request_ids: list[str]) -> list[str]:
        """Injectable seam for `_disconnect_pump`: which of `request_ids` are
        disconnected right now, per the shared DisconnectRegistry. Degrades to
        "none disconnected" on any failure, not just a dead actor — this pump is
        shared by every concurrent request on the replica, so letting an
        unhandled exception escape would silently kill disconnect detection for
        all of them at once (until a later request happens to restart the pump),
        not just break one request's poll the way the old per-request loop did.
        """
        try:
            return await infer_config.get_disconnect_registry().is_set_many.remote(request_ids)
        except RayActorError:
            logger.warning("Disconnect registry unavailable; assuming clients connected")
            infer_config.reset_disconnect_registry()
            return []
        except Exception as exc:
            logger.warning("Unexpected error polling disconnect registry; assuming clients connected: %r", exc)
            return []

    def backend_died(self, reason: str) -> NoReturn:
        """Report the death to the deploy coordinator, then kill this replica.
        Never returns; a failed report is logged and exits anyway.

        The report is submitted, not awaited: Serve retries an ActorDiedError, so
        time spent alive with a dead backend turns that retry into a client 502."""
        try:
            config = self.model_config
            ceiling = config.autoscaling_config.max_replicas if config.autoscaling_config else config.num_replicas
            self._deploy_coordinator().report_replica_death.remote(
                os.environ.get("MSHIP_GATEWAY_NAME", ""),
                serve.get_replica_context().app_name,
                config.name,
                ceiling,
                reason,
            )
        except Exception:
            logger.exception("Failed to report backend death for '%s'", self.model_config.name)
        logger.error("Exiting actor for '%s': %s", self.model_config.name, reason)
        os._exit(1)

    def _deploy_coordinator(self):
        if self._coordinator is None:
            from modelship.infer.deploy_coordinator import get_or_create_coordinator

            self._coordinator = get_or_create_coordinator()
        return self._coordinator

    @abstractmethod
    def shutdown(self) -> None:
        """Synchronously release resources (engine processes, GPU memory, etc.).

        Called during graceful teardown. Subclasses must implement to clean up
        loader-specific resources.
        """

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def warmup(self) -> None:
        """Run a minimal inference pass to warm up the model (CUDA kernels, caches, etc.).

        Subclasses should override this to send a tiny dummy request through
        their actual inference path. The default is a no-op for loaders that
        don't need warmup.
        """

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        """Prepare the request, then dispatch to the loader's stream/no-stream seam.

        `_prepare_chat` is the single place a loader shapes/renders the request and
        can fail — a pre-generation failure (bad content, context overflow, ...)
        comes back here as a plain `ErrorResponse`, matching `create_response` and
        OpenAI's own behavior: a streaming request that fails before any token is
        produced gets a normal HTTP 4xx JSON body, not a `200` SSE stream carrying
        an error chunk.
        """
        prepared = await self._prepare_chat(request, raw_request)
        if isinstance(prepared, ErrorResponse):
            return prepared
        if request.stream:
            return self._create_chat_completion_stream(request, prepared, raw_request)
        return await self._create_chat_completion_no_stream(request, prepared, raw_request)

    async def create_response(
        self, request: ResponsesRequest, raw_request: RawRequestProxy, *, response_id: str | None = None
    ) -> ErrorResponse | ResponseObject | AsyncGenerator[dict[str, Any], None]:
        """Responses counterpart of `create_chat_completion` — see its docstring.
        `response_id`, when given, is threaded to the streaming seam (unused on
        non-stream: a background run always sets `stream=True`)."""
        prepared = await self._prepare_responses(request, raw_request)
        if isinstance(prepared, ErrorResponse):
            return prepared
        if request.stream:
            return self._create_response_stream(request, prepared, raw_request, response_id=response_id)
        return await self._create_response_no_stream(request, prepared, raw_request)

    async def _prepare_chat(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | Prepared:
        """Shape/validate/render a chat request into whatever the loader's
        stream/no-stream seams need. Default: unsupported. Loaders that override
        `create_chat_completion`'s seams must override this instead."""
        return _NOT_SUPPORTED

    async def _prepare_responses(
        self, request: ResponsesRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | Prepared:
        """Responses counterpart of `_prepare_chat`."""
        return _NOT_SUPPORTED

    def _create_chat_completion_stream(
        self, request: ChatCompletionRequest, prepared: Prepared, raw_request: RawRequestProxy
    ) -> AsyncGenerator[str, None]:
        """Streaming chat seam. `prepared` is whatever `_prepare_chat` returned —
        rendering/validation already succeeded, so this only drives generation and
        encodes SSE chunks. Unreachable unless a loader overrides `_prepare_chat`
        without overriding this."""
        raise NotImplementedError

    async def _create_chat_completion_no_stream(
        self, request: ChatCompletionRequest, prepared: Prepared, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse:
        """Non-streaming chat seam. See `_create_chat_completion_stream`."""
        raise NotImplementedError

    def _create_response_stream(
        self,
        request: ResponsesRequest,
        prepared: Prepared,
        raw_request: RawRequestProxy,
        *,
        response_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming Responses seam. See `_create_chat_completion_stream`. `response_id`
        is forwarded straight to `_stream_responses`."""
        raise NotImplementedError

    async def _create_response_no_stream(
        self, request: ResponsesRequest, prepared: Prepared, raw_request: RawRequestProxy
    ) -> ErrorResponse | ResponseObject:
        """Non-streaming Responses seam. See `_create_chat_completion_stream`."""
        raise NotImplementedError

    async def create_embedding(self, request: EmbeddingRequest, raw_request: RawRequestProxy) -> ErrorResponse:
        return _NOT_SUPPORTED

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED
