"""`PipelineCalls` in a pipeline replica: a session's calls to its pipeline's models through their Serve handles,
with the client's headers, identity and the disconnect registry bound."""

from collections.abc import AsyncGenerator
from typing import Any, cast

from ray.exceptions import RayActorError, RayTaskError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

from modelship.infer.pipelines.assistant.calls import Answer
from modelship.infer.pipelines.routes import ModelRoutes, ModelState
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    SttCapabilities,
    SttStreamOptions,
    SttStreamText,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    validation_error_from_cause,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.utils import random_uuid

logger = get_logger("realtime")


async def _respond_stream(
    handle: DeploymentHandle,
    request: ResponsesRequest,
    headers: dict[str, str],
    registry: Any,
    identity: str,
    session_id: str,
) -> AsyncGenerator[Any, None]:
    """*handle*'s streamed answer to *request*, routed on *session_id*; a validation failure in the replica comes
    back as an ErrorResponse. Closed early, it cancels the call and tells the replica through *registry*."""
    req_id = random_uuid()
    stream = cast(
        "DeploymentResponseGenerator[Any]",
        handle.respond.options(stream=True, session_id=session_id).remote(request, headers, registry, req_id, identity),
    )
    finished = False
    try:
        try:
            async for event in stream:
                yield event
        except RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, BaseException) else None
            if not isinstance(cause, ValueError):
                raise
            yield validation_error_from_cause(cause)
        finished = True
    finally:
        if not finished:
            await _stop_call(stream, registry, req_id)


async def _stop_call(stream: DeploymentResponseGenerator, registry: Any, req_id: str) -> None:
    """Stops the model call behind *stream*, which its caller left early."""
    # cancelling the task that awaits the stream doesn't reach the replica
    stream.cancel()
    try:
        await registry.set.remote(req_id)
    except RayActorError:
        logger.warning("Disconnect registry unavailable; lost the cancel of %s", req_id)


class ModelCalls:
    """A session's `PipelineCalls`. *routes* is the replica's model table."""

    def __init__(self, routes: ModelRoutes, headers: dict[str, str], registry: Any, identity: str):
        self._routes = routes
        self._headers = headers
        self._registry = registry
        self._identity = identity

    def model_state(self, model: str) -> ModelState:
        return self._routes.state(model)

    def respond(self, model: str, request: ResponsesRequest, session_id: str) -> Answer | None:
        handle = self._routes.handle(model)
        if handle is None:
            return None
        return _respond_stream(handle, request, self._headers, self._registry, self._identity, session_id)

    async def stt_capabilities(self, model: str) -> SttCapabilities | None:
        handle = self._routes.handle(model)
        if handle is None:
            return None
        return await handle.stt_capabilities.remote()

    async def transcribe(
        self, model: str, audio: bytes, request: TranscriptionRequest, session_id: str
    ) -> TranscriptionResponse | TranscriptionResponseVerbose | ErrorResponse | None:
        return await self._request("transcribe", model, session_id, audio, request)

    async def transcribe_push(
        self, model: str, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> SttStreamText | ErrorResponse | None:
        return await self._stream_call("transcribe_push", model, key, pcm, rate, options, random_uuid(), self._identity)

    async def transcribe_finish(
        self, model: str, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> SttStreamText | ErrorResponse | None:
        return await self._stream_call(
            "transcribe_finish", model, key, pcm, rate, options, random_uuid(), self._identity
        )

    async def transcribe_close(self, model: str, key: str) -> None:
        await self._stream_call("transcribe_close", model, key)

    async def speak(
        self, model: str, request: SpeechRequest, session_id: str
    ) -> RawSpeechResponse | ErrorResponse | None:
        return await self._request("speak", model, session_id, request)

    async def _request(self, method: str, model: str, session_id: str, *args: Any) -> Any:
        """Calls handle method *method* of *model* with *args* the way the gateway calls a model: the client's
        headers, the disconnect registry, a request id and the client's identity follow them, and the call is routed
        on *session_id*. The one item it streams back; None for none, or while nothing serves *model*."""
        handle = self._routes.handle(model)
        if handle is None:
            return None
        req_id = random_uuid()
        call = getattr(handle, method).options(stream=True, session_id=session_id)
        stream = cast(
            "DeploymentResponseGenerator[Any]",
            call.remote(*args, self._headers, self._registry, req_id, self._identity),
        )
        finished = False
        try:
            results = [result async for result in stream]
            finished = True
        finally:
            if not finished:
                await _stop_call(stream, self._registry, req_id)
        return results[0] if results else None

    async def _stream_call(self, method: str, model: str, key: str, *args: Any) -> Any:
        """Calls handle method *method* of *model* on stream *key*, with *args* after the key; None while nothing
        serves *model*. The call carries the key as its session_id, which pins the stream to the replica holding
        it."""
        handle = self._routes.handle(model)
        if handle is None:
            return None
        return await getattr(handle, method).options(session_id=key).remote(key, *args)
