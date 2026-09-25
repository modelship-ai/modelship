import asyncio
import contextlib
import json
import os
import time
from http import HTTPStatus
from typing import Annotated, Any, cast

import ray
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from ray import serve
from ray.exceptions import RayActorError, RayTaskError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from modelship.infer import replica_coordinator
from modelship.infer.infer_config import RequestWatcher, get_disconnect_registry
from modelship.logging import configure_logging, get_logger
from modelship.metrics import (
    GATEWAY_RECONCILES_TOTAL,
    GATEWAY_ROUTING_GENERATION,
    GATEWAY_WATCH_ERRORS_TOTAL,
    MODELS_LOADED,
    REQUEST_DURATION_SECONDS,
    REQUEST_ERRORS_TOTAL,
    REQUEST_IN_PROGRESS,
    REQUEST_TOTAL,
    STREAM_CHUNKS_TOTAL,
    stamp_gateway,
)
from modelship.openai.auth import ApiKeyMiddleware, check_ws_auth, get_api_keys, resolve_identity
from modelship.openai.mcp import loop as mcp_loop
from modelship.openai.protocol import (
    TERMINAL_EVENT_TYPES,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompactRequest,
    EmbeddingRequest,
    EmbeddingResponse,
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
    create_error_response,
    encode_sse_event,
    error_ws_frame,
    failure_event_from_frames,
    frame_sse,
)
from modelship.openai.protocol.responses.adapter import UnsupportedResponsesFeatureError
from modelship.openai.state import responses as responses_state
from modelship.openai.utils import responses as responses_utils
from modelship.openai.utils.chat import encode_error_sse
from modelship.state import get_state_store
from modelship.utils import random_uuid

logger = get_logger("api")

_DEFAULT_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB
# Backoff before retrying the gateway watch loop after a transient coordinator error.
_WATCH_RETRY_S = 5.0
# Cap on a WS connection's local store:false continuation cache; oldest entry evicted first.
_WS_CONN_CACHE_MAX = 16


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int = _DEFAULT_MAX_BODY_BYTES):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (limit: {self.max_bytes} bytes)"},
            )
        return await call_next(request)


def build_app():
    # Runs at import time in both the driver process (binding the deployment) and
    # each replica process — before ModelshipAPI.__init__'s configure_logging() call
    # — so it must stay logging-free; ModelshipAPI.__init__ logs the outcome below instead.
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    max_body_bytes = int(os.environ.get("MSHIP_MAX_REQUEST_BODY_BYTES", _DEFAULT_MAX_BODY_BYTES))
    app.add_middleware(PayloadSizeLimitMiddleware, max_bytes=max_body_bytes)

    api_keys = get_api_keys()
    if api_keys:
        app.add_middleware(ApiKeyMiddleware, api_keys=api_keys)

    @app.exception_handler(RequestValidationError)
    async def log_validation_error(request: Request, exc: RequestValidationError):
        logger.warning("%s %s -> 422 validation error: %s", request.method, request.url.path, exc.errors())
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(responses_utils.ResponsesApiError)
    async def log_responses_api_error(request: Request, exc: responses_utils.ResponsesApiError):
        # MRO-based lookup means this wins over the plain-HTTPException handler
        # below, rendering the full OpenAI error envelope instead of {"detail": ...}.
        logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.err._http_status, exc.err.error.message)
        return _error_response(exc.err)

    @app.exception_handler(HTTPException)
    async def log_http_exception(request: Request, exc: HTTPException):
        logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def log_unhandled_exception(request: Request, exc: Exception):
        logger.exception("%s %s -> 500: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = build_app()


class OpenAiModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "modelship"


class OpenaiModelList(BaseModel):
    object: str = "list"
    data: list[OpenAiModelCard] = []


def _error_response(result: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=result.model_dump(mode="json"), status_code=result._http_status)


def _stream_error_chunk(endpoint: str, model: str, created_frame: Any, last_frame: Any) -> str:
    """SSE-encode a failure between chunks, after headers are already committed to 200
    — raising would just abort the connection with no body. Message is generic; details
    go to the server log instead. /v1/responses needs a typed terminal event rebuilt from
    the frames already sent; every other endpoint takes the bare `data:` error object."""
    message = "Internal error during generation"
    if endpoint == "create_response":
        return encode_sse_event(failure_event_from_frames(created_frame, last_frame, message, model=model))
    error = create_error_response(message=message, err_type="api_error", status_code=500)
    return encode_error_sse(error)


def _validation_error_from_cause(cause: BaseException) -> ErrorResponse:
    # OpenAI-style 400 for client-side validation failures (e.g. vLLM's
    # VLLMValidationError on context overflow, which subclasses ValueError).
    # A pydantic ValidationError's .args is only () locally; once it crosses the
    # Ray pickle boundary it becomes (model_title, [line_errors], ...) — args[0]
    # would be a bare model name like "TranscriptionRequest", not a useful
    # message, so always render those through str() instead.
    if isinstance(cause, ValidationError):
        base = str(cause)
    elif cause.args:
        base = cause.args[0]
    else:
        base = str(cause)
    return create_error_response(
        message=base,
        err_type="invalid_request_error",
        status_code=HTTPStatus.BAD_REQUEST,
        param=getattr(cause, "parameter", None),
    )


@serve.deployment
@serve.ingress(app)
class ModelshipAPI:
    def __init__(self, gateway_name: str):
        # Set up modelship-formatted application loggers at the driver's level;
        # MSHIP_LOG_* are forwarded to this replica via runtime_env (see serve_utils).
        configure_logging()
        max_body_bytes = int(os.environ.get("MSHIP_MAX_REQUEST_BODY_BYTES", _DEFAULT_MAX_BODY_BYTES))
        logger.info("Request body limit: %.4g MiB", max_body_bytes / 1024**2)
        api_keys = get_api_keys()
        if api_keys:
            logger.info("API key authentication enabled (%d key(s))", len(api_keys))
        else:
            logger.warning("API key authentication disabled (MSHIP_API_KEYS not set)")
        # model_name -> (app_name -> handle), keyed by app_name so _drop_apps can drop one deployment.
        self.models: dict[str, dict[str, DeploymentHandle]] = {}
        self.model_list: list[OpenAiModelCard] = []
        self.expected_models: list[str] = []
        self._started_at = time.time()
        self._gateway_name = gateway_name
        # /v1/responses conversation state. Construction is inert (no Ray call, no
        # connection), so it costs nothing for a gateway that never serves a stateful
        # request. The gateway owns this rather than the loaders: GET/DELETE carry no
        # model, so they could not be routed to one.
        self._state_store = get_state_store()
        # Detached background:true drain tasks; held so they aren't GC'd mid-run.
        self._background_tasks: set[asyncio.Task] = set()
        stamp_gateway(gateway_name)
        # Routing state is reconciled from the coordinator, the cluster-wide source
        # of truth — not pushed by the driver (a push hits only one replica). Each
        # replica runs a watch loop (started lazily on first request) that pulls a
        # snapshot whenever the coordinator's per-gateway generation advances, so
        # every replica — including restarted / autoscaled ones — converges.
        self._gen = 0  # last coordinator generation this replica reconciled to
        self._watch_task: asyncio.Task | None = None
        self._replica_coord = None  # cached replica-coordinator handle
        # Timing state — the first sync with a non-empty expected set stamps a start;
        # each model's first appearance records the gap since the previous arrival as
        # an approximate load duration.
        self._expected_set_at: float | None = None
        self._last_model_at: float | None = None
        self._all_ready_at: float | None = None
        self._model_load_times: dict[str, float] = {}

    def _register_deployment(self, app_name: str, model_name: str) -> bool:
        """Wire one deployment handle into the routing table. Returns True iff the
        model was newly added. Raises if the app handle isn't resolvable yet (e.g.
        controller lag / app not ready) so the caller can retry this generation
        instead of skipping the model. Sync so the reconcile applies atomically."""
        handle = serve.get_app_handle(app_name)
        newly_added = model_name not in self.models
        if newly_added:
            self.models[model_name] = {}
            self.model_list.append(OpenAiModelCard(id=model_name))
        self.models[model_name][app_name] = handle
        logger.info("Registered deployment: %s (model: %s)", app_name, model_name)
        return newly_added

    def _drop_apps(self, app_names: list[str]) -> list[str]:
        """Drop the given deployment app names from the routing tables. The owning
        model is found by reverse lookup. When a model loses its last deployment its
        model entry, card and load-time entry are also dropped. Returns the names of
        fully-removed models."""
        removed_models: list[str] = []
        for app_name in app_names:
            model_name = next((m for m, handles in self.models.items() if app_name in handles), None)
            if model_name is None:
                continue
            handles = self.models[model_name]
            del handles[app_name]
            logger.info("Unregistered deployment: %s (model: %s)", app_name, model_name)
            if not handles:
                del self.models[model_name]
                self.model_list = [c for c in self.model_list if c.id != model_name]
                self._model_load_times.pop(model_name, None)
                removed_models.append(model_name)
        return removed_models

    def _apply_routing(self, desired: dict[str, str]) -> None:
        """Reconcile the routing table to `desired` ({app_name: model_name}): add
        handles for newly-present apps and drop apps no longer present. Sync /
        await-free → atomic w.r.t. in-flight requests.

        Applies every app that can be registered (and any removals), then raises if
        any could not be. The caller leaves `_gen` unadvanced so the watch loop
        retries — re-pulling the latest snapshot — instead of skipping a model until
        the next unrelated deploy."""
        routed = {app for handles in self.models.values() for app in handles}

        failed: list[str] = []
        for app_name, model_name in desired.items():
            if app_name in routed:
                continue
            try:
                newly_added = self._register_deployment(app_name, model_name)
            except Exception:
                logger.warning("gateway: deferring registration of %s (not ready yet)", app_name, exc_info=True)
                failed.append(app_name)
                continue
            if newly_added:
                base = self._last_model_at or self._started_at
                self._model_load_times[model_name] = round(time.time() - base, 2)
                self._last_model_at = time.time()

        stale = [app for app in routed if app not in desired]
        if stale:
            self._drop_apps(stale)

        if failed:
            raise RuntimeError(f"deployments not yet registerable: {failed}")

    def _apply_snapshot(self, snapshot: dict) -> None:
        """Apply a coordinator routing snapshot to this replica (atomic mutation)."""
        new_gen = snapshot.get("generation", self._gen)
        # before the routes, which raise when an app isn't resolvable yet
        self.expected_models = list(snapshot.get("expected", []))
        self._apply_routing(snapshot.get("models", {}))
        if self.expected_models and self._expected_set_at is None:
            self._expected_set_at = self._last_model_at or time.time()
        if self.expected_models and self._all_ready_at is None and all(m in self.models for m in self.expected_models):
            self._all_ready_at = time.time()
        self._gen = new_gen
        MODELS_LOADED.set(len(self.models))
        GATEWAY_RECONCILES_TOTAL.inc()
        GATEWAY_ROUTING_GENERATION.set(new_gen)

    def _coord(self):
        if self._replica_coord is None:
            self._replica_coord = replica_coordinator.get_or_create_replica_coordinator()
        return self._replica_coord

    async def _coord_async(self):
        """Resolve (and cache) the replica-coordinator handle without blocking the event
        loop. Cached fast path is a no-op; only after a reset (coordinator restart) does
        this do work, and get_or_create's synchronous GCS lookup can stall on a
        recovering GCS — so hop it to a thread to keep concurrent requests flowing."""
        if self._replica_coord is None:
            self._replica_coord = await asyncio.to_thread(replica_coordinator.get_or_create_replica_coordinator)
        return self._replica_coord

    def _ensure_watching(self) -> None:
        """First-request hook: do one synchronous sync so this request isn't blocked
        on the loop's first tick, then start the background watch loop (once)."""
        if self._watch_task is not None:
            return
        self._sync_routing_blocking()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop yet; a later in-loop call starts it
        self._watch_task = loop.create_task(self._watch_loop())

    def _sync_routing_blocking(self) -> bool:
        """One-shot synchronous reconcile (first request). Blocking ray.get is fine
        here — it runs at most once per request until the loop is established."""
        try:
            snapshot = cast(dict, ray.get(self._coord().get_routing.remote(self._gateway_name)))
        except Exception:
            self._replica_coord = None  # re-resolve next time in case the handle went stale
            logger.debug("gateway: initial routing sync deferred; coordinator unavailable", exc_info=True)
            return False
        try:
            self._apply_snapshot(snapshot)
        except Exception:
            # A deployment isn't handle-able yet; defer to the watch loop rather than
            # failing the first request. The replica stays not-ready until it retries.
            logger.debug("gateway: initial routing apply deferred; deployment not ready", exc_info=True)
            return False
        return True

    async def _watch_loop(self) -> None:
        """Long-poll the coordinator and reconcile on every generation change. The
        await happens on the actor call; the reconcile mutation is await-free so it
        stays atomic w.r.t. in-flight requests."""
        while True:
            try:
                coord = await self._coord_async()
                gen = await coord.wait_for_change.remote(self._gateway_name, self._gen)
                if gen != self._gen:
                    snapshot = await coord.get_routing.remote(self._gateway_name)
                    self._apply_snapshot(snapshot)
            except asyncio.CancelledError:
                return
            except Exception:
                self._replica_coord = None
                GATEWAY_WATCH_ERRORS_TOTAL.inc()
                logger.debug("gateway: watch iteration failed; retrying", exc_info=True)
                await asyncio.sleep(_WATCH_RETRY_S)

    def __del__(self):
        task = getattr(self, "_watch_task", None)
        if task is not None and not task.done():
            task.cancel()

    async def list_deployments(self) -> dict[str, list[str]]:
        """Return model_name -> list of deployment app_names currently registered."""
        return {model_name: list(handles.keys()) for model_name, handles in self.models.items()}

    @staticmethod
    def _set_request_id(request_id: str) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    @staticmethod
    def _set_identity(raw_request: Request | WebSocket) -> str:
        from modelship.logging import identity_tier_var, identity_var

        identity, tier = resolve_identity(raw_request)
        identity_var.set(identity)
        identity_tier_var.set(tier)
        return identity

    def _get_handle(self, model_name: str | None) -> DeploymentHandle:
        self._ensure_watching()
        if model_name is None or model_name not in self.models:
            if model_name in self.expected_models:
                raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, detail="model not ready")
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")
        # The coordinator's table maps each model to one app.
        return next(reversed(self.models[model_name].values()))

    async def _await_first(self, response_gen, model: str, endpoint: str):
        """Await *response_gen*'s first item, translating a Ray-boundary failure into
        a clean 4xx/5xx ``JSONResponse``. Shared by ``_handle_response`` and
        ``compact_response`` — a caller must check ``isinstance(result, JSONResponse)``
        before treating the result as a payload item."""
        try:
            return await response_gen.__anext__()
        except RayTaskError as e:
            # Loader code raised across the Ray boundary. Treat ValueError-family
            # causes (e.g. vLLM's VLLMValidationError on context overflow) as
            # OpenAI-style 400s rather than masking them as 500s.
            cause = e.cause if isinstance(e.cause, BaseException) else None
            if isinstance(cause, ValueError | TypeError | OverflowError):
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "validation_error"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                logger.info("Validation error for model=%s: %s", model, cause)
                return _error_response(_validation_error_from_cause(cause))
            REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
            logger.exception("Initial response generation failed for model=%s", model)
            return JSONResponse(status_code=500, content={"detail": str(e)})
        except Exception as e:
            # Catch failures during initial generator creation or the very first yield
            REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
            logger.exception("Initial response generation failed for model=%s", model)
            return JSONResponse(status_code=500, content={"detail": str(e)})

    async def _handle_response(
        self,
        response_gen,
        watcher: RequestWatcher,
        model: str,
        endpoint: str,
        stream_media_type: str = "text/event-stream",
    ):
        start = time.monotonic()
        streaming = False
        REQUEST_IN_PROGRESS.set(1, tags={"model": model, "endpoint": endpoint})
        try:
            first = await self._await_first(response_gen, model, endpoint)
            if isinstance(first, JSONResponse):
                return first

            if isinstance(first, ErrorResponse):
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "inference_error"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                logger.info(
                    "Inference error endpoint=%s model=%s type=%s code=%s param=%s message=%.200s",
                    endpoint,
                    model,
                    first.error.type,
                    first.error.code,
                    first.error.param,
                    first.error.message,
                )
                return _error_response(first)

            if isinstance(first, Response):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                return first

            if isinstance(first, RawSpeechResponse):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                return Response(content=first.audio, media_type=first.media_type)

            if isinstance(
                first,
                ChatCompletionResponse
                | ResponseObject
                | EmbeddingResponse
                | TranscriptionResponse
                | TranscriptionResponseVerbose
                | TranslationResponse
                | TranslationResponseVerbose
                | ImageGenerationResponse,
            ):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                return JSONResponse(content=first.model_dump(mode="json"))

            # streaming — first chunk already consumed, chain it back
            async def _stream():
                # Held for _stream_error_chunk: opening frame carries the Responses
                # envelope, last one the sequence number. Parsed only on the error path.
                last = first
                try:
                    STREAM_CHUNKS_TOTAL.inc(tags={"model": model})
                    yield first
                    async for chunk in response_gen:
                        STREAM_CHUNKS_TOTAL.inc(tags={"model": model})
                        last = chunk
                        yield chunk
                    REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                except Exception:
                    REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "stream_error"})
                    REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                    logger.exception("stream failed mid-response endpoint=%s model=%s", endpoint, model)
                    yield _stream_error_chunk(endpoint, model, first, last)
                    yield "data: [DONE]\n\n"
                finally:
                    REQUEST_DURATION_SECONDS.observe(
                        time.monotonic() - start, tags={"model": model, "endpoint": endpoint}
                    )
                    REQUEST_IN_PROGRESS.set(0, tags={"model": model, "endpoint": endpoint})
                    watcher.stop()

            streaming = True
            return StreamingResponse(content=_stream(), media_type=stream_media_type)
        except Exception:
            # Fallback for anything else not caught above
            REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
            raise
        finally:
            # Non-streaming paths (including CancelledError — a BaseException that
            # skips the except clauses above) finalize on return/raise here; the
            # streaming path does it in _stream()'s finally after the stream drains.
            if not streaming:
                duration = time.monotonic() - start
                REQUEST_DURATION_SECONDS.observe(duration, tags={"model": model, "endpoint": endpoint})
                REQUEST_IN_PROGRESS.set(0, tags={"model": model, "endpoint": endpoint})
                watcher.stop()

    @app.get("/health")
    async def health(self):
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    def _readyz_body(self) -> dict:
        self._ensure_watching()
        expected = list(self.expected_models)
        pending = [m for m in expected if m not in self.models] if expected else []
        ready = bool(expected) and len(pending) == 0
        time_to_ready: float | None = None
        if self._all_ready_at is not None and self._expected_set_at is not None:
            time_to_ready = round(self._all_ready_at - self._expected_set_at, 2)
        return {
            "status": "ok",
            "ready": ready,
            "uptime_s": round(time.time() - self._started_at, 1),
            "time_to_ready_s": time_to_ready,
            "models_loaded": sorted(self.models.keys()),
            "models_expected": expected,
            "models_pending": pending,
            "model_load_times_s": dict(self._model_load_times),
        }

    @app.get("/readyz")
    async def readyz(self):
        body = self._readyz_body()
        if body["ready"]:
            return body
        return JSONResponse(status_code=503, content=body)

    @app.get("/v1/models", response_model=OpenaiModelList)
    async def list_models(self):
        self._ensure_watching()
        return OpenaiModelList(data=self.model_list)

    @app.get("/v1/models/{model}", response_model=OpenAiModelCard)
    async def model_info(self, model: str) -> OpenAiModelCard:
        self._ensure_watching()
        for card in self.model_list:
            if card.id == model:
                return card
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint="create_chat_completion")
        headers = dict(raw_request.headers)
        # Materialize any lazy pydantic ValidatorIterators (from Iterable-typed fields
        # like tool_calls) in place — they can't be pickled across the Ray boundary.
        # Do NOT re-validate via model_validate/model_validate_json as pydantic will
        # re-wrap Iterable fields in a new ValidatorIterator.
        for msg in request.messages:
            if isinstance(msg, dict) and "tool_calls" in msg:
                tc = msg["tool_calls"]
                if not isinstance(tc, list):
                    msg["tool_calls"] = list(tc)  # type: ignore[arg-type]
        logger.info(
            "chat_completion model=%s messages=%d stream=%s max_tokens=%s",
            model,
            len(request.messages),
            request.stream,
            request.max_tokens,
        )
        logger.debug("chat_completion full request: %s", request.model_dump_json())
        response_gen = handle.generate.options(stream=True).remote(request, headers, watcher.registry, req_id, identity)
        return await self._handle_response(response_gen, watcher, model, "create_chat_completion")

    @app.post("/v1/responses")
    async def create_response(self, request: ResponsesRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint="create_response")
        headers = dict(raw_request.headers)

        if request.background:
            # Run continues detached after this response sends; a live disconnect-watch
            # would self-abort it once the HTTP response closes.
            watcher.stop()
            if request.store is False:
                raise responses_utils.background_store_false_error()

        # Resolve conversation history here, before the Ray hop: the loader only ever
        # sees a flat `input`, and a store outage fails the request before any GPU
        # work starts. Costs nothing when previous_response_id is unset.
        if request.previous_response_id is not None:
            try:
                request.input = await responses_utils.resolve_history(self._state_store, identity, request)
            except BaseException:
                # Any failure here — not just the expected 404/503 HTTPException — must
                # still stop the watcher: _handle_response hasn't taken ownership of it
                # yet, so nothing else will cancel its polling task.
                watcher.stop()
                raise
            # Already item-list-shaped (resolve_history's return type) — avoid re-normalizing.
            input_items_for_turn = request.input
        else:
            input_items_for_turn = responses_utils.as_input_items(request.input)

        logger.info(
            "responses model=%s input_items=%s max_output_tokens=%s stream=%s store=%s previous_response_id=%s"
            " background=%s",
            model,
            1 if isinstance(request.input, str) else len(request.input),
            request.max_output_tokens,
            bool(request.stream),
            request.store is not False,
            request.previous_response_id,
            bool(request.background),
        )

        if request.background:
            # Captured before run_background forces request.stream=True on its own copy.
            wants_stream = bool(request.stream)
            response_dict = await responses_utils.start_background(
                self._state_store, identity, request, req_id, input_items_for_turn
            )
            task = asyncio.ensure_future(
                responses_utils.run_background(
                    handle,
                    request,
                    headers,
                    req_id,
                    identity,
                    store=self._state_store,
                    response_id=response_dict["id"],
                    input_items=input_items_for_turn,
                    stream_buffer=wants_stream,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if wants_stream:
                return StreamingResponse(
                    frame_sse(responses_utils.tail_background_events(self._state_store, identity, response_dict["id"])),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=response_dict, status_code=200)

        if mcp_loop.wants_mcp(request):
            response_gen = mcp_loop.run_mcp_response(handle, request, headers, watcher.registry, req_id, identity)
        else:
            # Under stream=True the remote() result is an async-iterable
            # DeploymentResponseGenerator; Ray's stub widens it to a union because it
            # doesn't overload on the stream literal, so narrow it explicitly.
            response_gen = cast(
                "DeploymentResponseGenerator[Any]",
                handle.respond.options(stream=True).remote(request, headers, watcher.registry, req_id, identity),
            )
        if request.store is not False:
            response_gen = responses_utils.persist_response(
                response_gen,
                self._state_store,
                identity=identity,
                input_items=input_items_for_turn,
            )
        # SSE-frame + [DONE] happen here, after persistence, so persist_response still sees plain event dicts.
        response_gen = frame_sse(response_gen)
        return await self._handle_response(response_gen, watcher, model, "create_response")

    @app.websocket("/v1/responses")
    async def responses_ws(self, websocket: WebSocket):
        """WebSocket transport for ``/v1/responses``: one socket, many sequential
        turns. Each frame in is a `{"type": "response.create", ...}` body; each
        frame out is one raw event dict (`json.dumps`d, no SSE framing, no `[DONE]`).
        """
        # BaseHTTPMiddleware skips websocket connections, so auth runs here, before accept().
        if not await check_ws_auth(websocket):
            return
        await websocket.accept()
        identity = self._set_identity(websocket)
        headers = dict(websocket.headers)
        conn_cache: dict[str, dict] = {}
        registry = get_disconnect_registry()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        current_req_id: str | None = None

        async def _reader() -> None:
            # Only one coroutine may call websocket.receive() at a time, so a single
            # reader task queues frames — this also lets it catch a mid-turn disconnect.
            nonlocal current_req_id
            try:
                while True:
                    raw = await websocket.receive_text()
                    await queue.put(raw)
            except WebSocketDisconnect:
                if current_req_id is not None:
                    try:
                        await registry.set.remote(current_req_id)
                    except RayActorError:
                        logger.warning("Disconnect registry unavailable; lost WS disconnect for %s", current_req_id)
                await queue.put(None)

        reader_task = asyncio.ensure_future(_reader())
        try:
            while True:
                raw = await queue.get()
                if raw is None:
                    return
                current_req_id = random_uuid()
                try:
                    await self._run_ws_turn(websocket, identity, headers, raw, conn_cache, registry, current_req_id)
                finally:
                    current_req_id = None
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task

    async def _run_ws_turn(
        self,
        websocket: WebSocket,
        identity: str,
        headers: dict[str, str],
        raw: str,
        conn_cache: dict[str, dict],
        registry: Any,
        req_id: str,
    ) -> None:
        """Run one `response.create` turn to completion. Never raises back into the
        socket loop — any failure is sent as an error frame instead, so one bad turn
        never kills the connection."""
        self._set_request_id(req_id)

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(
                error_ws_frame(
                    create_error_response("WebSocket frame is not valid JSON.", err_type="invalid_request_error")
                )
            )
            return
        if not isinstance(msg, dict) or msg.get("type") != "response.create":
            await websocket.send_text(
                error_ws_frame(
                    create_error_response(
                        'WebSocket frame must be a JSON object with "type": "response.create".',
                        err_type="invalid_request_error",
                    )
                )
            )
            return

        body = {k: v for k, v in msg.items() if k != "type"}
        # Not meaningful on an always-streaming transport; drop them so extra="allow" doesn't pass them through.
        for unsupported in ("stream", "stream_options", "background"):
            body.pop(unsupported, None)
        try:
            request = ResponsesRequest(**body)
        except ValidationError as exc:
            await websocket.send_text(error_ws_frame(responses_utils.responses_validation_error(exc)))
            return
        request.stream = True
        model = request.model or ""
        endpoint = "create_response_ws"

        prev_id = request.previous_response_id
        try:
            if prev_id is not None and prev_id in conn_cache:
                # A continuation this socket itself produced with store:false.
                this_turn = responses_utils.as_input_items(request.input) if request.input is not None else []
                request.input = [*responses_state.history_items(conn_cache[prev_id]), *this_turn]
                input_items_for_turn = request.input
            elif prev_id is not None:
                request.input = await responses_utils.resolve_history(self._state_store, identity, request)
                input_items_for_turn = request.input
            else:
                # request.input is left untouched here; only the persist/cache bookkeeping below needs item-list form.
                input_items_for_turn = responses_utils.as_input_items(request.input)
        except responses_utils.ResponsesApiError as exc:
            await websocket.send_text(error_ws_frame(exc.err))
            return

        try:
            handle = self._get_handle(request.model)
        except HTTPException as exc:
            err = (
                exc.err
                if isinstance(exc, responses_utils.ResponsesApiError)
                else create_error_response(
                    str(exc.detail),
                    err_type="invalid_request_error" if exc.status_code < 500 else "api_error",
                    status_code=exc.status_code,
                )
            )
            await websocket.send_text(error_ws_frame(err))
            return

        start = time.monotonic()
        REQUEST_IN_PROGRESS.set(1, tags={"model": model, "endpoint": endpoint})
        failed = False
        terminal_response: dict[str, Any] | None = None
        try:
            if mcp_loop.wants_mcp(request):
                response_gen = mcp_loop.run_mcp_response(handle, request, headers, registry, req_id, identity)
            else:
                response_gen = cast(
                    "DeploymentResponseGenerator[Any]",
                    handle.respond.options(stream=True).remote(request, headers, registry, req_id, identity),
                )
            if request.store is not False:
                # Same helper HTTP uses — persists before yielding, so a store
                # failure can still flip the terminal event to response.failed.
                response_gen = responses_utils.persist_response(
                    response_gen, self._state_store, identity=identity, input_items=input_items_for_turn
                )
            async for item in response_gen:
                STREAM_CHUNKS_TOTAL.inc(tags={"model": model})
                if isinstance(item, ErrorResponse):
                    await websocket.send_text(error_ws_frame(item))
                    failed = True
                    continue
                if not isinstance(item, dict):
                    continue  # pragma: no cover - ResponseObject is unreachable; WS always forces stream=True
                await websocket.send_text(json.dumps(item))
                event_type = item.get("type")
                if event_type == "response.failed":
                    failed = True
                elif event_type in TERMINAL_EVENT_TYPES:
                    terminal_response = item.get("response")
        except RayTaskError as exc:
            cause = exc.cause if isinstance(exc.cause, BaseException) else None
            if isinstance(cause, ValueError | TypeError | OverflowError):
                err = _validation_error_from_cause(cause)
            else:
                logger.exception("responses request %s failed over websocket", req_id)
                err = create_error_response(
                    "Internal error during generation",
                    err_type="api_error",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            await websocket.send_text(error_ws_frame(err))
            failed = True
        except Exception:
            logger.exception("responses request %s failed over websocket", req_id)
            await websocket.send_text(
                error_ws_frame(
                    create_error_response(
                        "Internal error during generation",
                        err_type="api_error",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                )
            )
            failed = True
        finally:
            REQUEST_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": model, "endpoint": endpoint})
            REQUEST_IN_PROGRESS.set(0, tags={"model": model, "endpoint": endpoint})

        if failed:
            REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "inference_error"})
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
            # Evict so a retry of the same previous_response_id gets a clean 404, not a replay of what broke it.
            if prev_id is not None:
                conn_cache.pop(prev_id, None)
            return

        REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
        if request.store is False and terminal_response is not None:
            response_id = terminal_response.get("id")
            if response_id:
                conn_cache[response_id] = {"input_items": input_items_for_turn, "response": terminal_response}
                while len(conn_cache) > _WS_CONN_CACHE_MAX:
                    conn_cache.pop(next(iter(conn_cache)))

    @app.get("/v1/responses/{response_id}")
    async def get_response(
        self, response_id: str, raw_request: Request, stream: bool = False, starting_after: int = -1
    ):
        self._set_request_id(random_uuid())
        identity = self._set_identity(raw_request)
        # Confirms existence/ownership up front, so an unknown id still 404s cleanly —
        # once a StreamingResponse starts, an in-loop miss can only end the stream quietly.
        snapshot = await responses_utils.load_snapshot(self._state_store, identity, response_id)
        if stream:
            return StreamingResponse(
                frame_sse(
                    responses_utils.tail_background_events(
                        self._state_store, identity, response_id, after_sequence=starting_after
                    )
                ),
                media_type="text/event-stream",
            )
        # Orphan detection: a dead drain task surfaces as `failed` here rather than hanging forever.
        snapshot = await responses_utils.reconcile_staleness(self._state_store, identity, response_id, snapshot)
        # Stored verbatim, so this is a passthrough — no re-derivation to drift.
        return JSONResponse(content=snapshot["response"])

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(self, response_id: str, raw_request: Request):
        self._set_request_id(random_uuid())
        identity = self._set_identity(raw_request)
        response = await responses_utils.cancel_background(self._state_store, identity, response_id)
        return JSONResponse(content=response)

    @app.delete("/v1/responses/{response_id}")
    async def delete_response(self, response_id: str, raw_request: Request):
        self._set_request_id(random_uuid())
        identity = self._set_identity(raw_request)
        # Read first: delete is idempotent by contract, so it alone can't tell an
        # unknown id from a real removal.
        snapshot = await responses_utils.load_snapshot(self._state_store, identity, response_id)
        # DELETE on an in-flight background run implies cancel.
        await responses_utils.signal_background_cancel_if_in_progress(identity, response_id, snapshot)
        await responses_utils.delete_snapshot(self._state_store, identity, response_id)
        return JSONResponse(content={"id": response_id, "object": "response", "deleted": True})

    @app.get("/v1/responses/{response_id}/input_items")
    async def get_response_input_items(self, response_id: str, raw_request: Request):
        self._set_request_id(random_uuid())
        identity = self._set_identity(raw_request)
        snapshot = await responses_utils.load_snapshot(self._state_store, identity, response_id)
        # Chronological; OpenAI defaults to order=desc, which we don't support yet.
        items = snapshot.get("input_items") or []
        return JSONResponse(
            content={
                "object": "list",
                "data": items,
                "first_id": items[0].get("id") if items and isinstance(items[0], dict) else None,
                "last_id": items[-1].get("id") if items and isinstance(items[-1], dict) else None,
                "has_more": False,
            }
        )

    @app.post("/v1/responses/compact")
    async def compact_response(self, request: CompactRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model
        endpoint = "compact_response"
        handle = self._get_handle(model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint=endpoint)
        headers = dict(raw_request.headers)

        start = time.monotonic()
        REQUEST_IN_PROGRESS.set(1, tags={"model": model, "endpoint": endpoint})
        try:
            items = await responses_utils.resolve_history_items(
                self._state_store,
                identity,
                previous_response_id=request.previous_response_id,
                input_=request.input,
            )
            if not items:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value, detail="cannot compact an empty conversation."
                )

            logger.info(
                "compact_response model=%s previous_response_id=%s items=%d",
                model,
                request.previous_response_id,
                len(items),
            )

            try:
                chat_request = responses_utils.build_summarization_request(model, items, request.instructions)
            except UnsupportedResponsesFeatureError as e:
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "validation_error"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                return _error_response(create_error_response(str(e), err_type="invalid_request_error"))

            response_gen = handle.generate.options(stream=True).remote(
                chat_request, headers, watcher.registry, req_id, identity
            )
            first = await self._await_first(response_gen, model, endpoint)
            if isinstance(first, JSONResponse):
                return first
            if isinstance(first, ErrorResponse):
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "inference_error"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                logger.info(
                    "Inference error endpoint=%s model=%s type=%s code=%s param=%s message=%.200s",
                    endpoint,
                    model,
                    first.error.type,
                    first.error.code,
                    first.error.param,
                    first.error.message,
                )
                return _error_response(first)

            assert isinstance(first, ChatCompletionResponse)
            summary_text = first.choices[0].message.content or ""
            summary_items = [{"type": "message", "role": "assistant", "content": summary_text}]
            resource = responses_utils.build_compaction(summary_items=summary_items, usage=first.usage)
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
            return JSONResponse(content=resource.model_dump(mode="json"))
        finally:
            REQUEST_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": model, "endpoint": endpoint})
            REQUEST_IN_PROGRESS.set(0, tags={"model": model, "endpoint": endpoint})
            watcher.stop()

    @app.post("/v1/embeddings")
    async def create_embeddings(self, request: EmbeddingRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint="create_embeddings")
        headers = dict(raw_request.headers)
        logger.info("embeddings model=%s", model)
        response_gen = handle.embed.options(stream=True).remote(request, headers, watcher.registry, req_id, identity)
        return await self._handle_response(response_gen, watcher, model, "create_embeddings")

    @app.post("/v1/audio/transcriptions")
    async def create_transcriptions(self, request: Annotated[TranscriptionRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint="create_transcriptions")
        headers = dict(raw_request.headers)
        logger.info("transcription model=%s", model)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        try:
            audio_data = await request.file.read()
        finally:
            await request.file.close()
        request_no_file = TranscriptionRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.transcribe.options(stream=True).remote(
            audio_data, request_no_file, headers, watcher.registry, req_id, identity
        )
        return await self._handle_response(response_gen, watcher, model, "create_transcriptions")

    @app.post("/v1/audio/translations")
    async def create_translations(self, request: Annotated[TranslationRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=model, endpoint="create_translations")
        headers = dict(raw_request.headers)
        logger.info("translation model=%s", model)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        try:
            audio_data = await request.file.read()
        finally:
            await request.file.close()
        request_no_file = TranslationRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.translate.options(stream=True).remote(
            audio_data, request_no_file, headers, watcher.registry, req_id, identity
        )
        return await self._handle_response(response_gen, watcher, model, "create_translations")

    @app.post("/v1/audio/speech")
    async def create_speech(self, request: SpeechRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        logger.info("speech model=%s voice=%s format=%s", request.model, request.voice, request.response_format)
        logger.debug("speech full request: %s", request.model_dump_json())
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=request.model, endpoint="create_speech")
        headers = dict(raw_request.headers)
        response_gen = handle.speak.options(stream=True).remote(request, headers, watcher.registry, req_id, identity)
        return await self._handle_response(response_gen, watcher, request.model, "create_speech")

    @app.post("/v1/images/generations")
    async def create_image(self, request: ImageGenerationRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        logger.info(
            "image_generation model=%s prompt=%r n=%d size=%s", request.model, request.prompt, request.n, request.size
        )
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=request.model, endpoint="create_image")
        headers = dict(raw_request.headers)
        response_gen = handle.imagine.options(stream=True).remote(request, headers, watcher.registry, req_id, identity)
        return await self._handle_response(response_gen, watcher, request.model, "create_image")

    @app.post("/v1/images/edits")
    async def create_image_edit(self, request: Annotated[ImageEditRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        logger.info(
            "image_edit model=%s prompt=%r n=%d size=%s mask=%s",
            request.model,
            request.prompt,
            request.n,
            request.size,
            request.mask is not None,
        )
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=request.model, endpoint="create_image_edit")
        headers = dict(raw_request.headers)
        # Read image bytes before crossing the process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file fields.
        # `image` is Optional on the model (to accept the `image[]` alias) but the validator
        # guarantees it is set post-validation.
        assert request.image is not None
        try:
            image_data = await request.image.read()
            mask_data = await request.mask.read() if request.mask is not None else None
        finally:
            await request.image.close()
            if request.mask is not None:
                await request.mask.close()
        request_no_file = ImageEditRequest.model_construct(**request.model_dump(exclude={"image", "mask"}))
        response_gen = handle.edit_image.options(stream=True).remote(
            image_data, mask_data, request_no_file, headers, watcher.registry, req_id, identity
        )
        return await self._handle_response(response_gen, watcher, request.model, "create_image_edit")

    @app.post("/v1/images/variations")
    async def create_image_variation(self, request: Annotated[ImageVariationRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        identity = self._set_identity(raw_request)
        logger.info("image_variation model=%s n=%d size=%s", request.model, request.n, request.size)
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, req_id, model=request.model, endpoint="create_image_variation")
        headers = dict(raw_request.headers)
        # Read image bytes before crossing the process boundary — UploadFile is not serializable.
        # `image` is Optional on the model (to accept the `image[]` alias) but the validator
        # guarantees it is set post-validation.
        assert request.image is not None
        try:
            image_data = await request.image.read()
        finally:
            await request.image.close()
        request_no_file = ImageVariationRequest.model_construct(**request.model_dump(exclude={"image"}))
        response_gen = handle.vary_image.options(stream=True).remote(
            image_data, request_no_file, headers, watcher.registry, req_id, identity
        )
        return await self._handle_response(response_gen, watcher, request.model, "create_image_variation")
