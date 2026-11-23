"""One `/v1/realtime` WebSocket session. The gateway owns the session; the pipeline names the models it runs
over. So far a session opens, reports and updates itself; every other client event is answered as not supported
yet."""

import json
from collections.abc import Collection, Mapping
from typing import Any

from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from modelship.logging import get_logger
from modelship.openai.protocol.realtime import (
    SESSION_OBJECTS,
    RealtimeError,
    RealtimeSession,
    SessionFields,
    error_event,
    session_event,
)
from modelship.utils import random_uuid
from modelship.utils.config_schema import PipelineConfig

logger = get_logger("realtime")

# RFC 6455 policy violation: the session can't open as asked
_CLOSE_REFUSED = 1008
_UPDATABLE = ("instructions", "output_modalities", "max_output_tokens")


def _refusal(
    name: str | None, pipelines: Mapping[str, dict], known: Collection[str], serving: Collection[str]
) -> RealtimeError | None:
    """Why a session on pipeline *name* can't open, or None. *pipelines* is the gateway's pipeline table, *known*
    its models and *serving* those it can route to now."""
    if not name:
        return RealtimeError(
            message="Pass the pipeline to use as /v1/realtime?model=<name>.", code="missing_model", param="model"
        )
    route = pipelines.get(name)
    if route is None:
        return RealtimeError(message=f"No pipeline {name!r} on this gateway.", code="model_not_found", param="model")
    config = PipelineConfig.model_validate(route["config"])
    for model in filter(None, (config.stt, config.llm, config.tts)):
        if model not in known:
            logger.warning("Pipeline %r uses model %r, which this gateway doesn't have", name, model)
            return RealtimeError(
                message=f"Pipeline {name!r} uses model {model!r}, which this gateway doesn't have.",
                code="model_not_found",
            )
        if model not in serving:
            return RealtimeError(
                message=f"Pipeline {name!r} uses model {model!r}, which isn't ready yet.", code="model_not_ready"
            )
    if not route["ready"]:
        return RealtimeError(message=f"Pipeline {name!r} isn't ready yet.", code="model_not_ready")
    return None


def _new_session(config: PipelineConfig) -> RealtimeSession:
    kind = "realtime" if config.llm is not None else "transcription"
    defaults = config.session_defaults
    return RealtimeSession(
        type=kind,
        object=SESSION_OBJECTS[kind],
        id=f"sess_{random_uuid()[:20]}",
        model=config.name,
        **(defaults.model_dump(exclude_none=True) if defaults is not None else {}),
    )


async def _send(websocket: WebSocket, event: dict) -> None:
    await websocket.send_text(json.dumps(event))


async def serve_session(
    websocket: WebSocket, pipelines: Mapping[str, dict], known: Collection[str], serving: Collection[str]
) -> None:
    """Runs one session until the client leaves."""
    await websocket.accept()
    name = websocket.query_params.get("model")
    refusal = _refusal(name, pipelines, known, serving)
    if refusal is not None:
        await _send(websocket, error_event(refusal))
        await websocket.close(code=_CLOSE_REFUSED)
        return
    assert name is not None
    session = _new_session(PipelineConfig.model_validate(pipelines[name]["config"]))
    logger.info("Realtime session %s opened on pipeline %s", session.id, name)
    await _send(websocket, session_event("session.created", session))
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is None:
                await _send(websocket, error_event(RealtimeError(message="Send events as JSON text frames.")))
                continue
            await _handle(websocket, session, message["text"])
    except WebSocketDisconnect:
        pass
    logger.info("Realtime session %s closed", session.id)


async def _handle(websocket: WebSocket, session: RealtimeSession, text: str) -> None:
    try:
        event: Any = json.loads(text)
    except json.JSONDecodeError:
        await _send(websocket, error_event(RealtimeError(message="The event is not valid JSON.", code="invalid_json")))
        return
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        error = RealtimeError(message='An event is a JSON object with a "type".', code="invalid_event")
        await _send(websocket, error_event(error))
        return
    client_event = event.get("event_id") if isinstance(event.get("event_id"), str) else None
    if event["type"] == "session.update":
        error = _update(session, event.get("session"))
        if error is not None:
            error.event_id = client_event
            await _send(websocket, error_event(error))
            return
        await _send(websocket, session_event("session.updated", session))
        return
    error = RealtimeError(
        message=f"{event['type']} is not supported yet.", code="unsupported_event", event_id=client_event
    )
    await _send(websocket, error_event(error))


def _update(session: RealtimeSession, fields: Any) -> RealtimeError | None:
    """Applies a `session.update`'s fields to *session*; the error to answer with, or None."""
    if not isinstance(fields, dict):
        return RealtimeError(message="session.update needs a session object.", param="session")
    try:
        update = SessionFields.model_validate(fields)
    except ValidationError as e:
        problem = e.errors()[0]
        param = ".".join(["session", *(str(part) for part in problem["loc"])])
        return RealtimeError(message=f"{param}: {problem['msg']}", code="invalid_value", param=param)
    if update.type is not None and update.type != session.type:
        return RealtimeError(
            message=f"This session is a {session.type} session; its type can't change.",
            code="invalid_value",
            param="session.type",
        )
    for field in _UPDATABLE:
        if field in update.model_fields_set:
            setattr(session, field, getattr(update, field))
    if ignored := sorted(update.model_extra or {}):
        logger.debug("Realtime session %s: ignoring session fields not used yet: %s", session.id, ", ".join(ignored))
    return None
