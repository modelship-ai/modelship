"""One `/v1/realtime` WebSocket session. The gateway owns the session; the pipeline names the models it runs
over. So far a session holds a text conversation and answers it with the pipeline's LLM; audio events are answered
as not supported yet."""

import asyncio
import json
from collections.abc import Callable, Collection, Mapping
from typing import Any

from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from modelship.logging import get_logger
from modelship.openai.protocol.realtime import (
    SESSION_OBJECTS,
    RealtimeError,
    RealtimeResponse,
    RealtimeSession,
    ResponseParams,
    SessionFields,
    error_event,
    server_event,
    session_event,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.openai.realtime.conversation import Conversation, as_input, new_item
from modelship.openai.realtime.response import ActiveResponse, Respond
from modelship.utils import random_uuid
from modelship.utils.config_schema import PipelineConfig

logger = get_logger("realtime")

# RFC 6455 policy violation: the session can't open as asked
_CLOSE_REFUSED = 1008
_UPDATABLE = ("instructions", "output_modalities", "max_output_tokens", "reasoning")
_REALTIME_ONLY = frozenset({"conversation.item.create", "response.create", "response.cancel"})


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
    fields = defaults.model_dump(exclude_none=True) if defaults is not None else {}
    if kind == "realtime":
        fields.setdefault("output_modalities", ["text"])
    return RealtimeSession(
        type=kind, object=SESSION_OBJECTS[kind], id=f"sess_{random_uuid()[:20]}", model=config.name, **fields
    )


def _invalid(prefix: str, error: ValidationError) -> RealtimeError:
    problem = error.errors()[0]
    param = ".".join([prefix, *(str(part) for part in problem["loc"])])
    return RealtimeError(message=f"{param}: {problem['msg']}", code="invalid_value", param=param)


def _audio_output(param: str) -> RealtimeError:
    return RealtimeError(message="Audio output is not supported yet.", code="unsupported_value", param=param)


async def serve_session(
    websocket: WebSocket,
    pipelines: Mapping[str, dict],
    known: Collection[str],
    serving: Collection[str],
    respond: Respond,
) -> None:
    """Runs one session until the client leaves. *pipelines* is the gateway's pipeline table, *known* its models,
    *serving* those it can route to now, and *respond* how the session calls a model."""
    await websocket.accept()
    name = websocket.query_params.get("model")
    refusal = _refusal(name, pipelines, known, serving)
    if refusal is not None:
        await websocket.send_text(json.dumps(error_event(refusal)))
        await websocket.close(code=_CLOSE_REFUSED)
        return
    assert name is not None
    await Session(websocket, PipelineConfig.model_validate(pipelines[name]["config"]), respond).run()


class Session:
    def __init__(self, websocket: WebSocket, config: PipelineConfig, respond: Respond):
        self.websocket = websocket
        self.config = config
        self.respond = respond
        self.state = _new_session(config)
        self.conversation = Conversation()
        self.active: ActiveResponse | None = None
        self._sending = asyncio.Lock()
        self._closed = False

    async def run(self) -> None:
        logger.info("Realtime session %s opened on pipeline %s", self.state.id, self.config.name)
        await self.send(session_event("session.created", self.state))
        if self.state.type == "realtime":
            conversation = {"id": self.conversation.id, "object": "realtime.conversation"}
            await self.send(server_event("conversation.created", conversation=conversation))
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is None:
                    await self.send(error_event(RealtimeError(message="Send events as JSON text frames.")))
                    continue
                await self._handle(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            self._closed = True
            if self.active is not None:
                await self.active.abandon()
        logger.info("Realtime session %s closed", self.state.id)

    async def send(self, event: dict) -> None:
        """Sends *event*, unless the client has left. A response sends from its own task."""
        text = json.dumps(event)
        async with self._sending:
            if self._closed:
                return
            try:
                await self.websocket.send_text(text)
            except (WebSocketDisconnect, RuntimeError, OSError):
                self._closed = True

    async def _handle(self, text: str) -> None:
        try:
            event: Any = json.loads(text)
        except json.JSONDecodeError:
            await self.send(error_event(RealtimeError(message="The event is not valid JSON.", code="invalid_json")))
            return
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            error = RealtimeError(message='An event is a JSON object with a "type".', code="invalid_event")
            await self.send(error_event(error))
            return
        kind = event["type"]
        handler = self._handlers().get(kind)
        if handler is None:
            error = RealtimeError(message=f"{kind} is not supported yet.", code="unsupported_event")
        elif kind in _REALTIME_ONLY and self.state.type != "realtime":
            error = RealtimeError(message=f"A transcription session takes no {kind}.", code="invalid_event")
        else:
            error = await handler(event)
        if error is not None:
            error.event_id = event.get("event_id") if isinstance(event.get("event_id"), str) else None
            await self.send(error_event(error))

    def _handlers(self) -> dict[str, Callable[[dict], Any]]:
        return {
            "session.update": self._session_update,
            "conversation.item.create": self._item_create,
            "conversation.item.retrieve": self._item_retrieve,
            "conversation.item.delete": self._item_delete,
            "response.create": self._response_create,
            "response.cancel": self._response_cancel,
        }

    async def _session_update(self, event: dict) -> RealtimeError | None:
        fields = event.get("session")
        if not isinstance(fields, dict):
            return RealtimeError(message="session.update needs a session object.", param="session")
        try:
            update = SessionFields.model_validate(fields)
        except ValidationError as e:
            return _invalid("session", e)
        if update.type is not None and update.type != self.state.type:
            return RealtimeError(
                message=f"This session is a {self.state.type} session; its type can't change.",
                code="invalid_value",
                param="session.type",
            )
        if update.output_modalities is not None and "audio" in update.output_modalities:
            return _audio_output("session.output_modalities")
        for name in _UPDATABLE:
            if name in update.model_fields_set:
                setattr(self.state, name, getattr(update, name))
        if ignored := sorted(update.model_extra or {}):
            logger.debug("Realtime session %s: ignoring session fields not used yet: %s", self.state.id, ignored)
        await self.send(session_event("session.updated", self.state))
        return None

    def _missing_item(self, item_id: Any, param: str) -> RealtimeError:
        return RealtimeError(message=f"No item {item_id!r} in the conversation.", code="item_not_found", param=param)

    async def _item_create(self, event: dict) -> RealtimeError | None:
        item = new_item(event.get("item"))
        if isinstance(item, RealtimeError):
            return item
        previous = event.get("previous_item_id")
        if previous is not None and previous != "root" and self.conversation.get(previous) is None:
            return self._missing_item(previous, "previous_item_id")
        if self.conversation.get(item["id"]) is not None:
            return RealtimeError(
                message=f"The conversation already has an item {item['id']!r}.", code="invalid_value", param="item.id"
            )
        before = self.conversation.insert(item, previous)
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        await self.send(server_event("conversation.item.done", previous_item_id=before, item=item))
        return None

    async def _item_retrieve(self, event: dict) -> RealtimeError | None:
        item = self.conversation.get(event.get("item_id"))
        if item is None:
            return self._missing_item(event.get("item_id"), "item_id")
        await self.send(server_event("conversation.item.retrieved", item=item))
        return None

    async def _item_delete(self, event: dict) -> RealtimeError | None:
        item_id = event.get("item_id")
        if not self.conversation.delete(item_id):
            return self._missing_item(item_id, "item_id")
        await self.send(server_event("conversation.item.deleted", item_id=item_id))
        return None

    def _input(self, raw_items: list[dict]) -> list[dict] | RealtimeError:
        """A response's own input items; an `item_reference` names an item of the conversation."""
        items = []
        for i, raw in enumerate(raw_items):
            if raw.get("type") == "item_reference":
                item = self.conversation.get(raw.get("id"))
                if item is None:
                    return self._missing_item(raw.get("id"), f"response.input[{i}].id")
                items.append(item)
                continue
            item = new_item(raw)
            if isinstance(item, RealtimeError):
                item.param = f"response.input[{i}]{(item.param or 'item').removeprefix('item')}"
                return item
            items.append(item)
        return items

    async def _response_create(self, event: dict) -> RealtimeError | None:
        if self.active is not None and not self.active.done():
            return RealtimeError(
                message="A response is in progress; wait for its response.done or cancel it.",
                code="conversation_already_has_active_response",
            )
        raw = event.get("response")
        if raw is not None and not isinstance(raw, dict):
            return RealtimeError(message="response.create's response is an object.", param="response")
        try:
            params = ResponseParams.model_validate(raw or {})
        except ValidationError as e:
            return _invalid("response", e)
        modalities = params.output_modalities or self.state.output_modalities or ["text"]
        if "audio" in modalities:
            return _audio_output("response.output_modalities")
        if ignored := sorted(params.model_extra or {}):
            logger.debug("Realtime session %s: ignoring response fields not used yet: %s", self.state.id, ignored)
        items = self.conversation.items if params.input is None else self._input(params.input)
        if isinstance(items, RealtimeError):
            return items
        llm = self.config.llm
        assert llm is not None
        max_output_tokens = params.max_output_tokens or self.state.max_output_tokens or "inf"
        reasoning = params.reasoning or self.state.reasoning
        request = ResponsesRequest(
            model=llm,
            input=as_input(items),
            instructions=params.instructions if params.instructions is not None else self.state.instructions,
            max_output_tokens=None if max_output_tokens == "inf" else max_output_tokens,
            reasoning={"effort": reasoning.effort} if reasoning is not None and reasoning.effort else None,
            stream=True,
            store=False,
        )
        answer = self.respond(llm, request, self.state.id)
        if answer is None:
            return RealtimeError(message=f"Pipeline model {llm!r} isn't ready.", code="model_not_ready")
        in_conversation = params.conversation != "none"
        response = RealtimeResponse(
            id=f"resp_{random_uuid()[:20]}",
            conversation_id=self.conversation.id if in_conversation else None,
            output_modalities=modalities,
            max_output_tokens=max_output_tokens,
            metadata=params.metadata,
        )
        conversation = self.conversation if in_conversation else None
        self.active = ActiveResponse(response, llm, answer, conversation, self.send)
        return None

    async def _response_cancel(self, event: dict) -> RealtimeError | None:
        active = self.active
        response_id = event.get("response_id")
        if active is None or active.done() or response_id not in (None, active.response.id):
            return RealtimeError(message="No response to cancel is in progress.", code="response_cancel_not_active")
        active.cancel()
        return None
