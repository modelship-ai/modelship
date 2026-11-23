"""One `/v1/realtime` WebSocket session. The gateway owns the session; the pipeline names the models it runs
over. So far a session holds a conversation of text and of the client's committed audio turns, which the pipeline's
stt model transcribes, and answers it in text with the pipeline's LLM."""

import asyncio
import base64
import binascii
import json
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    SttCapabilities,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.openai.protocol.realtime import (
    CLIENT_EVENTS,
    SESSION_OBJECTS,
    AudioAppend,
    AudioClear,
    AudioCommit,
    AudioInput,
    AudioPcm,
    ClientEvent,
    InputAudio,
    Item,
    ItemCreate,
    ItemDelete,
    ItemReference,
    ItemRetrieve,
    RealtimeError,
    RealtimeResponse,
    RealtimeSession,
    ResponseCancel,
    ResponseCreate,
    SessionUpdate,
    UserMessage,
    error_event,
    server_event,
    session_event,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.openai.realtime.audio import G711_RATE, MIN_SECONDS, InputAudioBuffer, to_pcm16
from modelship.openai.realtime.calls import Answer, PipelineCalls
from modelship.openai.realtime.conversation import Conversation, as_input
from modelship.openai.realtime.response import ActiveResponse
from modelship.utils import random_uuid
from modelship.utils.config_schema import PipelineConfig

logger = get_logger("realtime")

# RFC 6455 policy violation: the session can't open as asked
_CLOSE_REFUSED = 1008
_UPDATABLE = (
    "instructions",
    "output_modalities",
    "max_output_tokens",
    "reasoning",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
)
_REALTIME_ONLY = frozenset({"conversation.item.create", "response.create", "response.cancel"})
# the union members pydantic names in a validation error's location
_UNION_TAGS = frozenset(
    {
        "user",
        "system",
        "assistant",
        "message",
        "function_call",
        "function_call_output",
        "item_reference",
        "input_text",
        "input_audio",
        "audio/pcm",
        "audio/pcmu",
        "audio/pcma",
    }
)

Handler = Callable[[Any], Awaitable[RealtimeError | None]]


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


def _param(loc: tuple[int | str, ...]) -> str:
    """A validation error's location as a param: `item.content[0].type`."""
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int) and parts:
            parts[-1] += f"[{part}]"
        elif part not in _UNION_TAGS:
            parts.append(str(part))
    return ".".join(parts)


def _invalid(error: ValidationError) -> RealtimeError:
    """The error for a client event that isn't valid."""
    problem = error.errors()[0]
    if not problem["loc"]:
        if problem["type"] == "json_invalid":
            return RealtimeError(message="The event is not valid JSON.", code="invalid_json")
        if problem["type"] == "union_tag_invalid":
            tag = problem.get("ctx", {}).get("tag")
            return RealtimeError(message=f"{tag} is not supported yet.", code="unsupported_event")
        return RealtimeError(message='An event is a JSON object with a "type".', code="invalid_event")
    # the first part is the event's type
    param = _param(problem["loc"][1:])
    return RealtimeError(message=f"{param}: {problem['msg']}", code="invalid_value", param=param)


def _event_id(text: str) -> str | None:
    """The `event_id` of an event that didn't validate, if it has one."""
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    found = raw.get("event_id") if isinstance(raw, dict) else None
    return found if isinstance(found, str) else None


def _audio_output(param: str) -> RealtimeError:
    return RealtimeError(message="Audio output is not supported yet.", code="unsupported_value", param=param)


async def _stt_capabilities(config: PipelineConfig, calls: PipelineCalls) -> SttCapabilities | RealtimeError | None:
    """What the pipeline's stt model can do, asked once for the session; None without one."""
    if config.stt is None:
        return None
    try:
        found = await calls.stt_capabilities(config.stt)
    except Exception:
        logger.exception("Pipeline %r: model %r didn't say what it can do", config.name, config.stt)
        return RealtimeError(
            message=f"Pipeline {config.name!r} couldn't ask model {config.stt!r} what it can do.",
            code="model_not_ready",
        )
    if found is None:
        return RealtimeError(
            message=f"Pipeline {config.name!r} uses model {config.stt!r} as stt, which doesn't transcribe.",
            code="model_not_supported",
        )
    return found


async def _refuse(websocket: WebSocket, error: RealtimeError) -> None:
    await websocket.send_text(json.dumps(error_event(error)))
    await websocket.close(code=_CLOSE_REFUSED)


async def serve_session(
    websocket: WebSocket,
    pipelines: Mapping[str, dict],
    known: Collection[str],
    serving: Collection[str],
    calls: PipelineCalls,
) -> None:
    """Runs one session until the client leaves. *pipelines* is the gateway's pipeline table, *known* its models,
    *serving* those it can route to now, and *calls* how the session calls them."""
    await websocket.accept()
    name = websocket.query_params.get("model")
    refusal = _refusal(name, pipelines, known, serving)
    if refusal is not None:
        await _refuse(websocket, refusal)
        return
    assert name is not None
    config = PipelineConfig.model_validate(pipelines[name]["config"])
    stt = await _stt_capabilities(config, calls)
    if isinstance(stt, RealtimeError):
        await _refuse(websocket, stt)
        return
    await Session(websocket, config, calls, stt).run()


class Session:
    def __init__(self, websocket: WebSocket, config: PipelineConfig, calls: PipelineCalls, stt: SttCapabilities | None):
        self.websocket = websocket
        self.config = config
        self.calls = calls
        # what the stt model can do, asked once at open; None without one
        self.stt = stt
        self.state = _new_session(config)
        self.conversation = Conversation()
        self.active: ActiveResponse | None = None
        self.buffer = InputAudioBuffer()
        # the language the stt model detected on the session's first turn
        self.language: str | None = None
        # the last transcription queued; each runs after the one before it
        self._transcribing: asyncio.Task | None = None
        self._transcriptions: set[asyncio.Task] = set()
        self._sending = asyncio.Lock()
        self._closed = False
        self._handlers: dict[str, Handler] = {
            "session.update": self._session_update,
            "conversation.item.create": self._item_create,
            "conversation.item.retrieve": self._item_retrieve,
            "conversation.item.delete": self._item_delete,
            "response.create": self._response_create,
            "response.cancel": self._response_cancel,
            "input_audio_buffer.append": self._audio_append,
            "input_audio_buffer.commit": self._audio_commit,
            "input_audio_buffer.clear": self._audio_clear,
        }

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
            for task in self._transcriptions:
                task.cancel()
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
            event: ClientEvent = CLIENT_EVENTS.validate_json(text)
        except ValidationError as e:
            error = _invalid(e)
            error.event_id = _event_id(text)
            await self.send(error_event(error))
            return
        if event.type in _REALTIME_ONLY and self.state.type != "realtime":
            error = RealtimeError(message=f"A transcription session takes no {event.type}.", code="invalid_event")
        else:
            error = await self._handlers[event.type](event)
        if error is not None:
            error.event_id = event.event_id
            await self.send(error_event(error))

    async def _session_update(self, event: SessionUpdate) -> RealtimeError | None:
        update = event.session
        if update.type is not None and update.type != self.state.type:
            return RealtimeError(
                message=f"This session is a {self.state.type} session; its type can't change.",
                code="invalid_value",
                param="session.type",
            )
        if update.output_modalities is not None and "audio" in update.output_modalities:
            return _audio_output("session.output_modalities")
        audio_input = (
            update.audio.input if update.audio is not None and "input" in update.audio.model_fields_set else None
        )
        if audio_input is not None:
            error = self._check_audio_input(audio_input)
            if error is not None:
                return error
            for name in ("format", "transcription"):
                if name in audio_input.model_fields_set:
                    setattr(self.state.audio.input, name, getattr(audio_input, name))
        for name in _UPDATABLE:
            if name in update.model_fields_set:
                setattr(self.state, name, getattr(update, name))
        if ignored := sorted(update.model_extra or {}):
            logger.debug("Realtime session %s: ignoring session fields not used yet: %s", self.state.id, ignored)
        await self.send(session_event("session.updated", self.state))
        return None

    def _check_audio_input(self, given: AudioInput) -> RealtimeError | None:
        if given.turn_detection is not None:
            return RealtimeError(
                message="Turn detection is not supported yet; commit each turn with input_audio_buffer.commit.",
                code="unsupported_value",
                param="session.audio.input.turn_detection",
            )
        changes_format = "format" in given.model_fields_set and given.format != self.state.audio.input.format
        if changes_format and self.buffer.seconds:
            return RealtimeError(
                message="Clear the input audio buffer before changing its format.",
                code="invalid_value",
                param="session.audio.input.format",
            )
        return None

    def _missing_item(self, item_id: str, param: str) -> RealtimeError:
        return RealtimeError(message=f"No item {item_id!r} in the conversation.", code="item_not_found", param=param)

    async def _item_create(self, event: ItemCreate) -> RealtimeError | None:
        item, previous = event.item, event.previous_item_id
        if (error := _audio_in_item(item, "item")) is not None:
            return error
        if previous is not None and previous != "root" and self.conversation.get(previous) is None:
            return self._missing_item(previous, "previous_item_id")
        if self.conversation.get(item.id) is not None:
            return RealtimeError(
                message=f"The conversation already has an item {item.id!r}.", code="invalid_value", param="item.id"
            )
        before = self.conversation.insert(item, previous)
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        await self.send(server_event("conversation.item.done", previous_item_id=before, item=item))
        return None

    async def _item_retrieve(self, event: ItemRetrieve) -> RealtimeError | None:
        item = self.conversation.get(event.item_id)
        if item is None:
            return self._missing_item(event.item_id, "item_id")
        await self.send(server_event("conversation.item.retrieved", item=item))
        return None

    async def _item_delete(self, event: ItemDelete) -> RealtimeError | None:
        if not self.conversation.delete(event.item_id):
            return self._missing_item(event.item_id, "item_id")
        await self.send(server_event("conversation.item.deleted", item_id=event.item_id))
        return None

    def _input(self, given: list[Item | ItemReference]) -> list[Item] | RealtimeError:
        """A response's own input items, each reference replaced by the conversation item it names."""
        items = []
        for i, item in enumerate(given):
            if isinstance(item, ItemReference):
                found = self.conversation.get(item.id)
                if found is None:
                    return self._missing_item(item.id, f"response.input[{i}].id")
                item = found
            elif (error := _audio_in_item(item, f"response.input[{i}]")) is not None:
                return error
            items.append(item)
        return items

    async def _response_create(self, event: ResponseCreate) -> RealtimeError | None:
        if self.active is not None and not self.active.done():
            return RealtimeError(
                message="A response is in progress; wait for its response.done or cancel it.",
                code="conversation_already_has_active_response",
            )
        params = event.response
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
        tools = params.tools if params.tools is not None else self.state.tools
        tool_choice = params.tool_choice if params.tool_choice is not None else self.state.tool_choice
        parallel = params.parallel_tool_calls
        instructions = params.instructions if params.instructions is not None else self.state.instructions
        pending = self._transcribing

        async def start() -> Answer | RealtimeError:
            # the turns committed before this response, transcribed
            if pending is not None:
                await asyncio.wait([pending])
            request = ResponsesRequest(
                model=llm,
                input=as_input(items),
                instructions=instructions,
                max_output_tokens=None if max_output_tokens == "inf" else max_output_tokens,
                reasoning={"effort": reasoning.effort} if reasoning is not None and reasoning.effort else None,
                tools=[tool.model_dump(exclude_none=True) for tool in tools] if tools else None,
                tool_choice=tool_choice.model_dump() if isinstance(tool_choice, BaseModel) else tool_choice,
                parallel_tool_calls=parallel if parallel is not None else self.state.parallel_tool_calls,
                stream=True,
                store=False,
            )
            answer = self.calls.respond(llm, request, self.state.id)
            if answer is None:
                return RealtimeError(message=f"Pipeline model {llm!r} isn't ready.", code="model_not_ready")
            return answer

        in_conversation = params.conversation != "none"
        response = RealtimeResponse(
            id=f"resp_{random_uuid()[:20]}",
            conversation_id=self.conversation.id if in_conversation else None,
            output_modalities=modalities,
            max_output_tokens=max_output_tokens,
            metadata=params.metadata,
        )
        conversation = self.conversation if in_conversation else None
        self.active = ActiveResponse(response, llm, start, conversation, self.send)
        return None

    async def _response_cancel(self, event: ResponseCancel) -> RealtimeError | None:
        active = self.active
        if active is None or active.done() or event.response_id not in (None, active.response.id):
            return RealtimeError(message="No response to cancel is in progress.", code="response_cancel_not_active")
        active.cancel()
        return None

    def _no_stt(self) -> RealtimeError | None:
        if self.config.stt is not None:
            return None
        return RealtimeError(
            message=f"Pipeline {self.config.name!r} has no stt model to transcribe audio.", code="invalid_event"
        )

    async def _audio_append(self, event: AudioAppend) -> RealtimeError | None:
        if (error := self._no_stt()) is not None:
            return error
        try:
            audio = base64.b64decode(event.audio, validate=True)
        except binascii.Error:
            return RealtimeError(message="audio is not valid base64.", code="invalid_value", param="audio")
        audio_format = self.state.audio.input.format
        rate = audio_format.rate if isinstance(audio_format, AudioPcm) else G711_RATE
        problem = self.buffer.append(to_pcm16(audio, audio_format.type), rate)
        if problem is not None:
            return RealtimeError(message=problem, code="input_audio_buffer_full", param="audio")
        return None

    async def _audio_commit(self, event: AudioCommit) -> RealtimeError | None:
        if (error := self._no_stt()) is not None:
            return error
        if self.buffer.seconds < MIN_SECONDS:
            return RealtimeError(
                message=f"The input audio buffer holds {self.buffer.seconds * 1000:.0f} ms; a commit needs at least"
                f" {MIN_SECONDS * 1000:.0f} ms.",
                code="input_audio_buffer_commit_empty",
            )
        wav, seconds = self.buffer.take()
        item = UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])
        before = self.conversation.insert(item)
        await self.send(server_event("input_audio_buffer.committed", previous_item_id=before, item_id=item.id))
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        task = asyncio.ensure_future(self._transcribe(item, wav, seconds, self._transcribing))
        self._transcribing = task
        self._transcriptions.add(task)
        task.add_done_callback(self._transcriptions.discard)
        return None

    async def _audio_clear(self, event: AudioClear) -> RealtimeError | None:
        self.buffer.clear()
        await self.send(server_event("input_audio_buffer.cleared"))
        return None

    async def _transcribe(self, item: UserMessage, wav: bytes, seconds: float, previous: asyncio.Task | None) -> None:
        """Fills *item*'s transcript once the turn before it is done."""
        if previous is not None:
            await asyncio.wait([previous])
        stt = self.config.stt
        assert stt is not None
        settings = self.state.audio.input.transcription
        language = (settings.language if settings is not None else None) or self.language
        detect = language is None and self.stt is not None and self.stt.reports_language
        request = TranscriptionRequest.model_construct(
            file=None,
            model=stt,
            language=language,
            prompt=(settings.prompt or "") if settings is not None else "",
            response_format="verbose_json" if detect else "json",
            timestamp_granularities=[],
            stream=False,
            temperature=0.0,
        )
        try:
            result = await self.calls.transcribe(stt, wav, request, self.state.id)
        except Exception:
            logger.exception("Realtime session %s: transcribing %s failed", self.state.id, item.id)
            result = None
        part = item.content[0]
        assert isinstance(part, InputAudio)
        if isinstance(result, TranscriptionResponse | TranscriptionResponseVerbose):
            part.transcript = result.text
            if detect and isinstance(result, TranscriptionResponseVerbose) and result.language not in ("", "unknown"):
                self.language = result.language
            usage = {"type": "duration", "seconds": round(seconds, 3)}
            await self.send(
                server_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item.id,
                    content_index=0,
                    transcript=result.text,
                    usage=usage,
                )
            )
        else:
            if isinstance(result, ErrorResponse):
                error = result.error.model_dump(include={"type", "code", "message", "param"})
            else:
                error = {"type": "server_error", "code": "server_error", "message": f"Model {stt!r} didn't answer."}
            await self.send(
                server_event(
                    "conversation.item.input_audio_transcription.failed", item_id=item.id, content_index=0, error=error
                )
            )
        if self.conversation.get(item.id) is not None:
            before = self.conversation.before(item.id)
            await self.send(server_event("conversation.item.done", previous_item_id=before, item=item))


def _audio_in_item(item: Item, param: str) -> RealtimeError | None:
    """The error for audio sent inside an item, not through the input audio buffer."""
    if isinstance(item, UserMessage):
        for i, part in enumerate(item.content):
            if isinstance(part, InputAudio):
                return RealtimeError(
                    message="Audio in an item is not supported yet; append it to the input audio buffer.",
                    code="unsupported_value",
                    param=f"{param}.content[{i}]",
                )
    return None
