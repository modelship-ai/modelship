"""One `/v1/realtime` WebSocket session. The gateway owns the session; the pipeline names the models it runs
over. So far a session holds a conversation of text and of the client's audio turns, which the client commits or
server VAD finds and the pipeline's stt model transcribes, and answers it with the pipeline's LLM, in text or
spoken by its tts model. A turn server VAD finds cuts the answer off, and the client truncates a spoken answer to
what it played."""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    SttCapabilities,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.openai.protocol.realtime import (
    CLIENT_EVENTS,
    SESSION_OBJECTS,
    AssistantMessage,
    AudioAppend,
    AudioClear,
    AudioCommit,
    AudioG711,
    AudioInput,
    AudioOutput,
    AudioPcm,
    ClientEvent,
    InputAudio,
    Item,
    ItemCreate,
    ItemDelete,
    ItemReference,
    ItemRetrieve,
    ItemTruncate,
    OutputAudio,
    RealtimeError,
    RealtimeResponse,
    RealtimeSession,
    ResponseAudio,
    ResponseAudioOutput,
    ResponseCancel,
    ResponseCreate,
    SemanticVad,
    ServerVad,
    SessionAudio,
    SessionUpdate,
    UserMessage,
    Voice,
    VoiceId,
    error_event,
    server_event,
    server_vad,
    session_event,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.openai.realtime.audio import (
    G711_RATE,
    MAX_SECONDS,
    MIN_SECONDS,
    InputAudioBuffer,
    bytes_per_second,
    from_wav,
    to_pcm16,
)
from modelship.openai.realtime.calls import Answer, PipelineCalls
from modelship.openai.realtime.conversation import Conversation, as_input
from modelship.openai.realtime.playback import Playback
from modelship.openai.realtime.response import ActiveResponse, NewSpeech
from modelship.openai.realtime.speech import SendAudio, Speech, Spoken
from modelship.openai.realtime.turns import TurnDetector
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
_REALTIME_ONLY = frozenset(
    {"conversation.item.create", "conversation.item.truncate", "response.create", "response.cancel"}
)
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
        "output_text",
        "output_audio",
        "audio/pcm",
        "audio/pcmu",
        "audio/pcma",
        "str",
        "VoiceId",
        "server_vad",
        "semantic_vad",
    }
)
# a spoken answer goes out in pieces of this much audio
_AUDIO_DELTA_SECONDS = 0.5

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


def _turn_detection(config: PipelineConfig) -> ServerVad | None:
    """A new session's turn detection: server VAD on a pipeline with stt, unless its defaults say otherwise."""
    if config.stt is None:
        return None
    defaults = config.session_defaults
    given = defaults.audio.input if defaults is not None and defaults.audio is not None else None
    if given is None or "turn_detection" not in given.model_fields_set:
        return server_vad()
    if given.turn_detection is None:
        return None
    return ServerVad.model_validate(given.turn_detection.model_dump())


def _new_session(config: PipelineConfig) -> RealtimeSession:
    kind = "realtime" if config.llm is not None else "transcription"
    defaults = config.session_defaults
    fields = defaults.model_dump(exclude_none=True, exclude={"audio"}) if defaults is not None else {}
    audio = SessionAudio(input=AudioInput(turn_detection=_turn_detection(config)))
    if kind == "realtime":
        fields.setdefault("output_modalities", ["audio"] if config.tts is not None else ["text"])
        audio.output = AudioOutput()
    fields["audio"] = audio
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


def _voice_name(voice: Voice | None) -> str | None:
    return voice.id if isinstance(voice, VoiceId) else voice


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
        # the audio of the committed turns not transcribed yet
        self._queued_seconds = 0.0
        # server VAD, while the session's turn detection is on
        self.turns: TurnDetector | None = None
        # automatic responses waiting for their turn's transcript
        self._replies: set[asyncio.Task] = set()
        # whether the client heard that a turn server VAD found waits for room in the transcription queue
        self._busy_reported = False
        # a turn server VAD found went unanswered because the user talked again
        self._reply_owed = False
        # scans still finishing after server VAD was turned off
        self._retired_scans: set[asyncio.Task] = set()
        # the client playing the session's spoken answers
        self.playback = Playback()
        # what each spoken message said, by item id
        self._spoken: dict[str, Spoken] = {}
        self._sending = asyncio.Lock()
        self._closed = False
        self._handlers: dict[str, Handler] = {
            "session.update": self._session_update,
            "conversation.item.create": self._item_create,
            "conversation.item.retrieve": self._item_retrieve,
            "conversation.item.delete": self._item_delete,
            "conversation.item.truncate": self._item_truncate,
            "response.create": self._response_create,
            "response.cancel": self._response_cancel,
            "input_audio_buffer.append": self._audio_append,
            "input_audio_buffer.commit": self._audio_commit,
            "input_audio_buffer.clear": self._audio_clear,
        }

    async def run(self) -> None:
        logger.info("Realtime session %s opened on pipeline %s", self.state.id, self.config.name)
        await self._detect_turns()
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
            for task in (*self._transcriptions, *self._replies, *self._retired_scans):
                task.cancel()
            if self.turns is not None:
                await self.turns.close()
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
        asks_audio = update.output_modalities is not None and "audio" in update.output_modalities
        if asks_audio and (error := self._no_tts("session.output_modalities")) is not None:
            return error
        audio = update.audio
        audio_input = audio.input if audio is not None and "input" in audio.model_fields_set else None
        if audio_input is not None:
            error = self._check_audio_input(audio_input)
            if error is not None:
                return error
            for name in ("format", "transcription", "turn_detection"):
                if name in audio_input.model_fields_set:
                    setattr(self.state.audio.input, name, getattr(audio_input, name))
            await self._detect_turns()
        audio_output = audio.output if audio is not None else None
        if audio_output is not None and self.state.audio.output is not None:
            for name in ("format", "voice", "speed"):
                if name in audio_output.model_fields_set:
                    setattr(self.state.audio.output, name, getattr(audio_output, name))
        for name in _UPDATABLE:
            if name in update.model_fields_set:
                setattr(self.state, name, getattr(update, name))
        if ignored := sorted(update.model_extra or {}):
            logger.debug("Realtime session %s: ignoring session fields not used yet: %s", self.state.id, ignored)
        await self.send(session_event("session.updated", self.state))
        return None

    def _check_audio_input(self, given: AudioInput) -> RealtimeError | None:
        param = "session.audio.input.turn_detection"
        detection = given.turn_detection
        if isinstance(detection, SemanticVad):
            return RealtimeError(
                message="semantic_vad is not supported yet; use server_vad.", code="unsupported_value", param=param
            )
        if isinstance(detection, ServerVad):
            if detection.idle_timeout_ms is not None:
                return RealtimeError(
                    message="idle_timeout_ms is not supported yet.",
                    code="unsupported_value",
                    param=f"{param}.idle_timeout_ms",
                )
            if self.config.stt is None:
                return RealtimeError(
                    message=f"Pipeline {self.config.name!r} has no stt model to take audio.",
                    code="invalid_value",
                    param=param,
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
        self._spoken.pop(event.item_id, None)
        await self.send(server_event("conversation.item.deleted", item_id=event.item_id))
        return None

    async def _item_truncate(self, event: ItemTruncate) -> RealtimeError | None:
        item = self.conversation.get(event.item_id)
        if item is None:
            return self._missing_item(event.item_id, "item_id")
        if not isinstance(item, AssistantMessage):
            return RealtimeError(
                message="Only an assistant message can be truncated.", code="invalid_value", param="item_id"
            )
        index = event.content_index
        part = item.content[index] if index < len(item.content) else None
        if not isinstance(part, OutputAudio):
            return RealtimeError(
                message=f"Part {index} of item {item.id!r} holds no audio.", code="invalid_value", param="content_index"
            )
        active = self.active
        if active is not None and not active.done() and active.has_item(item.id):
            # the client stopped playing an answer still being spoken
            active.cancel()
            self.playback.stop()
            await asyncio.wait([active.task])
        spoken = self._spoken.get(item.id, Spoken())
        played = round(spoken.ms)
        if event.audio_end_ms > played:
            return RealtimeError(
                message=f"audio_end_ms is past the end of the part's audio, at {played} ms.",
                code="invalid_value",
                param="audio_end_ms",
            )
        part.transcript = spoken.truncate(event.audio_end_ms, part.transcript)
        await self.send(
            server_event(
                "conversation.item.truncated",
                item_id=item.id,
                content_index=index,
                audio_end_ms=event.audio_end_ms,
            )
        )
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
        if "audio" in modalities and (error := self._no_tts("response.output_modalities")) is not None:
            return error
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
        speech = None
        if "audio" in modalities:
            output = self.state.audio.output
            assert output is not None
            given = params.audio.output if params.audio is not None else None
            audio_format = given.format if given is not None and given.format is not None else output.format
            voice = given.voice if given is not None and given.voice is not None else output.voice
            response.audio = ResponseAudio(output=ResponseAudioOutput(format=audio_format, voice=voice))
            speech = self._speech(audio_format, voice, output.speed)
        conversation = self.conversation if in_conversation else None
        self.active = ActiveResponse(response, llm, start, conversation, self.send, speech)
        if in_conversation:
            self._reply_owed = False
        return None

    def _no_tts(self, param: str) -> RealtimeError | None:
        if self.config.tts is not None:
            return None
        return RealtimeError(
            message=f"Pipeline {self.config.name!r} has no tts model to speak with; ask for text output.",
            code="invalid_value",
            param=param,
        )

    def _speech(self, audio_format: AudioPcm | AudioG711, voice: Voice | None, speed: float) -> NewSpeech:
        """A spoken answer's speech: the pipeline's tts model says it in *voice* at *speed*, sent as *audio_format*."""
        tts = self.config.tts
        assert tts is not None
        kind = audio_format.type
        rate = audio_format.rate if isinstance(audio_format, AudioPcm) else G711_RATE
        per_second = bytes_per_second(kind, rate)
        step = int(per_second * _AUDIO_DELTA_SECONDS) // 2 * 2
        name = _voice_name(voice)

        async def speak(text: str) -> bytes | RealtimeError:
            request = SpeechRequest(input=text, model=tts, voice=name, speed=speed, response_format="wav")
            try:
                result = await self.calls.speak(tts, request, self.state.id)
            except Exception:
                logger.exception("Realtime session %s: speaking with %r failed", self.state.id, tts)
                result = None
            if isinstance(result, RawSpeechResponse):
                return result.audio
            if isinstance(result, ErrorResponse):
                error = result.error
                return RealtimeError(message=error.message, type=error.type, code=error.code)
            return RealtimeError(
                message=f"Pipeline model {tts!r} didn't speak the answer.", type="server_error", code="server_error"
            )

        def encode(wav: bytes) -> bytes:
            return from_wav(wav, kind, rate)

        def new(item_id: str, send: SendAudio) -> Speech:
            # made while its response runs
            response = self.active
            assert response is not None

            async def played(audio: bytes) -> bool:
                if not await send(audio):
                    return False
                # audio a cancel overtook doesn't play: the client stopped
                if not response.cancelled:
                    self.playback.sent(len(audio) / per_second)
                return True

            speech = Speech(speak, encode, played, step, per_second)
            self._spoken[item_id] = speech.spoken
            return speech

        return new

    async def _response_cancel(self, event: ResponseCancel) -> RealtimeError | None:
        active = self.active
        if active is None or active.done() or event.response_id not in (None, active.response.id):
            return RealtimeError(message="No response to cancel is in progress.", code="response_cancel_not_active")
        active.cancel()
        # a client that cancels stops playing
        self.playback.stop()
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
        except ValueError:
            # binascii.Error for bad base64, ValueError for non-ASCII text
            return RealtimeError(message="audio is not valid base64.", code="invalid_value", param="audio")
        audio_format = self.state.audio.input.format
        rate = audio_format.rate if isinstance(audio_format, AudioPcm) else G711_RATE
        pcm = to_pcm16(audio, audio_format.type)
        if self.config.duplex == "half" and self.playback.playing():
            # the answer's echo: silence in its place keeps the timeline
            pcm = bytes(len(pcm))
        problem = self.buffer.append(pcm, rate)
        if problem is not None:
            return RealtimeError(message=problem, code="input_audio_buffer_full", param="audio")
        if self.turns is not None:
            self.turns.appended()
        return None

    def _busy(self, seconds: float) -> RealtimeError | None:
        if self._queued_seconds + seconds <= MAX_SECONDS:
            return None
        return RealtimeError(
            message=f"The turns waiting to be transcribed hold {self._queued_seconds:.0f} s, and a session queues"
            f" at most {MAX_SECONDS:.0f} s; commit again once they are transcribed.",
            code="input_audio_transcription_busy",
        )

    async def _audio_commit(self, event: AudioCommit) -> RealtimeError | None:
        if (error := self._no_stt()) is not None:
            return error
        if self.buffer.seconds < MIN_SECONDS:
            return RealtimeError(
                message=f"The input audio buffer holds {self.buffer.seconds * 1000:.0f} ms; a commit needs at least"
                f" {MIN_SECONDS * 1000:.0f} ms.",
                code="input_audio_buffer_commit_empty",
            )
        if (error := self._busy(self.buffer.seconds)) is not None:
            return error
        wav, seconds = self.buffer.take()
        turn = self.turns.reset() if self.turns is not None else None
        if turn is not None:
            # the client ended the turn server VAD had open
            end = round(self.buffer.end_ms)
            await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=end, item_id=turn))
        await self._commit(wav, seconds, turn)
        return None

    async def _commit(self, wav: bytes, seconds: float, turn: str | None = None) -> tuple[UserMessage, asyncio.Task]:
        """Adds a turn of the client's audio, as the user message *turn* names, and queues its transcription."""
        item = UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])
        if turn is not None:
            item.id = turn
        before = self.conversation.insert(item)
        await self.send(server_event("input_audio_buffer.committed", previous_item_id=before, item_id=item.id))
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        task = asyncio.ensure_future(self._transcribe(item, wav, seconds, self._transcribing))
        self._transcribing = task
        self._transcriptions.add(task)
        self._queued_seconds += seconds
        task.add_done_callback(lambda done: self._transcribed(done, seconds))
        return item, task

    def _transcribed(self, task: asyncio.Task, seconds: float) -> None:
        self._transcriptions.discard(task)
        self._queued_seconds -= seconds

    async def _audio_clear(self, event: AudioClear) -> RealtimeError | None:
        self.buffer.clear()
        if self.turns is not None:
            self.turns.reset()
        await self.send(server_event("input_audio_buffer.cleared"))
        return None

    async def _detect_turns(self) -> None:
        """Starts, retunes or stops server VAD, as the session's turn detection says."""
        settings = self.state.audio.input.turn_detection
        if not isinstance(settings, ServerVad):
            if self.turns is not None:
                # not cancelled: a scan in flight stops at its next step, and a commit it began completes
                self.turns.reset()
                if (task := self.turns.task) is not None:
                    self._retired_scans.add(task)
                    task.add_done_callback(self._retired_scans.discard)
                self.turns = None
            return
        if self.turns is not None:
            self.turns.settings = settings
            return
        self.turns = TurnDetector(
            self.buffer,
            settings,
            self._scan,
            self._turn_started,
            self._turn_ended,
            self._scan_failed,
            self._min_speech,
        )

    def _min_speech(self) -> int:
        """The speech a turn needs before it starts now: more while an answer plays, which it would cut off."""
        if self.config.duplex == "full" and self.playback.playing():
            return self.config.min_interruption_ms
        return 0

    async def _scan(
        self, audio: bytes, rate: int, threshold: float, silence_duration_ms: int
    ) -> list[tuple[int, int | None, int]] | None:
        return await self.calls.scan(self.config.name, audio, rate, threshold, silence_duration_ms, self.state.id)

    async def _turn_started(self, turn: str, audio_start_ms: int) -> None:
        await self.send(server_event("input_audio_buffer.speech_started", audio_start_ms=audio_start_ms, item_id=turn))
        # a client stops playing at speech_started
        self.playback.stop()
        if self._interrupts():
            active = self.active
            if active is not None and not active.done() and active.in_conversation:
                active.cancel("turn_detected")

    def _interrupts(self) -> bool:
        """Whether a turn server VAD finds cuts the answer off."""
        settings = self.state.audio.input.turn_detection
        return isinstance(settings, ServerVad) and settings.interrupt_response

    async def _turn_ended(self, turn: str, audio_start_ms: int, audio_end_ms: int) -> bool:
        """Commits the turn server VAD found; False while the transcription queue is full, which keeps the turn and
        its audio for a later scan."""
        if (error := self._busy((audio_end_ms - audio_start_ms) / 1000)) is not None:
            if not self._busy_reported:
                self._busy_reported = True
                error.message = (
                    f"The turns waiting to be transcribed hold {self._queued_seconds:.0f} s, and a session queues at"
                    f" most {MAX_SECONDS:.0f} s; the turn just heard is kept and committed once there is room."
                )
                await self.send(error_event(error))
            return False
        self._busy_reported = False
        # before any await, so a commit or clear from the client can't take the same audio
        wav, seconds = self.buffer.take(audio_start_ms, audio_end_ms)
        await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=audio_end_ms, item_id=turn))
        settings = self.state.audio.input.turn_detection
        answers = isinstance(settings, ServerVad) and settings.create_response and self.state.type == "realtime"
        # a turn that ends while the model answers gets no answer of its own
        responding = self.active is not None and not self.active.done()
        item, transcribed = await self._commit(wav, seconds, turn)
        if answers and not responding:
            reply = asyncio.ensure_future(self._reply(item, transcribed))
            self._replies.add(reply)
            reply.add_done_callback(self._replies.discard)
        return True

    async def _reply(self, item: UserMessage, transcribed: asyncio.Task) -> None:
        """Answers the turn *item* once it is transcribed, unless it said nothing and no turn before it waits for an
        answer, or another answer began."""
        await asyncio.wait([transcribed])
        part = item.content[0]
        said = isinstance(part, InputAudio) and bool((part.transcript or "").strip())
        if self.conversation.get(item.id) is None or (self.active is not None and not self.active.done()):
            return
        if self._interrupts() and self.turns is not None and self.turns.turn is not None:
            # the user is talking again, which would cut the answer off: the next turn's answer covers this one
            self._reply_owed = self._reply_owed or said
            return
        if not said and not self._reply_owed:
            return
        error = await self._response_create(ResponseCreate(type="response.create"))
        if error is not None:
            await self.send(error_event(error))

    async def _scan_failed(self) -> None:
        await self.send(
            error_event(
                RealtimeError(
                    message=f"Pipeline {self.config.name!r} can't detect turns now: its audio app didn't answer."
                    " Its audio stays buffered, and input_audio_buffer.commit still works.",
                    type="server_error",
                    code="turn_detection_unavailable",
                )
            )
        )

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
