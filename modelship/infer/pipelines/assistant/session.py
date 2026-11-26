"""One `/v1/realtime` session, held by an assistant pipeline's replica; the gateway relays the client's WebSocket
to it as a `Channel`, and the pipeline names the models it runs over. A session holds a conversation of text and of
the client's audio turns, which the client commits or server VAD finds and the pipeline's stt model transcribes,
and answers it with the pipeline's LLM, in text or spoken by its tts model. A turn server VAD finds cuts the answer
off, and the client truncates a spoken answer to what it played."""

import asyncio
import base64
import json
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel

from modelship.infer.base_pipeline import CLOSE_REFUSED
from modelship.infer.pipelines.assistant.calls import Answer, PipelineCalls
from modelship.infer.pipelines.assistant.client_events import parse_event
from modelship.infer.pipelines.assistant.codecs import G711_RATE, bytes_per_second, from_wav, to_pcm16
from modelship.infer.pipelines.assistant.conversation import Conversation, as_input
from modelship.infer.pipelines.assistant.input.input_buffer import MIN_SECONDS, InputAudioBuffer
from modelship.infer.pipelines.assistant.input.stt_stream import Streamed, TurnStream
from modelship.infer.pipelines.assistant.input.turns import MAX_TURN_MS, NewFinder, SpeechFinder, TurnDetector
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.infer.pipelines.assistant.output.response import ActiveResponse, NewSpeech
from modelship.infer.pipelines.assistant.output.speech import SendAudio, Speech, Spoken
from modelship.infer.pipelines.assistant.tasks import BackgroundTasks
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
)
from modelship.openai.protocol.realtime import (
    SESSION_OBJECTS,
    AssistantMessage,
    AudioAppend,
    AudioClear,
    AudioCommit,
    AudioG711,
    AudioInput,
    AudioOutput,
    AudioPcm,
    InputAudio,
    Item,
    ItemCreate,
    ItemDelete,
    ItemReference,
    ItemRetrieve,
    ItemTruncate,
    OutputAudio,
    OutputText,
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
from modelship.utils import random_uuid
from modelship.utils.config_schema import AssistantPipelineConfig

logger = get_logger("realtime")

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
_STT_ONLY = frozenset({"input_audio_buffer.append", "input_audio_buffer.commit"})
# a spoken answer goes out in pieces of this much audio
_AUDIO_DELTA_SECONDS = 0.5
# OpenAI's maximum duration of a session
MAX_SESSION_S = 3600
# RFC 6455 normal closure: the session reached its maximum duration
CLOSE_EXPIRED = 1000
# how long a closing session waits for its streamed transcriptions to close
_STREAM_CLOSE_S = 2.0
# with wake words: no turn starts without one
_ASLEEP = (math.inf, math.inf)
# an answer that ends in one of these asks something (Home Assistant's continue_conversation)
_ASKS = ("?", ";", "？")

Handler = Callable[[Any], Awaitable[RealtimeError | None]]


class Channel(Protocol):
    """The client's WebSocket, as the gateway relays it."""

    async def receive(self) -> str | bytes | None:
        """The client's next frame; None once the client has left."""
        ...

    async def send(self, text: str) -> None:
        """Sends *text* to the client as a text frame."""
        ...

    async def close(self, code: int) -> None:
        """Closes the client's WebSocket with *code*."""
        ...


def _refusal(config: AssistantPipelineConfig, calls: PipelineCalls) -> RealtimeError | None:
    """Why a session on *config* can't open now, or None: each of its models has to be serving."""
    for model in config.models():
        state = calls.model_state(model)
        if state == "unknown":
            logger.warning("Pipeline %r uses model %r, which its gateway doesn't have", config.name, model)
            return RealtimeError(
                message=f"Pipeline {config.name!r} uses model {model!r}, which this gateway doesn't have.",
                code="model_not_found",
            )
        if state == "starting":
            return RealtimeError(
                message=f"Pipeline {config.name!r} uses model {model!r}, which isn't ready yet.", type="server_error"
            )
    return None


def _turn_detection(config: AssistantPipelineConfig) -> ServerVad | None:
    """A new session's turn detection: server VAD on a pipeline with stt, unless its defaults say otherwise."""
    if config.assistant_config.stt is None:
        return None
    defaults = config.assistant_config.session_defaults
    given = defaults.audio.input if defaults is not None and defaults.audio is not None else None
    if given is None or "turn_detection" not in given.model_fields_set:
        return server_vad()
    if given.turn_detection is None:
        return None
    return ServerVad.model_validate(given.turn_detection.model_dump())


def _new_session(config: AssistantPipelineConfig) -> RealtimeSession:
    assistant = config.assistant_config
    kind = "realtime" if assistant.llm is not None else "transcription"
    defaults = assistant.session_defaults
    fields = defaults.model_dump(exclude_none=True, exclude={"audio"}) if defaults is not None else {}
    audio = SessionAudio(input=AudioInput(turn_detection=_turn_detection(config)))
    if kind == "realtime":
        fields.setdefault("output_modalities", ["audio"] if assistant.tts is not None else ["text"])
        audio.output = AudioOutput()
    fields["audio"] = audio
    return RealtimeSession(
        type=kind,
        object=SESSION_OBJECTS[kind],
        id=f"sess_{random_uuid()[:20]}",
        model=config.name,
        expires_at=int(time.time() + MAX_SESSION_S),
        **fields,
    )


def _voice_name(voice: Voice | None) -> str | None:
    return voice.id if isinstance(voice, VoiceId) else voice


def _asks(response: RealtimeResponse) -> bool:
    """Whether *response*'s last message ends in a question."""
    for item in reversed(response.output):
        if isinstance(item, AssistantMessage) and item.content:
            part = item.content[-1]
            said = part.text if isinstance(part, OutputText) else part.transcript
            return said.rstrip().endswith(_ASKS)
    return False


async def _stt_capabilities(
    config: AssistantPipelineConfig, calls: PipelineCalls
) -> SttCapabilities | RealtimeError | None:
    """What the pipeline's stt model can do, asked once for the session; None without one."""
    stt = config.assistant_config.stt
    if stt is None:
        return None
    try:
        found = await calls.stt_capabilities(stt)
    except Exception:
        logger.exception("Pipeline %r: model %r didn't say what it can do", config.name, stt)
        return RealtimeError(
            message=f"Pipeline {config.name!r} couldn't ask model {stt!r} what it can do.", type="server_error"
        )
    if found is None:
        return RealtimeError(
            message=f"Pipeline {config.name!r} uses model {stt!r} as stt, which doesn't transcribe.",
            code="invalid_value",
            param="model",
        )
    return found


async def refuse(channel: Channel, error: RealtimeError) -> None:
    """Tells the client why its session can't open, and closes its WebSocket."""
    await channel.send(json.dumps(error_event(error)))
    await channel.close(CLOSE_REFUSED)


async def serve_session(
    channel: Channel, config: AssistantPipelineConfig, calls: PipelineCalls, finder: NewFinder | None = None
) -> None:
    """Runs one session on *config* until the client leaves; *calls* is how it calls the pipeline's models, and
    *finder* how server VAD finds speech and wake words, None without speech detection."""
    refusal = _refusal(config, calls)
    if refusal is not None:
        await refuse(channel, refusal)
        return
    stt = await _stt_capabilities(config, calls)
    if isinstance(stt, RealtimeError):
        await refuse(channel, stt)
        return
    await Session(channel, config, calls, stt, finder).run()


class Session:
    def __init__(
        self,
        channel: Channel,
        config: AssistantPipelineConfig,
        calls: PipelineCalls,
        stt: SttCapabilities | None,
        finder: NewFinder | None = None,
    ):
        self.channel = channel
        self.config = config
        # the pipeline's assistant settings
        self.assistant = config.assistant_config
        self.calls = calls
        # what the stt model can do, asked once at open; None without one
        self.stt = stt
        self._finder = finder
        self.state = _new_session(config)
        self.conversation = Conversation()
        self.active: ActiveResponse | None = None
        self.buffer = InputAudioBuffer()
        # the language the stt model detected on the session's first turn
        self.language: str | None = None
        # the last transcription queued; each runs after the one before it
        self._transcribing: asyncio.Task | None = None
        self._transcriptions = BackgroundTasks()
        # with a streaming stt model: the stream of the turn in progress, or of the buffer when the client commits
        self._stream: TurnStream | None = None
        # the session's calls to its streaming stt model go one at a time: Serve routes none while another is in flight
        self._stt_lane = asyncio.Lock()
        # streams closing
        self._closing = BackgroundTasks()
        # server VAD, while the session's turn detection is on
        self.turns: TurnDetector | None = None
        # automatic responses waiting for their turn's transcript
        self._replies = BackgroundTasks()
        # a turn server VAD found went unanswered because the user talked again
        self._reply_owed = False
        # feeds still finishing after server VAD was turned off
        self._retired_feeds = BackgroundTasks()
        # the client playing the session's spoken answers
        self.playback = Playback()
        # the wake words a realtime session's server VAD turns start at; None for none
        self._wake = self.assistant.wake if self.state.type == "realtime" else None
        # with wake words: where a turn may start without one, on the timeline; a follow-up's window opens it
        self._awake = _ASLEEP
        if self._wake is not None and self.assistant.duplex == "half":
            # the wake words are heard in the audio the session mutes while its answer plays
            self.buffer.keep_sent()
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
        expiry = asyncio.ensure_future(self._expire())
        try:
            while (message := await self.channel.receive()) is not None:
                if isinstance(message, bytes):
                    await self.send(error_event(RealtimeError(message="Send events as JSON text frames.")))
                    continue
                await self._handle(message)
        finally:
            self._closed = True
            # a closed session waits in reference cycles for the cyclic GC; its audio goes now
            self.buffer.clear()
            self._drop_stream()
            expiry.cancel()
            for tasks in (self._transcriptions, self._replies, self._retired_feeds):
                tasks.cancel()
            # a cancelled streamed transcription closes its stream as it ends
            await self._transcriptions.wait(_STREAM_CLOSE_S)
            await self._closing.wait(_STREAM_CLOSE_S)
            if self.turns is not None:
                await self.turns.close()
            if self.active is not None:
                await self.active.abandon()
        logger.info("Realtime session %s closed", self.state.id)

    async def _expire(self) -> None:
        """Ends the session once it reaches its maximum duration, as OpenAI does."""
        await asyncio.sleep(MAX_SESSION_S)
        logger.info("Realtime session %s reached its maximum duration", self.state.id)
        error = RealtimeError(
            message=f"Your session hit the maximum duration of {MAX_SESSION_S / 60:g} minutes.", code="session_expired"
        )
        await self.send(error_event(error))
        await self.channel.close(CLOSE_EXPIRED)

    async def send(self, event: dict) -> None:
        """Sends *event*, unless the client has left. A response sends from its own task."""
        text = json.dumps(event)
        async with self._sending:
            if not self._closed:
                await self.channel.send(text)

    async def _handle(self, text: str) -> None:
        event = parse_event(text, self._event_types())
        if isinstance(event, RealtimeError):
            await self.send(error_event(event))
            return
        error = await self._handlers[event.type](event)
        if error is not None:
            error.event_id = event.event_id
            await self.send(error_event(error))

    def _event_types(self) -> list[str]:
        """The client event types this session takes."""
        return [
            kind
            for kind in self._handlers
            if (self.state.type == "realtime" or kind not in _REALTIME_ONLY)
            and (self.assistant.stt is not None or kind not in _STT_ONLY)
        ]

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
                message="semantic_vad is not supported yet; use server_vad.", code="invalid_value", param=param
            )
        if isinstance(detection, ServerVad):
            if detection.idle_timeout_ms is not None:
                return RealtimeError(
                    message="idle_timeout_ms is not supported yet.",
                    code="invalid_value",
                    param=f"{param}.idle_timeout_ms",
                )
            if self.assistant.stt is None:
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
        llm = self.assistant.llm
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
                return RealtimeError(message=f"Pipeline model {llm!r} isn't ready.", type="server_error")
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
        active = self.active = ActiveResponse(response, llm, start, conversation, self.send, speech)
        active.task.add_done_callback(lambda _: self._answered(active))
        if in_conversation:
            self._reply_owed = False
        return None

    def _answered(self, active: ActiveResponse) -> None:
        """With wake words, opens the follow-up's window once *active*'s answer has played, as it says."""
        if self._wake is None or not active.in_conversation or active.response.status != "completed":
            return
        follow_up = self._wake.follow_up
        if follow_up.mode == "off" or (follow_up.mode == "question" and not _asks(active.response)):
            return
        opens = self.buffer.end_ms + self.playback.remaining() * 1000
        self._awake = (opens, opens + follow_up.listen_s * 1000)

    def _window(self) -> tuple[float, float] | None:
        """Where speech may start a turn now: anywhere without wake words, else in the follow-up's window, or while
        an answer runs when the follow-up is always on."""
        if self._wake is None:
            return None
        answering = (self.active is not None and not self.active.done()) or self.playback.playing()
        if self._wake.follow_up.mode == "always" and answering:
            return (-math.inf, math.inf)
        return self._awake

    def _no_tts(self, param: str) -> RealtimeError | None:
        if self.assistant.tts is not None:
            return None
        return RealtimeError(
            message=f"Pipeline {self.config.name!r} has no tts model to speak with; ask for text output.",
            code="invalid_value",
            param=param,
        )

    def _speech(self, audio_format: AudioPcm | AudioG711, voice: Voice | None, speed: float) -> NewSpeech:
        """A spoken answer's speech: the pipeline's tts model says it in *voice* at *speed*, sent as *audio_format*."""
        tts = self.assistant.tts
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

    async def _audio_append(self, event: AudioAppend) -> RealtimeError | None:
        try:
            audio = base64.b64decode(event.audio, validate=True)
        except ValueError:
            # binascii.Error for bad base64, ValueError for non-ASCII text
            return RealtimeError(message="audio is not valid base64.", code="invalid_value", param="audio")
        audio_format = self.state.audio.input.format
        rate = audio_format.rate if isinstance(audio_format, AudioPcm) else G711_RATE
        pcm = sent = to_pcm16(audio, audio_format.type)
        if self.assistant.duplex == "half" and self.playback.playing():
            # the answer's echo: silence in its place keeps the timeline
            pcm = bytes(len(pcm))
        problem = self.buffer.append(pcm, rate, sent)
        if problem is not None:
            return RealtimeError(message=problem, param="audio")
        if self.turns is not None:
            self.turns.appended()
        elif self._streams() and self._stream is None:
            # the client commits: its buffer streams from its start
            self._stream = self._open_stream(self.buffer.start_ms)
        if self._stream is not None:
            self._stream.feed()
        return None

    async def _audio_commit(self, event: AudioCommit) -> RealtimeError | None:
        if self.buffer.seconds < MIN_SECONDS:
            return RealtimeError(
                message=f"The input audio buffer holds {self.buffer.seconds * 1000:.0f} ms; a commit needs at least"
                f" {MIN_SECONDS * 1000:.0f} ms.",
                code="input_audio_buffer_commit_empty",
            )
        audio: bytes | Streamed
        if self._streams():
            # the turn server VAD has open streams from its start; else the buffer from its start
            stream = self._stream or self._open_stream(self.buffer.start_ms)
            self._stream = None
            audio = Streamed(stream, self.buffer.pcm(stream.sent_ms, self.buffer.end_ms))
            seconds = (self.buffer.end_ms - stream.start_ms) / 1000
            self.buffer.clear()
        else:
            audio, seconds = self.buffer.take()
        turn = self.turns.reset() if self.turns is not None else None
        if turn is not None:
            # the client ended the turn server VAD had open
            end = round(self.buffer.end_ms)
            await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=end, item_id=turn))
        await self._commit(audio, seconds, turn)
        return None

    async def _commit(
        self, audio: bytes | Streamed | None, seconds: float, turn: str | None = None
    ) -> tuple[UserMessage, asyncio.Task]:
        """Adds a turn of the client's audio, as the user message *turn* names, and queues its transcription: of
        *audio*, a WAV, or by finishing its stream; None for a turn without speech, whose transcript is empty."""
        item = UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])
        if turn is not None:
            item.id = turn
        before = self.conversation.insert(item)
        await self.send(server_event("input_audio_buffer.committed", previous_item_id=before, item_id=item.id))
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        if audio is None:
            transcription = self._untranscribed(item, seconds, self._transcribing)
        elif isinstance(audio, Streamed):
            transcription = self._transcribe_stream(item, audio, seconds, self._transcribing)
        else:
            transcription = self._transcribe(item, audio, seconds, self._transcribing)
        task = self._transcribing = self._transcriptions.spawn(transcription)
        return item, task

    async def _audio_clear(self, event: AudioClear) -> RealtimeError | None:
        self.buffer.clear()
        self._drop_stream()
        if self.turns is not None:
            self.turns.reset()
        await self.send(server_event("input_audio_buffer.cleared"))
        return None

    def _streams(self) -> bool:
        """Whether the session's stt model takes its turns as they arrive."""
        return self.stt is not None and self.stt.streaming

    def _open_stream(self, start_ms: float, turn: str | None = None) -> TurnStream:
        """A stream of the buffer's audio from *start_ms*: the turn *turn* server VAD found, which gets deltas, or
        the buffer the client commits."""
        stt = self.assistant.stt
        assert stt is not None and self.stt is not None
        settings = self.state.audio.input.transcription
        language = (settings.language if settings is not None else None) or self.language

        async def delta(text: str) -> None:
            event = "conversation.item.input_audio_transcription.delta"
            await self.send(server_event(event, item_id=turn, content_index=0, delta=text))

        key = f"{self.state.id}:{turn or f'buffer_{random_uuid()[:12]}'}"
        options = SttStreamOptions(language=language)
        chunk = self.stt.chunk_ms or 0
        turns = self.turns
        if turn is None or turns is None:
            return TurnStream(self.calls, stt, key, self.buffer, start_ms, chunk, options, self._stt_lane)
        # audio server VAD hasn't heard yet may be past the turn's end
        return TurnStream(
            self.calls, stt, key, self.buffer, start_ms, chunk, options, self._stt_lane, delta, lambda: turns.heard_ms
        )

    def _drop_stream(self) -> None:
        """Closes the stream of the audio that won't be committed."""
        stream, self._stream = self._stream, None
        if stream is not None:
            self._close_later(stream)

    def _close_later(self, stream: TurnStream) -> None:
        self._closing.spawn(stream.close())

    def _heard(self) -> None:
        """Server VAD heard more audio: the turn's stream may push it, and a turn a wake word opened streams from
        its start once speech past the wake word is heard."""
        turns = self.turns
        opens = self._stream is None and self._streams() and turns is not None and not turns.waiting
        if opens and turns is not None and turns.turn is not None:
            turn, start = turns.turn
            self._stream = self._open_stream(start, turn)
        if self._stream is not None:
            self._stream.feed()

    async def _detect_turns(self) -> None:
        """Starts, retunes or stops server VAD, as the session's turn detection says."""
        settings = self.state.audio.input.turn_detection
        if not isinstance(settings, ServerVad):
            if self.turns is not None:
                # not cancelled: a feed in flight stops at its next step, and a commit it began completes
                self.turns.reset()
                if (task := self.turns.task) is not None:
                    self._retired_feeds.add(task)
                self.turns = None
                # the open turn's stream; the buffer streams from its start with the next append
                self._drop_stream()
            return
        if self.turns is not None:
            self.turns.settings = settings
            return
        # the buffer's stream, which the client was to commit
        self._drop_stream()
        self.turns = TurnDetector(
            self.buffer,
            settings,
            self._open_finder,
            self._turn_started,
            self._turn_ended,
            self._detection_failed,
            self._min_speech,
            # a streamed turn costs its model no more as it grows
            max_turn_ms=None if self._streams() else MAX_TURN_MS,
            progressed=self._heard,
            listen_ms=self._wake.listen_s * 1000 if self._wake is not None else None,
            window=self._window,
        )

    def _min_speech(self) -> int:
        """The speech a turn needs before it starts now: more while an answer plays, which it would cut off."""
        if self.assistant.duplex == "full" and self.playback.playing():
            return self.assistant.min_interruption_ms
        return 0

    def _open_finder(self, rate: int, start_ms: float) -> SpeechFinder:
        if self._finder is None:
            raise RuntimeError(f"pipeline {self.config.name!r} has no speech detection")
        return self._finder(rate, start_ms, self._wake is not None)

    async def _turn_started(self, turn: str, audio_start_ms: int, woken: bool) -> None:
        """A turn server VAD found starts, where its wake word ended when *woken*."""
        # a follow-up's window opens for one turn
        self._awake = _ASLEEP
        await self.send(server_event("input_audio_buffer.speech_started", audio_start_ms=audio_start_ms, item_id=turn))
        if self._streams():
            # its deltas go out after its speech_started; a wake word's turn streams once it hears speech
            self._drop_stream()
            if not woken:
                self._stream = self._open_stream(audio_start_ms, turn)
                self._stream.feed()
        # a client stops playing at speech_started
        self.playback.stop()
        if self._interrupts():
            active = self.active
            if active is not None and not active.done() and active.in_conversation:
                active.cancel("turn_detected")
                # the answer cut off is owed, even if this turn says nothing; one a wake word stopped is not
                self._reply_owed = not woken

    def _interrupts(self) -> bool:
        """Whether a turn server VAD finds cuts the answer off."""
        settings = self.state.audio.input.turn_detection
        return isinstance(settings, ServerVad) and settings.interrupt_response

    async def _turn_ended(self, turn: str, audio_start_ms: int, audio_end_ms: int, spoken: bool) -> None:
        """Commits the turn server VAD found; one without speech, after a wake word nobody followed, is committed
        untranscribed and gets no answer."""
        # before any await, so a commit or clear from the client can't take the same audio
        audio: bytes | Streamed | None
        if not spoken:
            self._drop_stream()
            self.buffer.drop_before(audio_end_ms)
            audio, seconds = None, (audio_end_ms - audio_start_ms) / 1000
        elif self._streams():
            stream = self._stream if self._stream is not None else self._open_stream(audio_start_ms, turn)
            self._stream = None
            audio = Streamed(stream, self.buffer.pcm(stream.sent_ms, audio_end_ms))
            seconds = (audio_end_ms - stream.start_ms) / 1000
            self.buffer.drop_before(audio_end_ms)
        else:
            audio, seconds = self.buffer.take(audio_start_ms, audio_end_ms)
        await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=audio_end_ms, item_id=turn))
        settings = self.state.audio.input.turn_detection
        answers = isinstance(settings, ServerVad) and settings.create_response and self.state.type == "realtime"
        # a turn that ends while the model answers gets no answer of its own; one it cut off doesn't count
        responding = self.active is not None and not self.active.done() and not self.active.cancelled
        item, transcribed = await self._commit(audio, seconds, turn)
        if answers and not responding and spoken:
            self._replies.spawn(self._reply(item, transcribed))

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

    async def _detection_failed(self) -> None:
        await self.send(
            error_event(
                RealtimeError(
                    message=f"Pipeline {self.config.name!r} can't detect turns now: its speech detection failed."
                    " Its audio stays buffered, and input_audio_buffer.commit still works.",
                    type="server_error",
                    code="server_error",
                )
            )
        )

    async def _transcribe(self, item: UserMessage, wav: bytes, seconds: float, previous: asyncio.Task | None) -> None:
        """Fills *item*'s transcript once the turn before it is done."""
        if previous is not None:
            await asyncio.wait([previous])
        stt = self.assistant.stt
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
        if isinstance(result, TranscriptionResponse | TranscriptionResponseVerbose):
            if detect and isinstance(result, TranscriptionResponseVerbose) and result.language not in ("", "unknown"):
                self.language = result.language
            await self._transcribed(item, seconds, result.text)
        elif isinstance(result, ErrorResponse):
            error = result.error.model_dump(include={"type", "code", "message", "param"})
            await self._transcribed(item, seconds, error=error)
        else:
            error = {"type": "server_error", "code": "server_error", "message": f"Model {stt!r} didn't answer."}
            await self._transcribed(item, seconds, error=error)

    async def _untranscribed(self, item: UserMessage, seconds: float, previous: asyncio.Task | None) -> None:
        """Gives *item*, which holds no speech, its empty transcript once the turn before it is done."""
        if previous is not None:
            await asyncio.wait([previous])
        await self._transcribed(item, seconds, "")

    async def _transcribe_stream(
        self, item: UserMessage, audio: Streamed, seconds: float, previous: asyncio.Task | None
    ) -> None:
        """Fills *item*'s transcript by finishing its stream, once the turn before it is done."""
        try:
            if previous is not None:
                await asyncio.wait([previous])
            result = await audio.stream.finish(audio.rest)
        except asyncio.CancelledError:
            # the session closed before the stream finished
            self._close_later(audio.stream)
            raise
        if isinstance(result, SttStreamText):
            if self.language is None and self.stt is not None and self.stt.reports_language and result.language:
                self.language = result.language
            await self._transcribed(item, seconds, result.text)
        else:
            await self._transcribed(item, seconds, error=result)

    async def _transcribed(
        self, item: UserMessage, seconds: float, transcript: str | None = None, error: dict | None = None
    ) -> None:
        """Gives *item* its *transcript*, or tells the client its transcription failed with *error*."""
        if transcript is not None:
            part = item.content[0]
            assert isinstance(part, InputAudio)
            part.transcript = transcript
            usage = {"type": "duration", "seconds": round(seconds, 3)}
            await self.send(
                server_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item.id,
                    content_index=0,
                    transcript=transcript,
                    usage=usage,
                )
            )
        else:
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
                    code="invalid_value",
                    param=f"{param}.content[{i}]",
                )
    return None
