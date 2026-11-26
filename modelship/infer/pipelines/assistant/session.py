"""One `/v1/realtime` session, held by an assistant pipeline's replica; the gateway relays the client's WebSocket
to it as a `Channel`, and the pipeline names the models it runs over. A session holds a conversation of text and of
the client's audio turns, which the client commits or server VAD finds and the pipeline's stt model transcribes,
and answers it with the pipeline's LLM, in text or spoken by its tts model. A turn server VAD finds cuts the answer
off, and the client truncates a spoken answer to what it played.

`Session` runs the session's life and dispatches the client's events. The work lives in its parts: `session_config`
(settings), `items` (conversation items), `input/` (the audio buffer, server VAD, transcription) and `output/`
(responses, speech, playback)."""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from modelship.infer.base_pipeline import CLOSE_REFUSED
from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.infer.pipelines.assistant.client_events import parse_event
from modelship.infer.pipelines.assistant.codecs import G711_RATE, to_pcm16
from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.input.gate import TurnGate
from modelship.infer.pipelines.assistant.input.input_buffer import MIN_SECONDS, InputAudioBuffer
from modelship.infer.pipelines.assistant.input.transcripts import (
    SilentTurn,
    SttContext,
    TranscriptionQueue,
    TurnAudio,
)
from modelship.infer.pipelines.assistant.input.turn_end import TurnClassifier, turn_end
from modelship.infer.pipelines.assistant.input.turn_input import RecordedInput, StreamedInput, TurnInput
from modelship.infer.pipelines.assistant.input.turns import MAX_TURN_MS, TurnDetector, VadStream, VadStreamFactory
from modelship.infer.pipelines.assistant.items import ConversationItems
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.infer.pipelines.assistant.output.responder import Responder
from modelship.infer.pipelines.assistant.output.response import ActiveResponse
from modelship.infer.pipelines.assistant.output.speech import Spoken
from modelship.infer.pipelines.assistant.session_config import apply_update, check_update, new_session
from modelship.infer.pipelines.assistant.tasks import BackgroundTasks
from modelship.logging import get_logger
from modelship.openai.protocol import SttCapabilities
from modelship.openai.protocol.realtime import (
    AudioAppend,
    AudioClear,
    AudioCommit,
    AudioPcm,
    InputAudio,
    RealtimeError,
    ResponseCancel,
    ResponseCreate,
    SessionUpdate,
    UserMessage,
    VoiceTurnDetection,
    error_event,
    server_event,
    session_event,
)
from modelship.utils.config_schema import AssistantPipelineConfig

logger = get_logger("realtime")

_REALTIME_ONLY = frozenset(
    {"conversation.item.create", "conversation.item.truncate", "response.create", "response.cancel"}
)
_STT_ONLY = frozenset({"input_audio_buffer.append", "input_audio_buffer.commit"})
# OpenAI's maximum duration of a session
MAX_SESSION_S = 3600
# RFC 6455 normal closure: the session reached its maximum duration
CLOSE_EXPIRED = 1000
# how long a closing session waits for its streamed transcriptions to close
_STREAM_CLOSE_S = 2.0

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
    channel: Channel,
    config: AssistantPipelineConfig,
    calls: PipelineCalls,
    vad: VadStreamFactory | None = None,
    classifier: TurnClassifier | None = None,
) -> None:
    """Runs one session on *config* until the client leaves; *calls* is how it calls the pipeline's models, *vad*
    opens what server turn detection runs to find speech and wake words, and *classifier* tells semantic VAD whether
    a turn is finished; each None without speech detection."""
    refusal = _refusal(config, calls)
    if refusal is not None:
        await refuse(channel, refusal)
        return
    stt = await _stt_capabilities(config, calls)
    if isinstance(stt, RealtimeError):
        await refuse(channel, stt)
        return
    await Session(channel, config, calls, stt, vad, classifier).run()


class Session:
    """One session: it handles the client's events, and is the `TurnEvents` its server VAD reports to."""

    def __init__(
        self,
        channel: Channel,
        config: AssistantPipelineConfig,
        calls: PipelineCalls,
        stt: SttCapabilities | None,
        vad: VadStreamFactory | None = None,
        classifier: TurnClassifier | None = None,
    ):
        self.channel = channel
        self.config = config
        # the pipeline's assistant settings
        self.assistant = config.assistant_config
        self._vad = vad
        self._classifier = classifier
        self.state = new_session(config, MAX_SESSION_S)
        self.conversation = Conversation()
        self.buffer = InputAudioBuffer()
        # the client playing the session's spoken answers
        self.playback = Playback()
        # what each spoken message said, by item id
        spoken: dict[str, Spoken] = {}
        self.items = ConversationItems(self.conversation, self.send, spoken, self._stop_speaking)
        self.responder = Responder(config, calls, self.state, self.conversation, self.playback, spoken, self.send)
        self.active: ActiveResponse | None = None
        # server VAD, while the session's turn detection is on
        self.turns: TurnDetector | None = None
        # how the turns reach the stt model, and their transcription; a session without one takes no audio
        self.input: TurnInput = RecordedInput(self.buffer)
        self.transcriptions: TranscriptionQueue | None = None
        if stt is not None and self.assistant.stt is not None:
            context = SttContext(calls, self.assistant.stt, stt, self.state)
            self.transcriptions = TranscriptionQueue(context, self.conversation, self.send)
            if context.streams:
                self.input = StreamedInput(context, self.buffer, lambda: self.turns, self.send)
        # the wake words a realtime session's server VAD turns start at; None for none
        wake = self.assistant.wake if self.state.type == "realtime" else None
        self.gate = TurnGate(self.assistant, wake, self.buffer, self.playback, self._answering)
        if wake is not None and self.assistant.duplex == "half":
            # the wake words are heard in the audio the session mutes while its answer plays
            self.buffer.keep_sent()
        # automatic responses waiting for their turn's transcript
        self._replies = BackgroundTasks()
        # a turn server VAD found, or an answer it cut off, still needs an answer: the next turn's answers it
        self._unanswered_turn = False
        # feeds still finishing after server VAD was turned off
        self._retired_feeds = BackgroundTasks()
        self._sending = asyncio.Lock()
        self._closed = False
        self._handlers: dict[str, Handler] = {
            "session.update": self._session_update,
            "conversation.item.create": self.items.create,
            "conversation.item.retrieve": self.items.retrieve,
            "conversation.item.delete": self.items.delete,
            "conversation.item.truncate": self.items.truncate,
            "response.create": self._response_create,
            "response.cancel": self._response_cancel,
            "input_audio_buffer.append": self._audio_append,
            "input_audio_buffer.commit": self._audio_commit,
            "input_audio_buffer.clear": self._audio_clear,
        }

    # the session's life

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
            expiry.cancel()
            await self._wind_down()
        logger.info("Realtime session %s closed", self.state.id)

    async def _wind_down(self) -> None:
        """Stops everything the session runs, once its client left."""
        # a closed session waits in reference cycles for the cyclic GC; its audio goes now
        self.buffer.clear()
        self.input.drop()
        self._replies.cancel()
        self._retired_feeds.cancel()
        if self.transcriptions is not None:
            # a cancelled streamed transcription closes its stream as it ends
            self.transcriptions.cancel()
            await self.transcriptions.wait(_STREAM_CLOSE_S)
        await self.input.wait_closed(_STREAM_CLOSE_S)
        if self.turns is not None:
            await self.turns.close()
        if self.active is not None:
            await self.active.abandon()

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
        if (error := check_update(self.state, update, self.config, self.buffer.seconds > 0)) is not None:
            return error
        if apply_update(self.state, update):
            await self._detect_turns()
        if ignored := sorted(update.model_extra or {}):
            logger.debug("Realtime session %s: ignoring session fields not used yet: %s", self.state.id, ignored)
        await self.send(session_event("session.updated", self.state))
        return None

    # responses

    async def _response_create(self, event: ResponseCreate) -> RealtimeError | None:
        if self._answering():
            return RealtimeError(
                message="A response is in progress; wait for its response.done or cancel it.",
                code="conversation_already_has_active_response",
            )
        pending = self.transcriptions.last if self.transcriptions is not None else None
        active = self.responder.start(event.response, pending)
        if isinstance(active, RealtimeError):
            return active
        self.active = active
        active.task.add_done_callback(lambda _: self.gate.answered(active.response, active.in_conversation))
        if active.in_conversation:
            self._unanswered_turn = False
        return None

    async def _response_cancel(self, event: ResponseCancel) -> RealtimeError | None:
        active = self.active
        if active is None or active.done() or event.response_id not in (None, active.response.id):
            return RealtimeError(message="No response to cancel is in progress.", code="response_cancel_not_active")
        active.cancel()
        # a client that cancels stops playing
        self.playback.stop()
        return None

    def _answering(self) -> bool:
        """Whether a response is in progress."""
        return self.active is not None and not self.active.done()

    async def _stop_speaking(self, item_id: str) -> None:
        """Stops the response in progress if it speaks message *item_id*: the client stopped playing it."""
        active = self.active
        if active is not None and not active.done() and active.has_item(item_id):
            active.cancel()
            self.playback.stop()
            await asyncio.wait([active.task])

    # the input audio buffer

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
            self.turns.on_audio()
        self.input.on_audio()
        return None

    async def _audio_commit(self, event: AudioCommit) -> RealtimeError | None:
        if self.buffer.seconds < MIN_SECONDS:
            return RealtimeError(
                message=f"The input audio buffer holds {self.buffer.seconds * 1000:.0f} ms; a commit needs at least"
                f" {MIN_SECONDS * 1000:.0f} ms.",
                code="input_audio_buffer_commit_empty",
            )
        # the turn server VAD has open, or the buffer, from its start
        audio = self.input.take(self.buffer.start_ms, self.buffer.end_ms, None)
        turn = self.turns.reset() if self.turns is not None else None
        if turn is not None:
            # the client ended the turn server VAD had open
            end = round(self.buffer.end_ms)
            await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=end, item_id=turn))
        await self._commit(audio, turn)
        return None

    async def _commit(self, audio: TurnAudio, turn: str | None = None) -> tuple[UserMessage, asyncio.Task]:
        """Adds a turn of the client's audio, as the user message *turn* names, and queues its transcription; the
        task that transcribes it."""
        assert self.transcriptions is not None
        item = UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])
        if turn is not None:
            item.id = turn
        before = self.conversation.insert(item)
        await self.send(server_event("input_audio_buffer.committed", previous_item_id=before, item_id=item.id))
        await self.send(server_event("conversation.item.added", previous_item_id=before, item=item))
        return item, self.transcriptions.add(item, audio)

    async def _audio_clear(self, event: AudioClear) -> RealtimeError | None:
        self.buffer.clear()
        self.input.drop()
        if self.turns is not None:
            self.turns.reset()
        await self.send(server_event("input_audio_buffer.cleared"))
        return None

    # server VAD

    async def _detect_turns(self) -> None:
        """Starts, retunes or stops server VAD, as the session's turn detection says."""
        settings = self.state.audio.input.turn_detection
        if not isinstance(settings, VoiceTurnDetection):
            if self.turns is not None:
                # not cancelled: a feed in flight stops at its next step, and a commit it began completes
                self.turns.reset()
                if (task := self.turns.task) is not None:
                    self._retired_feeds.add(task)
                self.turns = None
                # the open turn's stream; the buffer streams from its start with the next append
                self.input.drop()
            return
        end = turn_end(settings, self._classifier)
        if self.turns is not None:
            self.turns.end = end
            return
        # the buffer's stream, which the client was to commit
        self.input.drop()
        self.turns = TurnDetector(
            self.buffer,
            end,
            self._open_vad,
            self,
            self.gate,
            # a streamed turn costs its model no more as it grows
            max_turn_ms=None if self.input.streams else MAX_TURN_MS,
        )

    def _open_vad(self, rate: int, start_ms: float, wake_words: bool) -> VadStream:
        if self._vad is None:
            raise RuntimeError(f"pipeline {self.config.name!r} has no speech detection")
        return self._vad(rate, start_ms, wake_words)

    def _interrupts(self) -> bool:
        """Whether a turn server VAD finds cuts the answer off."""
        settings = self.state.audio.input.turn_detection
        return isinstance(settings, VoiceTurnDetection) and settings.interrupt_response

    # TurnEvents: what server VAD tells the session

    def vad_progressed(self) -> None:
        self.input.vad_progressed()

    async def turn_started(self, turn: str, audio_start_ms: int, woken: bool) -> None:
        """A turn server VAD found starts, where its wake word ended when *woken*."""
        self.gate.turn_started()
        await self.send(server_event("input_audio_buffer.speech_started", audio_start_ms=audio_start_ms, item_id=turn))
        self.input.turn_started(turn, audio_start_ms, woken)
        # a client stops playing at speech_started
        self.playback.stop()
        if self._interrupts():
            active = self.active
            if active is not None and not active.done() and active.in_conversation:
                active.cancel("turn_detected")
                # the answer cut off is owed, even if this turn says nothing; one a wake word stopped is not
                self._unanswered_turn = not woken

    async def turn_ended(self, turn: str, audio_start_ms: int, audio_end_ms: int, spoken: bool) -> None:
        """Commits the turn server VAD found; one without speech, after a wake word nobody followed, is committed
        untranscribed and gets no answer."""
        # the turn's audio leaves the buffer before any await, so a commit or clear from the client can't take it too
        audio: TurnAudio
        if spoken:
            audio = self.input.take(audio_start_ms, audio_end_ms, turn)
        else:
            self.input.drop()
            self.buffer.drop_before(audio_end_ms)
            audio = SilentTurn((audio_end_ms - audio_start_ms) / 1000)
        await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=audio_end_ms, item_id=turn))
        settings = self.state.audio.input.turn_detection
        answers = (
            isinstance(settings, VoiceTurnDetection) and settings.create_response and self.state.type == "realtime"
        )
        # a turn that ends while the model answers gets no answer of its own; one it cut off doesn't count
        responding = self.active is not None and not self.active.done() and not self.active.cancelled
        item, transcribed = await self._commit(audio, turn)
        if answers and not responding and spoken:
            self._replies.spawn(self._reply(item, transcribed))

    async def detection_failed(self) -> None:
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

    async def turn_end_failed(self) -> None:
        await self.send(
            error_event(
                RealtimeError(
                    message=f"Pipeline {self.config.name!r} couldn't tell whether the turn ended: its turn classifier"
                    " failed. The turn ends after the longest wait its eagerness allows, unless speech follows.",
                    type="server_error",
                    code="turn_classifier_failed",
                )
            )
        )

    async def _reply(self, item: UserMessage, transcribed: asyncio.Task) -> None:
        """Answers the turn *item* once it is transcribed, unless it said nothing and no turn before it waits for an
        answer, or another answer began."""
        await asyncio.wait([transcribed])
        part = item.content[0]
        said = isinstance(part, InputAudio) and bool((part.transcript or "").strip())
        if self.conversation.get(item.id) is None or self._answering():
            return
        if self._interrupts() and self.turns is not None and self.turns.turn is not None:
            # the user is talking again, which would cut the answer off: the next turn's answer covers this one
            self._unanswered_turn = self._unanswered_turn or said
            return
        if not said and not self._unanswered_turn:
            return
        error = await self._response_create(ResponseCreate(type="response.create"))
        if error is not None:
            await self.send(error_event(error))
