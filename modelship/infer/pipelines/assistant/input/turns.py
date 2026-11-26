"""Server turn detection for a realtime session. `TurnDetector` feeds the client's audio, as it arrives, to a
`VadStream`, and finds the session's turns on its timeline (the ms of audio the client has sent since the session
opened).

A turn starts at the first speech heard, less the prefix padding. Once its speech pauses, its `TurnEnd` says where
it ends: after `silence_duration_ms` of silence (server_vad), or at once if a classifier hears it finished and else
when the user speaks again or after the wait its eagerness allows (semantic_vad). With wake words a turn also starts
where a wake word ends, and other speech starts one only inside the window the session's `TurnRules` give. The
detector is a state machine: `Idle` between turns, `Speaking` in a turn, `Paused` in a turn whose speech paused and
may go on, and `AwaitingSpeech` in a turn a wake word opened until speech past the wake word is heard."""

import asyncio
import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.infer.pipelines.assistant.input.turn_end import TurnClassifierError, TurnEnd
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import item_id
from modelship.utils.vad import SpeechSegment

logger = get_logger("realtime")

# while feeding fails, it is tried again with this much new audio
RETRY_STEP_MS = 300
# a turn a batch speech-to-text model transcribes, run this long without a pause, ends here
MAX_TURN_MS = 30_000
# the audio before the end fed that holds speech too short to open a segment yet
_LEAD_MS = 1000
# speech that ends this soon after a wake word is the wake word's own
_WAKE_TAIL_MS = 250


class VadStream(Protocol):
    """What server VAD runs over one stream of the session's audio: the speech in it, and the wake words when it
    was opened to hear them (`modelship.utils.voice.VoiceDetector`)."""

    rate: int
    # where a wake word ended in the last call's audio, each time one was heard
    wake_ends_ms: list[float]

    def detect(
        self, pcm16: bytes, threshold: float, silence_duration_ms: int, sent_pcm16: bytes | None = None
    ) -> list[SpeechSegment]:
        """The speech segments that closed in *pcm16*, then the one still open. The wake words are heard in
        *sent_pcm16*, the same stretch as the client sent it, when given."""
        ...


# opens a VadStream on 16-bit PCM at a rate, its timeline starting at a ms, hearing wake words too when told so
VadStreamFactory = Callable[[int, float, bool], VadStream]


@dataclass(frozen=True)
class Window:
    """Where speech may start a turn: from *start_ms* until *end_ms* on the session's timeline."""

    start_ms: float
    end_ms: float


# speech starts a turn anywhere
ANYWHERE = Window(-math.inf, math.inf)
# speech starts no turn; only a wake word does
SHUT = Window(math.inf, math.inf)


class TurnRules(Protocol):
    """When speech may start a turn now (`TurnGate` in `gate.py`)."""

    # with wake words: how long a turn a wake word opened waits for speech; None without wake words
    listen_ms: float | None

    def min_speech_ms(self) -> int:
        """How long speech must last before it starts a turn now."""
        ...

    def window(self) -> Window:
        """Where speech may start a turn now."""
        ...


class _NoRules:
    """Speech of any length starts a turn anywhere, and there are no wake words."""

    listen_ms: float | None = None

    def min_speech_ms(self) -> int:
        return 0

    def window(self) -> Window:
        return ANYWHERE


NO_RULES: TurnRules = _NoRules()


class TurnEvents(Protocol):
    """What the detector tells the session."""

    async def turn_started(self, turn: str, audio_start_ms: int, woken: bool) -> None:
        """Turn *turn* starts at *audio_start_ms*, where its wake word ended when *woken*."""
        ...

    async def turn_ended(self, turn: str, audio_start_ms: int, audio_end_ms: int, spoken: bool) -> None:
        """Turn *turn* ends; the session takes its audio out of the buffer before it awaits anything. A turn a wake
        word opened that no speech followed is not *spoken*."""
        ...

    async def detection_failed(self) -> None:
        """Feeding started failing; the buffer keeps all its audio while it fails."""
        ...

    async def turn_end_failed(self) -> None:
        """The classifier couldn't tell whether a turn ended; it ends after the longest wait unless speech follows."""
        ...

    def vad_progressed(self) -> None:
        """A feed is done, and `vad_through_ms` moved."""
        ...


@dataclass(frozen=True)
class OpenTurn:
    """A turn in progress: its item id, and where it starts on the timeline."""

    item_id: str
    start_ms: float


@dataclass(frozen=True)
class Idle:
    """No turn is open."""


@dataclass
class Speaking:
    """In *turn*, whose speech was last heard at *speech_ms*."""

    turn: OpenTurn
    speech_ms: float


@dataclass(frozen=True)
class Paused:
    """In *turn*, whose speech stopped at *speech_ms* and may go on: without speech it ends at *gives_up_ms*."""

    turn: OpenTurn
    speech_ms: float
    gives_up_ms: float


@dataclass(frozen=True)
class AwaitingSpeech:
    """In *turn*, which a wake word opened: speech that ends by *own_until_ms* is the wake word's own, and without
    speech past it the turn ends, unspoken, at *gives_up_ms*."""

    turn: OpenTurn
    own_until_ms: float
    gives_up_ms: float


TurnState = Idle | Speaking | Paused | AwaitingSpeech


class _StaleFeedError(Exception):
    """The buffer was committed or cleared during a feed."""


class TurnDetector:
    """Finds the turns in *buffer*, ending them as *end* says, with the VAD streams *open_vad* opens, and tells
    *events*. *rules* say when speech may start a turn. A turn without a pause is cut at *max_turn_ms*, or never with
    None."""

    def __init__(
        self,
        buffer: InputAudioBuffer,
        end: TurnEnd,
        open_vad: VadStreamFactory,
        events: TurnEvents,
        rules: TurnRules = NO_RULES,
        max_turn_ms: int | None = MAX_TURN_MS,
    ):
        self.buffer = buffer
        self.end = end
        self._open_vad = open_vad
        self._events = events
        self._rules = rules
        self._max_turn_ms = math.inf if max_turn_ms is None else max_turn_ms
        self._state: TurnState = Idle()
        # the VAD stream the buffer's audio is fed to; None until the next feed opens one
        self._vad: VadStream | None = None
        # the end of the audio fed so far
        self._fed_ms = buffer.end_ms
        # the end of the audio whose turns have all started and ended: a turn open past it ends after it
        self.vad_through_ms = buffer.end_ms
        # where the last turn ended; no turn starts before it
        self._floor_ms = buffer.start_ms
        self._failing = False
        # the end of the audio the last failed feed was given
        self._tried_ms = self._fed_ms
        # bumped by `reset`, so a feed in flight over audio that is gone changes nothing
        self._epoch = 0
        self._feed_epoch = 0
        self._task: asyncio.Task | None = None

    @property
    def turn(self) -> OpenTurn | None:
        """The turn in progress, if any."""
        return None if isinstance(self._state, Idle) else self._state.turn

    @property
    def awaiting_speech(self) -> bool:
        """Whether the turn in progress is one a wake word opened that has heard no speech past it yet."""
        return isinstance(self._state, AwaitingSpeech)

    @property
    def task(self) -> asyncio.Task | None:
        """The feed in flight, if any."""
        return self._task if self._task is not None and not self._task.done() else None

    def on_audio(self) -> None:
        """The client appended audio: feeds it, once the feed in flight is done."""
        if self._due() and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self._run())

    def reset(self) -> str | None:
        """Forgets the turn in progress, as the buffer was committed or cleared; its item id, or None."""
        found = self.turn.item_id if self.turn is not None else None
        self._state = Idle()
        self._epoch += 1
        self._vad = None
        self._fed_ms = self._tried_ms = self._floor_ms = self.vad_through_ms = self.buffer.end_ms
        return found

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _due(self) -> bool:
        if self._failing:
            return self.buffer.end_ms - self._tried_ms >= RETRY_STEP_MS
        return self.buffer.end_ms > self._fed_ms

    async def _run(self) -> None:
        try:
            while self._due():
                await self._feed_once()
        except _StaleFeedError:
            pass
        except Exception:
            logger.exception("Turn detection failed")

    async def _feed_once(self) -> None:
        # one feed runs on the turn end it started with
        turn_end, buffer = self.end, self.buffer
        self._feed_epoch = epoch = self._epoch
        min_speech = self._rules.min_speech_ms()
        start, end = max(self._fed_ms, buffer.start_ms), buffer.end_ms
        vad = self._vad_from(start)
        found = await self._detect(vad, start, end, turn_end)
        if epoch != self._epoch:
            return
        if found is None:
            await self._feed_failed(end)
            return
        self._failing = False
        self._fed_ms = self._tried_ms = end
        listen_ms = self._rules.listen_ms
        if vad.wake_ends_ms and isinstance(self._state, Idle) and listen_ms is not None:
            await self._woken(vad.wake_ends_ms[0], listen_ms)
        await self._apply(found, end, turn_end, min_speech)
        if epoch == self._epoch:
            self.vad_through_ms = end
            self._events.vad_progressed()
        if isinstance(self._state, Idle):
            self._trim_buffer(found, end, turn_end)

    def _vad_from(self, start: float) -> VadStream:
        """The VAD stream to feed the audio from *start* to: a new one after a failure, a reset or a new rate."""
        rate = self.buffer.rate
        assert rate is not None
        vad = self._vad
        if vad is None or vad.rate != rate or start != self._fed_ms:
            vad = self._vad = self._open_vad(rate, start, self._rules.listen_ms is not None)
        return vad

    async def _detect(self, vad: VadStream, start: float, end: float, turn_end: TurnEnd) -> list[SpeechSegment] | None:
        """Runs *vad* over the buffer's audio from *start* to *end* on a thread; its segments, or None if it failed."""
        buffer = self.buffer
        # wake words are heard in the audio as the client sent it, where the buffer holds silence in its place
        sent = buffer.sent_pcm(start, end) if buffer.keeps_sent else None
        try:
            return await asyncio.to_thread(
                vad.detect, buffer.pcm(start, end), turn_end.threshold, turn_end.pause_ms, sent
            )
        except Exception:
            logger.exception("Detecting speech failed")
            return None

    async def _feed_failed(self, end: float) -> None:
        # the next feed opens a new VAD stream on the audio this one was given
        self._vad = None
        self._tried_ms = end
        if not self._failing:
            self._failing = True
            await self._events.detection_failed()

    def _trim_buffer(self, found: list[SpeechSegment], end: float, turn_end: TurnEnd) -> None:
        """Between turns, drops the audio no turn can start in: it keeps the open segment from where it may start a
        turn, and what a turn's prefix padding needs."""
        low = self._rules.window().start_ms
        opened = [max(segment.start_ms, low) for segment in found if segment.end_ms is None]
        held = min([end - _LEAD_MS, *opened])
        self.buffer.drop_before(held - turn_end.prefix_padding_ms)

    async def _woken(self, at_ms: float, listen_ms: float) -> None:
        """Opens a turn where a wake word ends, at *at_ms*, which waits *listen_ms* for speech."""
        if at_ms <= self._floor_ms:
            # in a turn that ended already
            return
        await self._start(max(at_ms, self.buffer.start_ms), listen_ms)

    def _opening(self, segment: SpeechSegment, turn_end: TurnEnd, min_speech: int) -> float | None:
        """Where *segment* starts a turn now, or None: speech that starts outside the window or lasts shorter than
        *min_speech* inside it starts none."""
        window = self._rules.window()
        first = max(segment.start_ms, window.start_ms)
        speech_end = segment.speech_end_ms
        if speech_end <= first or speech_end - first < min_speech or first > window.end_ms:
            return None
        padded = segment.start_ms - turn_end.prefix_padding_ms
        return max(self._floor_ms, self.buffer.start_ms, padded, window.start_ms)

    def _check_current(self) -> None:
        """Stops the feed when the buffer was committed or cleared while it waited on the session."""
        if self._epoch != self._feed_epoch:
            raise _StaleFeedError

    async def _apply(self, segments: list[SpeechSegment], end: float, turn_end: TurnEnd, min_speech: int) -> None:
        """Moves the state machine through one feed's segments, then through the time the feed covered."""
        heard = False
        for segment in segments:
            if segment.end_ms is not None and segment.end_ms <= self._floor_ms:
                # a turn that ended already holds it
                continue
            state = self._state
            if isinstance(state, AwaitingSpeech):
                if segment.speech_end_ms <= state.own_until_ms:
                    # the wake word's own speech
                    continue
                self._state = Speaking(state.turn, state.turn.start_ms)
            elif isinstance(state, Paused):
                # the user goes on after a pause
                self._state = Speaking(state.turn, state.speech_ms)
            elif isinstance(state, Idle):
                opening = self._opening(segment, turn_end, min_speech)
                if opening is None:
                    if segment.end_ms is None:
                        # it may last long enough, or the window open, by the next feed
                        break
                    # it starts no turn, now or on a later feed
                    self._floor_ms = segment.end_ms
                    continue
                await self._start(opening)
            heard = True
            if not await self._speech(segment, turn_end):
                break
        await self._elapsed(end, heard, turn_end)

    async def _speech(self, segment: SpeechSegment, turn_end: TurnEnd) -> bool:
        """Speech in the turn in progress: an open segment moves where its speech was last heard, a closed one pauses
        it. Whether the feed's later segments may hold more speech."""
        state = self._state
        assert isinstance(state, Speaking)
        if segment.end_ms is None:
            state.speech_ms = segment.speech_end_ms
            return False
        state.speech_ms = segment.end_ms
        await self._paused(segment.end_ms, turn_end)
        return True

    async def _paused(self, speech_end_ms: float, turn_end: TurnEnd) -> None:
        """The turn in progress went quiet at *speech_end_ms*: it ends where *turn_end* says, or waits for more
        speech."""
        state = self._state
        assert isinstance(state, Speaking)
        try:
            at_ms = await turn_end.ends_at(self.buffer, state.turn.start_ms, speech_end_ms)
        except TurnClassifierError:
            logger.exception("Classifying the end of a turn failed")
            at_ms = None
            await self._events.turn_end_failed()
        self._check_current()
        if at_ms is not None:
            await self._close(speech_end_ms, at_ms)
        else:
            self._state = Paused(state.turn, speech_end_ms, turn_end.gives_up_ms(speech_end_ms))

    async def _elapsed(self, end: float, heard: bool, turn_end: TurnEnd) -> None:
        """Ends the turn in progress where the audio fed up to *end* says it ends without more speech: its pause
        lasted, the wake word's listening ran out, or it reached the longest a turn lasts."""
        fed_to = min(end, self.buffer.end_ms)
        state = self._state
        if isinstance(state, Speaking) and not heard and end - state.speech_ms >= turn_end.pause_ms:
            # the turn's speech ended before this feed's audio
            await self._paused(state.speech_ms, turn_end)
        state = self._state
        if isinstance(state, Paused) and fed_to >= state.gives_up_ms:
            # no speech followed the pause
            await self._close(state.speech_ms, state.gives_up_ms)
        state = self._state
        if isinstance(state, AwaitingSpeech) and fed_to >= state.gives_up_ms:
            # no speech followed the wake word
            await self._end(state.gives_up_ms)
        turn = self.turn
        if turn is not None and fed_to - turn.start_ms >= self._max_turn_ms:
            await self._end(turn.start_ms + self._max_turn_ms)

    async def _close(self, speech_end_ms: float, at_ms: float) -> None:
        """Ends the turn in progress at *at_ms*, its speech ending at *speech_end_ms*: in turns of the longest a turn
        lasts, when the speech ran longer."""
        turn = self.turn
        assert turn is not None
        longest = self._max_turn_ms
        while speech_end_ms > turn.start_ms + longest:
            cut = turn.start_ms + longest
            await self._end(cut)
            turn = await self._start(cut)
        await self._end(min(at_ms, turn.start_ms + longest))

    async def _start(self, at_ms: float, listen_ms: float | None = None) -> OpenTurn:
        """Opens a turn at *at_ms*: a wake word's when *listen_ms* says how long it waits for speech."""
        self._check_current()
        turn = OpenTurn(item_id(), at_ms)
        woken = listen_ms is not None
        if listen_ms is None:
            self._state = Speaking(turn, at_ms)
        else:
            self._state = AwaitingSpeech(turn, at_ms + _WAKE_TAIL_MS, at_ms + listen_ms)
        await self._events.turn_started(turn.item_id, round(at_ms), woken)
        # the session may have committed or cleared the buffer while it heard of the turn
        self._check_current()
        return turn

    async def _end(self, at_ms: float) -> None:
        self._check_current()
        state = self._state
        assert not isinstance(state, Idle)
        spoken = not isinstance(state, AwaitingSpeech)
        turn = state.turn
        # while the session ends the turn, it no longer waits for speech
        self._state = Speaking(turn, at_ms)
        await self._events.turn_ended(turn.item_id, round(turn.start_ms), round(at_ms), spoken)
        if self._epoch == self._feed_epoch:
            self._state = Idle()
            self._floor_ms = at_ms
