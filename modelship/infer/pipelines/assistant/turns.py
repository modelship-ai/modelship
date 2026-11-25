"""Server VAD for a realtime session: the client's audio is fed to a speech stream as it arrives, and each turn is
found on the session's timeline. A turn starts at the first speech heard, less the prefix padding, and ends once the
silence after it lasts `silence_duration_ms`."""

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable
from typing import Protocol

from modelship.infer.pipelines.assistant.audio import InputAudioBuffer
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import ServerVad, item_id
from modelship.utils.vad import SpeechSegment

logger = get_logger("realtime")

# while feeding fails, it is tried again with this much new audio
RETRY_STEP_MS = 300
# a turn a batch speech-to-text model transcribes, run this long without a pause, ends here
MAX_TURN_MS = 30_000
# the audio before the end fed that holds speech too short to open a segment yet
_LEAD_MS = 1000


class SpeechFinder(Protocol):
    """Finds the speech in one stream of the session's audio, as `SpeechStream` does."""

    rate: int

    def feed(self, pcm16: bytes, threshold: float, silence_duration_ms: int) -> list[SpeechSegment]: ...


# opens a finder on 16-bit PCM at a rate, its timeline starting at a ms
OpenFinder = Callable[[int, float], SpeechFinder]
Started = Callable[[str, int], Awaitable[None]]
Ended = Callable[[str, int, int], Awaitable[None]]
Failed = Callable[[], Awaitable[None]]
# how long speech must last, in ms, before it starts a turn now
MinSpeech = Callable[[], int]


class _StaleFeedError(Exception):
    """The buffer was committed or cleared during a feed."""


class TurnDetector:
    """Finds the turns in *buffer* with the speech finders *open_finder* opens, as *settings* say. *started* gets
    each turn's item id and where it starts; *ended* gets its item id, start and end, and takes its audio out of the
    buffer; *failed* hears of each stretch of feeds that fail, during which the buffer keeps all its
    audio. Speech shorter than *min_speech* says starts no turn. A turn without a pause is cut at *max_turn_ms*, or
    never with None."""

    def __init__(
        self,
        buffer: InputAudioBuffer,
        settings: ServerVad,
        open_finder: OpenFinder,
        started: Started,
        ended: Ended,
        failed: Failed,
        min_speech: MinSpeech = lambda: 0,
        max_turn_ms: int | None = MAX_TURN_MS,
    ):
        self.buffer = buffer
        self.settings = settings
        self._open_finder = open_finder
        self._started = started
        self._ended = ended
        self._failed = failed
        self._min_speech = min_speech
        self._max_turn_ms = math.inf if max_turn_ms is None else max_turn_ms
        # the finder the buffer's audio is fed to; None until the next feed opens one
        self._finder: SpeechFinder | None = None
        # the end of the audio fed so far
        self._fed_ms = buffer.end_ms
        # where the last turn ended; no turn starts before it
        self._floor_ms = buffer.start_ms
        # the turn in progress: its item id and start
        self.turn: tuple[str, float] | None = None
        # where speech was last heard in the turn in progress
        self._speech_ms = 0.0
        self._failing = False
        # the end of the audio the last failed feed was given
        self._tried_ms = self._fed_ms
        # bumped by `reset`, so a feed in flight over audio that is gone changes nothing
        self._epoch = 0
        self._feed_epoch = 0
        self._task: asyncio.Task | None = None

    def appended(self) -> None:
        """Feeds the new audio, once the feed in flight is done."""
        if self._due() and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self._run())

    def reset(self) -> str | None:
        """Forgets the turn in progress, as the buffer was committed or cleared; its item id, or None."""
        found = self.turn[0] if self.turn is not None else None
        self.turn = None
        self._epoch += 1
        self._finder = None
        self._fed_ms = self._tried_ms = self._floor_ms = self.buffer.end_ms
        return found

    @property
    def task(self) -> asyncio.Task | None:
        """The feed in flight, if any."""
        return self._task if self._task is not None and not self._task.done() else None

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
        # one feed runs on the settings it started with
        settings, buffer = self.settings, self.buffer
        self._feed_epoch = epoch = self._epoch
        rate = buffer.rate
        assert rate is not None
        min_speech = self._min_speech()
        start, end = max(self._fed_ms, buffer.start_ms), buffer.end_ms
        finder = self._finder
        if finder is None or finder.rate != rate or start != self._fed_ms:
            finder = self._finder = self._open_finder(rate, start)
        try:
            found = await asyncio.to_thread(
                finder.feed, buffer.pcm(start, end), settings.threshold, settings.silence_duration_ms
            )
        except Exception:
            logger.exception("Detecting speech failed")
            found = None
        if epoch != self._epoch:
            return
        if found is None:
            # the next feed opens a new finder on the audio this one was given
            self._finder = None
            self._tried_ms = end
            if not self._failing:
                self._failing = True
                await self._failed()
            return
        self._failing = False
        self._fed_ms = self._tried_ms = end
        await self._apply(found, end, settings, min_speech)
        if self.turn is None:
            # keep the open segment and what a turn's prefix padding needs
            held = min([end - _LEAD_MS, *(segment.start_ms for segment in found if segment.end_ms is None)])
            buffer.drop_before(held - settings.prefix_padding_ms)

    def _check_current(self) -> None:
        """Stops the feed when the buffer was committed or cleared while it waited on the session."""
        if self._epoch != self._feed_epoch:
            raise _StaleFeedError

    async def _apply(self, segments: list[SpeechSegment], end: float, settings: ServerVad, min_speech: int) -> None:
        heard = False
        for segment in segments:
            begin, stop, speech_end = segment.start_ms, segment.end_ms, segment.speech_end_ms
            if stop is not None and stop <= self._floor_ms:
                # a turn that ended already holds it
                continue
            if self.turn is None and speech_end - begin < min_speech:
                if stop is None:
                    # it may last long enough by the next feed
                    break
                # too short to start a turn, now or on a later feed
                self._floor_ms = stop
                continue
            heard = True
            if self.turn is None:
                await self._start(max(self._floor_ms, self.buffer.start_ms, begin - settings.prefix_padding_ms))
            if stop is None:
                self._speech_ms = speech_end
                break
            self._speech_ms = stop
            await self._close(stop, stop + settings.silence_duration_ms)
        # the turn's speech ended before this feed's audio
        ended_before = self.turn is not None and not heard and end - self._speech_ms >= settings.silence_duration_ms
        if ended_before:
            await self._close(self._speech_ms, self._speech_ms + settings.silence_duration_ms)
        longest = self._max_turn_ms
        if self.turn is not None and min(end, self.buffer.end_ms) - self.turn[1] >= longest:
            await self._end(self.turn[1] + longest)

    async def _close(self, speech_end_ms: float, at_ms: float) -> None:
        """Ends the turn in progress at *at_ms*, its speech ending at *speech_end_ms*: in turns of the longest a turn
        lasts, when the speech ran longer."""
        assert self.turn is not None
        longest = self._max_turn_ms
        while speech_end_ms > self.turn[1] + longest:
            cut = self.turn[1] + longest
            await self._end(cut)
            await self._start(cut)
        await self._end(min(at_ms, self.turn[1] + longest))

    async def _start(self, at_ms: float) -> None:
        self._check_current()
        self.turn = (item_id(), at_ms)
        self._speech_ms = at_ms
        await self._started(self.turn[0], round(at_ms))

    async def _end(self, at_ms: float) -> None:
        self._check_current()
        assert self.turn is not None
        turn_id, start = self.turn
        # the session takes the turn's audio before it awaits anything
        await self._ended(turn_id, round(start), round(at_ms))
        if self._epoch == self._feed_epoch:
            self.turn = None
            self._floor_ms = at_ms
