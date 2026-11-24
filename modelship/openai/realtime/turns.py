"""Server VAD for a realtime session: the client's audio is scanned for speech as it arrives, by the pipeline's
audio app, and each turn is found on the session's timeline. A turn starts at the first speech heard, less the
prefix padding, and ends once the silence after it lasts `silence_duration_ms`."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from modelship.logging import get_logger
from modelship.openai.protocol.realtime import ServerVad, item_id
from modelship.openai.realtime.audio import InputAudioBuffer

logger = get_logger("realtime")

# a scan comes due with this much new audio
SCAN_STEP_MS = 300
# the audio before the new audio a scan also covers, for the detector to settle
WARMUP_MS = 1000
# the most audio one scan covers
MAX_SCAN_MS = 30_000
# a turn that runs this long without a pause ends here
MAX_TURN_MS = 30_000

# scans 16-bit PCM at a rate with a threshold and a silence duration: the speech in it as (start, end, speech end)
# ms from its start, the end None while speech may go on and the speech end where the speech heard in it ends; None
# while nothing serves the scan
Scan = Callable[[bytes, int, float, int], Awaitable[list[tuple[int, int | None, int]] | None]]
Started = Callable[[str, int], Awaitable[None]]
# True once it took the turn's audio out of the buffer; False leaves the turn open, to end on a later scan
Ended = Callable[[str, int, int], Awaitable[bool]]
Failed = Callable[[], Awaitable[None]]
# how long speech must last, in ms, before it starts a turn now
MinSpeech = Callable[[], int]


class _StaleScanError(Exception):
    """The buffer was committed or cleared during a scan."""


class TurnDetector:
    """Finds the turns in *buffer* with *scan*, as *settings* say. *started* gets each turn's item id and where it
    starts; *ended* gets its item id, start and end, and takes its audio out of the buffer or refuses it; *failed*
    hears of each stretch of scans that fail, during which the buffer keeps all its audio. Speech shorter than
    *min_speech* says starts no turn."""

    def __init__(
        self,
        buffer: InputAudioBuffer,
        settings: ServerVad,
        scan: Scan,
        started: Started,
        ended: Ended,
        failed: Failed,
        min_speech: MinSpeech = lambda: 0,
    ):
        self.buffer = buffer
        self.settings = settings
        self._scan = scan
        self._started = started
        self._ended = ended
        self._failed = failed
        self._min_speech = min_speech
        # the end of the audio scanned so far
        self._scanned_ms = buffer.end_ms
        # where the last turn ended; no turn starts before it
        self._floor_ms = buffer.start_ms
        # the turn in progress: its item id and start
        self.turn: tuple[str, float] | None = None
        # where speech was last heard in the turn in progress
        self._speech_ms = 0.0
        self._failing = False
        # the end of the audio the last failed scan was sent
        self._tried_ms = self._scanned_ms
        # bumped by `reset`, so a scan in flight over audio that is gone changes nothing
        self._epoch = 0
        self._scan_epoch = 0
        self._task: asyncio.Task | None = None

    def appended(self) -> None:
        """Scans the new audio once there is enough of it."""
        if self._due() and (self._task is None or self._task.done()):
            self._task = asyncio.ensure_future(self._run())

    def reset(self) -> str | None:
        """Forgets the turn in progress, as the buffer was committed or cleared; its item id, or None."""
        found = self.turn[0] if self.turn is not None else None
        self.turn = None
        self._epoch += 1
        self._scanned_ms = self._tried_ms = self._floor_ms = self.buffer.end_ms
        return found

    @property
    def task(self) -> asyncio.Task | None:
        """The scan in flight, if any."""
        return self._task if self._task is not None and not self._task.done() else None

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _due(self) -> bool:
        # while scans fail, a new one is tried with each step of new audio
        since = self._tried_ms if self._failing else self._scanned_ms
        return self.buffer.end_ms - since >= SCAN_STEP_MS

    async def _run(self) -> None:
        try:
            while self._due():
                await self._scan_once()
        except _StaleScanError:
            pass
        except Exception:
            logger.exception("Turn detection failed")

    async def _scan_once(self) -> None:
        # one scan runs on the settings it started with
        settings, buffer = self.settings, self.buffer
        self._scan_epoch = epoch = self._epoch
        rate = buffer.rate
        assert rate is not None
        min_speech = self._min_speech()
        # the audio a scan looks back over: a whole silence, or speech too short to start a turn yet
        context = max(settings.silence_duration_ms, min_speech) + WARMUP_MS
        end = buffer.end_ms
        # after failed scans, a short one finds out whether the audio app answers again
        probing = self._failing
        start = max(buffer.start_ms, end - context if probing else min(self._scanned_ms - WARMUP_MS, end - context))
        end = min(end, start + MAX_SCAN_MS)
        try:
            found = await self._scan(buffer.pcm(start, end), rate, settings.threshold, settings.silence_duration_ms)
        except Exception:
            logger.exception("Scanning for speech failed")
            found = None
        if epoch != self._epoch:
            return
        if found is None:
            self._tried_ms = end
            if not self._failing:
                self._failing = True
                await self._failed()
            return
        self._failing = False
        if probing:
            # the next scans take the audio the failed ones missed, from where the last good scan stopped
            return
        self._scanned_ms = self._tried_ms = end
        segments = [
            (start + begin, None if stop is None else start + stop, start + speech_end)
            for begin, stop, speech_end in found
        ]
        await self._apply(segments, end, settings, min_speech)
        if self.turn is None:
            # keep what the next scan and a turn's prefix padding need
            buffer.drop_before(end - context - settings.prefix_padding_ms)

    def _check_current(self) -> None:
        """Stops the scan when the buffer was committed or cleared while it waited on the session."""
        if self._epoch != self._scan_epoch:
            raise _StaleScanError

    async def _apply(
        self, segments: list[tuple[float, float | None, float]], end: float, settings: ServerVad, min_speech: int
    ) -> None:
        heard = False
        for begin, stop, speech_end in segments:
            if stop is not None and stop <= self._floor_ms:
                # a turn that ended already holds it
                continue
            if self.turn is None and speech_end - begin < min_speech:
                if stop is None:
                    # it may last long enough by the next scan
                    break
                # too short to start a turn, now or on a later scan
                self._floor_ms = stop
                continue
            heard = True
            if self.turn is None:
                await self._start(max(self._floor_ms, self.buffer.start_ms, begin - settings.prefix_padding_ms))
            if stop is None:
                self._speech_ms = end
                break
            self._speech_ms = stop
            if not await self._close(stop, stop + settings.silence_duration_ms):
                return
        # the turn's speech ended before this scan's audio
        ended_before = self.turn is not None and not heard and end - self._speech_ms >= settings.silence_duration_ms
        if ended_before and not await self._close(self._speech_ms, self._speech_ms + settings.silence_duration_ms):
            return
        if self.turn is not None and min(end, self.buffer.end_ms) - self.turn[1] >= MAX_TURN_MS:
            await self._end(self.turn[1] + MAX_TURN_MS)

    async def _close(self, speech_end_ms: float, at_ms: float) -> bool:
        """Ends the turn in progress at *at_ms*, its speech ending at *speech_end_ms*: in 30 s turns, when the
        speech ran longer. False when a turn was refused and stays open."""
        assert self.turn is not None
        while speech_end_ms > self.turn[1] + MAX_TURN_MS:
            cut = self.turn[1] + MAX_TURN_MS
            if not await self._end(cut):
                return False
            await self._start(cut)
        return await self._end(min(at_ms, self.turn[1] + MAX_TURN_MS))

    async def _start(self, at_ms: float) -> None:
        self._check_current()
        self.turn = (item_id(), at_ms)
        self._speech_ms = at_ms
        await self._started(self.turn[0], round(at_ms))

    async def _end(self, at_ms: float) -> bool:
        self._check_current()
        assert self.turn is not None
        turn_id, start = self.turn
        # the session takes the turn's audio before it awaits anything
        if not await self._ended(turn_id, round(start), round(at_ms)):
            return False
        if self._epoch == self._scan_epoch:
            self.turn = None
            self._floor_ms = at_ms
        return True
