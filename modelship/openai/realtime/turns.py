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

# scans 16-bit PCM at a rate with a threshold and a silence duration: the speech in it as (start, end) ms from
# its start, the end None while speech goes on; None while nothing serves the scan
Scan = Callable[[bytes, int, float, int], Awaitable[list[tuple[int, int | None]] | None]]
Started = Callable[[str, int], Awaitable[None]]
# True once it took the turn's audio out of the buffer; False leaves the turn open, to end on a later scan
Ended = Callable[[str, int, int], Awaitable[bool]]
Failed = Callable[[], Awaitable[None]]


class TurnDetector:
    """Finds the turns in *buffer* with *scan*, as *settings* say. *started* gets each turn's item id and where it
    starts; *ended* gets its item id, start and end, and takes its audio out of the buffer or refuses it; *failed*
    hears of each stretch of scans that fail, during which the buffer keeps all its audio."""

    def __init__(
        self,
        buffer: InputAudioBuffer,
        settings: ServerVad,
        scan: Scan,
        started: Started,
        ended: Ended,
        failed: Failed,
    ):
        self.buffer = buffer
        self.settings = settings
        self._scan = scan
        self._started = started
        self._ended = ended
        self._failed = failed
        # the end of the audio scanned so far
        self._scanned_ms = buffer.end_ms
        # where the last turn ended; no turn starts before it
        self._floor_ms = buffer.start_ms
        # the turn in progress: its item id and start
        self.turn: tuple[str, float] | None = None
        # where speech was last heard in the turn in progress
        self._speech_ms = 0.0
        self._failing = False
        # bumped by `reset`, so a scan in flight over audio that is gone changes nothing
        self._epoch = 0
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
        self._scanned_ms = self._floor_ms = self.buffer.end_ms
        return found

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _due(self) -> bool:
        return self.buffer.end_ms - self._scanned_ms >= SCAN_STEP_MS

    async def _run(self) -> None:
        try:
            while self._due():
                await self._scan_once()
        except Exception:
            logger.exception("Turn detection failed")

    async def _scan_once(self) -> None:
        # one scan runs on the settings it started with
        settings, buffer, epoch = self.settings, self.buffer, self._epoch
        rate = buffer.rate
        assert rate is not None
        context = settings.silence_duration_ms + WARMUP_MS
        end = buffer.end_ms
        start = max(buffer.start_ms, min(self._scanned_ms - WARMUP_MS, end - context))
        end = min(end, start + MAX_SCAN_MS)
        try:
            found = await self._scan(buffer.pcm(start, end), rate, settings.threshold, settings.silence_duration_ms)
        except Exception:
            logger.exception("Scanning for speech failed")
            found = None
        if epoch != self._epoch:
            return
        self._scanned_ms = end
        if found is None:
            if not self._failing:
                self._failing = True
                await self._failed()
            return
        self._failing = False
        segments = [(start + begin, None if stop is None else start + stop) for begin, stop in found]
        await self._apply(segments, end, settings)
        if self.turn is None:
            # keep what the next scan and a turn's prefix padding need
            buffer.drop_before(end - context - settings.prefix_padding_ms)

    async def _apply(self, segments: list[tuple[float, float | None]], end: float, settings: ServerVad) -> None:
        heard = False
        for begin, stop in segments:
            if stop is not None and stop <= self._floor_ms:
                # a turn that ended already holds it
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
        self.turn = (item_id(), at_ms)
        self._speech_ms = at_ms
        await self._started(self.turn[0], round(at_ms))

    async def _end(self, at_ms: float) -> bool:
        assert self.turn is not None
        turn_id, start = self.turn
        if not await self._ended(turn_id, round(start), round(at_ms)):
            return False
        self.turn = None
        self._floor_ms = at_ms
        return True
