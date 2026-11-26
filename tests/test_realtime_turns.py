"""Server VAD's turn finding: a session's audio fed as it arrives to a fake speech finder that hears speech wherever
the audio is loud, and the turns it finds on the session's timeline."""

import asyncio
import math
import threading

import numpy as np
import pytest

from modelship.infer.pipelines.assistant.audio import InputAudioBuffer
from modelship.infer.pipelines.assistant.turns import MAX_TURN_MS, TurnDetector
from modelship.openai.protocol.realtime import ServerVad
from modelship.utils.vad import SpeechSegment

RATE = 24000


def _tone(seconds: float) -> bytes:
    count = int(seconds * RATE)
    return (np.sin(2 * np.pi * 440 * np.arange(count) / RATE) * 8000).astype("<i2").tobytes()


def _quiet(seconds: float) -> bytes:
    return bytes(int(seconds * RATE) * 2)


def wake_word(seconds: float = 0.6) -> bytes:
    """What the fake finders hear as a wake word: a tone louder than speech."""
    count = int(seconds * RATE)
    return (np.sin(2 * np.pi * 440 * np.arange(count) / RATE) * 25000).astype("<i2").tobytes()


class FakeVad:
    """Opens speech finders like the pipeline's: speech wherever a 10 ms frame is loud, a segment opening at its
    first loud frame and closing once *silence_duration_ms* of quiet follows it; a finder opened to hear wake words
    hears one end where a stretch of very loud frames ends. Records each finder's rate and start, and the ms each
    feed is given; *failing* makes feeds fail, and *gate*, set, makes each feed wait for it."""

    def __init__(self) -> None:
        self.failing = False
        self.opened: list[tuple[int, float]] = []
        self.feeds: list[float] = []
        self.gate: threading.Event | None = None

    def open(self, rate: int, start_ms: float, wake: bool = False) -> "_FakeFinder":
        self.opened.append((rate, start_ms))
        return _FakeFinder(self, rate, start_ms, wake)


class _FakeFinder:
    def __init__(self, vad: FakeVad, rate: int, start_ms: float, wake: bool = False) -> None:
        self.vad = vad
        self.rate = rate
        self._wake = wake
        self.woke: list[float] = []
        # where the next frame starts
        self._at = start_ms
        self._unfed = b""
        # the same audio as the client sent it
        self._unheard = b""
        self._start: float | None = None
        self._last = 0.0
        self._waking = False

    def feed(
        self, pcm16: bytes, threshold: float, silence_duration_ms: int, wake_pcm16: bytes | None = None
    ) -> list[SpeechSegment]:
        if self.vad.gate is not None:
            self.vad.gate.wait()
        if self.vad.failing:
            raise RuntimeError("the detector broke")
        self.vad.feeds.append(len(pcm16) // 2 * 1000 / self.rate)
        self._unfed += pcm16
        self._unheard += pcm16 if wake_pcm16 is None else wake_pcm16
        self.woke = []
        frame = self.rate // 100 * 2
        found = []
        while len(self._unfed) >= frame:
            samples = np.frombuffer(self._unfed[:frame], dtype="<i2")
            heard = np.frombuffer(self._unheard[:frame], dtype="<i2")
            self._unfed, self._unheard = self._unfed[frame:], self._unheard[frame:]
            at, self._at = self._at, self._at + 10
            waking = self._wake and np.abs(heard.astype(np.int32)).max() > 20000
            if self._waking and not waking:
                self.woke.append(at)
            self._waking = waking
            if np.abs(samples.astype(np.int32)).max() > 1000:
                self._start = at if self._start is None else self._start
                self._last = at + 10
            elif self._start is not None and self._at - self._last >= silence_duration_ms:
                found.append(SpeechSegment(self._start, self._last, self._last))
                self._start = None
        if self._start is not None:
            found.append(SpeechSegment(self._start, None, self._last))
        return found


class _Turns:
    """A detector over a fresh buffer, recording what it reports."""

    def __init__(
        self, settings: ServerVad | None = None, max_turn_ms: int | None = MAX_TURN_MS, listen_ms: float | None = None
    ) -> None:
        self.buffer = InputAudioBuffer()
        self.vad = FakeVad()
        self.events: list[tuple] = []
        # the turns a wake word opened, and those that ended without speech
        self.woken: list[str] = []
        self.unspoken: list[str] = []
        # commit the buffer as the client does while the session sends speech_started
        self.commit_on_start = False
        # set, an ending waits for it after taking its audio
        self.sending: asyncio.Event | None = None
        # the speech a turn needs before it starts, as while an answer plays
        self.min_speech = 0
        # with wake words: where speech may start a turn without one
        self.window: tuple[float, float] | None = None
        wake = listen_ms is not None
        self.detector = TurnDetector(
            self.buffer,
            settings or ServerVad(type="server_vad"),
            lambda rate, start_ms: self.vad.open(rate, start_ms, wake),
            self._started,
            self._ended,
            self._failed,
            lambda: self.min_speech,
            max_turn_ms=max_turn_ms,
            listen_ms=listen_ms,
            window=lambda: self.window,
        )

    async def _started(self, turn: str, at: int, woken: bool) -> None:
        self.events.append(("started", turn, at))
        if woken:
            self.woken.append(turn)
        if self.commit_on_start:
            self.buffer.take()
            self.detector.reset()

    async def _ended(self, turn: str, start: int, end: int, spoken: bool) -> None:
        self.buffer.take(start, end)
        if self.sending is not None:
            # the session sends its events after it takes the audio
            await self.sending.wait()
        self.events.append(("ended", turn, start, end))
        if not spoken:
            self.unspoken.append(turn)

    async def _failed(self) -> None:
        self.events.append(("failed",))

    async def feed(self, *parts: bytes, step: float | None = 0.1) -> None:
        """Appends *parts*, *step* seconds at a time as a live client does, or each at once."""
        audio = b"".join(parts)
        size = len(audio) if step is None else int(step * RATE) * 2
        for start in range(0, len(audio), size):
            assert self.buffer.append(audio[start : start + size], RATE) is None
            self.detector.appended()
            await self.settle()

    async def settle(self) -> None:
        while self.detector._task is not None and not self.detector._task.done():
            await self.detector._task

    def turns(self) -> list[tuple[int, int]]:
        return [(event[2], event[3]) for event in self.events if event[0] == "ended"]


class TestTurns:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", [0.1, 0.02, None], ids=["live", "small-appends", "all-at-once"])
    async def test_a_turn_starts_at_its_speech_less_the_padding_and_ends_after_the_silence(self, step):
        turns = _Turns()
        await turns.feed(_quiet(1), _tone(1), _quiet(1), step=step)
        (started, ended) = turns.events
        assert started[0] == "started" and started[2] == 700
        assert ended == ("ended", started[1], 700, 2500)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", [0.1, None], ids=["live", "all-at-once"])
    async def test_each_utterance_is_its_own_turn(self, step):
        turns = _Turns()
        await turns.feed(_quiet(1), _tone(1), _quiet(1), _tone(1), _quiet(1), step=step)
        assert turns.turns() == [(700, 2500), (2700, 4500)]
        first, second = (event[1] for event in turns.events if event[0] == "started")
        assert first != second

    @pytest.mark.asyncio
    async def test_a_pause_shorter_than_the_silence_keeps_the_turn(self):
        turns = _Turns()
        await turns.feed(_quiet(1), _tone(0.5), _quiet(0.3), _tone(0.5), _quiet(1))
        assert turns.turns() == [(700, 2800)]

    @pytest.mark.asyncio
    async def test_the_padding_stops_at_the_last_turn_and_the_buffers_start(self):
        turns = _Turns()
        await turns.feed(_tone(1), _quiet(0.6), _tone(1), _quiet(1))
        assert turns.turns() == [(0, 1500), (1500, 3100)]

    @pytest.mark.asyncio
    async def test_the_settings_shape_the_turn(self):
        turns = _Turns(ServerVad(type="server_vad", prefix_padding_ms=100, silence_duration_ms=200))
        await turns.feed(_quiet(1), _tone(1), _quiet(1))
        assert turns.turns() == [(900, 2200)]

    @pytest.mark.asyncio
    async def test_a_turn_without_a_pause_ends_at_30_seconds_and_the_speech_goes_on_in_the_next(self):
        turns = _Turns()
        await turns.feed(_quiet(0.5), _tone(31), _quiet(1), step=None)
        assert turns.turns() == [(200, 30200), (30200, 32000)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", [0.1, None], ids=["live", "all-at-once"])
    async def test_without_a_longest_turn_a_turn_ends_only_at_a_pause(self, step):
        turns = _Turns(max_turn_ms=None)
        await turns.feed(_quiet(0.5), _tone(31), _quiet(1), step=step)
        assert turns.turns() == [(200, 32000)]

    @pytest.mark.asyncio
    async def test_speech_still_going_leaves_the_turn_open(self):
        turns = _Turns()
        await turns.feed(_quiet(1), _tone(1))
        assert [event[0] for event in turns.events] == ["started"]
        assert turns.detector.turn is not None

    @pytest.mark.asyncio
    async def test_a_reset_forgets_the_open_turn(self):
        turns = _Turns()
        await turns.feed(_quiet(1), _tone(1))
        turn = turns.detector.reset()
        turns.buffer.clear()
        await turns.feed(_quiet(1))
        assert turn == turns.events[0][1]
        assert [event[0] for event in turns.events] == ["started"]
        assert turns.detector.turn is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", [0.1, 0.02, None], ids=["live", "small-appends", "all-at-once"])
    async def test_each_append_is_fed_once_to_one_finder(self, step):
        turns = _Turns()
        await turns.feed(_quiet(3), step=step)
        assert sum(turns.vad.feeds) == pytest.approx(3000)
        assert turns.vad.opened == [(RATE, 0.0)]
        assert turns.detector._fed_ms == turns.buffer.end_ms

    @pytest.mark.asyncio
    async def test_appends_during_a_feed_go_together_in_the_next(self):
        turns = _Turns()
        turns.vad.gate = threading.Event()
        for _ in range(5):
            assert turns.buffer.append(_quiet(0.1), RATE) is None
            turns.detector.appended()
            await asyncio.sleep(0.01)
        turns.vad.gate.set()
        await turns.settle()
        assert turns.vad.feeds == [pytest.approx(100), pytest.approx(400)]

    @pytest.mark.asyncio
    async def test_the_buffer_keeps_only_what_a_turn_would_need_between_turns(self):
        turns = _Turns()
        await turns.feed(_quiet(10))
        # the lead for speech too short to open a segment yet, and the padding
        assert turns.buffer.seconds <= 1 + 0.3 + 0.1
        assert turns.buffer.end_ms == pytest.approx(10000)

    @pytest.mark.asyncio
    async def test_a_commit_or_a_new_rate_opens_a_new_finder_where_the_buffer_ends(self):
        turns = _Turns()
        await turns.feed(_quiet(1))
        turns.buffer.take()
        turns.detector.reset()
        await turns.feed(_quiet(1))
        turns.buffer.take()
        assert turns.buffer.append(bytes(1600 * 2), 16000) is None
        turns.detector.appended()
        await turns.settle()
        assert turns.vad.opened == [(RATE, 0.0), (RATE, 1000.0), (16000, 2000.0)]

    @pytest.mark.asyncio
    async def test_failing_feeds_are_reported_once_per_outage_and_keep_the_audio(self):
        turns = _Turns()
        turns.vad.failing = True
        await turns.feed(_quiet(5))
        assert turns.events == [("failed",)]
        assert turns.buffer.seconds == pytest.approx(5)
        turns.vad.failing = False
        await turns.feed(_tone(1), _quiet(1))
        turns.vad.failing = True
        await turns.feed(_quiet(1))
        assert [event[0] for event in turns.events] == ["failed", "started", "ended", "failed"]

    @pytest.mark.asyncio
    async def test_speech_during_an_outage_is_found_once_feeds_work_again(self):
        turns = _Turns()
        turns.vad.failing = True
        await turns.feed(_quiet(1), _tone(1), _quiet(1))
        turns.vad.failing = False
        await turns.feed(_quiet(0.3))
        assert [event[0] for event in turns.events] == ["failed", "started", "ended"]
        assert turns.turns() == [(700, 2500)]
        # a new finder took the audio from where the last good feed stopped
        assert turns.vad.opened[-1] == (RATE, 0.0)

    @pytest.mark.asyncio
    async def test_a_commit_while_the_session_reports_a_turn_stops_the_feed(self):
        turns = _Turns()
        turns.commit_on_start = True
        await turns.feed(_quiet(1), _tone(1), _quiet(1), step=None)
        assert [event[0] for event in turns.events] == ["started"]
        assert turns.detector.turn is None
        assert turns.buffer.seconds == 0

    @pytest.mark.asyncio
    async def test_a_reset_lets_a_commit_under_way_finish(self):
        # what turning server VAD off does: the feed isn't cancelled, so audio taken is committed
        turns = _Turns()
        turns.sending = asyncio.Event()
        assert turns.buffer.append(_quiet(1) + _tone(1) + _quiet(1), RATE) is None
        turns.detector.appended()
        task = turns.detector.task
        while turns.buffer.start_ms < 2500:
            await asyncio.sleep(0)
        turns.detector.reset()
        turns.sending.set()
        assert task is not None
        await task
        assert turns.turns() == [(700, 2500)]
        assert turns.detector.turn is None

    @pytest.mark.asyncio
    async def test_a_feed_runs_on_the_settings_it_started_with(self):
        turns = _Turns()
        turns.vad.gate = threading.Event()
        assert turns.buffer.append(_quiet(1) + _tone(1) + _quiet(1), RATE) is None
        turns.detector.appended()
        await asyncio.sleep(0)
        # a session.update while the feed runs
        turns.detector.settings = ServerVad(type="server_vad", prefix_padding_ms=0, silence_duration_ms=200)
        turns.vad.gate.set()
        await turns.settle()
        assert turns.turns()[0] == (700, 2500)

    @pytest.mark.asyncio
    async def test_a_feed_in_flight_over_reset_audio_changes_nothing(self):
        turns = _Turns()
        assert turns.buffer.append(_quiet(1) + _tone(1), RATE) is None
        turns.detector.appended()
        turns.detector.reset()
        turns.buffer.clear()
        await turns.settle()
        assert turns.events == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("min_speech", "expected"), [(0, [(200, 1300), (1500, 3300)]), (500, [(1500, 3300)])])
    async def test_speech_shorter_than_min_speech_starts_no_turn(self, min_speech, expected):
        turns = _Turns()
        turns.min_speech = min_speech
        # a cough, then speech that lasts
        await turns.feed(_quiet(0.5), _tone(0.3), _quiet(1), _tone(1), _quiet(1))
        assert turns.turns() == expected

    @pytest.mark.asyncio
    async def test_speech_that_lasts_starts_its_turn_dated_back(self):
        turns = _Turns()
        turns.min_speech = 500
        await turns.feed(_quiet(1), _tone(1), _quiet(1))
        (started, ended) = turns.events
        assert started[2] == 700
        assert ended[2:] == (700, 2500)

    @pytest.mark.asyncio
    async def test_min_speech_longer_than_the_silence_still_starts_a_turn(self):
        # the open segment keeps its start while it waits to last long enough
        turns = _Turns()
        turns.min_speech = 2000
        await turns.feed(_quiet(1), _tone(3), _quiet(1))
        assert turns.turns() == [(700, 4500)]

    @pytest.mark.asyncio
    async def test_closing_stops_the_feed_in_flight(self):
        turns = _Turns()
        assert turns.buffer.append(_quiet(1), RATE) is None
        turns.detector.appended()
        await turns.detector.close()
        assert turns.detector._task is not None and turns.detector._task.done()


ASLEEP = (math.inf, math.inf)


class TestWakeWords:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", [0.1, None], ids=["live", "all-at-once"])
    async def test_a_wake_word_opens_a_turn_where_it_ends_held_by_the_speech_after_it(self, step):
        turns = _Turns(listen_ms=10_000)
        turns.window = ASLEEP
        await turns.feed(_quiet(0.5), wake_word(0.6), _quiet(0.3), _tone(1), _quiet(1), step=step)
        assert turns.turns() == [(1100, 2900)]
        assert turns.woken == [turns.events[0][1]] and turns.unspoken == []

    @pytest.mark.asyncio
    async def test_a_pause_after_the_wake_word_waits_for_the_speech(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = ASLEEP
        await turns.feed(_quiet(0.5), wake_word(0.6), _quiet(1), _tone(1), _quiet(1))
        assert turns.turns() == [(1100, 3600)]
        assert turns.unspoken == []

    @pytest.mark.asyncio
    async def test_a_wake_word_nobody_follows_ends_its_turn_without_speech_once_the_listening_ends(self):
        turns = _Turns(listen_ms=2000)
        turns.window = ASLEEP
        await turns.feed(_quiet(0.5), wake_word(0.6), _quiet(3))
        assert turns.turns() == [(1100, 3100)]
        assert turns.unspoken == turns.woken

    @pytest.mark.asyncio
    async def test_speech_without_a_wake_word_starts_no_turn_and_isnt_kept(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = ASLEEP
        await turns.feed(_quiet(0.5), _tone(5), _quiet(1))
        assert turns.events == []
        # what a turn's prefix padding and too-short speech need, not the speech
        assert turns.buffer.seconds <= 1.4

    @pytest.mark.asyncio
    async def test_speech_starts_a_turn_in_the_window_only(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = (1000, 3000)
        await turns.feed(_quiet(1.5), _tone(1), _quiet(1), _tone(1), _quiet(1))
        assert turns.turns() == [(1200, 3000)]
        assert turns.woken == []

    @pytest.mark.asyncio
    async def test_speech_that_began_before_the_window_starts_its_turn_where_the_window_opens(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = (1500, 5000)
        await turns.feed(_quiet(1), _tone(2), _quiet(1))
        assert turns.turns() == [(1500, 3500)]

    @pytest.mark.asyncio
    async def test_a_wake_word_in_an_open_turn_is_part_of_it(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = (-math.inf, math.inf)
        await turns.feed(_quiet(0.5), _tone(0.5), wake_word(0.6), _quiet(1))
        assert turns.turns() == [(200, 2100)]
        assert turns.woken == []

    @pytest.mark.asyncio
    async def test_the_wake_word_is_heard_in_the_audio_the_client_sent_where_the_buffer_holds_silence(self):
        turns = _Turns(listen_ms=10_000)
        turns.window = ASLEEP
        turns.buffer.keep_sent()
        sent = _quiet(0.5) + wake_word(0.6) + _quiet(0.2) + _tone(1) + _quiet(1)
        # the answer plays over the wake word, so the buffer holds silence for it
        muted = bytes(int(1.1 * RATE) * 2) + sent[int(1.1 * RATE) * 2 :]
        size = int(0.1 * RATE) * 2
        for start in range(0, len(sent), size):
            assert turns.buffer.append(muted[start : start + size], RATE, sent[start : start + size]) is None
            turns.detector.appended()
            await turns.settle()
        assert turns.turns() == [(1100, 2800)]
        assert len(turns.woken) == 1

    @pytest.mark.asyncio
    async def test_without_wake_words_a_finder_hears_none(self):
        turns = _Turns()
        await turns.feed(_quiet(0.5), wake_word(0.6), _quiet(1))
        # the wake word is speech like any other
        assert turns.turns() == [(200, 1600)]
        assert turns.woken == []
