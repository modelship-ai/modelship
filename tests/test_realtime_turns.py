"""Server VAD's turn finding: a session's audio scanned as it arrives, with a fake scan that hears speech wherever
the audio is loud, and the turns it finds on the session's timeline."""

import asyncio

import numpy as np
import pytest

from modelship.openai.protocol.realtime import ServerVad
from modelship.openai.realtime.audio import InputAudioBuffer
from modelship.openai.realtime.turns import MAX_SCAN_MS, TurnDetector

RATE = 24000


def _tone(seconds: float) -> bytes:
    count = int(seconds * RATE)
    return (np.sin(2 * np.pi * 440 * np.arange(count) / RATE) * 8000).astype("<i2").tobytes()


def _quiet(seconds: float) -> bytes:
    return bytes(int(seconds * RATE) * 2)


class FakeVad:
    """Scans like the audio app, statelessly: speech wherever a 10 ms frame is loud, a segment closing once
    *silence_duration_ms* of quiet follows it. Records each scan's length in ms; *failing* makes scans fail."""

    def __init__(self) -> None:
        self.failing = False
        self.scans: list[int] = []
        # set, each scan waits for it, as for the audio app's answer
        self.gate: asyncio.Event | None = None

    async def scan(self, audio: bytes, rate: int, threshold: float, silence_duration_ms: int):
        if self.gate is not None:
            await self.gate.wait()
        if self.failing:
            return None
        samples = np.frombuffer(audio, dtype="<i2")
        self.scans.append(len(samples) * 1000 // rate)
        frame = rate // 100
        segments: list[tuple[int, int | None, int]] = []
        start = last = None
        for index in range(0, len(samples) - frame + 1, frame):
            at = index * 1000 // rate
            if np.abs(samples[index : index + frame].astype(np.int32)).max() > 1000:
                start = at if start is None else start
                last = at + 10
            elif start is not None and last is not None and at + 10 - last >= silence_duration_ms:
                segments.append((start, last, last))
                start = None
        if start is not None and last is not None:
            segments.append((start, None, last))
        return segments


class _Turns:
    """A detector over a fresh buffer, recording what it reports."""

    def __init__(self, settings: ServerVad | None = None) -> None:
        self.buffer = InputAudioBuffer()
        self.vad = FakeVad()
        self.events: list[tuple] = []
        # how many endings to refuse, as a full transcription queue does
        self.refusals = 0
        # commit the buffer as the client does while the session sends speech_started
        self.commit_on_start = False
        # set, an ending waits for it after taking its audio
        self.sending: asyncio.Event | None = None
        # the speech a turn needs before it starts, as while an answer plays
        self.min_speech = 0
        self.detector = TurnDetector(
            self.buffer,
            settings or ServerVad(type="server_vad"),
            self.vad.scan,
            self._started,
            self._ended,
            self._failed,
            lambda: self.min_speech,
        )

    async def _started(self, turn: str, at: int) -> None:
        self.events.append(("started", turn, at))
        if self.commit_on_start:
            self.buffer.take()
            self.detector.reset()

    async def _ended(self, turn: str, start: int, end: int) -> bool:
        if self.refusals:
            self.refusals -= 1
            self.events.append(("refused", turn))
            return False
        self.buffer.take(start, end)
        if self.sending is not None:
            # the session sends its events after it takes the audio
            await self.sending.wait()
        self.events.append(("ended", turn, start, end))
        return True

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
    async def test_no_audio_goes_unscanned_and_no_scan_passes_its_limit(self):
        turns = _Turns()
        await turns.feed(_quiet(70), step=None)
        assert max(turns.vad.scans) <= MAX_SCAN_MS
        assert len(turns.vad.scans) >= 3
        assert turns.detector._scanned_ms == turns.buffer.end_ms

    @pytest.mark.asyncio
    async def test_the_buffer_keeps_only_what_the_next_scan_needs_between_turns(self):
        turns = _Turns()
        await turns.feed(_quiet(10))
        # the silence, the warm-up and the padding, and the audio not scanned yet
        assert turns.buffer.seconds <= 0.5 + 1 + 0.3 + 0.3
        assert turns.buffer.end_ms == pytest.approx(10000)

    @pytest.mark.asyncio
    async def test_failing_scans_are_reported_once_per_outage_and_keep_the_audio(self):
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
    async def test_speech_during_an_outage_is_found_once_scans_answer_again(self):
        turns = _Turns()
        turns.vad.failing = True
        await turns.feed(_quiet(1), _tone(1), _quiet(1))
        turns.vad.failing = False
        await turns.feed(_quiet(0.3))
        assert [event[0] for event in turns.events] == ["failed", "started", "ended"]
        assert turns.turns() == [(700, 2500)]

    @pytest.mark.asyncio
    async def test_a_commit_while_the_session_reports_a_turn_stops_the_scan(self):
        turns = _Turns()
        turns.commit_on_start = True
        await turns.feed(_quiet(1), _tone(1), _quiet(1), step=None)
        assert [event[0] for event in turns.events] == ["started"]
        assert turns.detector.turn is None
        assert turns.buffer.seconds == 0

    @pytest.mark.asyncio
    async def test_a_reset_lets_a_commit_under_way_finish(self):
        # what turning server VAD off does: the scan isn't cancelled, so audio taken is committed
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
    async def test_a_scan_runs_on_the_settings_it_started_with(self):
        turns = _Turns()
        turns.vad.gate = asyncio.Event()
        assert turns.buffer.append(_quiet(1) + _tone(1) + _quiet(1), RATE) is None
        turns.detector.appended()
        await asyncio.sleep(0)
        # a session.update while the scan waits for the audio app
        turns.detector.settings = ServerVad(type="server_vad", prefix_padding_ms=0, silence_duration_ms=200)
        turns.vad.gate.set()
        await turns.settle()
        assert turns.turns()[0] == (700, 2500)

    @pytest.mark.asyncio
    async def test_a_refused_ending_keeps_the_turn_and_its_audio_and_ends_it_on_a_later_scan(self):
        turns = _Turns()
        turns.refusals = 2
        await turns.feed(_quiet(1), _tone(1), _quiet(1.5))
        assert [event[0] for event in turns.events] == ["started", "refused", "refused", "ended"]
        started, *_, ended = turns.events
        assert ended == ("ended", started[1], 700, 2500)

    @pytest.mark.asyncio
    async def test_a_scan_in_flight_over_reset_audio_changes_nothing(self):
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
        # each scan looks back far enough to see the whole of the speech
        turns = _Turns()
        turns.min_speech = 2000
        await turns.feed(_quiet(1), _tone(3), _quiet(1))
        assert turns.turns() == [(700, 4500)]

    @pytest.mark.asyncio
    async def test_closing_stops_the_scan_in_flight(self):
        turns = _Turns()
        assert turns.buffer.append(_quiet(1), RATE) is None
        turns.detector.appended()
        await turns.detector.close()
        assert turns.detector._task is not None and turns.detector._task.done()
