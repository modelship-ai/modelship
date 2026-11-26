"""`/v1/realtime` sessions on a streaming speech-to-text model: each turn's audio pushed as it arrives, at least a
chunk at a time and one call at a time, its text sent as deltas while server VAD's turn goes on, the commit
finishing the stream, a failed or restarted stream failing the turn, and every stream closed that won't be
committed."""

import asyncio
import time

from modelship.openai.protocol import ErrorInfo, ErrorResponse, SttCapabilities, SttStreamText
from tests.test_realtime_session import (
    ASSISTANT,
    SCRIBE,
    _append,
    _client,
    _commit,
    _open,
    _pcm,
    _quiet,
    _route,
    _stream,
    _tone,
    _until,
)
from tests.test_realtime_turns import wake_word

STREAMING = SttCapabilities(streaming=True, reports_language=False, chunk_ms=160)
# SCRIBE with the client committing each turn
SCRIBE_MANUAL = {**SCRIBE, "session_defaults": {"audio": {"input": {"turn_detection": None}}}}
RATE = 24000
CHUNK_BYTES = RATE * 2 * 160 // 1000
# the fake model's text grows a word with each this much audio
WORD_MS = 500
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"


class _StreamStt:
    """The streaming speech-to-text model "whisper": each stream says the next of *scripts*, a word per WORD_MS of
    audio it holds, and all of it once finished. Records each call as (op, key, bytes) and each stream's options.
    *fail_push* fails that push (1-based) with an ErrorResponse; *lose* makes every stream forget the audio before
    each push, as one opened again on another replica does; each call takes *delay*."""

    def __init__(self, *scripts: str, delay: float = 0.0):
        self.scripts = list(scripts)
        self.delay = delay
        self.fail_push: int | None = None
        self.lose = False
        # finishes never answer
        self.hang_finish = False
        self.calls: list[tuple[str, str, int]] = []
        self.options: dict[str, object] = {}
        self._streams: dict[str, list] = {}
        self._in_flight = 0
        self.most_in_flight = 0

    async def _take(self, op: str, key: str, pcm: bytes, options) -> list:
        self.calls.append((op, key, len(pcm)))
        self._in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self._in_flight -= 1
        if key not in self._streams:
            self._streams[key] = [self.scripts.pop(0).split(), 0]
            self.options[key] = options
        stream = self._streams[key]
        stream[1] = len(pcm) if self.lose else stream[1] + len(pcm)
        return stream

    async def push(self, key: str, pcm: bytes, rate: int, options):
        pushes = sum(1 for call in self.calls if call[0] == "push") + 1
        words, held = await self._take("push", key, pcm, options)
        if pushes == self.fail_push:
            return ErrorResponse(error=ErrorInfo(message="decoder blew up", type="server_error"))
        ms = held / 2 / rate * 1000
        return SttStreamText(text=" ".join(words[: int(ms // WORD_MS)]), audio_ms=ms)

    async def finish(self, key: str, pcm: bytes, rate: int, options):
        words, held = await self._take("finish", key, pcm, options)
        if self.hang_finish:
            await asyncio.Event().wait()
        del self._streams[key]
        return SttStreamText(text=" ".join(words), audio_ms=held / 2 / rate * 1000)

    async def close(self, key: str) -> None:
        self.calls.append(("close", key, 0))
        self._streams.pop(key, None)

    async def transcribe(self, audio, request):
        raise AssertionError("a streaming model got a batch transcription")

    def ops(self, *kinds: str) -> list[tuple[str, str, int]]:
        return [call for call in self.calls if call[0] in kinds]


def _session(stt: _StreamStt, pipeline: dict = SCRIBE):
    client = _client(_route(pipeline), stt=STREAMING, transcriber=stt)  # type: ignore[arg-type]
    return client.websocket_connect(f"/v1/realtime?model={pipeline['name']}")


def _settle(ws) -> None:
    """Waits until the session has handled everything sent so far, and its pushes have run."""
    ws.send_json({"type": "session.update", "session": {}})
    _until(ws, "session.updated")
    time.sleep(0.2)


def _eventually(check, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


class TestClientCommits:
    def test_the_buffer_streams_as_it_arrives_and_the_commit_finishes_it(self):
        stt = _StreamStt("turn the kitchen lights on")
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            for _ in range(20):
                _append(ws, _pcm(0.1))
            _settle(ws)
            pushed_before_commit = len(stt.ops("push"))
            events = _commit(ws)
        assert pushed_before_commit > 0
        pushes, finishes = stt.ops("push"), stt.ops("finish")
        assert all(size >= CHUNK_BYTES for _, _, size in pushes)
        assert len(finishes) == 1 and finishes[0][1] == pushes[0][1]
        assert sum(size for _, _, size in pushes + finishes) == 2 * RATE * 2
        assert DELTA not in [event["type"] for event in events]
        [completed] = [event for event in events if event["type"] == COMPLETED]
        assert completed["transcript"] == "turn the kitchen lights on"
        assert completed["usage"] == {"type": "duration", "seconds": 2.0}

    def test_a_commit_without_a_push_sends_the_whole_buffer_as_it_finishes(self):
        stt = _StreamStt("hi")
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            # less than a chunk: nothing is pushed before the commit
            _append(ws, _pcm(0.12))
            events = _commit(ws)
        assert [op for op, _, _ in stt.calls] == ["finish"]
        assert [event["transcript"] for event in events if event["type"] == COMPLETED] == ["hi"]

    def test_the_sessions_transcription_language_opens_the_stream(self):
        stt = _StreamStt("hallo")
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            update = {"audio": {"input": {"transcription": {"language": "de"}}}}
            ws.send_json({"type": "session.update", "session": update})
            _until(ws, "session.updated")
            _append(ws, _pcm(0.5))
            _commit(ws)
        [options] = stt.options.values()
        assert options.language == "de"  # type: ignore[attr-defined]

    def test_a_failed_push_fails_the_turn_and_closes_its_stream(self):
        stt = _StreamStt("lost")
        stt.fail_push = 1
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            _append(ws, _pcm(0.5))
            _settle(ws)
            _append(ws, _pcm(0.5))
            events = _commit(ws)
        [failed] = [event for event in events if event["type"] == FAILED]
        assert failed["error"] == {"type": "server_error", "code": None, "message": "decoder blew up"}
        assert [op for op, _, _ in stt.calls] == ["push", "close"]
        assert COMPLETED not in [event["type"] for event in events]

    def test_a_stream_that_lost_audio_fails_the_turn(self):
        stt = _StreamStt("lost")
        stt.lose = True
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            _append(ws, _pcm(0.3))
            _settle(ws)
            _append(ws, _pcm(0.3))
            _settle(ws)
            events = _commit(ws)
        [failed] = [event for event in events if event["type"] == FAILED]
        assert failed["error"]["type"] == "server_error"
        assert "failed to transcribe the turn as it streamed" in failed["error"]["message"]
        assert ("close", stt.calls[0][1], 0) in stt.calls
        assert not stt.ops("finish")

    def test_a_clear_closes_the_stream_and_the_session_closes_the_next(self):
        stt = _StreamStt("one", "two")
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            _append(ws, _pcm(0.5))
            _settle(ws)
            ws.send_json({"type": "input_audio_buffer.clear"})
            _until(ws, "input_audio_buffer.cleared")
            _append(ws, _pcm(0.5))
            _settle(ws)
        keys = list(dict.fromkeys(key for _, key, _ in stt.ops("push")))
        assert len(keys) == 2
        assert _eventually(lambda: [key for _, key, _ in stt.ops("close")] == keys)


class TestServerVad:
    def test_a_turns_text_streams_as_deltas_and_its_commit_completes_it(self):
        stt = _StreamStt("one two three four")
        with _session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(2), _quiet(1))
            events = _until(ws, "conversation.item.done")
            turn = next(event["item_id"] for event in events if event["type"] == "input_audio_buffer.speech_started")
            ws.send_json({"type": "conversation.item.retrieve", "item_id": turn})
            retrieved = _until(ws, "conversation.item.retrieved")[-1]
        kinds = [event["type"] for event in events]
        deltas = [event for event in events if event["type"] == DELTA]
        assert deltas
        assert all((event["item_id"], event["content_index"]) == (turn, 0) for event in deltas)
        assert kinds.index("input_audio_buffer.speech_started") < kinds.index(DELTA)
        assert max(i for i, kind in enumerate(kinds) if kind == DELTA) < kinds.index(COMPLETED)
        assert "one two three four".startswith("".join(event["delta"] for event in deltas))
        [completed] = [event for event in events if event["type"] == COMPLETED]
        assert (completed["item_id"], completed["transcript"]) == (turn, "one two three four")
        assert retrieved["item"]["content"][0]["transcript"] == "one two three four"
        # the stream is the turn's, from its start
        assert {key for _, key, _ in stt.ops("push", "finish")} == {stt.calls[0][1]}
        assert stt.calls[0][1].endswith(f":{turn}")

    def test_a_turns_stream_gets_its_audio_and_none_after_it(self):
        stt = _StreamStt("one", "two")
        with _session(stt) as ws:
            _open(ws)
            # faster than real time, the next turn arrives before server VAD ends the first
            _stream(ws, _quiet(1), _tone(1), _quiet(1), _tone(1), _quiet(1), step=1.0)
            events = _until(ws, "conversation.item.done") + _until(ws, "conversation.item.done")
        starts = {e["item_id"]: e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"}
        ends = {e["item_id"]: e["audio_end_ms"] for e in events if e["type"] == "input_audio_buffer.speech_stopped"}
        assert len(starts) == 2
        for turn, start in starts.items():
            sent = sum(size for _, key, size in stt.ops("push", "finish") if key.endswith(f":{turn}"))
            assert sent == (ends[turn] - start) * RATE * 2 // 1000, turn

    def test_a_session_that_ends_while_a_stream_finishes_closes_it(self):
        stt = _StreamStt("never")
        stt.hang_finish = True
        with _session(stt, SCRIBE_MANUAL) as ws:
            _open(ws)
            _append(ws, _pcm(0.5))
            ws.send_json({"type": "input_audio_buffer.commit"})
            _until(ws, "conversation.item.added")
            assert _eventually(lambda: bool(stt.ops("finish")))
        key = stt.ops("finish")[0][1]
        assert _eventually(lambda: ("close", key, 0) in stt.calls)

    def test_a_streamed_turn_is_not_cut_at_30_seconds(self):
        stt = _StreamStt("a long story")
        with _session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(0.5), _tone(31), _quiet(1), step=0.5)
            events = _until(ws, "conversation.item.done")
            _settle(ws)
        kinds = [event["type"] for event in events]
        assert kinds.count("input_audio_buffer.speech_started") == 1
        assert kinds.count("input_audio_buffer.committed") == 1

    def test_a_sessions_calls_go_one_at_a_time(self):
        stt = _StreamStt("one", "two", delay=0.05)
        with _session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1), _quiet(1), _tone(1), _quiet(1))
            _until(ws, "conversation.item.done")
            _until(ws, "conversation.item.done")
        assert stt.most_in_flight == 1
        assert len(stt.ops("finish")) == 2

    def test_turning_turn_detection_off_closes_the_open_turns_stream(self):
        stt = _StreamStt("cut")
        with _session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1))
            _settle(ws)
            ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": None}}}})
            _until(ws, "session.updated")
            key = stt.calls[0][1]
            assert _eventually(lambda: ("close", key, 0) in stt.calls)
        assert not stt.ops("finish")

    def test_a_client_commit_during_a_turn_finishes_the_turns_stream(self):
        stt = _StreamStt("short")
        with _session(stt) as ws:
            _open(ws)
            _stream(ws, _quiet(1), _tone(1))
            _settle(ws)
            events = _commit(ws)
        [completed] = [event for event in events if event["type"] == COMPLETED]
        assert completed["transcript"] == "short"
        assert {key for _, key, _ in stt.ops("push", "finish")} == {stt.calls[0][1]}


# a realtime session whose turns start at the wake word, answering none of them
WOKEN = {
    **ASSISTANT,
    "wake": {"words": ["hey_jarvis"], "listen_s": 1},
    "session_defaults": {"audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}}},
}


class TestWakeWords:
    def test_a_wake_word_nobody_follows_never_reaches_the_model(self):
        stt = _StreamStt()
        with _session(stt, WOKEN) as ws:
            _open(ws)
            _stream(ws, _quiet(0.5), wake_word(0.6), _quiet(2))
            events = _until(ws, COMPLETED)
            _settle(ws)
        assert events[-1]["transcript"] == ""
        assert DELTA not in [event["type"] for event in events]
        assert stt.calls == []

    def test_a_wake_words_turn_streams_from_where_the_wake_word_ends_once_it_hears_speech(self):
        stt = _StreamStt("turn the kitchen lights on")
        with _session(stt, WOKEN) as ws:
            _open(ws)
            _stream(ws, _quiet(0.5), wake_word(0.6), _quiet(0.3), _tone(2), _quiet(1))
            events = _until(ws, COMPLETED)
        started = next(event for event in events if event["type"] == "input_audio_buffer.speech_started")
        assert started["audio_start_ms"] == 1100
        [key] = {key for _, key, _ in stt.ops("push", "finish")}
        assert key.endswith(f":{started['item_id']}")
        # from the wake word's end to the turn's: 0.3 s pause, 2 s of speech and 0.5 s of silence
        assert sum(size for _, _, size in stt.ops("push", "finish")) == round(2.8 * RATE) * 2
        assert events[-1]["transcript"] == "turn the kitchen lights on"
