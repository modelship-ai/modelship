"""The realtime pipeline's small shared units: the relay's bounded `FrameQueue`, a session's `BackgroundTasks`, and
parsing a client frame into an event or its error."""

import asyncio
from typing import ClassVar

import pytest

from modelship.infer.pipelines import relay
from modelship.infer.pipelines.assistant.client_events import parse_event
from modelship.infer.pipelines.assistant.tasks import BackgroundTasks
from modelship.infer.pipelines.relay import FrameQueue
from modelship.openai.protocol.realtime import AudioClear, RealtimeError


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class TestFrameQueue:
    @pytest.mark.asyncio
    async def test_frames_come_out_in_order(self):
        queue = FrameQueue()
        await queue.put(["a", b"b"])
        await queue.put(["c"])
        assert [await queue.get(), await queue.get(), await queue.get()] == ["a", b"b", "c"]

    @pytest.mark.asyncio
    async def test_get_waits_for_a_frame(self):
        queue = FrameQueue()
        got = asyncio.ensure_future(queue.get())
        await _settle()
        assert not got.done()
        await queue.put(["a"])
        assert await got == "a"

    @pytest.mark.asyncio
    async def test_put_waits_while_more_than_the_cap_waits(self, monkeypatch):
        monkeypatch.setattr(relay, "MAX_QUEUED_BYTES", 5)
        queue = FrameQueue()
        put = asyncio.ensure_future(queue.put(["abc", "def"]))
        await _settle()
        assert not put.done()
        assert await queue.get() == "abc"
        await _settle()
        assert put.done()

    @pytest.mark.asyncio
    async def test_get_all_takes_every_frame_and_frees_the_cap(self, monkeypatch):
        monkeypatch.setattr(relay, "MAX_QUEUED_BYTES", 5)
        queue = FrameQueue()
        put = asyncio.ensure_future(queue.put(["abc", "def"]))
        await _settle()
        assert await queue.get_all() == ["abc", "def"]
        await _settle()
        assert put.done() and queue.empty

    @pytest.mark.asyncio
    async def test_a_closed_queue_gives_its_frames_then_none(self):
        queue = FrameQueue()
        await queue.put(["a"])
        queue.close()
        assert await queue.get() == "a"
        assert await queue.get() is None
        assert await queue.get_all() == []

    @pytest.mark.asyncio
    async def test_closing_ends_the_waits(self, monkeypatch):
        monkeypatch.setattr(relay, "MAX_QUEUED_BYTES", 1)
        queue, empty = FrameQueue(), FrameQueue()
        put = asyncio.ensure_future(queue.put(["abc"]))
        got = asyncio.ensure_future(empty.get())
        await _settle()
        queue.close()
        empty.close()
        await _settle()
        assert put.done() and await got is None


class TestBackgroundTasks:
    @pytest.mark.asyncio
    async def test_a_finished_task_is_dropped(self):
        tasks = BackgroundTasks()
        task = tasks.spawn(asyncio.sleep(0))
        assert len(tasks) == 1
        await task
        await _settle()
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_cancel_cancels_every_task(self):
        tasks = BackgroundTasks()
        spawned = tasks.spawn(asyncio.Event().wait())
        added = asyncio.ensure_future(asyncio.Event().wait())
        tasks.add(added)
        tasks.cancel()
        await tasks.wait(1.0)
        assert spawned.cancelled() and added.cancelled()

    @pytest.mark.asyncio
    async def test_wait_stops_at_its_timeout(self):
        tasks = BackgroundTasks()
        task = tasks.spawn(asyncio.Event().wait())
        await tasks.wait(0.01)
        assert not task.done()
        task.cancel()

    @pytest.mark.asyncio
    async def test_wait_with_no_tasks_returns_at_once(self):
        await asyncio.wait_for(BackgroundTasks().wait(10.0), timeout=1.0)


class TestParseEvent:
    ACCEPTED: ClassVar[list[str]] = ["input_audio_buffer.clear", "session.update"]

    def test_a_valid_event_parses(self):
        event = parse_event('{"type": "input_audio_buffer.clear", "event_id": "e1"}', self.ACCEPTED)
        assert isinstance(event, AudioClear) and event.event_id == "e1"

    def test_a_type_the_session_doesnt_take_is_the_error(self):
        error = parse_event('{"type": "response.cancel", "event_id": "e2"}', self.ACCEPTED)
        assert isinstance(error, RealtimeError)
        assert (error.code, error.param, error.event_id) == ("invalid_value", "type", "e2")

    def test_an_invalid_event_of_a_type_it_doesnt_take_names_the_type(self):
        error = parse_event('{"type": "response.create", "response": 1, "event_id": "e3"}', self.ACCEPTED)
        assert isinstance(error, RealtimeError) and error.param == "type" and error.event_id == "e3"

    def test_not_json(self):
        error = parse_event("{", self.ACCEPTED)
        assert isinstance(error, RealtimeError) and error.message == "The event is not valid JSON."

    def test_a_missing_type(self):
        error = parse_event("{}", self.ACCEPTED)
        assert isinstance(error, RealtimeError) and error.code == "missing_required_parameter"

    def test_an_invalid_field_is_named_by_its_path(self):
        error = parse_event('{"type": "session.update", "session": {"instructions": 3}}', self.ACCEPTED)
        assert isinstance(error, RealtimeError)
        assert error.code == "invalid_value" and error.param == "session.instructions"
