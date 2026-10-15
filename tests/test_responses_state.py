"""Tests for the /v1/responses conversation-state domain layer — record round-trip,
history rebuild, identity scoping, TTL, and availability-vs-absent semantics.

Runs against every backend the store supports, so the layer is proven on the default
memory:// and on the durable redis:// alike.
"""

import asyncio

import pytest

from modelship.openai.state import responses as responses_state
from modelship.state import MemoryStoreActor, StateStoreUnavailableError
from modelship.state.redis import RedisStateStore

_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


def _fake_redis_store():
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    s = RedisStateStore("redis://fake")
    s._sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    s._async_client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return s


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return _MemoryStore()
    return _fake_redis_store()


def _response(response_id: str = "resp_1", output: list | None = None) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "output": output if output is not None else [{"type": "message", "role": "assistant", "content": []}],
    }


def _without_ids(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "id"} for item in items]


async def _write(store, identity="u1", response_id="resp_1", *, response=None, parent=None, input_items=()):
    await responses_state.write_async(
        store,
        identity,
        response_id,
        response=response or _response(response_id),
        parent=parent,
        input_items=list(input_items),
    )


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_write_then_read(self, store):
        await _write(store, input_items=[{"type": "message", "role": "user"}])
        record = await responses_state.read_async(store, "u1", "resp_1")
        assert record is not None
        assert record["response"] == _response()
        items = await responses_state.input_items_async(store, "u1", record)
        assert _without_ids(items) == [{"type": "message", "role": "user"}]

    @pytest.mark.asyncio
    async def test_absent_returns_none(self, store):
        assert await responses_state.read_async(store, "u1", "nope") is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await _write(store)
        await responses_state.delete_async(store, "u1", "resp_1")
        assert await responses_state.read_async(store, "u1", "resp_1") is None
        # idempotent — deleting an absent record is not an error
        await responses_state.delete_async(store, "u1", "resp_1")


class TestIdentityScoping:
    """A bare response_id would let any caller fetch another's conversation."""

    @pytest.mark.asyncio
    async def test_other_identity_cannot_read(self, store):
        await _write(store)
        assert await responses_state.read_async(store, "u2", "resp_1") is None

    @pytest.mark.asyncio
    async def test_same_id_different_identities_are_independent(self, store):
        await _write(store, "u1", response=_response("a"))
        await _write(store, "u2", response=_response("b"))
        u1 = await responses_state.read_async(store, "u1", "resp_1")
        u2 = await responses_state.read_async(store, "u2", "resp_1")
        assert u1 is not None and u2 is not None
        assert u1["response"]["id"] == "a"
        assert u2["response"]["id"] == "b"


class TestHistory:
    @pytest.mark.asyncio
    async def test_rebuild_is_input_then_output(self, store):
        await _write(
            store,
            response=_response(output=[{"type": "message", "role": "assistant", "content": "yo"}]),
            input_items=[{"type": "message", "role": "user", "content": "hi"}],
        )
        record = await responses_state.read_async(store, "u1", "resp_1", with_output=False)
        history = await responses_state.history_async(store, "u1", record)
        assert [(item["role"], item["content"]) for item in history] == [("user", "hi"), ("assistant", "yo")]

    @pytest.mark.asyncio
    async def test_accumulates_across_turns(self, store):
        await _write(store, response=_response(output=[{"o": 1}]), input_items=[{"i": 1}])
        await _write(
            store,
            response_id="resp_2",
            response=_response("resp_2", output=[{"o": 2}]),
            parent={"log": "resp_1", "depth": 1},
            input_items=[{"i": 2}],
        )
        record = await responses_state.read_async(store, "u1", "resp_2", with_output=False)
        history = await responses_state.history_async(store, "u1", record)
        assert _without_ids(history) == [{"i": 1}, {"o": 1}, {"i": 2}, {"o": 2}]


class TestTerminalRace:
    @pytest.mark.parametrize("parent", [None, {"log": "resp_0", "depth": 1}])
    @pytest.mark.asyncio
    async def test_record_and_output_come_from_one_writer(self, store, monkeypatch, parent):
        if parent is not None:
            await _write(store, response_id="resp_0")
        await responses_state.write_background(
            store, "u1", "resp_1", response={**_response(output=[]), "status": "queued"}, parent=parent, input_items=[]
        )
        both_read = asyncio.Barrier(2)
        get_async = store.get_async

        async def get_after_both(key):
            value = await get_async(key)
            await both_read.wait()
            return value

        monkeypatch.setattr(store, "get_async", get_after_both)
        cancelled = {**_response(output=[]), "status": "cancelled"}
        completed = _response(output=[{"type": "message", "role": "assistant", "content": "done"}])
        await asyncio.gather(
            responses_state.write_terminal_if_not_terminal(store, "u1", "resp_1", response=cancelled),
            responses_state.write_terminal_if_not_terminal(store, "u1", "resp_1", response=completed),
        )
        monkeypatch.undo()

        record = await responses_state.read_async(store, "u1", "resp_1")
        assert record is not None
        assert record["response"] in (cancelled, completed)

    @pytest.mark.parametrize("parent", [None, {"log": "resp_0", "depth": 1}])
    @pytest.mark.asyncio
    async def test_a_record_deleted_after_the_read_stays_deleted(self, store, monkeypatch, parent):
        if parent is not None:
            await _write(store, response_id="resp_0")
        await responses_state.write_background(
            store, "u1", "resp_1", response={**_response(output=[]), "status": "queued"}, parent=parent, input_items=[]
        )
        read, deleted = asyncio.Event(), asyncio.Event()
        get_async = store.get_async

        async def get_until_deleted(key):
            value = await get_async(key)
            read.set()
            await deleted.wait()
            return value

        monkeypatch.setattr(store, "get_async", get_until_deleted)
        write = asyncio.create_task(
            responses_state.write_terminal_if_not_terminal(store, "u1", "resp_1", response=_response())
        )
        await read.wait()
        await responses_state.delete_async(store, "u1", "resp_1")
        deleted.set()
        await write
        monkeypatch.undo()

        assert await responses_state.read_async(store, "u1", "resp_1") is None
        assert await store.read_range_async("responses-log/u1/resp_1") == []
        assert len(await store.read_range_async("responses-log/u1/resp_0")) == (0 if parent is None else 1)


class TestMalformedRecord:
    @pytest.mark.parametrize(
        "value",
        [
            {"input_items": []},
            {"response": {"id": "resp_1"}, "input_items": []},
            {"response": {"id": "resp_1"}, "log": "resp_1"},
        ],
    )
    @pytest.mark.asyncio
    async def test_treated_as_missing(self, store, value):
        await store.set_async("responses/u1/resp_1", value)
        assert await responses_state.read_async(store, "u1", "resp_1") is None


class TestUnavailableVsAbsent:
    """A store outage must 503, never look like an unknown id."""

    @pytest.mark.asyncio
    async def test_unavailable_propagates(self):
        class Down:
            async def get_async(self, *a, **k):
                raise StateStoreUnavailableError("down")

        with pytest.raises(StateStoreUnavailableError):
            await responses_state.read_async(Down(), "u1", "resp_1")


class TestStreamBuffer:
    @pytest.mark.asyncio
    async def test_append_then_read_roundtrips(self, store):
        await responses_state.append_stream_event(
            store, "u1", "resp_1", {"type": "response.created", "sequence_number": 0}
        )
        await responses_state.append_stream_event(
            store, "u1", "resp_1", {"type": "response.output_text.delta", "sequence_number": 1}
        )
        events = await responses_state.read_stream_events_after(store, "u1", "resp_1", -1)
        assert [e["sequence_number"] for e in events] == [0, 1]

    @pytest.mark.asyncio
    async def test_read_after_sequence_excludes_seen(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 1})
        events = await responses_state.read_stream_events_after(store, "u1", "resp_1", 0)
        assert [e["sequence_number"] for e in events] == [1]

    @pytest.mark.asyncio
    async def test_read_absent_buffer_returns_empty(self, store):
        assert await responses_state.read_stream_events_after(store, "u1", "nope", -1) == []

    @pytest.mark.asyncio
    async def test_discard_removes_buffer(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        await responses_state.discard_stream_buffer(store, "u1", "resp_1")
        assert await responses_state.read_stream_events_after(store, "u1", "resp_1", -1) == []
        # idempotent — discarding an absent buffer is not an error
        await responses_state.discard_stream_buffer(store, "u1", "resp_1")

    @pytest.mark.asyncio
    async def test_scoped_by_identity(self, store):
        await responses_state.append_stream_event(store, "u1", "resp_1", {"sequence_number": 0})
        assert await responses_state.read_stream_events_after(store, "u2", "resp_1", -1) == []


class TestStreamBufferTtl:
    def test_defaults_to_600s(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", raising=False)
        assert responses_state.stream_buffer_ttl_seconds() == 600.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "30")
        assert responses_state.stream_buffer_ttl_seconds() == 30.0

    def test_non_positive_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "0")
        assert responses_state.stream_buffer_ttl_seconds() == 600.0

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_STREAM_BUFFER_TTL_S", "soon")
        assert responses_state.stream_buffer_ttl_seconds() == 600.0


class TestTtl:
    def test_defaults_to_30_days(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_TTL_S", raising=False)
        assert responses_state.ttl_seconds() == 30 * 24 * 60 * 60

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "60")
        assert responses_state.ttl_seconds() == 60

    def test_non_positive_disables_expiry(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "0")
        assert responses_state.ttl_seconds() is None

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "soon")
        assert responses_state.ttl_seconds() == 30 * 24 * 60 * 60

    @pytest.mark.asyncio
    async def test_record_expires(self, monkeypatch):
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "10")
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        await _write(store)
        assert await responses_state.read_async(store, "u1", "resp_1") is not None
        clock["t"] = 1011.0
        assert await responses_state.read_async(store, "u1", "resp_1") is None
