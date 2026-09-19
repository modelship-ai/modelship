"""Tests for the state-store URI factory and the memory/redis backends — sync and
async paths, TTL, prefix listing, and availability-vs-absent semantics.
"""

from unittest.mock import MagicMock

import pytest
from ray import exceptions as ray_exceptions

from modelship.state import (
    MemoryStateStore,
    MemoryStoreActor,
    StateStoreUnavailableError,
    get_state_store,
    state_store_from_uri,
)
from modelship.state import memory as memory_module
from modelship.state.redis import RedisStateStore

# The plain class behind @ray.remote, tested in-process without a Ray cluster.
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


def _fake_redis_store():
    """A RedisStateStore whose sync + async clients share one fakeredis server, so
    a value written via one path is visible from the other."""
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    s = RedisStateStore.__new__(RedisStateStore)
    s._url = "redis://fake"
    s._sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    s._async_client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return s


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return _MemoryStore()
    return _fake_redis_store()


class TestBackends:
    """Behaviour shared by every backend (parametrized via the `store` fixture)."""

    def test_set_get_roundtrip(self, store):
        store.set("a/b", {"x": [1, 2]})
        assert store.get("a/b") == {"x": [1, 2]}

    def test_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_delete(self, store):
        store.set("k", {"x": 1})
        store.delete("k")
        assert store.get("k") is None
        store.delete("k")  # idempotent

    def test_list_prefix(self, store):
        store.set("responses/u1/a", {"n": 1})
        store.set("responses/u1/b", {"n": 2})
        store.set("responses/u2/c", {"n": 3})
        store.set("responses/u10/c", {"n": 5})
        store.set("other/x", {"n": 4})
        # "responses/u1" must match only its own segment, not the sibling "u10" whose
        # name happens to share "u1" as a string prefix.
        assert sorted(store.list("responses/u1")) == ["responses/u1/a", "responses/u1/b"]
        assert sorted(store.list("responses")) == [
            "responses/u1/a",
            "responses/u1/b",
            "responses/u10/c",
            "responses/u2/c",
        ]

    def test_list_prefix_trailing_slash(self, store):
        # A trailing slash on the prefix must not turn the boundary check into a
        # literal "//" that can never match.
        store.set("responses/u1/a", {"n": 1})
        store.set("responses/u1/b", {"n": 2})
        assert sorted(store.list("responses/u1/")) == ["responses/u1/a", "responses/u1/b"]

    @pytest.mark.asyncio
    async def test_async_mirrors_sync(self, store):
        await store.set_async("k", {"x": 1})
        assert await store.get_async("k") == {"x": 1}
        assert await store.list_async("k") == ["k"]
        # cross-path visibility: a sync write is seen by an async read
        store.set("k2", {"y": 2})
        assert await store.get_async("k2") == {"y": 2}
        await store.delete_async("k")
        assert await store.get_async("k") is None

    @pytest.mark.asyncio
    async def test_append_then_read_from_roundtrips_in_order(self, store):
        for i in range(3):
            await store.append_async("log/a", {"sequence_number": i, "n": i * 10})
        assert await store.read_from_async("log/a") == [
            {"sequence_number": 0, "n": 0},
            {"sequence_number": 1, "n": 10},
            {"sequence_number": 2, "n": 20},
        ]

    @pytest.mark.asyncio
    async def test_read_from_excludes_after_sequence_and_below(self, store):
        for i in range(3):
            await store.append_async("log/a", {"sequence_number": i})
        assert await store.read_from_async("log/a", after_sequence=0) == [
            {"sequence_number": 1},
            {"sequence_number": 2},
        ]
        assert await store.read_from_async("log/a", after_sequence=2) == []

    @pytest.mark.asyncio
    async def test_read_from_absent_key_returns_empty(self, store):
        assert await store.read_from_async("log/nope") == []

    @pytest.mark.asyncio
    async def test_append_is_isolated_per_key(self, store):
        await store.append_async("log/a", {"sequence_number": 0})
        await store.append_async("log/b", {"sequence_number": 0})
        assert len(await store.read_from_async("log/a")) == 1
        assert len(await store.read_from_async("log/b")) == 1


class TestMemoryStoreActor:
    """The dict logic that lives inside MemoryStoreActor, exercised via the plain
    class (no Ray cluster needed)."""

    def test_isolates_stored_value_from_caller_mutation(self):
        store = _MemoryStore()
        payload = {"x": 1}
        store.set("k", payload)
        payload["x"] = 999
        assert store.get("k") == {"x": 1}
        got = store.get("k")
        assert isinstance(got, dict)
        got["x"] = 7
        assert store.get("k") == {"x": 1}

    def test_ttl_expires(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.set("k", {"x": 1}, ttl_seconds=10)
        assert store.get("k") == {"x": 1}
        clock["t"] = 1011.0
        assert store.get("k") is None
        assert store.list("k") == []

    def test_append_max_len_bounds_to_trailing_entries(self):
        store = _MemoryStore()
        for i in range(5):
            store.append("log/a", {"sequence_number": i}, max_len=3)
        assert store.read_from("log/a") == [{"sequence_number": 2}, {"sequence_number": 3}, {"sequence_number": 4}]

    def test_append_ttl_expires_like_set(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.append("log/a", {"sequence_number": 0}, ttl_seconds=10)
        assert store.read_from("log/a") == [{"sequence_number": 0}]
        clock["t"] = 1011.0
        assert store.read_from("log/a") == []

    def test_append_isolates_stored_event_from_caller_mutation(self):
        store = _MemoryStore()
        event = {"sequence_number": 0, "x": 1}
        store.append("log/a", event)
        event["x"] = 999
        assert store.read_from("log/a") == [{"sequence_number": 0, "x": 1}]


class TestMemoryStoreActorSweep:
    """Expired keys that are never read again are reclaimed by the write-path sweep;
    lazy expiry alone would pin them for the actor's lifetime."""

    def test_sweep_reclaims_key_never_read_again(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.set("stale", {"x": 1}, ttl_seconds=10)

        # Past both the TTL and the sweep interval; the next write sweeps. Reach into
        # _data rather than get()/list(), which would mask the leak via lazy expiry.
        clock["t"] = 1000.0 + 301.0
        store.set("fresh", {"x": 2}, ttl_seconds=10)
        assert "stale" not in store._data
        assert "fresh" in store._data

    def test_sweep_is_throttled(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.set("stale", {"x": 1}, ttl_seconds=10)

        # Expired, but within the sweep interval — the write must not pay for a scan.
        clock["t"] = 1011.0
        store.set("other", {"x": 2})
        assert "stale" in store._data

    def test_sweep_keeps_unexpired_and_no_ttl_keys(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.set("forever", {"x": 1})
        store.set("long", {"x": 2}, ttl_seconds=10_000)

        clock["t"] = 1000.0 + 301.0
        store.set("trigger", {"x": 3})
        assert store.get("forever") == {"x": 1}
        assert store.get("long") == {"x": 2}

    def test_sweep_disabled_by_non_positive_interval(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_SWEEP_INTERVAL_S", "0")
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = _MemoryStore()
        store.set("stale", {"x": 1}, ttl_seconds=10)

        clock["t"] = 1000.0 + 10_000.0
        store.set("other", {"x": 2})
        assert "stale" in store._data

    def test_bad_interval_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_SWEEP_INTERVAL_S", "not-a-number")
        assert memory_module._sweep_interval_s() == 300.0


class TestMemoryStateStoreClient:
    """The MemoryStateStore client: Ray-availability gating, delegation to the
    actor handle, and RayActorError -> StateStoreUnavailableError mapping."""

    def test_construction_is_inert(self, monkeypatch):
        # No ray.is_initialized / get_or_create call until first use.
        monkeypatch.setattr(memory_module.ray, "is_initialized", MagicMock(side_effect=AssertionError))
        MemoryStateStore()

    def test_raises_when_ray_not_initialized(self, monkeypatch):
        monkeypatch.setattr(memory_module.ray, "is_initialized", lambda: False)
        store = MemoryStateStore()
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")

    def test_delegates_get_to_actor_handle(self, monkeypatch):
        monkeypatch.setattr(memory_module.ray, "is_initialized", lambda: True)
        fake_handle = MagicMock()
        fake_handle.get.remote.return_value = "sentinel-ref"
        monkeypatch.setattr(memory_module, "get_or_create_memory_store_actor", lambda: fake_handle)
        monkeypatch.setattr(memory_module.ray, "get", lambda ref: {"x": 1} if ref == "sentinel-ref" else None)

        store = MemoryStateStore()
        assert store.get("k") == {"x": 1}
        fake_handle.get.remote.assert_called_once_with("k")

    def test_actor_error_raises_unavailable_and_drops_cached_handle(self, monkeypatch):
        monkeypatch.setattr(memory_module.ray, "is_initialized", lambda: True)
        fake_handle = MagicMock()
        fake_handle.get.remote.return_value = "ref"
        monkeypatch.setattr(memory_module, "get_or_create_memory_store_actor", lambda: fake_handle)
        monkeypatch.setattr(
            memory_module.ray,
            "get",
            MagicMock(side_effect=ray_exceptions.RayActorError()),
        )

        store = MemoryStateStore()
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")
        assert store._handle is None  # dropped so the next call re-resolves


class TestRedisStateStore:
    def test_ttl_sets_native_expiry(self):
        store = _fake_redis_store()
        store.set("k", {"x": 1}, ttl_seconds=100)
        assert store._sync_client.pttl("modelship/state/k") > 0
        store.set("k2", {"x": 1})
        assert store._sync_client.pttl("modelship/state/k2") == -1  # no expiry

    @pytest.mark.asyncio
    async def test_append_uses_a_native_stream(self):
        store = _fake_redis_store()
        await store.append_async("log/a", {"sequence_number": 0, "type": "x"})
        await store.append_async("log/a", {"sequence_number": 1, "type": "y"})
        assert store._sync_client.type("modelship/state/log/a") == "stream"
        assert await store.read_from_async("log/a") == [
            {"sequence_number": 0, "type": "x"},
            {"sequence_number": 1, "type": "y"},
        ]

    @pytest.mark.asyncio
    async def test_append_ttl_sets_native_expiry(self):
        store = _fake_redis_store()
        await store.append_async("log/a", {"sequence_number": 0}, ttl_seconds=100)
        assert store._sync_client.pttl("modelship/state/log/a") > 0

    def test_corrupt_value_treated_as_missing(self):
        store = _fake_redis_store()
        store._sync_client.set("modelship/state/k", "not json{")
        assert store.get("k") is None

    def test_connection_error_raises_unavailable(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        class Boom:
            def get(self, *a, **k):
                raise RedisConnectionError("down")

        store = RedisStateStore.__new__(RedisStateStore)
        store._url = "redis://fake"
        store._sync_client = Boom()
        store._async_client = None
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")


class TestStateStoreFromUri:
    # state_store_from_uri wraps the chosen backend in an instrumented proxy; the
    # concrete store the URI selected is exposed as `.inner`.
    def test_memory_scheme(self):
        assert isinstance(state_store_from_uri("memory://").inner, MemoryStateStore)

    def test_bare_value_treated_as_scheme(self):
        assert isinstance(state_store_from_uri("memory").inner, MemoryStateStore)

    def test_memory_scheme_rejects_host(self):
        with pytest.raises(ValueError, match="takes no host/path"):
            state_store_from_uri("memory://foo")

    def test_memory_scheme_rejects_path(self):
        with pytest.raises(ValueError, match="takes no host/path"):
            state_store_from_uri("memory:///foo")

    def test_file_scheme_no_longer_supported(self):
        with pytest.raises(ValueError, match="unknown state-store scheme"):
            state_store_from_uri("file:///tmp/mship-state-test")

    def test_redis_scheme_builds_redis_store(self):
        # Construction is inert (clients are lazy) — no server contacted.
        assert isinstance(state_store_from_uri("redis://cache:6379/0").inner, RedisStateStore)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unknown state-store scheme"):
            state_store_from_uri("postgres://host/db")

    def test_get_state_store_defaults_to_memory(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        assert isinstance(get_state_store().inner, MemoryStateStore)

    def test_get_state_store_reads_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://cache:6379/0")
        assert isinstance(get_state_store().inner, RedisStateStore)
