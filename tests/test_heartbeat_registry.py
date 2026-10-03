"""Tests for HeartbeatRegistry: the actor tracking liveness for background
``/v1/responses`` runs, separate from the response snapshot.

``_HeartbeatStore`` is tested directly for TTL behavior; the module-level
wrapper functions are tested against a fake registry handle mimicking Ray's
``.remote()`` dispatch, same pattern as `test_disconnect_registry.py`.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from ray.exceptions import RayActorError

from modelship.openai.state import responses as responses_state
from modelship.openai.state.responses import HeartbeatRegistry, _HeartbeatStore
from modelship.state import MemoryStoreActor

_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class
_Registry = HeartbeatRegistry.__ray_metadata__.modified_class


class TestHeartbeatStore:
    def test_heartbeat_then_is_alive(self):
        store = _HeartbeatStore(ttl_seconds=300.0)
        store.heartbeat("k1", "req-1")
        assert store.is_alive("k1") is True

    def test_is_alive_false_for_unknown_key(self):
        store = _HeartbeatStore(ttl_seconds=300.0)
        assert store.is_alive("nope") is False

    def test_req_id_returns_what_was_set(self):
        store = _HeartbeatStore(ttl_seconds=300.0)
        store.heartbeat("k1", "req-1")
        assert store.req_id("k1") == "req-1"

    def test_req_id_none_for_unknown_key(self):
        store = _HeartbeatStore(ttl_seconds=300.0)
        assert store.req_id("nope") is None

    def test_evicts_after_ttl(self):
        clock = {"t": 1000.0}
        store = _HeartbeatStore(ttl_seconds=300.0, now=lambda: clock["t"])

        store.heartbeat("k1", "req-1")
        assert store.is_alive("k1") is True

        clock["t"] += 299.0  # just inside the window
        assert store.is_alive("k1") is True

        clock["t"] += 2.0  # now past the 300s deadline
        assert store.is_alive("k1") is False
        assert store.req_id("k1") is None

    def test_heartbeat_refreshes_the_deadline(self):
        clock = {"t": 0.0}
        store = _HeartbeatStore(ttl_seconds=10.0, now=lambda: clock["t"])

        store.heartbeat("k1", "req-1")
        clock["t"] += 8.0
        store.heartbeat("k1", "req-1")  # refresh before expiry
        clock["t"] += 8.0  # 16s since the first heartbeat, but only 8s since the refresh
        assert store.is_alive("k1") is True

    def test_heartbeat_sweeps_expired_entries(self):
        clock = {"t": 0.0}
        store = _HeartbeatStore(ttl_seconds=10.0, now=lambda: clock["t"])

        store.heartbeat("stale", "req-stale")
        clock["t"] += 11.0  # "stale" is now expired
        store.heartbeat("fresh", "req-fresh")  # heartbeat() sweeps expired entries

        assert "stale" not in store._entries
        assert store.is_alive("fresh") is True


class TestHeartbeatRegistryActor:
    @pytest.mark.asyncio
    async def test_an_unknown_key_is_alive_while_the_registry_is_younger_than_its_ttl(self):
        reg = _Registry(30.0)
        assert await reg.is_alive("never-seen") is True

    @pytest.mark.asyncio
    async def test_an_unknown_key_is_stale_once_the_registry_outlives_its_ttl(self):
        reg = _Registry(30.0)
        reg._started -= 31.0
        assert await reg.is_alive("never-seen") is False

    @pytest.mark.asyncio
    async def test_a_live_key_is_alive_on_an_old_registry(self):
        reg = _Registry(30.0)
        reg._started -= 31.0
        await reg.heartbeat("k1", "req-1")
        assert await reg.is_alive("k1") is True


class TestGetHeartbeatRegistry:
    def test_creates_the_registry_on_the_head(self, monkeypatch):
        monkeypatch.setattr(responses_state, "_heartbeat_registry", None)
        options = MagicMock()
        with patch.object(HeartbeatRegistry, "options", options):
            responses_state.get_heartbeat_registry()
        assert options.call_args.kwargs["resources"] == {"node:__internal_head__": 0.001}

    @pytest.mark.asyncio
    async def test_a_dead_registry_drops_the_cached_handle(self, monkeypatch):
        dead = MagicMock()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_exception(RayActorError())
        dead.is_alive.remote.return_value = fut
        monkeypatch.setattr(responses_state, "_heartbeat_registry", dead)

        with pytest.raises(RayActorError):
            await responses_state.is_alive("u1", "resp_1")
        assert responses_state._heartbeat_registry is None


class _FakeHeartbeatRegistry:
    """Stand-in for the HeartbeatRegistry actor handle, backed by a real
    _HeartbeatStore so wrapper-function tests exercise real TTL/keying logic."""

    def __init__(self, ttl_seconds: float = 300.0):
        self._store = _HeartbeatStore(ttl_seconds)
        self.heartbeat = self._Method(self._heartbeat)
        self.is_alive = self._Method(self._is_alive)
        self.req_id = self._Method(self._req_id)

    async def _heartbeat(self, key, req_id):
        self._store.heartbeat(key, req_id)

    async def _is_alive(self, key):
        return self._store.is_alive(key)

    async def _req_id(self, key):
        return self._store.req_id(key)

    class _Method:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)


@pytest.fixture
def fake_registry():
    reg = _FakeHeartbeatRegistry()
    with patch("modelship.openai.state.responses.get_heartbeat_registry", return_value=reg):
        yield reg


class TestWrapperFunctions:
    @pytest.mark.asyncio
    async def test_heartbeat_then_is_alive(self, fake_registry):
        await responses_state.heartbeat("u1", "resp_1", "req-1")
        assert await responses_state.is_alive("u1", "resp_1") is True

    @pytest.mark.asyncio
    async def test_is_alive_false_when_never_heartbeat(self, fake_registry):
        assert await responses_state.is_alive("u1", "resp_nope") is False

    @pytest.mark.asyncio
    async def test_req_id_for_returns_the_seeded_req_id(self, fake_registry):
        await responses_state.heartbeat("u1", "resp_1", "req-1")
        assert await responses_state.req_id_for("u1", "resp_1") == "req-1"

    @pytest.mark.asyncio
    async def test_req_id_for_none_when_never_heartbeat(self, fake_registry):
        assert await responses_state.req_id_for("u1", "resp_nope") is None

    @pytest.mark.asyncio
    async def test_keys_are_scoped_by_identity(self, fake_registry):
        # Same response_id, different identity: must not collide, matching the
        # snapshot store's own identity-scoping discipline.
        await responses_state.heartbeat("u1", "resp_1", "req-u1")
        assert await responses_state.is_alive("u2", "resp_1") is False


class TestHeartbeatNeverTouchesTheSnapshot:
    """A heartbeat refresh calls a different actor entirely, so it cannot race
    or regress the response snapshot's terminal status."""

    @pytest.mark.asyncio
    async def test_heartbeat_after_a_terminal_write_leaves_the_snapshot_untouched(self, fake_registry):
        store = _MemoryStore()
        await responses_state.write_background(
            store, "u1", "resp_1", response={"id": "resp_1", "status": "in_progress", "output": []}, input_items=[]
        )
        terminal = {"id": "resp_1", "status": "cancelled", "output": []}
        await responses_state.write_terminal_if_not_terminal(store, "u1", "resp_1", response=terminal)

        # A heartbeat tick landing after the terminal write touches only the registry.
        await responses_state.heartbeat("u1", "resp_1", "req-1")

        snapshot = await responses_state.read_async(store, "u1", "resp_1")
        assert snapshot["response"]["status"] == "cancelled"
        assert "_mship" not in snapshot
