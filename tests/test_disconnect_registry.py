"""Disconnect propagation through the shared DisconnectRegistry actor.

Ray Serve is mocked out; the registry actor is stood in for by a fake whose
``.remote()`` mimics Ray actor-method dispatch (fire-and-forget for set/clear,
awaitable for is_set) so the keying contract can be exercised without a cluster.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ray.exceptions import RayActorError

from modelship.infer import infer_config
from modelship.infer.infer_config import DisconnectRegistry, RawRequestProxy, RequestWatcher, _DisconnectStore


@pytest.fixture(autouse=True)
def neutralize_request_watcher():
    """Override the conftest stub: this module exercises the real RequestWatcher
    watch loop and DisconnectRegistry interaction directly."""
    yield


class _FakeRegistry:
    """Stand-in for the DisconnectRegistry actor handle."""

    def __init__(self):
        self.disconnected: set[str] = set()
        self.set = self._Method(self._set)
        self.is_set = self._Method(self._is_set)
        self.is_set_many = self._Method(self._is_set_many)
        self.clear = self._Method(self._clear)

    async def _set(self, rid):
        self.disconnected.add(rid)

    async def _is_set(self, rid):
        return rid in self.disconnected

    async def _is_set_many(self, rids):
        return [rid for rid in rids if rid in self.disconnected]

    async def _clear(self, rid):
        self.disconnected.discard(rid)

    class _Method:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return asyncio.ensure_future(self._fn(*args))


class _DeadRegistry:
    """A registry handle whose every method raises RayActorError on await, as a
    dead Ray actor's methods do."""

    def __init__(self):
        self.set = self._Method()
        self.is_set = self._Method()
        self.is_set_many = self._Method()
        self.clear = self._Method()

    class _Method:
        def remote(self, *args):
            fut: asyncio.Future = asyncio.Future()
            fut.set_exception(RayActorError())
            return fut


@pytest.mark.asyncio
async def test_proxies_keyed_independently_on_one_registry():
    reg = _FakeRegistry()
    p1 = RawRequestProxy(reg, {}, "req-1")
    p2 = RawRequestProxy(reg, {}, "req-2")

    await reg.set.remote("req-1")

    assert await p1.is_disconnected() is True
    assert await p2.is_disconnected() is False

    await reg.clear.remote("req-1")
    assert await p1.is_disconnected() is False


@pytest.mark.asyncio
async def test_watcher_sets_on_disconnect_and_stop_leaves_entry_for_ttl():
    """stop() must NOT clear the entry — the deployment side may still need to
    read it; it's left for the registry to TTL-evict, stop() only cancels the watch task."""
    reg = _FakeRegistry()
    raw_request = MagicMock()
    raw_request.is_disconnected = AsyncMock(return_value=True)

    with patch("modelship.infer.infer_config.get_disconnect_registry", return_value=reg):
        watcher = RequestWatcher(raw_request, "req-9", model="m", endpoint="e")
        await watcher._task  # watch loop records the disconnect, then breaks

    assert "req-9" in reg.disconnected

    watcher.stop()
    await asyncio.sleep(0)  # nothing fires, but give any stray task a tick
    assert "req-9" in reg.disconnected  # survives stop() — deployment can still read it


def test_disconnect_store_evicts_after_ttl():
    clock = {"t": 1000.0}
    store = _DisconnectStore(ttl_seconds=300.0, now=lambda: clock["t"])

    store.set("req-1")
    assert store.is_set("req-1") is True

    clock["t"] += 299.0  # just inside the window
    assert store.is_set("req-1") is True

    clock["t"] += 2.0  # now past the 300s deadline
    assert store.is_set("req-1") is False


def test_disconnect_store_set_sweeps_expired_entries():
    clock = {"t": 0.0}
    store = _DisconnectStore(ttl_seconds=10.0, now=lambda: clock["t"])

    store.set("stale")
    clock["t"] += 11.0  # "stale" is now expired
    store.set("fresh")  # set() sweeps expired entries

    assert "stale" not in store._deadlines
    assert store.is_set("fresh") is True


def test_disconnect_store_clear_removes_entry():
    store = _DisconnectStore(ttl_seconds=300.0)
    store.set("req-1")
    store.clear("req-1")
    assert store.is_set("req-1") is False
    store.clear("never-set")  # clearing an absent id is a no-op


def test_disconnect_store_is_set_many_filters_to_disconnected_subset():
    clock = {"t": 0.0}
    store = _DisconnectStore(ttl_seconds=10.0, now=lambda: clock["t"])

    store.set("a")
    store.set("b")

    assert store.is_set_many(["a", "b", "never-set"]) == ["a", "b"]

    clock["t"] += 11.0  # past both deadlines
    assert store.is_set_many(["a", "b", "never-set"]) == []


@pytest.mark.asyncio
async def test_is_disconnected_degrades_and_reresolves_on_actor_death():
    """A dead registry actor must not fail a healthy in-flight request: the proxy
    degrades to 'still connected' and re-resolves the recreated actor for later polls."""
    healthy = _FakeRegistry()
    proxy = RawRequestProxy(_DeadRegistry(), {}, "req-1")

    with patch("modelship.infer.infer_config.get_disconnect_registry", return_value=healthy):
        assert await proxy.is_disconnected() is False  # degraded, not raised

    assert proxy._registry is healthy  # re-resolved to the live actor for later polls


@pytest.mark.asyncio
async def test_watch_reresolves_and_retries_set_on_actor_death():
    """When the registry actor dies, the watcher re-resolves and retries the set so
    the disconnect still lands on the recreated actor."""
    healthy = _FakeRegistry()
    raw_request = MagicMock()
    raw_request.is_disconnected = AsyncMock(return_value=True)

    # First resolve (in __init__) hands back the dead actor; the retry re-resolves to a live one.
    with patch("modelship.infer.infer_config.get_disconnect_registry", side_effect=[_DeadRegistry(), healthy]):
        watcher = RequestWatcher(raw_request, "req-2", model="m", endpoint="e")
        await watcher._task

    assert "req-2" in healthy.disconnected


def test_creates_the_registry_on_the_head(monkeypatch):
    monkeypatch.setattr(infer_config, "_disconnect_registry", None)
    options = MagicMock()
    with patch.object(DisconnectRegistry, "options", options):
        infer_config.get_disconnect_registry()
    assert options.call_args.kwargs["resources"] == {"node:__internal_head__": 0.001}
