"""DownloadLeases, driven directly: grants, renewals, reaping, and idle exit."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from modelship.infer import download_leases
from modelship.infer.download_leases import LEASE_SECONDS, DownloadLeases
from modelship.infer.sources import HfSource

_Leases = DownloadLeases.__ray_metadata__.modified_class
_SOURCE = HfSource("org/repo", "a" * 40, "model.gguf", None, None, None)


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures its own process's logging; here that's pytest's, for every later test
    monkeypatch.setattr(download_leases, "configure_logging", lambda: None)


def _fresh():
    leases = _Leases()
    leases._reaper.cancel()
    return leases


def _open():
    leases = _fresh()
    leases._grants_from = 0.0
    return leases


@pytest.fixture
def cleanup(monkeypatch):
    """Stands in for the cleanup task; the test resolves `cleanup.result`."""
    stub = MagicMock()

    def dispatch(node_id, source):
        stub.result = asyncio.get_running_loop().create_future()
        return stub.result

    stub.side_effect = dispatch
    monkeypatch.setattr(download_leases, "remove_leftovers_on", stub)
    return stub


@pytest.mark.asyncio
class TestGrants:
    async def test_refused_for_a_lease_period_after_start(self):
        leases = _fresh()
        assert await leases.acquire("k", _SOURCE, "a", "node") == "lease service starting"
        leases._grants_from = time.monotonic()
        assert await leases.acquire("k", _SOURCE, "a", "node") is None

    async def test_one_holder_per_key(self):
        leases = _open()
        assert await leases.acquire("k", _SOURCE, "a", "node") is None
        assert await leases.acquire("k", _SOURCE, "b", "node") == "held by a"
        assert await leases.acquire("other", _SOURCE, "b", "node") is None

    async def test_renew_extends_only_the_holders_lease(self):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node")
        leases._leases["k"] = leases._leases["k"]._replace(expires_at=0.0)
        assert await leases.renew("k", "a")
        assert leases._leases["k"].expires_at > time.monotonic() + LEASE_SECONDS - 1
        assert not await leases.renew("k", "b")
        assert not await leases.renew("unknown", "a")

    async def test_release_frees_only_for_the_holder(self):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node")
        await leases.release("k", "b")
        assert await leases.acquire("k", _SOURCE, "b", "node") == "held by a"
        await leases.release("k", "a")
        assert await leases.acquire("k", _SOURCE, "b", "node") is None


@pytest.mark.asyncio
class TestReaping:
    async def test_expired_lease_is_cleaned_on_its_node_before_the_key_frees(self, cleanup, caplog):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node-a")
        with caplog.at_level("WARNING"):
            leases._reap(time.monotonic() + LEASE_SECONDS + 1)
        await asyncio.sleep(0)
        assert "expired without release" in caplog.text
        cleanup.assert_called_once_with("node-a", _SOURCE)

        assert await leases.acquire("k", _SOURCE, "b", "node-b") == "cleaning up after a"
        assert not await leases.renew("k", "a")
        await leases.release("k", "a")
        assert "k" in leases._leases

        cleanup.result.set_result(2)
        await asyncio.sleep(0)
        assert await leases.acquire("k", _SOURCE, "b", "node-b") is None

    async def test_failed_cleanup_still_frees_the_key(self, cleanup):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node-a")
        leases._reap(time.monotonic() + LEASE_SECONDS + 1)
        await asyncio.sleep(0)
        cleanup.result.set_exception(RuntimeError("node gone"))
        await asyncio.sleep(0)
        assert await leases.acquire("k", _SOURCE, "b", "node-b") is None

    async def test_pending_cleanup_keeps_the_key(self, cleanup):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node-a")
        leases._reap(time.monotonic() + LEASE_SECONDS + 1)
        await asyncio.sleep(0)
        leases._reap(time.monotonic() + 10 * LEASE_SECONDS)
        await asyncio.sleep(0)
        cleanup.assert_called_once()
        assert await leases.acquire("k", _SOURCE, "b", "node-b") == "cleaning up after a"

    async def test_renewed_lease_is_not_reaped(self, cleanup):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node-a")
        leases._reap(time.monotonic() + LEASE_SECONDS - 1)
        await asyncio.sleep(0)
        cleanup.assert_not_called()


@pytest.fixture
def kill(monkeypatch):
    context = MagicMock()
    monkeypatch.setattr(download_leases.ray, "get_runtime_context", lambda: context)
    kill = MagicMock()
    monkeypatch.setattr(download_leases.ray, "kill", kill)
    kill.self_handle = context.current_actor
    return kill


@pytest.mark.asyncio
class TestIdleExit:
    async def test_idle_actor_kills_itself_once(self, kill):
        leases = _open()
        leases._last_call = 0.0
        leases._reap(time.monotonic())
        leases._reap(time.monotonic())
        kill.assert_called_once_with(kill.self_handle, no_restart=True)
        assert await leases.acquire("k", _SOURCE, "a", "node") == "lease service shutting down"

    async def test_not_idle_while_a_lease_is_held(self, kill):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node")
        leases._last_call = 0.0
        leases._reap(time.monotonic())
        kill.assert_not_called()

    async def test_not_idle_right_after_a_call(self, kill):
        leases = _open()
        await leases.acquire("k", _SOURCE, "a", "node")
        await leases.release("k", "a")
        leases._reap(time.monotonic())
        kill.assert_not_called()
