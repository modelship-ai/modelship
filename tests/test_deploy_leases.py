"""DeployLeases, driven directly: grants, renewals, reaping, and the client
context manager's acquire/release."""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from modelship.infer import deploy_leases
from modelship.infer.deploy_leases import LEASE_SECONDS, DeployLeaseError, DeployLeases, deploy_lease

_Leases = DeployLeases.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(deploy_leases, "configure_logging", lambda: None)


def _fresh():
    leases = _Leases()
    leases._reaper.cancel()
    return leases


@pytest.mark.asyncio
class TestGrants:
    async def test_granted_immediately_on_a_fresh_actor(self):
        assert await _fresh().acquire("node", "a") is None

    async def test_one_holder_per_node(self):
        leases = _fresh()
        assert await leases.acquire("node", "a") is None
        assert await leases.acquire("node", "b") == "held by a"
        assert await leases.acquire("other-node", "b") is None

    async def test_renew_extends_only_the_holders_lease(self):
        leases = _fresh()
        await leases.acquire("node", "a")
        leases._leases["node"] = leases._leases["node"]._replace(expires_at=0.0)
        assert await leases.renew("node", "a")
        assert leases._leases["node"].expires_at > time.monotonic() + LEASE_SECONDS - 1
        assert not await leases.renew("node", "b")
        assert not await leases.renew("unknown", "a")

    async def test_release_frees_only_for_the_holder(self):
        leases = _fresh()
        await leases.acquire("node", "a")
        await leases.release("node", "b")
        assert await leases.acquire("node", "b") == "held by a"
        await leases.release("node", "a")
        assert await leases.acquire("node", "b") is None


@pytest.mark.asyncio
class TestReaping:
    async def test_expired_lease_frees_the_node(self, caplog):
        leases = _fresh()
        await leases.acquire("node", "a")
        with caplog.at_level("WARNING"):
            leases._reap(time.monotonic() + LEASE_SECONDS + 1)
        assert "expired without release" in caplog.text
        assert await leases.acquire("node", "b") is None

    async def test_renewed_lease_is_not_reaped(self):
        leases = _fresh()
        await leases.acquire("node", "a")
        await leases.renew("node", "a")
        leases._reap(time.monotonic() + LEASE_SECONDS - 1)
        assert await leases.acquire("node", "b") == "held by a"


class _FakeHandle:
    """Stands in for the actor handle: `.remote()` calls resolve as awaited futures."""

    def __init__(self, leases):
        self._leases = leases
        self.released = []

    def __getattr__(self, name):
        method = getattr(self._leases, name)
        remote = MagicMock()
        remote.remote = lambda *args: asyncio.ensure_future(self._record(name, method, args))
        return remote

    async def _record(self, name, method, args):
        if name == "release":
            self.released.append(args)
        return await method(*args)


@pytest.mark.asyncio
class TestDeployLease:
    @pytest.fixture(autouse=True)
    def _no_ray(self, monkeypatch):
        ctx = MagicMock()
        ctx.get_node_id.return_value = "node-a"
        monkeypatch.setattr(deploy_leases.ray, "get_runtime_context", lambda: ctx)
        monkeypatch.setattr(deploy_leases, "_renew_until", lambda *a: None)
        monkeypatch.setattr(deploy_leases, "POLL_SECONDS", 0.01)

    async def test_holds_the_lease_for_the_block_then_releases(self, monkeypatch):
        leases = _fresh()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_leases", lambda: handle)

        async with deploy_lease("qwen"):
            assert await leases.acquire("node-a", "other") is not None
        assert handle.released
        assert await leases.acquire("node-a", "other") is None

    async def test_releases_when_the_block_raises(self, monkeypatch):
        leases = _fresh()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_leases", lambda: handle)

        with pytest.raises(RuntimeError):
            async with deploy_lease("qwen"):
                raise RuntimeError("engine init failed")
        assert await leases.acquire("node-a", "other") is None

    async def test_waits_for_the_node_then_proceeds(self, monkeypatch):
        leases = _fresh()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_leases", lambda: handle)
        await leases.acquire("node-a", "incumbent")

        async def free_it():
            await asyncio.sleep(0.05)
            await leases.release("node-a", "incumbent")

        freed = asyncio.create_task(free_it())
        async with deploy_lease("qwen"):
            pass
        await freed

    async def test_unreachable_lease_service_is_retryable(self, monkeypatch):
        from ray.exceptions import ActorUnavailableError

        def unreachable():
            handle = MagicMock()
            handle.acquire.remote.side_effect = ActorUnavailableError("gone", None)
            return handle

        monkeypatch.setattr(deploy_leases, "get_or_create_leases", unreachable)
        with pytest.raises(DeployLeaseError, match="unreachable"):
            async with deploy_lease("qwen"):
                pass


class TestRenewThread:
    def test_lost_lease_exits_the_process(self, monkeypatch):
        exited = []

        def fake_exit(code):
            exited.append(code)
            # the real os._exit never returns; SystemExit stands in for that
            raise SystemExit(code)

        monkeypatch.setattr(deploy_leases.os, "_exit", fake_exit)
        monkeypatch.setattr(deploy_leases.ray, "get", MagicMock(return_value=False))
        monkeypatch.setattr(deploy_leases, "RENEW_SECONDS", 0.01)

        stop = threading.Event()
        with pytest.raises(SystemExit):
            deploy_leases._renew_until(MagicMock(), "node-a", "holder", "qwen", stop)
        assert exited == [1]

    def test_stop_ends_the_thread_without_exiting(self, monkeypatch):
        exited = []
        monkeypatch.setattr(deploy_leases.os, "_exit", lambda code: exited.append(code))
        monkeypatch.setattr(deploy_leases.ray, "get", MagicMock(return_value=True))
        monkeypatch.setattr(deploy_leases, "RENEW_SECONDS", 0.01)

        stop = threading.Event()
        thread = threading.Thread(target=deploy_leases._renew_until, args=(MagicMock(), "n", "h", "m", stop))
        thread.start()
        time.sleep(0.05)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not exited
