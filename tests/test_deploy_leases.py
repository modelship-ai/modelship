"""The deploy-lease holder's side: the node and gateway context managers' acquire/release and the renew thread."""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from modelship.infer import deploy_coordinator, deploy_leases
from modelship.infer.deploy_coordinator import gateway_lease_key
from modelship.infer.deploy_leases import DeployLeaseError, deploy_lease, gateway_lease

_Coord = deploy_coordinator.DeployCoordinator.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(deploy_coordinator, "configure_logging", lambda: None)


def _fresh():
    leases = _Coord()
    leases._reaper.cancel()
    leases._grants_from = 0.0
    return leases


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
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)

        async with deploy_lease("qwen"):
            assert await leases.acquire("node-a", "other") is not None
        assert handle.released
        assert await leases.acquire("node-a", "other") is None

    async def test_releases_when_the_block_raises(self, monkeypatch):
        leases = _fresh()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)

        with pytest.raises(RuntimeError):
            async with deploy_lease("qwen"):
                raise RuntimeError("engine init failed")
        assert await leases.acquire("node-a", "other") is None

    async def test_waits_for_the_node_then_proceeds(self, monkeypatch):
        leases = _fresh()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)
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

        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", unreachable)
        with pytest.raises(DeployLeaseError, match="unreachable"):
            async with deploy_lease("qwen"):
                pass

    async def test_a_lost_lease_is_reported_as_loading_anyway(self, monkeypatch):
        handle = _FakeHandle(_fresh())
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)
        renewing = []
        monkeypatch.setattr(deploy_leases, "_renew_in_thread", lambda *args: renewing.append(args) or threading.Event())
        async with deploy_lease("qwen"):
            pass
        assert renewing[0][3] == "qwen: lost the deploy lease on this node; loading anyway"


class _Ref:
    """Stands in for an ObjectRef: awaitable, and what the patched ray.get resolves."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        yield from ()
        return self.value


class TestGatewayLease:
    @pytest.fixture
    def handle(self, monkeypatch):
        handle = MagicMock()
        handle.acquire.remote.return_value = _Ref(None)
        handle.write_effective.remote.return_value = _Ref(True)
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)
        monkeypatch.setattr(deploy_leases.ray, "get", lambda ref, **kwargs: ref.value)
        monkeypatch.setattr(deploy_leases, "_renew_in_thread", lambda *args: threading.Event())
        monkeypatch.setattr(deploy_leases, "POLL_SECONDS", 0.01)
        return handle

    def test_holds_the_gateways_lease_for_the_block_then_releases(self, handle):
        with gateway_lease("g"):
            handle.release.remote.assert_not_called()
        key, holder = handle.acquire.remote.call_args.args
        assert key == gateway_lease_key("g")
        handle.release.remote.assert_called_once_with(key, holder)

    def test_releases_when_the_block_raises(self, handle):
        with pytest.raises(RuntimeError), gateway_lease("g"):
            raise RuntimeError("merge failed")
        handle.release.remote.assert_called_once()

    def test_waits_while_something_else_holds_it(self, handle, caplog):
        handle.acquire.remote.side_effect = [_Ref("held by another deploy"), _Ref(None)]
        with caplog.at_level("INFO"), gateway_lease("g"):
            pass
        assert handle.acquire.remote.call_count == 2
        assert "Waiting to deploy to gateway 'g' (held by another deploy)" in caplog.text

    def test_each_hold_has_its_own_holder(self, handle):
        with gateway_lease("g"):
            pass
        with gateway_lease("g"):
            pass
        first, second = (call.args[1] for call in handle.acquire.remote.call_args_list)
        assert first != second

    def test_writes_the_effective_config_as_the_holder(self, handle):
        with gateway_lease("g") as lease:
            lease.write_effective([{"name": "m"}])
        holder = handle.acquire.remote.call_args.args[1]
        handle.write_effective.remote.assert_called_once_with("g", holder, [{"name": "m"}])

    def test_a_refused_write_raises(self, handle):
        handle.write_effective.remote.return_value = _Ref(False)
        with pytest.raises(DeployLeaseError, match="lost the deploy lease of gateway 'g'"), gateway_lease("g") as lease:
            lease.write_effective([{"name": "m"}])


class TestRenewThread:
    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch):
        monkeypatch.setattr(deploy_leases, "RENEW_SECONDS", 0.01)
        monkeypatch.setattr(deploy_leases, "LEASE_SECONDS", 0.05)

    def _renew(self, monkeypatch, stop, **mock_kwargs):
        get = MagicMock(**mock_kwargs)
        monkeypatch.setattr(deploy_leases.ray, "get", get)
        deploy_leases._renew_until(MagicMock(), "node-a", "holder", "lost the lease", stop)
        return get

    def test_a_refused_renewal_stops_renewing_and_warns(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            get = self._renew(monkeypatch, threading.Event(), return_value=False)
        assert get.call_count == 1
        assert "lost the lease (renewal refused)" in caplog.text

    def test_failures_are_retried_until_the_lease_would_have_expired(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            get = self._renew(monkeypatch, threading.Event(), side_effect=RuntimeError("actor gone"))
        assert get.call_count > 1
        assert "actor gone" in caplog.text

    def test_a_transient_failure_is_retried_without_a_warning(self, monkeypatch, caplog):
        stop = threading.Event()
        results = iter([RuntimeError("timed out"), True])

        def flaky(*args, **kwargs):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            stop.set()
            return result

        with caplog.at_level("WARNING"):
            get = self._renew(monkeypatch, stop, side_effect=flaky)
        assert get.call_count == 2
        assert caplog.text == ""

    def test_a_refusal_after_the_block_released_is_not_reported(self, monkeypatch, caplog):
        stop = threading.Event()

        def release_then_refuse(*args, **kwargs):
            stop.set()
            return False

        with caplog.at_level("WARNING"):
            self._renew(monkeypatch, stop, side_effect=release_then_refuse)
        assert caplog.text == ""

    def test_renewals_continue_until_stopped(self, monkeypatch):
        monkeypatch.setattr(deploy_leases.ray, "get", MagicMock(return_value=True))
        stop = threading.Event()
        thread = threading.Thread(
            target=deploy_leases._renew_until, args=(MagicMock(), "node-a", "holder", "lost the lease", stop)
        )
        thread.start()
        time.sleep(0.05)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
