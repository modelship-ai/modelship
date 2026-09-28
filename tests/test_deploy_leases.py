"""The deploy-lease holder's side: waiting for the node's lease."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ray.exceptions import ActorUnavailableError

from modelship.infer import deploy_coordinator, deploy_leases
from modelship.infer.deploy_leases import DeployLeaseError, wait_for_deploy_lease
from modelship.state import MemoryStoreActor

_Coord = deploy_coordinator.DeployCoordinator.__ray_metadata__.modified_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class

_HOLDER = {"app": "g.qwen-1", "deployment": "ModelDeployment", "replica": "r1", "label": "qwen/r1"}


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(deploy_coordinator, "configure_logging", lambda: None)


async def _leases():
    leases = _Coord()
    leases._lease_checker.cancel()
    leases._store = _MemoryStore()
    await leases._leases_loaded
    return leases


class _FakeHandle:
    """Stands in for the actor handle: `.remote()` calls resolve as awaited futures."""

    def __init__(self, leases):
        self._leases = leases
        self.calls = []

    def __getattr__(self, name):
        method = getattr(self._leases, name)
        remote = MagicMock()
        remote.remote = lambda *args: asyncio.ensure_future(self._record(name, method, args))
        return remote

    async def _record(self, name, method, args):
        self.calls.append((name, args))
        return await method(*args)


@pytest.mark.asyncio
class TestWaitForDeployLease:
    @pytest.fixture(autouse=True)
    def _no_ray(self, monkeypatch):
        ctx = MagicMock()
        ctx.get_node_id.return_value = "node-a"
        monkeypatch.setattr(deploy_leases.ray, "get_runtime_context", lambda: ctx)
        replica = SimpleNamespace(
            app_name="g.qwen-1", deployment="ModelDeployment", replica_id=SimpleNamespace(unique_id="r1")
        )
        monkeypatch.setattr(deploy_leases.serve, "get_replica_context", lambda: replica)
        monkeypatch.setattr(deploy_leases, "POLL_SECONDS", 0.01)

    async def test_asks_for_the_nodes_lease_as_its_replica(self, monkeypatch):
        leases = await _leases()
        handle = _FakeHandle(leases)
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: handle)
        await wait_for_deploy_lease("qwen")
        assert handle.calls == [("acquire", ("node-a", _HOLDER))]
        assert await leases.acquire("node-a", _HOLDER | {"replica": "r2"}) == "held by qwen/r1"

    async def test_waits_while_the_node_is_held(self, monkeypatch):
        leases = await _leases()
        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", lambda: _FakeHandle(leases))
        incumbent = _HOLDER | {"replica": "r0", "label": "other/r0"}
        await leases.acquire("node-a", incumbent)
        waiting = asyncio.ensure_future(wait_for_deploy_lease("qwen"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        await leases._release("node-a", deploy_coordinator._Lease(**incumbent))
        await asyncio.wait_for(waiting, 1)

    async def test_an_unreachable_deploy_coordinator_is_retried_then_raises(self, monkeypatch):
        lookups = []

        def unreachable():
            handle = MagicMock()
            handle.acquire.remote.side_effect = ActorUnavailableError("gone", None)
            lookups.append(handle)
            return handle

        monkeypatch.setattr(deploy_leases, "get_or_create_coordinator", unreachable)
        with pytest.raises(DeployLeaseError, match="unreachable"):
            await wait_for_deploy_lease("qwen")
        assert len(lookups) == deploy_leases._MAX_LOOKUPS
