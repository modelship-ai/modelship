"""The deploy coordinator, driven directly: leases, effective-config writes, replica-death counts and retiring.
Placement options live in test_actor_placement.py."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from modelship.deploy.effective_config import read_effective
from modelship.infer import deploy_coordinator
from modelship.infer.deploy_coordinator import LEASE_SECONDS, gateway_lease_key
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state import MemoryStoreActor

# The plain class behind @ray.remote; its methods are ordinary coroutines.
_Coord = deploy_coordinator.DeployCoordinator.__ray_metadata__.modified_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(deploy_coordinator, "configure_logging", lambda: None)


def _fresh():
    coord = _Coord()
    coord._reaper.cancel()
    coord._grants_from = 0.0
    coord._store = _MemoryStore()
    return coord


@pytest.mark.asyncio
class TestGrants:
    async def test_granted_immediately_on_a_fresh_actor(self):
        assert await _fresh().acquire("node", "a") is None

    async def test_one_holder_per_node(self):
        coord = _fresh()
        assert await coord.acquire("node", "a") is None
        assert await coord.acquire("node", "b") == "held by a"
        assert await coord.acquire("other-node", "b") is None

    async def test_renew_extends_only_the_holders_lease(self):
        coord = _fresh()
        await coord.acquire("node", "a")
        coord._leases["node"] = coord._leases["node"]._replace(expires_at=0.0)
        assert await coord.renew("node", "a")
        assert coord._leases["node"].expires_at > time.monotonic() + LEASE_SECONDS - 1
        assert not await coord.renew("node", "b")
        assert not await coord.renew("unknown", "a")

    async def test_release_frees_only_for_the_holder(self):
        coord = _fresh()
        await coord.acquire("node", "a")
        await coord.release("node", "b")
        assert await coord.acquire("node", "b") == "held by a"
        await coord.release("node", "a")
        assert await coord.acquire("node", "b") is None


@pytest.mark.asyncio
class TestStartupWindow:
    async def test_a_new_actor_grants_nothing_for_one_lease_period(self):
        coord = _Coord()
        coord._reaper.cancel()
        assert await coord.acquire("node", "a") == "lease service starting"
        assert coord._grants_from >= time.monotonic() + LEASE_SECONDS - 1

    async def test_grants_resume_once_the_window_passes(self):
        coord = _Coord()
        coord._reaper.cancel()
        coord._grants_from = time.monotonic()
        assert await coord.acquire("node", "a") is None

    async def test_an_actor_without_the_window_grants_at_once(self):
        coord = _Coord(startup_window=False)
        coord._reaper.cancel()
        assert await coord.acquire("node", "a") is None


@pytest.mark.asyncio
class TestReaping:
    async def test_expired_lease_frees_the_node(self, caplog):
        coord = _fresh()
        await coord.acquire("node", "a")
        with caplog.at_level("WARNING"):
            coord._reap(time.monotonic() + LEASE_SECONDS + 1)
        assert "expired without release" in caplog.text
        assert await coord.acquire("node", "b") is None

    async def test_renewed_lease_is_not_reaped(self):
        coord = _fresh()
        await coord.acquire("node", "a")
        await coord.renew("node", "a")
        coord._reap(time.monotonic() + LEASE_SECONDS - 1)
        assert await coord.acquire("node", "b") == "held by a"


@pytest.mark.asyncio
class TestWriteEffective:
    async def test_writes_for_the_gateways_holder(self):
        coord = _fresh()
        await coord.acquire(gateway_lease_key("g"), "a")
        assert await coord.write_effective("g", "a", [{"name": "m"}])
        assert read_effective(coord._store, "g") == [{"name": "m"}]

    async def test_refuses_anyone_else(self):
        coord = _fresh()
        await coord.acquire(gateway_lease_key("g"), "a")
        assert not await coord.write_effective("g", "b", [{"name": "m"}])
        assert read_effective(coord._store, "g") == []

    async def test_refuses_once_the_lease_has_expired(self):
        coord = _fresh()
        await coord.acquire(gateway_lease_key("g"), "a")
        coord._reap(time.monotonic() + LEASE_SECONDS + 1)
        assert not await coord.write_effective("g", "a", [{"name": "m"}])
        assert read_effective(coord._store, "g") == []

    async def test_another_gateways_lease_does_not_count(self):
        coord = _fresh()
        await coord.acquire(gateway_lease_key("other"), "a")
        assert not await coord.write_effective("g", "a", [{"name": "m"}])

    async def test_renews_the_lease(self):
        coord = _fresh()
        key = gateway_lease_key("g")
        await coord.acquire(key, "a")
        coord._leases[key] = coord._leases[key]._replace(expires_at=0.0)
        await coord.write_effective("g", "a", [{"name": "m"}])
        assert coord._leases[key].expires_at > time.monotonic() + LEASE_SECONDS - 1


@pytest.mark.asyncio
class TestReplicaDeathCounting:
    async def test_deaths_below_the_limit_do_not_retire(self):
        coord = _fresh()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
                await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_not_called()

    async def test_the_limiting_death_retires(self):
        coord = _fresh()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA):
                await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_awaited_once_with("qwen-aaaa")

    async def test_the_limit_scales_with_the_replica_count(self):
        coord = _fresh()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA * 4):
                await coord.report_replica_death("qwen-aaaa", 4, "engine died")
        assert retire.await_count == 1

    async def test_the_count_is_not_time_windowed(self):
        coord = _fresh()
        coord._deaths["qwen-aaaa"] = deploy_coordinator._DEATHS_PER_REPLICA - 1
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_awaited_once()

    async def test_deployments_are_counted_separately(self):
        coord = _fresh()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for name in ("qwen-aaaa", "kokoro-bbbb"):
                for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
                    await coord.report_replica_death(name, 1, "engine died")
        retire.assert_not_called()


_APP = ModelshipModelConfig.model_validate(
    {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "llama_server"}
).deployment_name("g")


@pytest.mark.asyncio
class TestRetire:
    async def test_deletes_the_app_under_its_gateways_lease(self):
        coord = _fresh()
        held = []
        with patch(
            "modelship.deploy.removal.serve.delete", lambda name: held.append(coord._leases.get(gateway_lease_key("g")))
        ):
            await coord._retire(_APP)
        assert held[0].holder == f"deploy coordinator retiring {_APP}"
        assert gateway_lease_key("g") not in coord._leases

    async def test_waits_while_a_deploy_holds_the_gateways_lease(self, monkeypatch):
        monkeypatch.setattr(deploy_coordinator, "POLL_SECONDS", 0.01)
        coord = _fresh()
        await coord.acquire(gateway_lease_key("g"), "a deploy")
        with patch("modelship.deploy.removal.serve.delete") as delete:
            retiring = asyncio.create_task(coord._retire(_APP))
            await asyncio.sleep(0.05)
            delete.assert_not_called()
            await coord.release(gateway_lease_key("g"), "a deploy")
            await asyncio.wait_for(retiring, 1)
        delete.assert_called_once_with(_APP)
