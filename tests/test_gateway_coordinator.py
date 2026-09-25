"""The gateway coordinator's passes: each gateway's model table, its generation, the
long-poll API gateway replicas use, and the deletion of unused apps under the gateway's
deploy lease. Exercises the undecorated class in-process, with Serve's status, the state
store and the deploy coordinator faked."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from ray.serve.schema import (
    ApplicationStatus,
    ApplicationStatusOverview,
    DeploymentStatus,
    DeploymentStatusOverview,
    DeploymentStatusTrigger,
)

from modelship.deploy.effective_config import write_effective
from modelship.infer import deploy_coordinator, gateway_coordinator
from modelship.infer.deploy_coordinator import DeployCoordinator, gateway_lease_key
from modelship.infer.gateway_coordinator import GatewayCoordinator
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state import MemoryStoreActor, StateStoreUnavailableError

# The plain classes behind @ray.remote — their async methods are ordinary
# coroutines, so both can be exercised in-process without a Ray cluster.
_Coord = GatewayCoordinator.__ray_metadata__.modified_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class
_DeployCoord = DeployCoordinator.__ray_metadata__.modified_class


def _raw(name: str, **overrides) -> dict:
    return {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server", **overrides}


def _app_name(raw: dict, gateway: str = "gw") -> str:
    return ModelshipModelConfig.model_validate(raw).deployment_name(gateway)


def _app(status=ApplicationStatus.RUNNING, running=1, deployed_at=0.0):
    deployment = DeploymentStatusOverview(
        status=DeploymentStatus.HEALTHY,
        status_trigger=DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED,
        replica_states={"RUNNING": running} if running else {"STARTING": 1},
        message="",
    )
    return ApplicationStatusOverview(
        status=status, message="", last_deployed_time_s=deployed_at, deployments={"d": deployment}
    )


OLD, NEW = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
LOADING = _app(ApplicationStatus.DEPLOYING, running=0, deployed_at=1)


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(gateway_coordinator, "configure_logging", lambda: None)


class _Handle:
    """`handle.method.remote(...)` over the cluster's deploy coordinator."""

    def __init__(self, cluster):
        self._cluster = cluster

    def __getattr__(self, name):
        return SimpleNamespace(remote=lambda *args: getattr(self._cluster.leases, name)(*args))


class _Cluster:
    """The Serve apps and effective configs a gateway coordinator reads, the deploy coordinator
    granting its gateway leases, and the apps it deletes."""

    def __init__(self, monkeypatch):
        self.apps = {"gw": _app()}
        self.store = _MemoryStore()
        self.deleted: list[str] = []
        self._leases = None
        monkeypatch.setattr(gateway_coordinator, "get_state_store", lambda: self.store)
        monkeypatch.setattr(gateway_coordinator.serve, "status", lambda: SimpleNamespace(applications=dict(self.apps)))
        monkeypatch.setattr(gateway_coordinator, "delete_apps_quietly", lambda names: self.deleted.extend(names))
        monkeypatch.setattr(deploy_coordinator, "configure_logging", lambda: None)
        monkeypatch.setattr(gateway_coordinator, "get_or_create_coordinator", lambda: _Handle(self))

    @property
    def leases(self):
        """Built on first use, from within the test's event loop."""
        if self._leases is None:
            self._leases = _DeployCoord(startup_window=False)
            self._leases._reaper.cancel()
        return self._leases

    def configure(self, *raws: dict, gateway: str = "gw") -> None:
        write_effective(self.store, gateway, list(raws))


@pytest.fixture
def cluster(monkeypatch):
    return _Cluster(monkeypatch)


def _coordinator(*watched: str):
    """A gateway coordinator whose passes are run by hand; call from within a running loop."""
    coord = _Coord()
    coord._passes.cancel()
    coord._watched |= set(watched)
    return coord


async def _pass(coord) -> None:
    await coord._compute()
    await asyncio.gather(*coord._deletions)


class TestTables:
    @pytest.mark.asyncio
    async def test_each_model_is_routed_to_its_serving_target(self, cluster):
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("gw")
        assert routing["models"] == {_app_name(NEW): "a"}
        assert routing["expected"] == ["a"]

    @pytest.mark.asyncio
    async def test_an_older_app_serves_until_the_target_can(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): LOADING}
        coord = _coordinator()
        await _pass(coord)
        assert (await coord.get_routing("gw"))["models"] == {_app_name(OLD): "a"}

    @pytest.mark.asyncio
    async def test_a_gateway_is_found_through_its_apps(self, cluster):
        cluster.configure(NEW, gateway="edge")
        cluster.apps |= {"edge": _app(), _app_name(NEW, "edge"): _app()}
        coord = _coordinator()
        await _pass(coord)
        assert coord._routing["edge"].models == {_app_name(NEW, "edge"): "a"}

    @pytest.mark.asyncio
    async def test_a_gateway_missing_from_serve_keeps_its_last_table(self, cluster):
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator("gw")
        await _pass(coord)
        del cluster.apps["gw"]
        del cluster.apps[_app_name(NEW)]
        await _pass(coord)
        assert (await coord.get_routing("gw"))["models"] == {_app_name(NEW): "a"}

    @pytest.mark.asyncio
    async def test_a_gateway_never_computed_gets_an_empty_table(self, cluster):
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("elsewhere")
        assert (routing["models"], routing["expected"]) == ({}, [])

    @pytest.mark.asyncio
    async def test_a_failed_status_read_keeps_the_last_tables(self, cluster, monkeypatch):
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)

        def unavailable():
            raise RuntimeError("controller restarting")

        monkeypatch.setattr(gateway_coordinator.serve, "status", unavailable)
        with pytest.raises(RuntimeError):
            await coord._compute()
        assert (await coord.get_routing("gw"))["models"] == {_app_name(NEW): "a"}


class TestGeneration:
    @pytest.mark.asyncio
    async def test_an_unchanged_table_keeps_the_generation(self, cluster):
        coord = _coordinator("gw")
        await _pass(coord)
        before = (await coord.get_routing("gw"))["generation"]
        await _pass(coord)
        assert (await coord.get_routing("gw"))["generation"] == before

    @pytest.mark.asyncio
    async def test_a_changed_table_advances_it(self, cluster):
        cluster.configure(NEW)
        coord = _coordinator("gw")
        await _pass(coord)
        before = (await coord.get_routing("gw"))["generation"]
        cluster.apps[_app_name(NEW)] = _app()
        await _pass(coord)
        assert (await coord.get_routing("gw"))["generation"] == before + 1

    @pytest.mark.asyncio
    async def test_it_is_per_gateway(self, cluster):
        cluster.apps["edge"] = _app()
        cluster.configure(NEW, gateway="edge")
        coord = _coordinator("gw", "edge")
        await _pass(coord)
        gw, edge = (await coord.get_routing("gw"))["generation"], (await coord.get_routing("edge"))["generation"]
        cluster.apps[_app_name(NEW, "edge")] = _app()
        await _pass(coord)
        assert (await coord.get_routing("gw"))["generation"] == gw
        assert (await coord.get_routing("edge"))["generation"] == edge + 1

    @pytest.mark.asyncio
    async def test_a_new_coordinator_starts_from_the_clock(self, cluster, monkeypatch):
        monkeypatch.setattr(gateway_coordinator.time, "time", lambda: 1000.0)
        coord = _coordinator()
        assert coord._first_generation == 1_000_000


class TestGetRouting:
    @pytest.mark.asyncio
    async def test_waits_for_the_first_pass(self, cluster):
        coord = _coordinator()
        read = asyncio.create_task(coord.get_routing("gw"))
        await asyncio.sleep(0)
        assert not read.done()
        await _pass(coord)
        assert (await read)["models"] == {}

    @pytest.mark.asyncio
    async def test_raises_when_no_pass_completes_in_time(self, cluster, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "_FIRST_PASS_TIMEOUT_S", 0.01)
        coord = _coordinator()
        with pytest.raises(TimeoutError):
            await coord.get_routing("gw")


class TestGetRetiring:
    @pytest.mark.asyncio
    async def test_lists_the_gateways_unused_apps(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        read = asyncio.create_task(coord.get_retiring("gw"))
        await asyncio.sleep(0)
        await _pass(coord)
        assert await read == [_app_name(OLD)]

    @pytest.mark.asyncio
    async def test_answers_from_a_pass_started_after_the_call(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await _pass(coord)
        read = asyncio.create_task(coord.get_retiring("gw"))
        await asyncio.sleep(0)
        assert not read.done()
        del cluster.apps[_app_name(OLD)]
        await _pass(coord)
        assert await read == []

    @pytest.mark.asyncio
    async def test_a_pass_already_running_does_not_answer(self, cluster, monkeypatch):
        release = threading.Event()
        status = gateway_coordinator.serve.status

        def slow_status():
            release.wait(5)
            return status()

        monkeypatch.setattr(gateway_coordinator.serve, "status", slow_status)
        coord = _coordinator()
        running = asyncio.create_task(coord._compute())
        await asyncio.sleep(0)
        read = asyncio.create_task(coord.get_retiring("gw"))
        await asyncio.sleep(0)
        release.set()
        await running
        await asyncio.sleep(0)
        assert not read.done()
        await _pass(coord)
        assert await read == []

    @pytest.mark.asyncio
    async def test_a_failed_pass_does_not_answer(self, cluster, monkeypatch):
        coord = _coordinator()
        read = asyncio.create_task(coord.get_retiring("gw"))
        await asyncio.sleep(0)

        def unavailable():
            raise RuntimeError("controller restarting")

        status = gateway_coordinator.serve.status
        monkeypatch.setattr(gateway_coordinator.serve, "status", unavailable)
        with pytest.raises(RuntimeError):
            await coord._compute()
        await asyncio.sleep(0)
        assert not read.done()
        monkeypatch.setattr(gateway_coordinator.serve, "status", status)
        await _pass(coord)
        assert await read == []

    @pytest.mark.asyncio
    async def test_a_gateway_never_computed_has_nothing_retiring(self, cluster):
        coord = _coordinator()
        read = asyncio.create_task(coord.get_retiring("elsewhere"))
        await asyncio.sleep(0)
        await _pass(coord)
        assert await read == []


class TestWaitForChange:
    @pytest.mark.asyncio
    async def test_returns_at_once_when_the_generation_differs(self, cluster):
        coord = _coordinator()
        await _pass(coord)
        current = (await coord.get_routing("gw"))["generation"]
        assert await coord.wait_for_change("gw", 0, timeout=5) == current

    @pytest.mark.asyncio
    async def test_blocks_then_wakes_on_a_change(self, cluster):
        cluster.configure(NEW)
        coord = _coordinator()
        await _pass(coord)
        current = (await coord.get_routing("gw"))["generation"]
        waiter = asyncio.create_task(coord.wait_for_change("gw", current, timeout=5))
        await asyncio.sleep(0)
        assert not waiter.done()
        cluster.apps[_app_name(NEW)] = _app()
        await _pass(coord)
        assert await asyncio.wait_for(waiter, 1) == current + 1

    @pytest.mark.asyncio
    async def test_times_out_returning_the_same_generation(self, cluster):
        coord = _coordinator()
        await _pass(coord)
        current = (await coord.get_routing("gw"))["generation"]
        assert await coord.wait_for_change("gw", current, timeout=0.01) == current

    @pytest.mark.asyncio
    async def test_reports_no_change_before_the_first_pass(self, cluster):
        coord = _coordinator()
        assert await coord.wait_for_change("gw", 7, timeout=0.01) == 7


class TestCleanup:
    @pytest.fixture(autouse=True)
    def _no_grace(self, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "_UNUSED_GRACE_SECONDS", 0)

    @pytest.mark.asyncio
    async def test_the_older_app_is_deleted_once_the_target_serves(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await _pass(coord)
        assert cluster.deleted == [_app_name(OLD)]

    @pytest.mark.asyncio
    async def test_an_older_app_still_serving_is_kept(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): LOADING}
        coord = _coordinator()
        await _pass(coord)
        assert cluster.deleted == []

    @pytest.mark.asyncio
    async def test_a_dropped_models_app_is_deleted(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(NEW): _app(), _app_name(_raw("b")): _app()}
        coord = _coordinator()
        await _pass(coord)
        assert cluster.deleted == [_app_name(_raw("b"))]

    @pytest.mark.asyncio
    async def test_nothing_is_deleted_within_the_grace_period(self, cluster, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "_UNUSED_GRACE_SECONDS", 60)
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await _pass(coord)
        assert cluster.deleted == []

    @pytest.mark.asyncio
    async def test_an_app_used_again_starts_its_grace_period_over(self, cluster, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "_UNUSED_GRACE_SECONDS", 60)
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await _pass(coord)
        assert _app_name(OLD) in coord._unused_since
        cluster.apps[_app_name(NEW)] = LOADING
        await _pass(coord)
        assert _app_name(OLD) not in coord._unused_since

    @pytest.mark.asyncio
    async def test_a_delete_in_progress_is_not_repeated(self, cluster, monkeypatch):
        release, calls = threading.Event(), []

        def slow_delete(names):
            calls.extend(names)
            release.wait(5)

        monkeypatch.setattr(gateway_coordinator, "delete_apps_quietly", slow_delete)
        cluster.configure(NEW)
        cluster.apps |= {_app_name(NEW): _app(), _app_name(_raw("b")): _app()}
        coord = _coordinator()
        await coord._compute()
        await coord._compute()
        release.set()
        await asyncio.gather(*coord._deletions)
        assert calls == [_app_name(_raw("b"))]

    @pytest.mark.asyncio
    async def test_a_missing_effective_config_deletes_nothing(self, cluster):
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await _pass(coord)
        assert cluster.deleted == []
        assert (await coord.get_routing("gw"))["models"] == {_app_name(NEW): "a"}

    @pytest.mark.asyncio
    async def test_an_unreadable_effective_config_deletes_nothing(self, cluster, monkeypatch, caplog):
        async def unavailable(key):
            raise StateStoreUnavailableError("redis down")

        cluster.configure(NEW)
        cluster.apps |= {_app_name(NEW): _app(), _app_name(_raw("b")): _app()}
        monkeypatch.setattr(cluster.store, "get_async", unavailable)
        coord = _coordinator()
        with caplog.at_level("WARNING"):
            await _pass(coord)
            await _pass(coord)
        assert cluster.deleted == []
        assert caplog.messages.count("Could not read the effective config of gateway gw") == 1


class TestCleanupUnderTheGatewayLease:
    @pytest.fixture(autouse=True)
    def _no_grace(self, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "_UNUSED_GRACE_SECONDS", 0)

    @pytest.fixture
    def dropped(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(NEW): _app(), _app_name(_raw("b")): _app()}
        return _app_name(_raw("b"))

    @pytest.mark.asyncio
    async def test_the_delete_holds_the_gateways_lease_then_releases_it(self, cluster, dropped, monkeypatch):
        held = []
        monkeypatch.setattr(
            gateway_coordinator,
            "delete_apps_quietly",
            lambda names: held.append(cluster.leases._leases.get(gateway_lease_key("gw"))),
        )
        await _pass(_coordinator())
        assert held[0].holder == f"gateway coordinator deleting {dropped}"
        assert gateway_lease_key("gw") not in cluster.leases._leases

    @pytest.mark.asyncio
    async def test_a_delete_waits_while_a_deploy_holds_the_lease(self, cluster, dropped):
        coord = _coordinator()
        await cluster.leases.acquire(gateway_lease_key("gw"), "a deploy")
        await _pass(coord)
        assert cluster.deleted == []
        assert dropped in coord._unused_since
        await cluster.leases.release(gateway_lease_key("gw"), "a deploy")
        await _pass(coord)
        assert cluster.deleted == [dropped]

    @pytest.mark.asyncio
    async def test_an_app_targeted_again_before_its_delete_is_kept(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): _app(deployed_at=1)}
        coord = _coordinator()
        await coord._compute()
        cluster.configure(OLD)
        await asyncio.gather(*coord._deletions)
        assert cluster.deleted == []

    @pytest.mark.asyncio
    async def test_the_lease_is_renewed_during_a_long_delete(self, cluster, dropped, monkeypatch):
        monkeypatch.setattr(gateway_coordinator, "RENEW_SECONDS", 0.01)
        monkeypatch.setattr(gateway_coordinator, "delete_apps_quietly", lambda names: time.sleep(0.1))
        renew, renewals = cluster.leases.renew, []

        async def counting(key, holder):
            renewals.append(holder)
            return await renew(key, holder)

        cluster.leases.renew = counting
        await _pass(_coordinator())
        assert renewals and set(renewals) == {f"gateway coordinator deleting {dropped}"}

    @pytest.mark.asyncio
    async def test_an_unreachable_deploy_coordinator_deletes_nothing(self, cluster, dropped, monkeypatch, caplog):
        def unreachable():
            raise RuntimeError("deploy coordinator gone")

        monkeypatch.setattr(gateway_coordinator, "get_or_create_coordinator", unreachable)
        coord = _coordinator()
        with caplog.at_level("WARNING"):
            await _pass(coord)
            await _pass(coord)
        assert cluster.deleted == []
        assert dropped in coord._unused_since
        assert caplog.messages.count("Could not ask for the deploy lease of gateway gw") == 1


def test_get_or_create_sets_max_restarts(monkeypatch):
    options = {}

    class _Options:
        def remote(self):
            return None

    def fake_options(**kwargs):
        options.update(kwargs)
        return _Options()

    monkeypatch.setattr(GatewayCoordinator, "options", fake_options)
    gateway_coordinator.get_or_create_gateway_coordinator()
    assert options["max_restarts"] == -1
    assert options["lifetime"] == "detached"
