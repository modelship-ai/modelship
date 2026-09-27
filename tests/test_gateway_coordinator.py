"""The gateway coordinator's passes: each gateway's model table from its routing version, its generation,
the long-poll API gateway replicas use, and the switch check a deploy runs on their reports. Exercises the
undecorated class in-process, with Serve's status and the deploy coordinator faked."""

import asyncio
from types import SimpleNamespace

import pytest
from ray.serve.schema import (
    ApplicationStatus,
    ApplicationStatusOverview,
    DeploymentStatus,
    DeploymentStatusOverview,
    DeploymentStatusTrigger,
)

from modelship.deploy.ledger import Version
from modelship.infer import gateway_coordinator
from modelship.infer.gateway_coordinator import GatewayCoordinator
from modelship.infer.infer_config import ModelshipModelConfig

# The plain class behind @ray.remote — its async methods are ordinary coroutines,
# so it can be exercised in-process without a Ray cluster.
_Coord = GatewayCoordinator.__ray_metadata__.modified_class


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


class _Cluster:
    """The Serve apps a gateway coordinator reads, and the routing versions the deploy coordinator gives it."""

    def __init__(self, monkeypatch):
        self.apps = {"gw": _app()}
        self.versions: dict[str, dict] = {}
        self.unreachable = False
        self._seq = 0
        monkeypatch.setattr(gateway_coordinator.serve, "status", lambda: SimpleNamespace(applications=dict(self.apps)))
        self.exists = True
        monkeypatch.setattr(gateway_coordinator, "find_coordinator", lambda: self if self.exists else None)

    @property
    def routing_versions(self):
        return SimpleNamespace(remote=self._routing_versions)

    async def _routing_versions(self, gateways):
        if self.unreachable:
            raise RuntimeError("deploy coordinator restarting")
        return {g: self.versions.get(g, {"seq": 0, "apps": {}}) for g in gateways}

    def configure(self, *raws: dict, gateway: str = "gw") -> int:
        """Routes *gateway* by *raws* under a new seq, which it returns."""
        self._seq += 1
        self.versions[gateway] = {"seq": self._seq, "apps": Version(0, list(raws)).apps(gateway)}
        return self._seq


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


class TestTables:
    @pytest.mark.asyncio
    async def test_each_model_is_routed_to_its_routing_versions_app(self, cluster):
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("gw")
        assert routing["models"] == {_app_name(NEW): "a"}
        assert routing["expected"] == ["a"]

    @pytest.mark.asyncio
    async def test_an_app_that_cannot_serve_leaves_its_model_out_even_with_another_app(self, cluster):
        cluster.configure(NEW)
        cluster.apps |= {_app_name(OLD): _app(), _app_name(NEW): LOADING}
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("gw")
        assert (routing["models"], routing["expected"]) == ({}, ["a"])

    @pytest.mark.asyncio
    async def test_a_gateway_without_a_committed_version_routes_nothing(self, cluster):
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("gw")
        assert (routing["models"], routing["expected"]) == ({}, [])

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

    @pytest.mark.asyncio
    async def test_without_a_deploy_coordinator_nothing_is_routed(self, cluster, caplog):
        cluster.exists = False
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)
        routing = await coord.get_routing("gw")
        assert (routing["models"], routing["expected"]) == ({}, [])
        assert caplog.messages == []

    @pytest.mark.asyncio
    async def test_an_unreachable_deploy_coordinator_keeps_the_last_routing_versions(self, cluster, caplog):
        cluster.configure(NEW)
        cluster.apps[_app_name(NEW)] = _app()
        coord = _coordinator()
        await _pass(coord)
        cluster.unreachable = True
        cluster.configure(OLD)
        cluster.apps[_app_name(OLD)] = _app()
        await _pass(coord)
        await _pass(coord)
        assert (await coord.get_routing("gw"))["models"] == {_app_name(NEW): "a"}
        assert caplog.messages.count("Could not read routing versions from the deploy coordinator") == 1


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
    async def test_a_new_routing_seq_advances_it_even_with_the_same_table(self, cluster):
        cluster.configure(NEW)
        coord = _coordinator("gw")
        await _pass(coord)
        before = await coord.get_routing("gw")
        seq = cluster.configure(NEW)
        await _pass(coord)
        after = await coord.get_routing("gw")
        assert (after["generation"], after["routing"]) == (before["generation"] + 1, seq)

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

    @pytest.mark.asyncio
    async def test_carries_the_routing_seq_its_table_came_from(self, cluster):
        seq = cluster.configure(NEW)
        coord = _coordinator("gw")
        await _pass(coord)
        assert (await coord.get_routing("gw"))["routing"] == seq


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

    @pytest.mark.asyncio
    async def test_records_the_replicas_routing_seq(self, cluster):
        coord = _coordinator()
        await coord.wait_for_change("gw", 7, timeout=0.01, replica_id="r1", routing=3)
        assert coord._reports["gw"]["r1"][0] == 3


class TestWaitSwitched:
    async def _two_replicas(self, cluster):
        cluster.apps["gw"] = _app(running=2)
        seq = cluster.configure(NEW)
        coord = _coordinator("gw")
        await _pass(coord)
        return coord, seq

    @pytest.mark.asyncio
    async def test_true_once_every_running_replica_reports_the_seq(self, cluster):
        coord, seq = await self._two_replicas(cluster)
        for replica in ("r1", "r2"):
            await coord.wait_for_change("gw", 0, timeout=0.01, replica_id=replica, routing=seq)
        assert await coord.wait_switched("gw", seq, 1.0)

    @pytest.mark.asyncio
    async def test_false_after_the_timeout_while_a_replica_has_not(self, cluster):
        coord, seq = await self._two_replicas(cluster)
        await coord.wait_for_change("gw", 0, timeout=0.01, replica_id="r1", routing=seq)
        await coord.wait_for_change("gw", 0, timeout=0.01, replica_id="r2", routing=seq - 1)
        assert not await coord.wait_switched("gw", seq, 0.05)

    @pytest.mark.asyncio
    async def test_a_report_arriving_during_the_wait_completes_it(self, cluster):
        coord, seq = await self._two_replicas(cluster)
        await coord.wait_for_change("gw", 0, timeout=0.01, replica_id="r1", routing=seq)
        waiting = asyncio.create_task(coord.wait_switched("gw", seq, 1.0))
        await asyncio.sleep(0.02)
        await coord.wait_for_change("gw", 0, timeout=0.01, replica_id="r2", routing=seq)
        assert await waiting

    @pytest.mark.asyncio
    async def test_a_stale_report_does_not_count(self, cluster, monkeypatch):
        coord, seq = await self._two_replicas(cluster)
        for replica in ("r1", "r2"):
            await coord.wait_for_change("gw", 0, timeout=0.01, replica_id=replica, routing=seq)
        at = coord._reports["gw"]["r2"][1]
        coord._reports["gw"]["r2"] = (seq, at - gateway_coordinator._REPORT_TTL_S - 1)
        assert not await coord.wait_switched("gw", seq, 0.05)

    @pytest.mark.asyncio
    async def test_false_until_a_pass_has_picked_up_the_seq(self, cluster):
        coord, _ = await self._two_replicas(cluster)
        newer = cluster.configure(OLD)
        for replica in ("r1", "r2"):
            await coord.wait_for_change("gw", 0, timeout=0.01, replica_id=replica, routing=newer)
        assert not await coord.wait_switched("gw", newer, 0.05)
        await _pass(coord)
        assert await coord.wait_switched("gw", newer, 0.05)

    @pytest.mark.asyncio
    async def test_a_gateway_with_no_running_replica_is_switched_once_the_seq_is_picked_up(self, cluster):
        cluster.apps["gw"] = _app(running=0)
        seq = cluster.configure(NEW)
        coord = _coordinator("gw")
        await _pass(coord)
        assert await coord.wait_switched("gw", seq, 0.05)


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
