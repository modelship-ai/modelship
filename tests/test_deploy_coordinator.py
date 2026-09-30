"""The deploy coordinator, driven directly: node leases, replica-death counts, and the per-gateway deploy
queue with fake workers. Placement options live in test_actor_placement.py."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from ray.exceptions import RayActorError
from ray.serve.schema import LoggingConfig

from modelship.deploy import worker as worker_module
from modelship.deploy.ledger import DeployRequest, Version, commit_version, read_versions
from modelship.infer import deploy_coordinator
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state import MemoryStoreActor
from modelship.state.base import StateStoreUnavailableError

# The plain class behind @ray.remote; its methods are ordinary coroutines.
_Coord = deploy_coordinator.DeployCoordinator.__ray_metadata__.modified_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(deploy_coordinator, "configure_logging", lambda: None)


def _fresh():
    coord = _Coord()
    coord._lease_checker.cancel()
    coord._store = _MemoryStore()
    return coord


async def _loaded(coord=None):
    """A fresh deploy coordinator once its stored leases are read back."""
    coord = coord or _fresh()
    await coord._leases_loaded
    return coord


def _holder(replica: str = "r1") -> dict:
    return {"app": "g.a-1", "deployment": "ModelDeployment", "replica": replica, "label": f"a/{replica}"}


def _replica(holder: dict) -> tuple[str, str, str]:
    return holder["app"], holder["deployment"], holder["replica"]


@pytest.mark.asyncio
class TestGrants:
    async def test_granted_once_the_stored_leases_are_read_back(self):
        coord = _fresh()
        assert await coord.acquire("node", _holder()) == "deploy lease service starting"
        await _loaded(coord)
        assert await coord.acquire("node", _holder()) is None

    async def test_one_holder_per_node(self):
        coord = await _loaded()
        assert await coord.acquire("node", _holder("r1")) is None
        assert await coord.acquire("node", _holder("r2")) == "held by a/r1"
        assert await coord.acquire("other-node", _holder("r2")) is None

    async def test_the_holders_retry_is_granted(self):
        coord = await _loaded()
        await coord.acquire("node", _holder())
        assert await coord.acquire("node", _holder()) is None

    async def test_a_grant_is_stored(self):
        coord = await _loaded()
        await coord.acquire("node", _holder())
        assert coord._store.get("deploy-leases/node") == _holder()

    async def test_a_failed_store_write_refuses_and_leaves_the_node_free(self, monkeypatch):
        coord = await _loaded()

        async def unavailable(*args, **kwargs):
            raise StateStoreUnavailableError("down")

        monkeypatch.setattr(coord._store, "set_async", unavailable)
        assert await coord.acquire("node", _holder("r1")) == "deploy lease store unavailable"
        monkeypatch.undo()
        assert await coord.acquire("node", _holder("r2")) is None


@pytest.mark.asyncio
class TestLeaseChecks:
    @pytest.fixture
    def replicas(self, monkeypatch) -> dict:
        states: dict = {}
        monkeypatch.setattr(deploy_coordinator, "_replica_states", lambda: states)
        return states

    async def test_a_running_replica_frees_its_lease(self, replicas):
        coord = await _loaded()
        await coord.acquire("node", _holder("r1"))
        replicas[_replica(_holder("r1"))] = "RUNNING"
        await coord._check_leases()
        assert coord._store.get("deploy-leases/node") is None
        assert await coord.acquire("node", _holder("r2")) is None

    @pytest.mark.parametrize("state", ["STARTING", "RECOVERING", "STOPPING"])
    async def test_a_replica_still_loading_keeps_its_lease(self, replicas, state):
        coord = await _loaded()
        await coord.acquire("node", _holder("r1"))
        replicas[_replica(_holder("r1"))] = state
        await coord._check_leases()
        await coord._check_leases()
        assert await coord.acquire("node", _holder("r2")) == "held by a/r1"

    async def test_an_unlisted_replica_is_freed_on_the_second_check(self, replicas):
        coord = await _loaded()
        await coord.acquire("node", _holder("r1"))
        await coord._check_leases()
        assert await coord.acquire("node", _holder("r2")) == "held by a/r1"
        await coord._check_leases()
        assert await coord.acquire("node", _holder("r2")) is None

    async def test_a_replica_listed_again_between_checks_is_kept(self, replicas):
        coord = await _loaded()
        await coord.acquire("node", _holder("r1"))
        await coord._check_leases()
        replicas[_replica(_holder("r1"))] = "STARTING"
        await coord._check_leases()
        del replicas[_replica(_holder("r1"))]
        await coord._check_leases()
        assert await coord.acquire("node", _holder("r2")) == "held by a/r1"

    async def test_a_failed_read_frees_nothing(self, monkeypatch):
        coord = await _loaded()
        await coord.acquire("node", _holder("r1"))

        def unreadable():
            raise RuntimeError("controller restarting")

        monkeypatch.setattr(deploy_coordinator, "_replica_states", unreadable)
        await coord._check_leases()
        await coord._check_leases()
        assert await coord.acquire("node", _holder("r2")) == "held by a/r1"

    async def test_serve_is_not_read_while_no_lease_is_held(self, monkeypatch):
        reads = []
        monkeypatch.setattr(deploy_coordinator, "_replica_states", lambda: reads.append(1) or {})
        await (await _loaded())._check_leases()
        assert reads == []

    async def test_a_restarted_deploy_coordinator_keeps_the_leases_it_reads_back(self, replicas):
        first = await _loaded()
        await first.acquire("node", _holder("r1"))
        restarted = _Coord()
        restarted._lease_checker.cancel()
        restarted._store = first._store
        await _loaded(restarted)
        assert await restarted.acquire("node", _holder("r2")) == "held by a/r1"
        replicas[_replica(_holder("r1"))] = "RUNNING"
        await restarted._check_leases()
        assert await restarted.acquire("node", _holder("r2")) is None


class TestReplicaStates:
    def test_every_listed_replica_with_its_state(self, monkeypatch):
        details = {
            "applications": {
                "g.a-1": {
                    "deployments": {
                        "ModelDeployment": {
                            "replicas": [
                                {"replica_id": "r1", "state": "RUNNING"},
                                {"replica_id": "r2", "state": "STARTING"},
                            ]
                        }
                    }
                },
                "g": {"deployments": {"ModelshipAPI": {"replicas": []}}},
            }
        }
        client = SimpleNamespace(get_serve_details=lambda: details)
        monkeypatch.setattr("ray.serve.context._get_global_client", lambda **kwargs: client)
        assert deploy_coordinator._replica_states() == {
            ("g.a-1", "ModelDeployment", "r1"): "RUNNING",
            ("g.a-1", "ModelDeployment", "r2"): "STARTING",
        }

    def test_serve_still_has_the_calls_it_reads(self):
        from ray.serve._private.client import ServeControllerClient
        from ray.serve._private.controller import ServeController
        from ray.serve.schema import ReplicaDetails

        assert hasattr(ServeControllerClient, "get_serve_details")
        assert hasattr(ServeController, "get_serve_instance_details")
        assert {"replica_id", "state"} <= set(ReplicaDetails.__fields__)


@pytest.mark.asyncio
class TestReplicaDeathCounting:
    async def test_deaths_below_the_limit_are_not_crash_looping(self):
        coord = _fresh()
        for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        assert await coord.crash_looping(["qwen-aaaa"]) == {}

    async def test_the_limiting_death_is_crash_looping_with_its_reason(self):
        coord = _fresh()
        for _ in range(deploy_coordinator._DEATHS_PER_REPLICA):
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        assert await coord.crash_looping(["qwen-aaaa", "other"]) == {"qwen-aaaa": "engine died"}

    async def test_the_limit_scales_with_the_replica_count(self):
        coord = _fresh()
        for _ in range(deploy_coordinator._DEATHS_PER_REPLICA * 4 - 1):
            await coord.report_replica_death("qwen-aaaa", 4, "engine died")
        assert await coord.crash_looping(["qwen-aaaa"]) == {}
        await coord.report_replica_death("qwen-aaaa", 4, "engine died")
        assert "qwen-aaaa" in await coord.crash_looping(["qwen-aaaa"])

    async def test_deployments_are_counted_separately(self):
        coord = _fresh()
        for name in ("qwen-aaaa", "kokoro-bbbb"):
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
                await coord.report_replica_death(name, 1, "engine died")
        assert await coord.crash_looping(["qwen-aaaa", "kokoro-bbbb"]) == {}

    async def test_forgetting_resets_the_count(self):
        coord = _fresh()
        for _ in range(deploy_coordinator._DEATHS_PER_REPLICA):
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        await coord.forget_deaths(["qwen-aaaa"])
        assert await coord.crash_looping(["qwen-aaaa"]) == {}

    async def test_a_death_past_the_limit_deletes_nothing(self, monkeypatch):
        monkeypatch.setattr(deploy_coordinator.serve, "delete", lambda *a, **k: pytest.fail("deleted an app"))
        coord = _fresh()
        for _ in range(deploy_coordinator._DEATHS_PER_REPLICA + 2):
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")


def _raw(name: str, **overrides) -> dict:
    return {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server", **overrides}


def _app_name(raw: dict, gateway: str = "g") -> str:
    return ModelshipModelConfig.model_validate(raw).deployment_name(gateway)


def _request(gateway: str = "g", models=None, mode="additive") -> DeployRequest:
    return DeployRequest(gateway, mode, "blue_green", models if models is not None else [_raw("a")], {})


class _Worker:
    """A deploy worker whose run the test finishes; a kill ends it with RayActorError."""

    def __init__(self, workers: "_Workers"):
        self.run_future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.runs: list[tuple] = []
        self.killed = False
        self.run = SimpleNamespace(remote=self._run)
        self.roll_back = SimpleNamespace(remote=lambda *args: workers.record_rollback(args))

    def _run(self, request, committed, switch_timeout):
        self.runs.append((request, committed))
        return self.run_future

    def finish(self, state: str = "succeeded") -> None:
        request = self.runs[0][0]
        self.run_future.set_result({"id": request.id, "state": state, "reason": "", "models": {}, "version": None})


class _Workers:
    def __init__(self):
        self.created: list[_Worker] = []
        self.rollbacks: list[tuple] = []
        # when set, a rollback routes back through it as the real worker does
        self.ledger = None
        # what each rollback's reset_routing returned
        self.resets: list[tuple] = []
        # when set, a rollback waits for it after its reset
        self.hold: asyncio.Event | None = None

    def create(self, coordinator):
        worker = _Worker(self)
        self.created.append(worker)
        return worker

    def record_rollback(self, args):
        self.rollbacks.append(args)
        return asyncio.ensure_future(self._roll_back(args[0]))

    async def _roll_back(self, gateway_name):
        if self.ledger is not None:
            self.resets.append(await self.ledger.reset_routing(gateway_name))
        if self.hold is not None:
            await self.hold.wait()
        return []

    def kill(self, handle):
        handle.killed = True
        if isinstance(handle, _Worker) and not handle.run_future.done():
            handle.run_future.set_exception(RayActorError())
            # retrieved here too: a kill from a cancelled task has no awaiter
            handle.run_future.add_done_callback(lambda future: future.exception())


@pytest.fixture
def workers(monkeypatch):
    fakes = _Workers()
    monkeypatch.setattr(worker_module, "create_worker", fakes.create)
    monkeypatch.setattr(deploy_coordinator.ray, "kill", fakes.kill)
    return fakes


def _ledger():
    """A fresh deploy coordinator whose calls to its own handle run in-process."""
    coord = _fresh()
    coord._self = lambda: SimpleNamespace(
        wait=SimpleNamespace(remote=lambda rid: asyncio.ensure_future(coord.wait(rid)))
    )
    return coord


@pytest.fixture(autouse=True)
def serve_apps(monkeypatch):
    """Serve's app list as the recovery reads it."""
    apps: list[str] = []
    monkeypatch.setattr(deploy_coordinator.serve, "status", lambda: SimpleNamespace(applications=dict.fromkeys(apps)))
    return apps


async def _until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.005)


def _hold_commits(monkeypatch) -> threading.Event:
    """Store writes by commit wait until the returned event is set."""
    release = threading.Event()

    def held(store, gateway_name, models):
        release.wait(2)
        return commit_version(store, gateway_name, models)

    monkeypatch.setattr(deploy_coordinator, "commit_version", held)
    return release


async def _settle() -> None:
    """Lets queued work, store reads included, run."""
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
class TestQueue:
    async def test_a_request_runs_in_a_worker_and_its_outcome_is_returned(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        assert receipt["behind"] is None
        await _settle()
        (worker,) = workers.created
        assert worker.runs[0][1] is None
        worker.finish()
        outcome = await receipt["outcome"]
        assert outcome["id"] == receipt["id"]
        assert outcome["state"] == "succeeded"
        assert worker.killed

    async def test_requests_on_one_gateway_run_one_after_another(self, workers):
        coord = _ledger()
        first = await coord.submit(_request())
        second = await coord.submit(_request())
        assert second["behind"] == first["id"]
        await _settle()
        assert len(workers.created) == 1
        workers.created[0].finish()
        await first["outcome"]
        await _settle()
        assert len(workers.created) == 2
        workers.created[1].finish()
        await second["outcome"]

    async def test_requests_on_different_gateways_run_side_by_side(self, workers):
        coord = _ledger()
        await coord.submit(_request("g"))
        await coord.submit(_request("edge"))
        await _settle()
        assert len(workers.created) == 2

    async def test_a_request_starts_from_the_committed_version(self, workers):
        coord = _ledger()
        commit_version(coord._store, "g", [_raw("a")])
        await coord.submit(_request())
        await _settle()
        assert workers.created[0].runs[0][1] == [_raw("a")]

    async def test_the_outcome_is_forgotten_once_collected(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        workers.created[0].finish()
        await receipt["outcome"]
        assert coord._requests == {}

    async def test_recovery_runs_before_the_first_request(self, workers, serve_apps):
        serve_apps.extend(["g", _app_name(_raw("a"))])
        coord = _ledger()
        await coord.submit(_request())
        await _settle()
        assert workers.rollbacks == [("g", True, 60.0)]
        assert len(workers.created) == 2


@pytest.mark.asyncio
class TestCancel:
    async def test_a_queued_request_is_dropped(self, workers):
        coord = _ledger()
        await coord.submit(_request())
        queued = await coord.submit(_request())
        result = await coord.cancel(queued["id"])
        assert (result["cancelled"], result["message"]) == (True, f"deploy {queued['id']} cancelled")
        assert result["outcome"] is queued["outcome"]
        assert (await queued["outcome"])["state"] == "cancelled"
        await _until(lambda: workers.created)
        workers.created[0].finish()
        await _settle()
        assert len(workers.created) == 1

    async def test_a_running_request_is_flagged_for_its_worker(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        assert (await coord.cancel(receipt["id"]))["outcome"] is receipt["outcome"]
        assert await coord.is_cancelled(receipt["id"])
        assert coord._requests[receipt["id"]].cancel_seen

    async def test_a_worker_that_does_not_stop_within_the_grace_is_killed_and_rolled_back(self, workers):
        coord = _ledger()
        coord._cancel_grace = 0.01
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
        outcome = await asyncio.wait_for(receipt["outcome"], 1)
        assert outcome["state"] == "cancelled"
        assert workers.rollbacks == [("g", False, 60.0)]

    async def test_a_worker_that_saw_the_cancel_is_left_to_roll_back(self, workers):
        coord = _ledger()
        coord._cancel_grace = 0.01
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
        await coord.is_cancelled(receipt["id"])
        await asyncio.sleep(0.05)
        assert not workers.created[0].killed
        workers.created[0].finish("cancelled")
        await receipt["outcome"]

    async def test_a_committed_request_cannot_be_cancelled(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        await coord.commit(receipt["id"], "g", [_raw("a")])
        result = await coord.cancel(receipt["id"])
        assert not result["cancelled"]
        assert "already committed" in result["message"]

    async def test_an_unknown_request(self, workers):
        assert await _ledger().cancel("nope") == {"cancelled": False, "message": "no queued or running deploy nope"}

    async def test_a_request_rolling_itself_back_cannot_be_cancelled(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.rolling_back(receipt["id"], "g")
        result = await coord.cancel(receipt["id"])
        assert result == {
            "cancelled": False,
            "message": f"deploy {receipt['id']} failed and is already being rolled back",
        }
        assert not await coord.is_cancelled(receipt["id"])

    async def test_a_cancelled_request_rolling_back_can_be_cancelled_again(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
        await coord.is_cancelled(receipt["id"])
        await coord.rolling_back(receipt["id"], "g")
        assert (await coord.cancel(receipt["id"]))["cancelled"]

    async def test_a_worker_rolling_back_is_not_killed_for_a_cancel_it_missed(self, workers):
        coord = _ledger()
        coord._cancel_grace = 0.01
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
        await coord.rolling_back(receipt["id"], "g")
        await asyncio.sleep(0.05)
        assert not workers.created[0].killed
        workers.created[0].finish("failed")
        assert (await receipt["outcome"])["state"] == "failed"

    async def test_a_dead_workers_rollback_cannot_be_cancelled(self, workers):
        coord = _ledger()
        workers.hold = asyncio.Event()
        receipt = await coord.submit(_request())
        await _settle()
        workers.kill(workers.created[0])
        await _until(lambda: workers.rollbacks)
        assert not (await coord.cancel(receipt["id"]))["cancelled"]
        workers.hold.set()
        assert (await receipt["outcome"])["state"] == "failed"


@pytest.mark.asyncio
class TestWorkerDeath:
    async def test_a_worker_that_dies_fails_the_request_and_rolls_it_back(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        workers.kill(workers.created[0])
        outcome = await receipt["outcome"]
        assert outcome["state"] == "failed"
        assert outcome["reason"].startswith("the deploy worker died")
        assert workers.rollbacks == [("g", True, 60.0)]

    async def test_a_worker_that_dies_during_a_slow_commit_succeeds_once_the_write_lands(self, workers, monkeypatch):
        coord = _ledger()
        workers.ledger = coord
        release = _hold_commits(monkeypatch)
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        commit = asyncio.ensure_future(coord.commit(receipt["id"], "g", [_raw("a")]))
        await _until(lambda: coord._requests[receipt["id"]].state == "committing")
        workers.kill(workers.created[0])
        await _settle()
        release.set()
        assert (await receipt["outcome"])["state"] == "succeeded"
        assert await commit == 1

    async def test_a_reset_during_a_commit_returns_the_version_it_writes(self, workers, monkeypatch):
        coord = _ledger()
        release = _hold_commits(monkeypatch)
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        commit = asyncio.ensure_future(coord.commit(receipt["id"], "g", [_raw("a")]))
        await _until(lambda: coord._requests[receipt["id"]].state == "committing")
        reset = asyncio.ensure_future(coord.reset_routing("g"))
        await _settle()
        release.set()
        assert (await reset)[1] == [_raw("a")]
        await commit

    async def test_a_commit_that_reaches_a_rollback_after_its_reset_is_refused(self, workers):
        coord = _ledger()
        workers.ledger = coord
        workers.hold = asyncio.Event()
        commit_version(coord._store, "g", [_raw("a")])
        receipt = await coord.submit(_request(models=[_raw("b")]))
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("b")])
        workers.kill(workers.created[0])
        await _until(lambda: workers.resets)
        with pytest.raises(ValueError, match="being rolled back"):
            await coord.commit(receipt["id"], "g", [_raw("b")])
        workers.hold.set()
        assert (await receipt["outcome"])["state"] == "failed"
        assert workers.resets[0][1] == [_raw("a")]
        assert read_versions(coord._store, "g")[0] == Version(1, [_raw("a")])

    async def test_a_switch_that_reaches_a_rollback_after_its_reset_is_refused(self, workers):
        coord = _ledger()
        workers.ledger = coord
        workers.hold = asyncio.Event()
        commit_version(coord._store, "g", [_raw("a")])
        receipt = await coord.submit(_request(models=[_raw("b")]))
        await _settle()
        workers.kill(workers.created[0])
        await _until(lambda: workers.resets)
        with pytest.raises(ValueError, match="being rolled back"):
            await coord.switch(receipt["id"], "g", [_raw("b")])
        workers.hold.set()
        assert (await receipt["outcome"])["state"] == "failed"
        assert (await coord.routing_versions(["g"]))["g"]["apps"] == {"a": _app_name(_raw("a"))}

    async def test_a_worker_that_dies_rolling_back_a_switch_waits_for_the_switch_back(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        await coord.rolling_back(receipt["id"], "g")
        workers.kill(workers.created[0])
        assert (await receipt["outcome"])["state"] == "failed"
        assert workers.rollbacks == [("g", True, 60.0)]

    async def test_a_worker_that_dies_after_the_commit_still_succeeds(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        await coord.commit(receipt["id"], "g", [_raw("a")])
        workers.kill(workers.created[0])
        assert (await receipt["outcome"])["state"] == "succeeded"
        assert workers.rollbacks == [("g", False, 60.0)]


@pytest.mark.asyncio
async def test_the_deploy_coordinator_reports_its_cluster_settings(monkeypatch):
    monkeypatch.setenv("MSHIP_STATE_STORE", "redis://head:6379/0")
    monkeypatch.setenv("MSHIP_METRICS", "false")
    monkeypatch.setenv("MSHIP_GATEWAY_MAX_REPLICAS", "8")
    settings = await _fresh().cluster_settings()
    assert settings["env"]["MSHIP_STATE_STORE"] == "redis://head:6379/0"
    assert settings["env"]["MSHIP_METRICS"] == "false"
    assert isinstance(settings["serve_logging_config"], LoggingConfig)
    assert settings["gateway_sizing"]["autoscaling_config"]["max_replicas"] == 8


@pytest.mark.asyncio
class TestRouting:
    async def test_a_gateway_routes_by_its_committed_version(self, workers):
        coord = _ledger()
        commit_version(coord._store, "g", [_raw("a")])
        versions = await coord.routing_versions(["g"])
        assert versions["g"]["apps"] == {"a": _app_name(_raw("a"))}

    async def test_a_gateway_without_a_version_has_an_empty_table(self, workers):
        assert (await _ledger().routing_versions(["g"]))["g"]["apps"] == {}

    async def test_a_switch_routes_by_the_proposed_models_under_a_new_seq(self, workers):
        coord = _ledger()
        before = (await coord.routing_versions(["g"]))["g"]["seq"]
        receipt = await coord.submit(_request())
        await _settle()
        seq = await coord.switch(receipt["id"], "g", [_raw("b")])
        assert seq > before
        assert await coord.routing_versions(["g"]) == {"g": {"seq": seq, "apps": {"b": _app_name(_raw("b"))}}}
        assert coord._running["g"].state == "switching"

    async def test_a_reset_routes_back_under_another_seq(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        seq = await coord.switch(receipt["id"], "g", [_raw("b")])
        assert (await coord.reset_routing("g"))[0] > seq
        assert (await coord.routing_versions(["g"]))["g"]["apps"] == {}

    async def test_a_reset_on_the_committed_version_keeps_the_seq(self, workers):
        coord = _ledger()
        seq = (await coord.routing_versions(["g"]))["g"]["seq"]
        assert await coord.reset_routing("g") == (seq, None)

    async def test_only_the_running_request_can_switch(self, workers):
        coord = _ledger()
        with pytest.raises(ValueError, match="not running"):
            await coord.switch("nope", "g", [])


@pytest.mark.asyncio
class TestCommit:
    async def test_writes_the_next_version(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        assert await coord.commit(receipt["id"], "g", [_raw("a")]) == 1
        assert read_versions(coord._store, "g")[0] == Version(1, [_raw("a")])
        assert coord._running["g"].state == "retiring"

    async def test_a_cancelled_request_commits_nothing(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
        assert await coord.commit(receipt["id"], "g", [_raw("a")]) is None
        assert read_versions(coord._store, "g") == (None, None)


class TestFindCoordinator:
    def test_none_when_nothing_created_it(self, monkeypatch):
        def missing(name, namespace):
            raise ValueError(f"Failed to look up actor {name}")

        monkeypatch.setattr(deploy_coordinator.ray, "get_actor", missing)
        assert deploy_coordinator.find_coordinator() is None

    def test_the_named_actor(self, monkeypatch):
        monkeypatch.setattr(deploy_coordinator.ray, "get_actor", lambda name, namespace: (name, namespace))
        assert deploy_coordinator.find_coordinator() == ("modelship-deploy-coordinator", "modelship")
