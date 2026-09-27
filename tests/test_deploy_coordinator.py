"""The deploy coordinator, driven directly: leases, effective-config writes, replica-death counts, and the
per-gateway deploy queue with fake workers. Placement options live in test_actor_placement.py."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from ray.exceptions import RayActorError
from ray.serve.schema import LoggingConfig

from modelship.deploy import worker as worker_module
from modelship.deploy.ledger import DeployRequest, Version, commit_version, read_effective, read_versions
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

    async def test_a_restarted_actor_waits_out_the_window_anyway(self, monkeypatch):
        monkeypatch.setattr(deploy_coordinator, "_reconstructed", lambda: True)
        coord = _Coord(startup_window=False)
        coord._reaper.cancel()
        assert await coord.acquire("node", "a") == "lease service starting"


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
    return DeployRequest(
        gateway, mode, "blue_green", models if models is not None else [_raw("a")], LoggingConfig(), {}
    )


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

    def create(self, coordinator):
        worker = _Worker(self)
        self.created.append(worker)
        return worker

    def record_rollback(self, args):
        self.rollbacks.append(args)
        done = asyncio.get_running_loop().create_future()
        done.set_result([])
        return done

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
        assert workers.rollbacks == [("g", None, True, 60.0)]
        assert len(workers.created) == 2


@pytest.mark.asyncio
class TestCancel:
    async def test_a_queued_request_is_dropped(self, workers):
        coord = _ledger()
        await coord.submit(_request())
        queued = await coord.submit(_request())
        assert await coord.cancel(queued["id"]) == f"deploy {queued['id']} cancelled"
        assert (await queued["outcome"])["state"] == "cancelled"
        await _until(lambda: workers.created)
        workers.created[0].finish()
        await _settle()
        assert len(workers.created) == 1

    async def test_a_running_request_is_flagged_for_its_worker(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.cancel(receipt["id"])
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
        assert workers.rollbacks == [("g", None, False, 60.0)]

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
        assert "already committed" in await coord.cancel(receipt["id"])

    async def test_an_unknown_request(self, workers):
        assert await _ledger().cancel("nope") == "no queued or running deploy nope"


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
        assert workers.rollbacks == [("g", None, True, 60.0)]

    async def test_a_worker_that_dies_after_the_commit_still_succeeds(self, workers):
        coord = _ledger()
        receipt = await coord.submit(_request())
        await _settle()
        await coord.switch(receipt["id"], "g", [_raw("a")])
        await coord.commit(receipt["id"], "g", [_raw("a")])
        workers.kill(workers.created[0])
        assert (await receipt["outcome"])["state"] == "succeeded"
        assert workers.rollbacks == [("g", [_raw("a")], False, 60.0)]


@pytest.mark.asyncio
class TestRouting:
    async def test_a_gateway_routes_by_its_committed_version(self, workers):
        coord = _ledger()
        commit_version(coord._store, "g", [_raw("a")])
        versions = await coord.routing_versions(["g"])
        assert versions["g"]["apps"] == {"a": _app_name(_raw("a"))}

    async def test_a_gateway_without_a_version_has_no_table(self, workers):
        assert (await _ledger().routing_versions(["g"]))["g"]["apps"] is None

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
        assert await coord.reset_routing("g") > seq
        assert (await coord.routing_versions(["g"]))["g"]["apps"] is None

    async def test_a_reset_on_the_committed_version_keeps_the_seq(self, workers):
        coord = _ledger()
        seq = (await coord.routing_versions(["g"]))["g"]["seq"]
        assert await coord.reset_routing("g") == seq

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
