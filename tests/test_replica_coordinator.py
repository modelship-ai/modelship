"""Tests for the replica coordinator's routing registry + watch API (the source of
truth gateway replicas reconcile from). Exercises the undecorated class directly,
in-process, without a Ray cluster."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ray.exceptions import ActorUnavailableError

from modelship.infer import replica_coordinator
from modelship.infer.replica_coordinator import RegistrationError, ReplicaCoordinator
from modelship.state import MemoryStoreActor

# The plain classes behind @ray.remote — their async methods are ordinary
# coroutines, so both can be exercised in-process without a Ray cluster.
_Coord = ReplicaCoordinator.__ray_metadata__.modified_class
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    # the actor configures logging for its whole process, here pytest's
    monkeypatch.setattr(replica_coordinator, "configure_logging", lambda: None)


@pytest.fixture
def coord():
    # get_state_store() returns a Ray-actor-backed client that requires a live
    # cluster; patch it to the plain in-process dict so these tests stay cluster-free.
    with patch.object(replica_coordinator, "get_state_store", return_value=_MemoryStore()):
        yield _Coord()


async def _route(coord, gateway, deployment, model):
    await coord.declare_deployment(gateway, deployment, model)
    return await coord.register_deployment(gateway, deployment, model)


class TestRoutingRegistry:
    @pytest.mark.asyncio
    async def test_register_records_and_bumps_generation(self, coord):
        assert (await coord.get_routing("gw"))["generation"] == 0
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {"qwen-aaaa": "qwen"}
        assert routing["generation"] == 1

    @pytest.mark.asyncio
    async def test_unregister_removes_and_bumps(self, coord):
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        await coord.unregister_deployment("gw", "qwen-aaaa")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {}
        assert routing["generation"] == 2

    @pytest.mark.asyncio
    async def test_set_expected_records_and_bumps(self, coord):
        await coord.set_expected("gw", ["qwen", "kokoro"])
        routing = await coord.get_routing("gw")
        assert routing["expected"] == ["qwen", "kokoro"]
        assert routing["generation"] == 1

    @pytest.mark.asyncio
    async def test_generation_is_per_gateway(self, coord):
        await _route(coord, "gw-a", "x-1", "x")
        assert (await coord.get_routing("gw-a"))["generation"] == 1
        assert (await coord.get_routing("gw-b"))["generation"] == 0

    @pytest.mark.asyncio
    async def test_register_evicts_prior_deployment_for_same_model(self, coord):
        await _route(coord, "gw", "qwen-OLD", "qwen")
        await _route(coord, "gw", "qwen-NEW", "qwen")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {"qwen-NEW": "qwen"}
        assert routing["generation"] == 2

    @pytest.mark.asyncio
    async def test_register_does_not_evict_other_models(self, coord):
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        await _route(coord, "gw", "kokoro-bbbb", "kokoro")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {"qwen-aaaa": "qwen", "kokoro-bbbb": "kokoro"}


class TestDeclaredRegistration:
    @pytest.mark.asyncio
    async def test_an_undeclared_deployment_is_refused(self, coord):
        assert not await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {}
        assert routing["generation"] == 0

    @pytest.mark.asyncio
    async def test_declaring_neither_routes_nor_bumps(self, coord):
        await coord.declare_deployment("gw", "qwen-aaaa", "qwen")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {}
        assert routing["generation"] == 0

    @pytest.mark.asyncio
    async def test_every_replica_after_the_first_is_a_no_op(self, coord):
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        assert await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        assert (await coord.get_routing("gw"))["generation"] == 1

    @pytest.mark.asyncio
    async def test_a_newer_declaration_refuses_the_older_deployment(self, coord):
        await coord.declare_deployment("gw", "qwen-OLD", "qwen")
        await coord.declare_deployment("gw", "qwen-NEW", "qwen")
        assert not await coord.register_deployment("gw", "qwen-OLD", "qwen")
        assert await coord.register_deployment("gw", "qwen-NEW", "qwen")

    @pytest.mark.asyncio
    async def test_a_replaced_deployment_cannot_take_routing_back(self, coord):
        await _route(coord, "gw", "qwen-OLD", "qwen")
        await _route(coord, "gw", "qwen-NEW", "qwen")
        assert not await coord.register_deployment("gw", "qwen-OLD", "qwen")
        assert (await coord.get_routing("gw"))["models"] == {"qwen-NEW": "qwen"}

    @pytest.mark.asyncio
    async def test_registering_a_routed_deployment_clears_its_declaration(self, coord):
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        await coord.declare_deployment("gw", "qwen-aaaa", "qwen")
        assert await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        assert coord._declared["gw"] == {}

    @pytest.mark.asyncio
    async def test_unregister_withdraws_a_declaration(self, coord):
        await coord.set_expected("gw", ["qwen"])
        await coord.declare_deployment("gw", "qwen-aaaa", "qwen")
        await coord.unregister_deployment("gw", "qwen-aaaa")
        assert not await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        assert (await coord.get_routing("gw"))["expected"] == []


class TestCutoverDeletesTheReplacedApp:
    @pytest.fixture
    def deleted(self, monkeypatch):
        deleted = []
        monkeypatch.setattr(replica_coordinator, "_WATCH_TIMEOUT_S", 0)
        monkeypatch.setattr("modelship.deploy.removal.delete_apps_quietly", lambda apps: deleted.extend(apps))
        return deleted

    @pytest.mark.asyncio
    async def test_the_replaced_app_is_deleted(self, coord, deleted):
        await _route(coord, "gw", "qwen-OLD", "qwen")
        await _route(coord, "gw", "qwen-NEW", "qwen")
        await asyncio.gather(*coord._deletions)
        assert deleted == ["qwen-OLD"]

    @pytest.mark.asyncio
    async def test_a_first_registration_deletes_nothing(self, coord, deleted):
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        assert not coord._deletions

    @pytest.mark.asyncio
    async def test_an_app_declared_again_before_the_delete_is_kept(self, coord, deleted, monkeypatch):
        monkeypatch.setattr(replica_coordinator, "_WATCH_TIMEOUT_S", 0.05)
        await _route(coord, "gw", "qwen-OLD", "qwen")
        await _route(coord, "gw", "qwen-NEW", "qwen")
        await coord.declare_deployment("gw", "qwen-OLD", "qwen")
        await asyncio.gather(*coord._deletions)
        assert deleted == []


class TestRegisterLoadedDeployment:
    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch):
        monkeypatch.setattr(replica_coordinator, "_REGISTER_RETRY_SECONDS", 0)

    def _lookup(self, monkeypatch, *results):
        handle = MagicMock()
        handle.register_deployment.remote = AsyncMock(side_effect=results)
        get_actor = MagicMock(return_value=handle)
        monkeypatch.setattr(replica_coordinator.ray, "get_actor", get_actor)
        return get_actor, handle

    @pytest.mark.asyncio
    async def test_registers_through_the_named_actor(self, monkeypatch):
        get_actor, handle = self._lookup(monkeypatch, True)
        await replica_coordinator.register_loaded_deployment("gw", "qwen-aaaa", "qwen")
        get_actor.assert_called_with(
            replica_coordinator.REPLICA_COORDINATOR_ACTOR_NAME, namespace=replica_coordinator.COORDINATOR_NAMESPACE
        )
        handle.register_deployment.remote.assert_called_once_with("gw", "qwen-aaaa", "qwen")

    @pytest.mark.asyncio
    async def test_a_refusal_warns_without_failing_the_replica(self, monkeypatch, caplog):
        self._lookup(monkeypatch, False)
        with caplog.at_level("WARNING"):
            await replica_coordinator.register_loaded_deployment("gw", "qwen-aaaa", "qwen")
        assert "not routing it" in caplog.text

    @pytest.mark.asyncio
    async def test_retries_through_a_coordinator_restart(self, monkeypatch):
        _, handle = self._lookup(monkeypatch, ActorUnavailableError("restarting", None), True)
        await replica_coordinator.register_loaded_deployment("gw", "qwen-aaaa", "qwen")
        assert handle.register_deployment.remote.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_once_the_attempts_run_out(self, monkeypatch):
        monkeypatch.setattr(replica_coordinator.ray, "get_actor", MagicMock(side_effect=ValueError("absent")))
        with pytest.raises(RegistrationError, match="unreachable"):
            await replica_coordinator.register_loaded_deployment("gw", "qwen-aaaa", "qwen")


class TestExpectedFollowsTheRegistry:
    """`_expected` is what /readyz measures against, so it has to track the registry."""

    @pytest.mark.asyncio
    async def test_unregister_drops_the_model_from_expected(self, coord):
        await coord.set_expected("gw", ["qwen", "kokoro"])
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        await coord.unregister_deployment("gw", "qwen-aaaa")
        assert (await coord.get_routing("gw"))["expected"] == ["kokoro"]

    @pytest.mark.asyncio
    async def test_another_deployment_for_the_model_keeps_it_expected(self, coord):
        await coord.set_expected("gw", ["qwen"])
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        # Bypass register_deployment's same-name eviction to get two live at once.
        coord._registry["gw"]["qwen-bbbb"] = "qwen"
        await coord.unregister_deployment("gw", "qwen-aaaa")
        assert (await coord.get_routing("gw"))["expected"] == ["qwen"]

    @pytest.mark.asyncio
    async def test_blue_green_cutover_keeps_the_model_expected(self, coord):
        await coord.set_expected("gw", ["qwen"])
        await _route(coord, "gw", "qwen-aaaa", "qwen")
        await _route(coord, "gw", "qwen-bbbb", "qwen")  # evicts qwen-aaaa
        await coord.unregister_deployment("gw", "qwen-aaaa")  # driver's late delete
        assert (await coord.get_routing("gw"))["expected"] == ["qwen"]
        assert (await coord.get_routing("gw"))["models"] == {"qwen-bbbb": "qwen"}

    @pytest.mark.asyncio
    async def test_unregistering_an_unknown_deployment_touches_nothing(self, coord):
        await coord.set_expected("gw", ["qwen"])
        await coord.unregister_deployment("gw", "qwen-aaaa")
        assert (await coord.get_routing("gw"))["expected"] == ["qwen"]

    @pytest.mark.asyncio
    async def test_a_never_registered_deployment_drops_by_model_name(self, coord):
        # Neither registered nor declared, so the model name can only come from the caller.
        await coord.set_expected("gw", ["qwen", "kokoro"])
        await coord.unregister_deployment("gw", "qwen-aaaa", "qwen")
        assert (await coord.get_routing("gw"))["expected"] == ["kokoro"]

    @pytest.mark.asyncio
    async def test_a_passed_model_name_still_respects_a_live_sibling(self, coord):
        await coord.set_expected("gw", ["qwen"])
        await _route(coord, "gw", "qwen-bbbb", "qwen")
        await coord.unregister_deployment("gw", "qwen-aaaa", "qwen")
        assert (await coord.get_routing("gw"))["expected"] == ["qwen"]


class TestWaitForChange:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_advanced(self, coord):
        await _route(coord, "gw", "x-1", "x")  # gen -> 1
        # A caller still at gen 0 must not block — the set already moved.
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=1)
        assert gen == 1

    @pytest.mark.asyncio
    async def test_blocks_then_wakes_on_change(self, coord):
        async def mutate():
            await asyncio.sleep(0.05)
            await _route(coord, "gw", "x-1", "x")

        task = asyncio.create_task(mutate())
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=2)
        await task
        assert gen == 1

    @pytest.mark.asyncio
    async def test_times_out_returning_same_generation(self, coord):
        gen = await coord.wait_for_change("gw", 0, timeout=0.05)
        assert gen == 0

    @pytest.mark.asyncio
    async def test_restart_lower_generation_returns_immediately(self, coord):
        # Replica last saw gen 5; a restarted coordinator is back at gen 0. Returning
        # at once (0 != 5) lets the replica re-sync instead of blocking forever.
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 5), timeout=1)
        assert gen == 0

    @pytest.mark.asyncio
    async def test_consecutive_cycles_advance(self, coord):
        await _route(coord, "gw", "x-1", "x")
        g1 = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=1)
        assert g1 == 1
        # Subscribe at g1; a second change wakes us with the next generation.
        waiter = asyncio.create_task(asyncio.wait_for(coord.wait_for_change("gw", g1), timeout=2))
        await asyncio.sleep(0.05)
        await coord.set_expected("gw", ["x"])
        assert await waiter == 2


class TestDurableState:
    """Registry + expected are written through a StateStore so a resurrected
    coordinator (max_restarts) reloads them instead of coming back empty."""

    @pytest.mark.asyncio
    async def test_reloads_registry_and_expected_from_shared_store(self):
        store = _MemoryStore()  # stand-in for a redis:// store across restarts
        with patch.object(replica_coordinator, "get_state_store", return_value=store):
            first = _Coord()
            await _route(first, "gw", "qwen-aaaa", "qwen")
            await first.set_expected("gw", ["qwen", "kokoro"])

            # A brand-new coordinator backed by the same store = a resurrected actor.
            second = _Coord()
        routing = await second.get_routing("gw")
        assert routing["models"] == {"qwen-aaaa": "qwen"}
        assert routing["expected"] == ["qwen", "kokoro"]
        # Generation is ephemeral — restart resets it to 0 (the gateway treats that
        # as "changed" and re-pulls).
        assert routing["generation"] == 0

    @pytest.mark.asyncio
    async def test_a_declaration_survives_a_restart(self):
        store = _MemoryStore()
        with patch.object(replica_coordinator, "get_state_store", return_value=store):
            await _Coord().declare_deployment("gw", "qwen-aaaa", "qwen")
            second = _Coord()
        assert await second.register_deployment("gw", "qwen-aaaa", "qwen")

    @pytest.mark.asyncio
    async def test_unregister_persists_removal(self):
        store = _MemoryStore()
        with patch.object(replica_coordinator, "get_state_store", return_value=store):
            first = _Coord()
            await _route(first, "gw", "qwen-aaaa", "qwen")
            await first.unregister_deployment("gw", "qwen-aaaa")
            second = _Coord()
        assert (await second.get_routing("gw"))["models"] == {}

    def test_get_or_create_sets_max_restarts(self):
        # Resurrection only helps because the actor auto-restarts; assert the option.
        with (
            patch.object(replica_coordinator.ray, "get_actor", side_effect=ValueError("absent")),
            patch.object(replica_coordinator.ReplicaCoordinator, "options") as options,
        ):
            options.return_value.remote.return_value = MagicMock()
            replica_coordinator.get_or_create_replica_coordinator()
        assert options.call_args.kwargs["max_restarts"] == -1
