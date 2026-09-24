"""Tests for the deploy coordinator; placement options live in test_actor_placement.py.
Reserve/release/liveness paths are untested."""

from unittest.mock import AsyncMock, patch

import pytest

from modelship.infer import deploy_coordinator


class TestReplicaDeathCounting:
    @staticmethod
    def _coord():
        # The plain class behind @ray.remote; its methods are ordinary coroutines.
        return deploy_coordinator.DeployCoordinator.__ray_metadata__.modified_class()

    @pytest.mark.asyncio
    async def test_deaths_below_the_limit_do_not_retire(self):
        coord = self._coord()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
                await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_limiting_death_retires(self):
        coord = self._coord()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA):
                await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_awaited_once_with("qwen-aaaa")

    @pytest.mark.asyncio
    async def test_the_limit_scales_with_the_replica_count(self):
        coord = self._coord()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for _ in range(deploy_coordinator._DEATHS_PER_REPLICA * 4):
                await coord.report_replica_death("qwen-aaaa", 4, "engine died")
        assert retire.await_count == 1

    @pytest.mark.asyncio
    async def test_the_count_is_not_time_windowed(self):
        coord = self._coord()
        coord._deaths["qwen-aaaa"] = deploy_coordinator._DEATHS_PER_REPLICA - 1
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            await coord.report_replica_death("qwen-aaaa", 1, "engine died")
        retire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deployments_are_counted_separately(self):
        coord = self._coord()
        with patch.object(coord, "_retire", new=AsyncMock()) as retire:
            for name in ("qwen-aaaa", "kokoro-bbbb"):
                for _ in range(deploy_coordinator._DEATHS_PER_REPLICA - 1):
                    await coord.report_replica_death(name, 1, "engine died")
        retire.assert_not_called()

    @pytest.mark.asyncio
    async def test_retire_deletes_the_app(self):
        coord = self._coord()
        with patch("modelship.deploy.removal.serve.delete") as delete:
            await coord._retire("qwen-aaaa")
        delete.assert_called_once_with("qwen-aaaa")
