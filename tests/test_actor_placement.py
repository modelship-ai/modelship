"""Placement options of the cluster-wide singleton actors."""

from unittest.mock import patch

import pytest

from modelship.infer import deploy_coordinator, deploy_leases, download_leases, gateway_coordinator
from modelship.state import memory

_HEAD_PINNED = {
    "get_if_exists": True,
    "lifetime": "detached",
    "resources": {"node:__internal_head__": 0.001},
    "scheduling_strategy": "DEFAULT",
}


@pytest.mark.parametrize(
    ("actor_cls", "getter", "max_restarts"),
    [
        (deploy_coordinator.DeployCoordinator, deploy_coordinator.get_or_create_coordinator, -1),
        (gateway_coordinator.GatewayCoordinator, gateway_coordinator.get_or_create_gateway_coordinator, -1),
        (memory.MemoryStoreActor, memory.get_or_create_memory_store_actor, -1),
        (download_leases.DownloadLeases, download_leases.get_or_create_leases, 0),
        (deploy_leases.DeployLeases, deploy_leases.get_or_create_leases, 0),
    ],
)
def test_singletons_are_pinned_to_the_head(actor_cls, getter, max_restarts):
    with patch.object(actor_cls, "options") as options:
        getter()
    kwargs = options.call_args.kwargs
    assert {k: kwargs[k] for k in _HEAD_PINNED} == _HEAD_PINNED
    assert kwargs["max_restarts"] == max_restarts
