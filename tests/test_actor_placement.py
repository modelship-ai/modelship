"""Placement options of the cluster-wide singleton actors and the operator probe."""

from unittest.mock import patch

import pytest

from modelship.infer import deploy_coordinator, download_leases, replica_coordinator
from modelship.state import memory

_NODE_ID = "ab" * 28

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
        (replica_coordinator.ReplicaCoordinator, replica_coordinator.get_or_create_replica_coordinator, -1),
        (memory.MemoryStoreActor, memory.get_or_create_memory_store_actor, -1),
        (download_leases.DownloadLeases, download_leases.get_or_create_leases, 0),
    ],
)
def test_singletons_are_pinned_to_the_head(actor_cls, getter, max_restarts):
    with patch.object(actor_cls, "options") as options:
        getter()
    kwargs = options.call_args.kwargs
    assert {k: kwargs[k] for k in _HEAD_PINNED} == _HEAD_PINNED
    assert kwargs["max_restarts"] == max_restarts


def test_operator_probe_runs_on_the_driver_node():
    with (
        patch.object(deploy_coordinator.ray, "get_runtime_context") as ctx,
        patch.object(deploy_coordinator.OperatorProbe, "options") as options,
    ):
        ctx.return_value.get_node_id.return_value = _NODE_ID
        deploy_coordinator.create_operator_probe()
    strategy = options.call_args.kwargs["scheduling_strategy"]
    assert (strategy.node_id, strategy.soft) == (_NODE_ID, False)
