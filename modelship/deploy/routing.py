"""A gateway's model table, computed from its routing version and Serve's application statuses."""

from collections.abc import Mapping
from dataclasses import dataclass

from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview

# Serve reports replica states as plain strings, though typed as its ReplicaState enum.
_RUNNING = "RUNNING"


@dataclass
class Routing:
    # app name -> model name, one app per model
    models: dict[str, str]
    # models the gateway reports ready only once they are in `models`
    expected: list[str]


def can_serve(app: ApplicationStatusOverview) -> bool:
    """RUNNING (which includes an app autoscaled to zero replicas) or with a running replica, and not being deleted."""
    if app.status == ApplicationStatus.DELETING:
        return False
    if app.status == ApplicationStatus.RUNNING:
        return True
    return any(
        state == _RUNNING and count > 0
        for deployment in app.deployments.values()
        for state, count in deployment.replica_states.items()
    )


def running_replicas(app: ApplicationStatusOverview) -> int:
    return sum(
        count
        for deployment in app.deployments.values()
        for state, count in deployment.replica_states.items()
        if state == _RUNNING
    )


def compute_routing(
    gateway_name: str, targets: Mapping[str, str], apps: Mapping[str, ApplicationStatusOverview]
) -> Routing | None:
    """Each model's app from *targets* (model -> app), while it can serve. None when *apps* lacks the gateway's own app."""
    if gateway_name not in apps:
        return None
    models = {app: model for model, app in targets.items() if app in apps and can_serve(apps[app])}
    return Routing(models=models, expected=list(targets))
