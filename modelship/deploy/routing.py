"""A gateway's model table, computed from its routing version and Serve's application statuses."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from ray.serve._private.common import ReplicaState
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview, DeploymentStatusOverview

_RUNNING = ReplicaState.RUNNING


@dataclass
class Routing:
    # app name -> model name, one app per model
    models: dict[str, str]
    # models the gateway reports ready only once they are in `models`
    expected: list[str]
    # pipeline name -> {"config": raw pipeline, "app": its app, "ready": whether that app can serve}
    pipelines: dict[str, dict] = field(default_factory=dict)


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


def status_overview(app: dict) -> ApplicationStatusOverview:
    return ApplicationStatusOverview(
        status=app["status"],
        message=app["message"],
        last_deployed_time_s=app["last_deployed_time_s"],
        deployments={
            name: DeploymentStatusOverview(
                status=deployment["status"],
                status_trigger=deployment["status_trigger"],
                replica_states=dict(Counter(ReplicaState(r["state"]) for r in deployment["replicas"])),
                message=deployment["message"],
            )
            for name, deployment in app["deployments"].items()
        },
    )


def running_replica_ids(app: dict) -> set[str]:
    return {
        replica["replica_id"]
        for deployment in app["deployments"].values()
        for replica in deployment["replicas"]
        if replica["state"] == _RUNNING
    }


def compute_routing(
    gateway_name: str,
    targets: Mapping[str, str],
    apps: Mapping[str, ApplicationStatusOverview],
    pipelines: Mapping[str, dict] | None = None,
) -> Routing | None:
    """Each model's app from *targets* (model -> app), while it can serve, and each of *pipelines* (name ->
    {"config", "app"}) with whether its app can serve. None when *apps* lacks the gateway's own app."""
    if gateway_name not in apps:
        return None
    models = {app: model for model, app in targets.items() if app in apps and can_serve(apps[app])}
    ready = {
        name: {**pipeline, "ready": (app := pipeline["app"]) in apps and can_serve(apps[app])}
        for name, pipeline in (pipelines or {}).items()
    }
    return Routing(models=models, expected=list(targets), pipelines=ready)
