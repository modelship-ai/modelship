"""A gateway's model table, computed from its target apps and Serve's application statuses."""

from collections.abc import Mapping
from dataclasses import dataclass

from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview

from modelship.utils.config_schema import parse_deployment_name

# Serve reports replica states as plain strings, though typed as its ReplicaState enum.
_RUNNING = "RUNNING"


@dataclass
class Routing:
    # app name -> model name, one app per model
    models: dict[str, str]
    # models the gateway reports ready only once they are in `models`
    expected: list[str]
    # this gateway's apps that neither serve a model nor are its target
    unused: set[str]
    # this gateway's apps that are unused or being deleted
    retiring: set[str]


def can_serve(app: ApplicationStatusOverview) -> bool:
    """At least one running replica, and not being deleted."""
    if app.status == ApplicationStatus.DELETING:
        return False
    return any(
        state == _RUNNING and count > 0
        for deployment in app.deployments.values()
        for state, count in deployment.replica_states.items()
    )


def compute_routing(
    gateway_name: str, targets: Mapping[str, str] | None, apps: Mapping[str, ApplicationStatusOverview]
) -> Routing | None:
    """Each model's app: its target when it can serve, else the newest of its other apps that can.
    *targets* maps model -> target app; None when unknown, which routes every model and marks nothing
    unused. None when *apps* lacks the gateway's own app."""
    if gateway_name not in apps:
        return None
    by_model: dict[str, list[str]] = {}
    for name in apps:
        parsed = parse_deployment_name(name)
        if parsed is not None and parsed[0] == gateway_name:
            by_model.setdefault(parsed[1], []).append(name)

    wanted = targets if targets is not None else dict.fromkeys(by_model, "")
    models: dict[str, str] = {}
    for model, target in wanted.items():
        if target in apps and can_serve(apps[target]):
            models[target] = model
            continue
        others = [name for name in by_model.get(model, []) if name != target and can_serve(apps[name])]
        if others:
            models[max(others, key=lambda name: apps[name].last_deployed_time_s)] = model

    present = {name for name, app in apps.items() if app.status != ApplicationStatus.DELETING}
    mine = {name for names in by_model.values() for name in names}
    if targets is None:
        return Routing(models=models, expected=sorted(set(models.values())), unused=set(), retiring=mine - present)
    expected = [model for model, target in targets.items() if target in present or model in models.values()]
    unused = (mine & present) - set(models) - set(targets.values())
    return Routing(models=models, expected=expected, unused=unused, retiring=unused | (mine - present))
