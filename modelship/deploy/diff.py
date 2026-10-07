"""What a deploy request changes on its gateway and in what order, from the request, the gateway's committed
version and Serve's apps."""

from dataclasses import dataclass, field
from typing import NamedTuple

from ray.serve.schema import ApplicationStatus

from modelship.deploy.ledger import DeployRequest, merge, to_config
from modelship.deploy.strategy import ServeApp, gateway_apps, serve_scaling
from modelship.infer.infer_config import ModelshipModelConfig

# Serve has stopped starting replicas for these; submitting again replaces the app.
_NOT_LIVE = (ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.DELETING)

# the order a request's actions run in
_LEFTOVERS, _SCALE_DOWN, _STOP, _DEPLOY, _SCALE_UP, _COMMIT, _RETIRE = range(1, 8)


class Rescale(NamedTuple):
    model: str
    app: str
    # the replica-count fields the app gets, as `ModelshipModelConfig.scaling` gives them
    scaling: dict


@dataclass
class Diff:
    gateway: str
    # add | replace | rescale | redeploy | remove | keep, for each model the request decides on
    actions: dict[str, str] = field(default_factory=dict)
    # the gateway's apps the committed version doesn't name
    leftovers: list[str] = field(default_factory=list)
    # rescales that raise no replica limit, a replaced app's among them
    scale_downs: list[Rescale] = field(default_factory=list)
    deploys: list[ModelshipModelConfig] = field(default_factory=list)
    # rescales that raise a replica limit
    scale_ups: list[Rescale] = field(default_factory=list)
    # the apps of the replaced and the removed models
    retired: list[str] = field(default_factory=list)
    # the gateway's models after the request; None without a version
    models: list[dict] | None = None
    # whether the request writes *models* as the gateway's next version
    commits: bool = False
    # (step, text) per action; step 0 for what stays as it is
    steps: list[tuple[int, str]] = field(default_factory=list)

    def lines(self) -> list[str]:
        """The diff as text, in the order it runs."""
        ordered = sorted(self.steps, key=lambda step: step[0] or _RETIRE + 1)
        return [f"{step}. {text}" if step else text for step, text in ordered]


def build_diff(request: DeployRequest, committed: list[dict] | None, apps: dict[str, ServeApp]) -> Diff:
    """The diff of *request* against the gateway's *committed* version and Serve's *apps*."""
    gateway = request.gateway
    current = {c.name: c for c in to_config(committed).models} if committed else {}
    named = {c.deployment_name(gateway) for c in current.values()}
    diff = Diff(gateway, models=committed)
    diff.leftovers = sorted(name for name in gateway_apps(apps, gateway) if name not in named)
    if diff.leftovers:
        diff.steps.append((_LEFTOVERS, f"delete leftover apps: {', '.join(diff.leftovers)}"))

    wanted = list(current.values())
    if request.mode != "bare" and request.models is not None:
        wanted = list(to_config(request.models).models)
        proposed = merge(committed or [], request.models, gateway, request.mode)
        diff.commits = committed is None or _deployed(proposed, gateway) != _deployed(committed, gateway)
        if diff.commits:
            diff.models = proposed
            diff.steps.append((_COMMIT, "switch the gateway and commit"))
    for config in wanted:
        _decide(diff, request, config, current.get(config.name), apps)

    names = {c.name for c in wanted}
    others = sorted(name for name in current if name not in names)
    if request.mode == "reconcile":
        for name in others:
            diff.actions[name] = "remove"
            diff.retired.append(current[name].deployment_name(gateway))
            diff.steps.append((_STOP if request.strategy == "stop_start" else _RETIRE, f"remove {name}"))
    elif others:
        diff.steps.append((0, f"keep, not in the request: {', '.join(others)}"))
    kept = sorted(model for model, action in diff.actions.items() if action == "keep")
    if kept:
        diff.steps.append((0, f"keep: {', '.join(kept)}"))
    diff.retired.sort()
    return diff


def _decide(
    diff: Diff,
    request: DeployRequest,
    config: ModelshipModelConfig,
    before: ModelshipModelConfig | None,
    apps: dict[str, ServeApp],
) -> None:
    """Adds *config*'s action to *diff*; *before* is the committed model of its name."""
    model, app = config.name, config.deployment_name(diff.gateway)
    want = serve_scaling(config)
    if before is None:
        diff.actions[model] = "add"
        diff.deploys.append(config)
        diff.steps.append((_DEPLOY, f"add {model}"))
        return
    old_app = before.deployment_name(diff.gateway)
    if old_app != app:
        diff.actions[model] = "replace"
        diff.deploys.append(config)
        diff.retired.append(old_app)
        diff.steps.append((_DEPLOY, f"replace {model}: {', '.join(_config_changes(before, config))}"))
        old = apps.get(old_app)
        if request.strategy == "blue_green" and old is not None and _shrinks(old, want):
            assert old.scaling is not None
            diff.scale_downs.append(Rescale(model, old_app, config.scaling()))
            diff.steps.append((_SCALE_DOWN, f"rescale {model}'s replaced app: {_scaling_changes(old.scaling, want)}"))
        return
    held = apps.get(app)
    if held is None or held.status in _NOT_LIVE:
        diff.actions[model] = "redeploy"
        diff.deploys.append(config)
        why = "missing from Serve" if held is None else f"{held.status.value} in Serve"
        diff.steps.append((_DEPLOY, f"redeploy {model} ({why})"))
    elif held.scaling != want:
        diff.actions[model] = "rescale"
        rises = held.scaling is None or _rises(held.scaling, want)
        (diff.scale_ups if rises else diff.scale_downs).append(Rescale(model, app, config.scaling()))
        note = " (Serve differs from the committed version)" if before.scaling() == config.scaling() else ""
        text = f"rescale {model}: {_scaling_changes(held.scaling, want)}{note}"
        diff.steps.append((_SCALE_UP if rises else _SCALE_DOWN, text))
    else:
        diff.actions[model] = "keep"


def _deployed(models: list[dict], gateway_name: str) -> dict[str, tuple[str, dict]]:
    """Model name -> its app and replica-count fields."""
    configs = [ModelshipModelConfig.model_validate(raw) for raw in models]
    return {c.name: (c.deployment_name(gateway_name), c.scaling()) for c in configs}


def _limits(scaling: dict) -> tuple[int, int]:
    """The fewest and the most replicas *scaling* allows."""
    if "num_replicas" in scaling:
        return scaling["num_replicas"], scaling["num_replicas"]
    return scaling["min_replicas"], scaling["max_replicas"]


def _rises(have: dict, want: dict) -> bool:
    return any(new > old for old, new in zip(_limits(have), _limits(want), strict=True))


def _shrinks(old: ServeApp, want: dict) -> bool:
    """Whether a replaced app is live and *want* lowers a replica limit of it without raising the other."""
    if old.status in _NOT_LIVE or old.scaling is None:
        return False
    return _limits(want) != _limits(old.scaling) and not _rises(old.scaling, want)


def _scaling_changes(have: dict | None, want: dict) -> str:
    if have is None:
        return f"unknown -> {_replicas(want)}"
    if have.keys() != want.keys():
        return f"{_replicas(have)} -> {_replicas(want)}"
    prefix = "" if "num_replicas" in want else "autoscaling_config."
    return ", ".join(f"{prefix}{key}: {have[key]} -> {want[key]}" for key in want if have[key] != want[key])


def _replicas(scaling: dict) -> str:
    low, high = _limits(scaling)
    return f"num_replicas {low}" if "num_replicas" in scaling else f"autoscaling_config {low}..{high}"


def _config_changes(before: ModelshipModelConfig, after: ModelshipModelConfig) -> list[str]:
    old, new = (_flat(config.model_dump(mode="json", exclude={"name"})) for config in (before, after))
    return [
        f"{path}: {old.get(path)} -> {new.get(path)}"
        for path in sorted(old.keys() | new.keys())
        if old.get(path) != new.get(path)
    ]


def _flat(value, prefix: str = "") -> dict[str, object]:
    """*value*'s leaves by dotted path."""
    if isinstance(value, dict):
        return {path: leaf for key, item in value.items() for path, leaf in _flat(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: value}
