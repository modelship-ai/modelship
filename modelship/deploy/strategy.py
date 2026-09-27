"""What a deploy request submits and deletes, planned against Serve's apps, and the submit itself."""

from dataclasses import dataclass

from ray import serve
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview, LoggingConfig

from modelship.deploy.actor_options import build_deployment_options
from modelship.deploy.ledger import RequestMode, Version, merge
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger
from modelship.utils.config_schema import parse_deployment_name

logger = get_logger("startup")

# Serve has stopped starting replicas for these; submitting again replaces the app.
_NOT_LIVE = (ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.DELETING)


@dataclass
class Plan:
    adds: list[ModelshipModelConfig]
    # the gateway's apps the request replaces or drops
    retired: list[str]


def gateway_apps(apps, gateway_name: str) -> dict[str, str]:
    """The gateway's model apps among *apps*, each mapped to its model name."""
    return {
        name: parsed[1]
        for name in apps
        if (parsed := parse_deployment_name(name)) is not None and parsed[0] == gateway_name
    }


def plan_request(
    mode: RequestMode,
    models: list[ModelshipModelConfig],
    committed: list[ModelshipModelConfig] | None,
    apps: dict[str, ApplicationStatusOverview],
    gateway_name: str,
) -> Plan:
    """What to submit and what to delete, against the gateway's apps in Serve."""
    wanted = (committed or []) if mode == "bare" else models
    wanted_apps = {c.deployment_name(gateway_name): c for c in wanted}
    adds = [c for name, c in wanted_apps.items() if name not in apps or apps[name].status in _NOT_LIVE]
    if mode == "bare":
        return Plan(adds, [])
    names = {c.name for c in wanted}
    retired = sorted(
        name
        for name, model in gateway_apps(apps, gateway_name).items()
        if name not in wanted_apps and (mode == "reconcile" or model in names)
    )
    return Plan(adds, retired)


def proposed_models(
    mode: RequestMode, models: list[dict] | None, committed: list[dict] | None, gateway_name: str
) -> list[dict] | None:
    """The model set the request commits; None when it leaves the committed version as it is."""
    if mode == "bare" or models is None:
        return None
    proposed = list(models) if mode == "reconcile" else merge(committed or [], models, gateway_name, "additive")
    if committed is not None and Version(0, proposed).apps(gateway_name) == Version(0, committed).apps(gateway_name):
        return None
    return proposed


def unnamed_apps(gateway_name: str, committed: list[dict] | None, apps) -> list[str]:
    """The gateway's apps that *committed* doesn't name; every one of them when there's no committed version."""
    keep = set(Version(0, committed).apps(gateway_name).values()) if committed is not None else set()
    return sorted(name for name in gateway_apps(apps, gateway_name) if name not in keep)


def submit_app(
    config: ModelshipModelConfig,
    gateway_name: str,
    serve_logging_config: LoggingConfig,
    env: dict[str, str] | None = None,
) -> None:
    """Hand one model to Serve and return; its replicas come up afterwards, and
    pend rather than fail when the cluster has no room for them yet."""
    deployment_name = config.deployment_name(gateway_name)
    deploy_opts = build_deployment_options(config, env)

    # Mutually exclusive, enforced at config validation — pass Serve exactly one.
    if config.autoscaling_config is not None:
        scaling_opts: dict = {"autoscaling_config": config.autoscaling_config.to_serve_dict()}
    else:
        scaling_opts = {"num_replicas": config.num_replicas}

    logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
    serve.run_many(
        [
            serve.RunTarget(
                target=ModelDeployment.options(
                    name=deployment_name,
                    max_constructor_retry_count=1,
                    logging_config=serve_logging_config,
                    **scaling_opts,
                    **deploy_opts,
                ).bind(config),
                name=deployment_name,
                route_prefix=None,
            )
        ],
        wait_for_applications_running=False,
    )
