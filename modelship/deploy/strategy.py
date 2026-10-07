"""What Serve holds for the model apps, and the submits that change it."""

from typing import NamedTuple

from ray import serve
from ray.serve.config import AutoscalingConfig as ServeAutoscalingConfig
from ray.serve.schema import ApplicationStatus, LoggingConfig

from modelship.deploy.actor_options import build_deployment_options
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger
from modelship.utils.config_schema import AutoscalingConfig, parse_deployment_name

logger = get_logger("startup")

_AUTOSCALING_FIELDS = tuple(AutoscalingConfig.model_fields)


class ServeApp(NamedTuple):
    status: ApplicationStatus
    # the replica-count fields its deployment is set to, shaped as `serve_scaling`; None without a deployment
    scaling: dict | None


class LiveApp(NamedTuple):
    """What Serve holds for an app's deployment."""

    config: ModelshipModelConfig
    runtime_env: dict
    version: str


def gateway_apps(apps, gateway_name: str) -> dict[str, str]:
    """The gateway's model apps among *apps*, each mapped to its model name."""
    return {
        name: parsed[1]
        for name in apps
        if (parsed := parse_deployment_name(name)) is not None and parsed[0] == gateway_name
    }


def serve_scaling(config: ModelshipModelConfig) -> dict:
    """The replica-count fields Serve holds for *config*: `num_replicas` alone, or the autoscaling fields
    with Serve's defaults filled in."""
    if config.autoscaling_config is None:
        return {"num_replicas": config.num_replicas}
    autoscaling = ServeAutoscalingConfig(**config.autoscaling_config.to_serve_dict())
    return {name: getattr(autoscaling, name) for name in _AUTOSCALING_FIELDS}


def serve_apps() -> dict[str, ServeApp]:
    """Every Serve app's status and the replica-count fields its deployment is set to."""
    from ray.serve.context import _get_global_client

    client = _get_global_client()
    assert client is not None
    apps = {}
    for name, app in client.get_serve_details().get("applications", {}).items():
        deployment = app["deployments"].get(name)
        apps[name] = ServeApp(ApplicationStatus(app["status"]), _set_scaling(deployment) if deployment else None)
    return apps


def _set_scaling(deployment: dict) -> dict:
    config = deployment["deployment_config"]
    autoscaling = config.get("autoscaling_config")
    if autoscaling is None:
        return {"num_replicas": config.get("num_replicas")}
    return {name: autoscaling.get(name) for name in _AUTOSCALING_FIELDS}


def submit_app(
    config: ModelshipModelConfig,
    gateway_name: str,
    serve_logging_config: LoggingConfig,
    env: dict[str, str] | None = None,
) -> None:
    """Hand one model to Serve and return; its replicas come up afterwards, and
    pend rather than fail when the cluster has no room for them yet."""
    deployment_name = config.deployment_name(gateway_name)
    logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
    _run(config, deployment_name, build_deployment_options(config, env), serve_logging_config)


def live_app(app_name: str) -> LiveApp | None:
    """The config, runtime_env and version of the app's deployment in Serve; None when Serve has none."""
    from ray.serve.context import _get_global_client

    client = _get_global_client()
    assert client is not None
    try:
        info, _ = client.get_deployment_info(app_name, app_name)
    except KeyError:
        return None
    replica = info.replica_config
    return LiveApp(replica.init_args[0], replica.ray_actor_options["runtime_env"], info.version)


def rescale_app(app_name: str, live: LiveApp, scaling: dict, serve_logging_config: LoggingConfig) -> None:
    """Re-submits the live app with *scaling* as its replica-count fields; its running replicas stay."""
    config = live.config.model_copy(update=scaling)
    deploy_opts = build_deployment_options(config, {})
    deploy_opts["ray_actor_options"]["runtime_env"] = live.runtime_env
    logger.info("Rescaling model: %s (deployment: %s)", config.name, app_name)
    _run(config, app_name, deploy_opts, serve_logging_config, live.version)


def _run(
    config: ModelshipModelConfig,
    deployment_name: str,
    deploy_opts: dict,
    serve_logging_config: LoggingConfig,
    version: str | None = None,
) -> None:
    # Mutually exclusive, enforced at config validation — pass Serve exactly one.
    if config.autoscaling_config is not None:
        scaling_opts: dict = {"autoscaling_config": config.autoscaling_config.to_serve_dict()}
    else:
        scaling_opts = {"num_replicas": config.num_replicas}

    deployment = ModelDeployment.options(
        name=deployment_name,
        max_constructor_retry_count=1,
        logging_config=serve_logging_config,
        **scaling_opts,
        **deploy_opts,
    )
    if version is not None:
        # Serve restarts every replica of an app re-submitted under another version.
        deployment._version = version
    serve.run_many(
        [serve.RunTarget(target=deployment.bind(config), name=deployment_name, route_prefix=None)],
        wait_for_applications_running=False,
    )
