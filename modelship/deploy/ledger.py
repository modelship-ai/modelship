"""Each gateway's deploy versions: the committed models and pipelines, and the version before it, in the state store.

The store holds raw, user-equivalent model dicts, not dumped validated configs: validation normalizes
``num_gpus``/``tensor_parallel_size`` non-idempotently, so a dumped config would reload with another fingerprint.
"""

from dataclasses import dataclass, field
from typing import Literal

from modelship.deploy.config import validate_models
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.state import StateStore
from modelship.utils.config_schema import parse_pipeline

logger = get_logger("startup")

DeployMode = Literal["additive", "reconcile"]
# bare: redeploy the committed version's missing apps
RequestMode = Literal["additive", "reconcile", "bare"]
ReplaceStrategy = Literal["blue_green", "stop_start"]

# State-store namespace; one key per gateway: "effective/<gateway-name>".
_NAMESPACE = "effective"


@dataclass
class DeployRequest:
    gateway: str
    mode: RequestMode
    strategy: ReplaceStrategy
    # raw model dicts as sent; None for a bare request
    models: list[dict] | None
    # per-deploy settings replicas read from their runtime_env
    env: dict[str, str]
    # raw pipeline dicts as sent; None when the request names none
    pipelines: list[dict] | None = None
    # set by the deploy coordinator
    id: str = ""


@dataclass(frozen=True)
class Version:
    number: int
    models: list[dict]
    pipelines: list[dict] = field(default_factory=list)

    def apps(self, gateway_name: str) -> dict[str, str]:
        """Model name -> the app it runs on."""
        configs = [ModelshipModelConfig.model_validate(raw) for raw in self.models]
        return {c.name: c.deployment_name(gateway_name) for c in configs}


def pipeline_apps(pipelines: list[dict], gateway_name: str) -> dict[str, str]:
    """Pipeline name -> its app."""
    configs = [parse_pipeline(raw) for raw in pipelines]
    return {c.name: c.deployment_name(gateway_name) for c in configs}


def _version(data) -> Version | None:
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return None
    pipelines = data.get("pipelines")
    return Version(int(data.get("version", 0)), data["models"], pipelines if isinstance(pipelines, list) else [])


def read_versions(store: StateStore, gateway_name: str) -> tuple[Version | None, Version | None]:
    """The gateway's committed version and the one before it; None where there is none."""
    data = store.get(f"{_NAMESPACE}/{gateway_name}")
    committed = _version(data)
    previous = _version(data.get("previous")) if committed is not None and isinstance(data, dict) else None
    return committed, previous


def commit_version(
    store: StateStore, gateway_name: str, models: list[dict], pipelines: list[dict] | None = None
) -> Version:
    """Writes *models* and *pipelines* as the gateway's next version, keeping the committed one as its previous."""
    committed, _ = read_versions(store, gateway_name)
    version = Version(committed.number + 1 if committed else 1, models, pipelines or [])
    value: dict = {"version": version.number, "models": models, "pipelines": version.pipelines}
    if committed is not None:
        value["previous"] = {"version": committed.number, "models": committed.models, "pipelines": committed.pipelines}
    store.set(f"{_NAMESPACE}/{gateway_name}", value)
    return version


def _identity(raw: dict, gateway_name: str) -> tuple[str, str]:
    """(deployment_name, model_name) for a raw model dict from one validation pass."""
    cfg = ModelshipModelConfig.model_validate(raw)
    return cfg.deployment_name(gateway_name), cfg.name


def merge(
    effective_raw: list[dict],
    input_raw: list[dict],
    gateway_name: str,
    mode: DeployMode,
) -> list[dict]:
    """Fold the user's input into the committed raw model set under *mode*.

    - additive: replace-by-name — the same app (same deployment_name) takes the
      input's entry and its replica-count fields; a different config sharing a model
      name replaces the existing entry for that name rather than joining it.
    - reconcile: input replaces the committed set entirely.

    Validates *input_raw* alone (not the merged result), so a model name reused with a
    different config in the input is rejected before it reaches a committed version.
    """
    to_config(input_raw)
    if mode == "reconcile":
        return list(input_raw)

    # dep_name -> raw dict, and model_name -> its current dep_name, both built in
    # one validation pass per dict so lookups below are O(1) instead of rescanning
    # (and re-validating) the whole accumulated set per input entry.
    merged: dict[str, dict] = {}
    dep_name_by_model_name: dict[str, str] = {}
    for m in effective_raw:
        dep_name, model_name = _identity(m, gateway_name)
        merged[dep_name] = m
        dep_name_by_model_name[model_name] = dep_name

    for d in input_raw:
        dep_name, model_name = _identity(d, gateway_name)
        if dep_name in merged:
            merged[dep_name] = d
            continue
        prior_dep_name = dep_name_by_model_name.get(model_name)
        if prior_dep_name is not None:
            _log_replacement(model_name, merged.pop(prior_dep_name), d)
        merged[dep_name] = d
        dep_name_by_model_name[model_name] = dep_name

    return list(merged.values())


def merge_pipelines(effective_raw: list[dict], input_raw: list[dict], mode: DeployMode) -> list[dict]:
    """Fold the input pipelines into the committed ones under *mode*: additive replaces by name, reconcile
    replaces the set."""
    if mode == "reconcile":
        return list(input_raw)
    names = {parse_pipeline(raw).name for raw in input_raw}
    kept = [raw for raw in effective_raw if parse_pipeline(raw).name not in names]
    return kept + list(input_raw)


def _log_replacement(model_name: str, prior: dict, incoming: dict) -> None:
    """A name already in the committed set is replaced, not joined. Pointing it at
    different weights is worth a warning; any other config change is routine."""
    if prior.get("model") == incoming.get("model"):
        logger.info("Model %r config changed; replacing the existing deployment.", model_name)
        return
    logger.warning(
        "Model %r is already deployed from %r and will be REPLACED by %r — one model name maps to "
        "exactly one deployment. Give one of them a distinct name to run both side by side.",
        model_name,
        prior.get("model"),
        incoming.get("model"),
    )


def to_config(raw_models: list[dict], raw_pipelines: list[dict] | None = None) -> ModelshipConfig:
    """Validate raw model and pipeline dicts into a ModelshipConfig for the deploy path."""
    return validate_models(raw_models, raw_pipelines)
