"""Per-gateway *effective config* — the durable desired-state for deploys.

Every deploy (``mship start`` / ``mship deploy``), whatever its mode, folds the user's input into
the gateway's effective set (additive = union; reconcile = replace), then
the deploy ALWAYS reconciles the live cluster to that effective set. Self-heal is
then just "re-run the deploy": it reads the persisted effective set and reconciles
onto an empty cluster, restoring the TRUE live set after the cluster dies — not
just whatever the last user input happened to contain.

The store holds **raw, user-equivalent model dicts**, NOT serialized validated
configs: ``ModelshipModelConfig``'s ``num_gpus``/``tensor_parallel_size``
normalization is not idempotent, so a dumped validated config fails (or silently
mutates its fingerprint) on reload. Raw input dicts reload exactly as written.

This is the deploy-domain layer over the generic ``modelship.state`` store.
"""

from typing import Literal

from modelship.deploy.config import validate_models
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.state import StateStore

logger = get_logger("startup")

DeployMode = Literal["additive", "reconcile"]

# State-store namespace; one key per gateway: "effective/<gateway-name>".
_NAMESPACE = "effective"


def resolve_mode(*, reconcile: bool) -> DeployMode:
    """Map the CLI flags to the effective-config merge verb."""
    return "reconcile" if reconcile else "additive"


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
    """Fold the user's input into the effective raw model set under *mode*.

    - additive: replace-by-name — identical config (same deployment_name) is an
      idempotent skip; a different config sharing a model name replaces the
      existing entry for that name rather than joining it.
    - reconcile: input replaces the effective set entirely.

    Validates *input_raw* alone (not the merged result) via ModelshipConfig, so a
    model name reused with a different config in this file is rejected before it
    ever reaches the persisted effective set; pre-existing effective state from
    before this rule existed is left alone.
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
            continue
        prior_dep_name = dep_name_by_model_name.get(model_name)
        if prior_dep_name is not None:
            _log_replacement(model_name, merged.pop(prior_dep_name), d)
        merged[dep_name] = d
        dep_name_by_model_name[model_name] = dep_name

    return list(merged.values())


def _log_replacement(model_name: str, prior: dict, incoming: dict) -> None:
    """A name already in the effective set is replaced, not joined. Pointing it at
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


def to_config(raw_models: list[dict]) -> ModelshipConfig:
    """Validate raw model dicts into a ModelshipConfig for the deploy path."""
    return validate_models(raw_models)


def read_effective(store: StateStore, gateway_name: str) -> list[dict]:
    """Return the persisted effective raw model set for *gateway_name* (empty if
    none yet)."""
    data = store.get(f"{_NAMESPACE}/{gateway_name}")
    if not isinstance(data, dict):
        return []
    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning("Effective config for gateway %r has non-list 'models'; treating as empty.", gateway_name)
        return []
    return models


async def read_targets(store: StateStore, gateway_name: str) -> dict[str, str] | None:
    """Model name -> the app it should run on, from the persisted effective config;
    None when *gateway_name* has none."""
    data = await store.get_async(f"{_NAMESPACE}/{gateway_name}")
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    configs = [ModelshipModelConfig.model_validate(d) for d in models]
    return {c.name: c.deployment_name(gateway_name) for c in configs}


def write_effective(store: StateStore, gateway_name: str, raw_models: list[dict]) -> None:
    """Persist the effective raw model set for *gateway_name*."""
    store.set(f"{_NAMESPACE}/{gateway_name}", {"models": raw_models})
