import argparse
import os
from pathlib import Path

import yaml
from pydantic_yaml import parse_yaml_raw_as

from modelship.logging import get_logger
from modelship.utils import is_pathy
from modelship.utils.cli import model_from_args
from modelship.utils.config_schema import ModelLoader, ModelshipConfig

logger = get_logger("startup")


def default_config_path(config_dir: Path | None = None) -> Path:
    """The default config/models.yaml path used absent an explicit --config."""
    config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config"
    return config_dir / "models.yaml"


def resolve_config_path(arg_path: str | None, config_dir: Path | None = None) -> str:
    """Resolve the models.yaml to deploy.

    Precedence:
    1. An explicit ``--config`` path always wins (most specific signal); it must exist.
    2. Otherwise the default ``config/models.yaml`` must exist.
    """
    if arg_path:
        if not os.path.exists(arg_path):
            raise FileNotFoundError(f"--config {arg_path} not found.")
        return arg_path

    default = default_config_path(config_dir)
    if default.exists():
        return str(default)

    raise FileNotFoundError(f"{default} not found. Copy an example config from config/examples/ to config/models.yaml.")


def load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def load_raw_models(arg_path: str | None) -> list[dict]:
    """Read the user's models.yaml as raw, pre-validation dicts.

    The effective-config store keeps raw dicts (not validated configs, which don't
    round-trip through num_gpus/tp normalization), so the deploy path merges at the
    raw-dict level; ``merge()`` validates this input before folding it in, and the
    merged result is validated again before deploy."""
    with open(resolve_config_path(arg_path)) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("models.yaml: top-level document must be a mapping with a 'models' key.")
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models.yaml: 'models' must be a list.")
    return models


def _is_whispercpp_builtin_ref(model: str) -> bool:
    """A bare pywhispercpp built-in name (e.g. `base.en`) — no repo or path
    separator. Validated in the actor; the driver may lack pywhispercpp."""
    return "/" not in model and not os.path.exists(model)


def _resolve_sherpa_onnx_source(cfg) -> None:
    """Not an HF repo, so no `check_model_source`. A local-directory `model:` is
    validated here at driver preflight; a bare registry name has nothing to pin
    since the tarball fetch happens in the actor."""
    from modelship.infer.sherpa_onnx.bundle import validate_bundle
    from modelship.infer.sherpa_onnx.registry import REGISTRY

    model = cfg.model
    assert model is not None  # validator guarantees this for built-in loaders
    if is_pathy(model):
        path = os.path.expanduser(model)
        name = os.path.basename(path.rstrip("/"))
        entry = REGISTRY[name]  # config validation already guarantees this key exists
        logger.info("Checking sherpa_onnx bundle for '%s': %s", cfg.name, path)
        validate_bundle(path, entry)
        logger.info("Checked '%s' (local bundle, no download)", cfg.name)
    else:
        logger.info(
            "Skipping source check for '%s': sherpa_onnx registry model %r is fetched by the actor", cfg.name, model
        )


def resolve_all_model_sources(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: check every built-in-loader model's source, without
    downloading any weight bytes.

    Populates `_pinned_source` (and, for llama_server, the mmproj pin) on each
    config in place; actual download happens per-replica in
    `BaseInfer.ensure_downloaded`. Raises on the first failure (auth,
    missing repo, missing file, glob-no-match) so the operator sees it before
    any Ray actor spins up.

    Note: HF_HOME / VLLM_CACHE_ROOT / FLASHINFER_WORKSPACE_BASE are set at module
    load time in modelship/driver.py — `huggingface_hub.HF_HOME` is latched at
    import, so setting them later doesn't take effect.
    """
    # Deferred: pulls huggingface_hub.
    from modelship.infer.model_resolver import check_model_source

    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.whispercpp and cfg.model and _is_whispercpp_builtin_ref(cfg.model):
            # pywhispercpp resolves/downloads its own built-in models; nothing to pin here.
            logger.info("Skipping source check for '%s': pywhispercpp built-in model %r", cfg.name, cfg.model)
            continue
        if cfg.loader == ModelLoader.sherpa_onnx:
            _resolve_sherpa_onnx_source(cfg)
            continue
        assert cfg.model is not None  # validator guarantees this for built-in loaders
        trust_remote_code = bool(cfg.vllm_engine_kwargs and cfg.vllm_engine_kwargs.trust_remote_code)
        logger.info("Checking model source for '%s': %s", cfg.name, cfg.model)
        cfg._pinned_source = check_model_source(cfg.model, trust_remote_code=trust_remote_code)
        logger.info("Checked '%s' (revision=%s)", cfg.name, cfg._pinned_source.revision or "local")

        if cfg.loader == ModelLoader.llama_server and cfg.llama_server_config and cfg.llama_server_config.mmproj:
            logger.info("Checking mmproj source for '%s': %s", cfg.name, cfg.llama_server_config.mmproj)
            cfg.llama_server_config._pinned_mmproj = check_model_source(
                cfg.llama_server_config.mmproj, trust_remote_code=trust_remote_code
            )

        # GGUF is not supported on the vllm loader (vLLM 0.24 dropped in-tree
        # GGUF). Reject early using the listed filename, before any download.
        if cfg.loader == ModelLoader.vllm and cfg._pinned_source.resolves_to_gguf:
            raise ValueError(
                f"Model '{cfg.name}' resolves to a GGUF file, which the vllm loader does not support "
                f"(vLLM 0.24 dropped in-tree GGUF). Use `loader: llama_server` for GGUF models, or point "
                f"the vllm loader at a non-GGUF checkpoint (safetensors, or an AWQ/GPTQ/FP8 quant)."
            )


def config_absent(arg_path: str | None) -> bool:
    """True when there's no file to load: no ``--config`` and no default file.
    An explicit ``--config`` that doesn't exist is still a hard error."""
    return arg_path is None and not default_config_path().exists()


def validate_models(raw_models: list[dict]) -> ModelshipConfig:
    """Validate raw model dicts into a ModelshipConfig. Both models.yaml and the
    ``--model`` flags land here."""
    return ModelshipConfig.model_validate({"models": raw_models})


def resolve_input_models(args: argparse.Namespace) -> list[dict] | None:
    """The raw model dicts this invocation asks for, or None when it asks for
    none. ``--model`` describes one entry; otherwise models.yaml supplies them.

    Ray-free, so the launcher can validate the result before importing ray.
    """
    # parse_args rejects this first; kept for callers that build a Namespace directly.
    if args.model is not None and args.config is not None:
        raise ValueError(
            "--model and --config are mutually exclusive: --model deploys a single model, "
            "--config deploys the set in a models.yaml. Add the model to the file instead."
        )
    cli_model = model_from_args(args)
    if cli_model is not None:
        return [cli_model]
    if config_absent(args.config):
        return None
    return load_raw_models(args.config)
