import argparse
import os
from typing import TYPE_CHECKING

import yaml
from pydantic_yaml import parse_yaml_raw_as

from modelship.logging import get_logger
from modelship.utils import is_pathy
from modelship.utils.cli import model_from_args
from modelship.utils.config_schema import ModelLoader, ModelshipConfig

if TYPE_CHECKING:
    from modelship.infer.sources import PinnedSource

logger = get_logger("startup")


def load_yaml_config(path: str) -> ModelshipConfig:
    with open(path) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def load_raw_models(path: str) -> list[dict]:
    """Read the --config models.yaml as raw, pre-validation dicts.

    The effective-config store keeps raw dicts (not validated configs, which don't
    round-trip through num_gpus/tp normalization), so the deploy path merges at the
    raw-dict level; ``merge()`` validates this input before folding it in, and the
    merged result is validated again before deploy."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"--config {path} not found.")
    with open(path) as f:
        return parse_raw_models(f.read())


def parse_raw_models(text: str) -> list[dict]:
    """models.yaml text as raw, pre-validation dicts."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("models.yaml: top-level document must be a mapping with a 'models' key.")
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models.yaml: 'models' must be a list.")
    return models


WHISPERCPP_REPO = "ggerganov/whisper.cpp"


def _whispercpp_source_ref(model: str) -> str:
    """A bare name (`base.en`) is shorthand for its ggml file in the whisper.cpp HF repo."""
    if is_pathy(model) or "/" in model or ":" in model or os.path.exists(model):
        return model
    return f"{WHISPERCPP_REPO}:ggml-{model}.bin"


def _pin_label(pinned: "PinnedSource") -> str:
    from modelship.infer.sources import ArchiveSource, HfSource

    match pinned:
        case HfSource():
            return f"revision={pinned.revision}"
        case ArchiveSource():
            return f"sha256={pinned.sha256[:12]}"
        case _:
            return "local"


def resolve_all_model_sources(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: check every built-in-loader model's source, without
    downloading any weight bytes.

    Populates `_pinned_source` (and, for llama_server, the mmproj pin) on each
    config in place; actual download happens per-replica in
    `BaseInfer.ensure_downloaded`. Raises on the first failure (auth,
    missing repo, missing file, glob-no-match) so the operator sees it before
    any Ray actor spins up.

    Note: `main (modelship.driver)` sets HF_HOME before this runs; huggingface_hub
    latches it at import.
    """
    # Deferred: pulls huggingface_hub.
    from modelship.infer.sherpa_onnx.bundle import bundle_source
    from modelship.infer.sources import check_model_source

    for cfg in yml_conf.models:
        assert cfg.model is not None  # validator guarantees this for built-in loaders
        if cfg.loader == ModelLoader.sherpa_onnx:
            logger.info("Checking sherpa_onnx bundle for '%s': %s", cfg.name, cfg.model)
            cfg._pinned_source = bundle_source(cfg.model)
            logger.info("Checked '%s' (%s)", cfg.name, _pin_label(cfg._pinned_source))
            continue
        trust_remote_code = bool(cfg.vllm_engine_kwargs and cfg.vllm_engine_kwargs.trust_remote_code)
        # cfg.model stays as written: it feeds the fingerprint.
        ref = _whispercpp_source_ref(cfg.model) if cfg.loader == ModelLoader.whispercpp else cfg.model
        logger.info("Checking model source for '%s': %s", cfg.name, ref)
        try:
            pinned = check_model_source(ref, trust_remote_code=trust_remote_code)
        except FileNotFoundError as e:
            if ref == cfg.model:
                raise
            raise FileNotFoundError(
                f"Model '{cfg.name}': {cfg.model!r} is not a whisper.cpp model name in {WHISPERCPP_REPO!r} "
                f"(looked for {ref.split(':', 1)[1]!r})"
            ) from e
        cfg._pinned_source = pinned
        logger.info("Checked '%s' (%s)", cfg.name, _pin_label(pinned))

        if cfg.loader == ModelLoader.llama_server and cfg.llama_server_config and cfg.llama_server_config.mmproj:
            mmproj = cfg.llama_server_config.mmproj
            logger.info("Checking mmproj source for '%s': %s", cfg.name, mmproj)
            pinned_mmproj = check_model_source(mmproj, trust_remote_code=trust_remote_code)
            if not pinned_mmproj.resolves_to_gguf:
                raise ValueError(
                    f"Model '{cfg.name}': mmproj {mmproj!r} does not resolve to a GGUF file. Pick one with the "
                    f"`:filename` syntax (glob supported), e.g. `repo:mmproj-*.gguf`."
                )
            cfg.llama_server_config._pinned_mmproj = pinned_mmproj

        # GGUF is not supported on the vllm loader (vLLM 0.24 dropped in-tree
        # GGUF). Reject early using the listed filename, before any download.
        if cfg.loader == ModelLoader.vllm and pinned.resolves_to_gguf:
            raise ValueError(
                f"Model '{cfg.name}' resolves to a GGUF file, which the vllm loader does not support "
                f"(vLLM 0.24 dropped in-tree GGUF). Use `loader: llama_server` for GGUF models, or point "
                f"the vllm loader at a non-GGUF checkpoint (safetensors, or an AWQ/GPTQ/FP8 quant)."
            )
        if cfg.loader == ModelLoader.llama_server and not pinned.resolves_to_gguf:
            raise ValueError(
                f"Model '{cfg.name}': {cfg.model!r} does not resolve to a GGUF file, which the llama_server loader "
                f"needs. Pick one with the `:filename` syntax (glob supported), e.g. `repo:*Q4_K_M.gguf`."
            )


def validate_models(raw_models: list[dict]) -> ModelshipConfig:
    """Validate raw model dicts into a ModelshipConfig. Both models.yaml and the
    ``--model`` flags land here."""
    return ModelshipConfig.model_validate({"models": raw_models})


def resolve_input_models(args: argparse.Namespace) -> list[dict] | None:
    """The raw model dicts this invocation asks for, or None when it asks for
    none. ``--model`` describes one entry; otherwise the ``--config`` file supplies them.

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
    if getattr(args, "config_from_job", False):
        from modelship.remote import job_models_yaml

        return parse_raw_models(job_models_yaml())
    if args.config is None:
        return None
    return load_raw_models(args.config)
