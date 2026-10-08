"""vLLM tool-call / reasoning parser-name detection, run inside the vllm actor.

Auto-detects which vLLM-native parser (``vllm.tool_parsers.ToolParserManager`` /
``vllm.reasoning.ReasoningParserManager``) a model calls for, with explicit config
always taking precedence. Names returned here are vLLM's own registered names.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("infer.vllm.parsing.detect")

# A fixed, minimal conversation to render templates against when probing their
# boolean toggle defaults. Content is irrelevant — only which branches the
# template takes for a given kwarg matters.
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]


def classify_tool_template(template: str) -> str | None:
    """Map a chat-template string to a vLLM tool-call parser name based on markers.

    - Returns "gemma4" if `<|tool_call>` markers are found.
    - Returns "functiongemma" if `<start_function_call>` markers are found.
    - Returns "qwen3_coder" if `<function=` or `<parameter=` markers are found.
    - Returns "hermes" if `<tool_call>` markers are found.
    - Returns "mistral" if `[TOOL_CALLS]` markers are found.
    - Returns "llama3_json" if `<|python_tag|>` markers are found.
    - Returns "unknown" if tool logic is found but format markers are not recognized.
    - Returns ``None`` if no chat template / no tool logic is detected.
    """
    if "tools" not in template and "tool_calls" not in template and "function" not in template:
        return None

    if "<|tool_call>" in template:
        return "gemma4"

    if "<start_function_call>" in template:
        return "functiongemma"

    # Qwen3-Coder shares the ``<tool_call>`` envelope with Hermes but the
    # body is XML rather than JSON. The ``<function=`` / ``<parameter=``
    # markers only appear in Qwen3-Coder templates, so check them before
    # the ``<tool_call>`` -> hermes branch below.
    if "<function=" in template or "<parameter=" in template:
        return "qwen3_coder"

    if "<tool_call>" in template:
        return "hermes"

    if "[TOOL_CALLS]" in template:
        return "mistral"

    if "<|python_tag|>" in template or "<|eom_id|>" in template:
        return "llama3_json"

    return "unknown"


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _named_in(registered: set[str], text: str | None) -> list[str]:
    haystack = _alnum(text or "")
    return sorted((name for name in registered if _alnum(name) in haystack), key=lambda name: (-len(name), name))


def _fits_template(name: str, tokenizer: Any, template: str) -> bool:
    from vllm.reasoning import ReasoningParserManager as VllmReasoningParserManager

    try:
        parser = VllmReasoningParserManager.get_reasoning_parser(name)(tokenizer)
    except Exception:
        return False
    markers = [marker for marker in (parser.reasoning_start_str, parser.reasoning_end_str) if marker]
    return not markers or any(marker in template for marker in markers)


def _is_tool_opt_out(cfg: ModelshipModelConfig) -> bool:
    """Explicit "no auto tool-call detection" signal: ``enable_auto_tool_choice: false``."""
    return cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs.enable_auto_tool_choice is False


def _is_reasoning_opt_out(cfg: ModelshipModelConfig) -> bool:
    """Explicit "no auto reasoning detection" signal: ``enable_reasoning: false``."""
    return cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs.enable_reasoning is False


def discover_template_vars(template_src: str) -> set[str]:
    """Undeclared (caller-supplied) variables a Jinja template reads.

    These are the names ``chat_template_kwargs`` can influence — the set we probe
    for boolean toggle defaults. Runs at deployment init, so a template Jinja can't
    even parse must not abort startup — it just means no toggle defaults get pinned.
    """
    import jinja2
    from jinja2 import meta as jinja2_meta

    env = jinja2.Environment()
    try:
        return jinja2_meta.find_undeclared_variables(env.parse(template_src))
    except jinja2.TemplateError:
        logger.warning("Chat template failed to parse for toggle-var discovery; skipping default pinning.")
        return set()


def detect_boolean_defaults(candidates: set[str], render: Callable[..., str]) -> dict[str, bool]:
    """Detect each candidate var's boolean default by rendering with it forced on/off.

    A var is a boolean toggle iff ``render(var=True)`` and ``render(var=False)``
    differ. Its default is whichever of the two the base render (no kwarg for that
    var) matches *exactly*. If the base matches neither or both, the var isn't a
    plain boolean (e.g. a token string interpolated verbatim) and is skipped. A
    render that raises for a given var skips that var; a base render that raises
    aborts entirely (``{}``) so the caller leaves the config untouched.
    """
    try:
        base = render()
    except Exception:
        return {}

    defaults: dict[str, bool] = {}
    for var in candidates:
        try:
            on = render(**{var: True})
            off = render(**{var: False})
        except Exception:
            continue
        if on == off:
            continue  # inert for this probe — not a toggle we can pin
        if base == on and base != off:
            defaults[var] = True
        elif base == off and base != on:
            defaults[var] = False
        # base matching neither/both -> ambiguous (non-bool); skip.
    return defaults


def detect_template_toggle_defaults(template_src: str, tokenizer: Any) -> dict[str, bool]:
    """Each chat-template boolean toggle's own default, for pinning into config.

    Renders a fixed probe conversation through ``tokenizer.apply_chat_template``
    with each discovered variable forced True/False (see ``detect_boolean_defaults``).
    Variables that are real ``apply_chat_template`` parameters (``add_generation_prompt``,
    ``tools``, ``messages``, ...) are excluded — pinning them into
    ``chat_template_kwargs`` would collide with vLLM's own explicit argument at
    request time (``TypeError: multiple values``).
    """
    candidates = discover_template_vars(template_src)
    try:
        reserved = set(inspect.signature(tokenizer.apply_chat_template).parameters)
    except (TypeError, ValueError):
        # Some tokenizer wrappers expose an uninspectable `apply_chat_template` (e.g.
        # a C-extension or mocked callable). Falling back to no exclusions is safe:
        # any reserved name that slips through as a candidate just fails its own
        # render probe below, which `detect_boolean_defaults` already tolerates.
        reserved = set()
    candidates -= reserved
    if not candidates:
        return {}

    def render(**kwargs: bool) -> str:
        return tokenizer.apply_chat_template(_PROBE_MESSAGES, tokenize=False, add_generation_prompt=True, **kwargs)

    return detect_boolean_defaults(candidates, render)


def resolve_tool_parser(cfg: ModelshipModelConfig, template: str | None) -> str | None:
    """Resolve the tool-call parser name to hand to ``OnlineRenderer``.

    Precedence: opt-out -> None; explicit ``tool_call_parser`` (validated against
    vLLM's registry, raises if unknown) -> that name; else classify the chat
    template and validate the detected name against the registry (warn +
    disable if unrecognized or unregistered); else None.
    """
    if _is_tool_opt_out(cfg):
        logger.info("Tool-call resolution skipped for '%s' (explicit opt-out).", cfg.name)
        return None

    from vllm.tool_parsers import ToolParserManager as VllmToolParserManager

    registered = set(VllmToolParserManager.list_registered())

    explicit = cfg.vllm_engine_kwargs.tool_call_parser
    if explicit is not None:
        if explicit not in registered:
            raise ValueError(
                f"Model '{cfg.name}' configures tool_call_parser={explicit!r} "
                f"which is not registered. Available: {sorted(registered) or '(none)'}."
            )
        logger.info("Using explicit tool_call_parser=%r for '%s'", explicit, cfg.name)
        return explicit

    if template is None:
        logger.info("No chat template found for '%s'; tool-call detection skipped.", cfg.name)
        return None

    detected = classify_tool_template(template)
    if detected is None:
        logger.info("No tool-calling support detected for '%s'; tools disabled.", cfg.name)
        return None
    if detected == "unknown":
        logger.warning(
            "Model '%s' chat template references tools but uses unrecognized markers; tool calling disabled.",
            cfg.name,
        )
        return None
    if detected not in registered:
        logger.warning(
            "Model '%s' uses tool format %r but no parser is registered; tool calling disabled.", cfg.name, detected
        )
        return None
    logger.info("Auto-detected tool_call_parser=%r for '%s'", detected, cfg.name)
    return detected


def resolve_reasoning_parser(
    cfg: ModelshipModelConfig, template: str | None, tokenizer: Any, hf_config: Any
) -> str | None:
    """Resolve the reasoning parser name to hand to the engine and ``OnlineRenderer``."""
    if _is_reasoning_opt_out(cfg):
        logger.info("Reasoning resolution skipped for '%s' (explicit opt-out).", cfg.name)
        return None

    from vllm.reasoning import ReasoningParserManager as VllmReasoningParserManager

    registered = set(VllmReasoningParserManager.list_registered())

    explicit = cfg.vllm_engine_kwargs.reasoning_parser
    if explicit is not None:
        if explicit not in registered:
            raise ValueError(
                f"Model '{cfg.name}' configures reasoning_parser={explicit!r} "
                f"which is not registered. Available: {sorted(registered) or '(none)'}."
            )
        logger.info("Using explicit reasoning_parser=%r for '%s'", explicit, cfg.name)
        return explicit

    if template is None:
        logger.info("No chat template found for '%s'; reasoning detection skipped.", cfg.name)
        return None

    sources = (
        ("model reference", cfg.model),
        ("model_type", hf_config.model_type),
        ("architecture", (hf_config.architectures or [None])[0]),
    )
    for source, text in sources:
        for name in _named_in(registered, text):
            if _fits_template(name, tokenizer, template):
                logger.info("Auto-detected reasoning_parser=%r for '%s' from its %s", name, cfg.name, source)
                return name

    # qwen3 reads the template's switch; deepseek_r1 does not.
    fallback = "qwen3" if "enable_thinking" in template else "deepseek_r1"
    if fallback not in registered:
        logger.warning("vLLM registers no %r reasoning parser; none detected for '%s'.", fallback, cfg.name)
        return None
    if _fits_template(fallback, tokenizer, template):
        logger.info("Auto-detected reasoning_parser=%r for '%s' from its chat template", fallback, cfg.name)
        return fallback
    return None
