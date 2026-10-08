"""vLLM tool-call / reasoning parser-name detection, run inside the vllm actor.

Covers `resolve_tool_parser` / `resolve_reasoning_parser` resolving the parser
name `init_serving_chat` hands to `OnlineRenderer`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.reasoning import ReasoningParser as VllmReasoningParser
from vllm.reasoning import ReasoningParserManager as VllmReasoningParserManager
from vllm.tool_parsers import ToolParserManager as VllmToolParserManager

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
)
from modelship.infer.vllm.parsing import detect
from modelship.infer.vllm.parsing.detect import (
    classify_tool_template,
    detect_boolean_defaults,
    detect_template_toggle_defaults,
    discover_template_vars,
    resolve_reasoning_parser,
    resolve_tool_parser,
)


def _make_cfg(**overrides) -> ModelshipModelConfig:
    base = {
        "name": "m",
        "model": "some/model",
        "usecase": ModelUsecase.generate,
        "loader": ModelLoader.vllm,
    }
    base.update(overrides)
    return ModelshipModelConfig(**base)


class TestClassifyToolTemplate:
    """``classify_tool_template`` must return names matching vLLM's own
    ``ToolParserManager`` registry, which ``resolve_tool_parser`` validates against."""

    def test_no_tool_markers_returns_none(self):
        assert classify_tool_template("plain template with no markers") is None

    def test_gemma4_marker(self):
        assert classify_tool_template("{% if tools %}<|tool_call>{% endif %}") == "gemma4"

    def test_function_gemma_marker_matches_vllm_name(self):
        # vLLM registers this parser as "functiongemma" (no underscore).
        assert classify_tool_template("{% if tools %}<start_function_call>{% endif %}") == "functiongemma"

    def test_qwen3_coder_function_marker_routes_ahead_of_hermes(self):
        # The chat template mentions tools (gating clause) and contains
        # ``<function=`` — must not fall through to Hermes.
        template = "{% if tools %}<tool_call>\n<function={{ name }}>{% endif %}"
        assert classify_tool_template(template) == "qwen3_coder"

    def test_qwen3_coder_parameter_marker(self):
        template = "{% if tools %}<parameter={{ key }}>value</parameter>{% endif %}"
        assert classify_tool_template(template) == "qwen3_coder"

    def test_hermes_template_without_function_marker_stays_hermes(self):
        template = '{% if tools %}<tool_call>{"name": "x"}</tool_call>{% endif %}'
        assert classify_tool_template(template) == "hermes"

    def test_mistral_marker(self):
        assert classify_tool_template("{% if tools %}[TOOL_CALLS]{% endif %}") == "mistral"

    def test_llama3_json_marker(self):
        assert classify_tool_template("{% if tools %}<|python_tag|>{% endif %}") == "llama3_json"

    def test_unrecognized_markers_returns_unknown(self):
        assert classify_tool_template("{% if tools %}some tool syntax{% endif %}") == "unknown"


def _hf(model_type: str | None = None, architecture: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_type=model_type, architectures=[architecture] if architecture else None)


def _parser(start: str | None = "<think>", end: str | None = "</think>", error: Exception | None = None) -> type:
    class Parser:
        reasoning_start_str = start
        reasoning_end_str = end

        def __init__(self, tokenizer: Any) -> None:
            if error:
                raise error

    return Parser


@pytest.fixture
def registry(monkeypatch) -> dict[str, Any]:
    parsers: dict[str, Any] = {}

    def get(name: str) -> Any:
        if isinstance(parsers[name], Exception):
            raise parsers[name]
        return parsers[name]

    monkeypatch.setattr(VllmReasoningParserManager, "list_registered", lambda: list(parsers))
    monkeypatch.setattr(VllmReasoningParserManager, "get_reasoning_parser", get)
    return parsers


class TestResolveReasoningParsers:
    def test_explicit_parser_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="deepseek_r1"))
        assert resolve_reasoning_parser(cfg, None, None, _hf()) == "deepseek_r1"

    def test_explicit_opt_out_leaves_none(self, registry):
        registry["acme9"] = _parser()
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_reasoning=False))
        assert resolve_reasoning_parser(cfg, "<think>", None, _hf(model_type="acme9")) is None

    def test_no_template_leaves_none(self, registry):
        registry["acme9"] = _parser(start=None, end=None)
        assert resolve_reasoning_parser(_make_cfg(), None, None, _hf(model_type="acme9")) is None

    def test_explicit_wins_over_auto(self, registry):
        registry.update(acme9=_parser(), other7=_parser())
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="other7"))
        assert resolve_reasoning_parser(cfg, "<think>", None, _hf(model_type="acme9")) == "other7"

    def test_unknown_explicit_raises(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="not-a-real-parser"))
        with pytest.raises(ValueError, match="not-a-real-parser"):
            resolve_reasoning_parser(cfg, None, None, _hf())

    def test_does_not_mutate_chat_template_kwargs(self):
        # Toggle defaults are pinned by detect_template_toggle_defaults at init,
        # not here — the resolver is pure name resolution.
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="gemma4"))
        resolve_reasoning_parser(cfg, None, None, _hf())
        assert cfg.chat_template_kwargs == {}

    def test_the_names_in_models_yaml_are_not_searched(self, registry):
        registry["acme9"] = _parser()
        cfg = _make_cfg(name="acme9", model="org/Acme-9-Chat")
        assert resolve_reasoning_parser(cfg, "<think>", None, _hf()) is None

    def test_model_type_outranks_architecture(self, registry):
        registry.update(acme9=_parser(), other7=_parser())
        hf_config = _hf(model_type="acme9", architecture="Other7ForCausalLM")
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, hf_config) == "acme9"

    def test_a_name_in_the_architecture_is_picked(self, registry):
        registry["other7"] = _parser()
        hf_config = _hf(architecture="Other7ForCausalLM")
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, hf_config) == "other7"

    def test_the_longest_name_wins(self, registry):
        registry.update(step3=_parser(), step3p5=_parser())
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, _hf(model_type="step3p5")) == "step3p5"

    def test_a_parser_whose_markers_are_absent_is_skipped(self, registry):
        registry["acme9"] = _parser()
        assert resolve_reasoning_parser(_make_cfg(), "plain", None, _hf(model_type="acme9")) is None

    def test_a_parser_without_markers_is_picked_on_its_name(self, registry):
        registry["acme9"] = _parser(start=None, end=None)
        assert resolve_reasoning_parser(_make_cfg(), "plain", None, _hf(model_type="acme9")) == "acme9"

    def test_a_parser_that_cannot_be_built_is_skipped(self, registry):
        registry.update(step3=_parser(), step3p5=_parser(error=RuntimeError("no think tokens")))
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, _hf(model_type="step3p5")) == "step3"

    def test_a_parser_that_fails_to_load_is_skipped(self, registry):
        registry["acme9"] = ImportError("optional dependency")
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, _hf(model_type="acme9")) is None

    def test_fallback_with_the_thinking_switch_is_qwen3(self, registry):
        registry.update(qwen3=_parser(), deepseek_r1=_parser())
        template = "{% if enable_thinking %}<think>{% endif %}"
        assert resolve_reasoning_parser(_make_cfg(), template, None, _hf()) == "qwen3"

    def test_fallback_without_the_thinking_switch_is_deepseek_r1(self, registry):
        registry.update(qwen3=_parser(), deepseek_r1=_parser())
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, _hf()) == "deepseek_r1"

    def test_no_fallback_for_a_template_without_its_markers(self, registry):
        registry.update(qwen3=_parser(), deepseek_r1=_parser())
        assert resolve_reasoning_parser(_make_cfg(), "plain", None, _hf()) is None

    def test_no_fallback_when_vllm_does_not_register_it(self, registry):
        registry["acme9"] = _parser()
        assert resolve_reasoning_parser(_make_cfg(), "<think>", None, _hf()) is None


class TestVllmReasoningRegistry:
    def test_both_fallback_parsers_are_registered(self):
        assert {"qwen3", "deepseek_r1"} <= set(VllmReasoningParserManager.list_registered())

    def test_a_parser_declares_its_markers(self):
        assert isinstance(VllmReasoningParser.reasoning_start_str, property)
        assert isinstance(VllmReasoningParser.reasoning_end_str, property)


class TestDiscoverTemplateVars:
    def test_finds_undeclared_vars(self):
        src = "{% if enable_thinking %}x{% endif %}{{ messages }}"
        assert discover_template_vars(src) == {"enable_thinking", "messages"}

    def test_plain_string_has_no_vars(self):
        assert discover_template_vars("just some text") == set()


class TestDetectBooleanDefaults:
    """A chat template's boolean-toggle default is recovered by rendering it
    forced on vs off and matching against the no-kwarg base render."""

    @staticmethod
    def _render_for(src: str):
        import jinja2

        template = jinja2.Environment().from_string(src)
        return lambda **kw: template.render(**kw)

    def test_opt_in_toggle_defaults_false(self):
        # Gemma shape: primer only when explicitly enabled -> default False.
        render = self._render_for("{% if enable_thinking %}<think>{% endif %}A")
        assert detect_boolean_defaults({"enable_thinking"}, render) == {"enable_thinking": False}

    def test_opt_out_toggle_defaults_true(self):
        # Qwen shape: thinking on unless explicitly disabled -> default True.
        render = self._render_for("{% if enable_thinking is not defined or enable_thinking %}<think>{% endif %}A")
        assert detect_boolean_defaults({"enable_thinking"}, render) == {"enable_thinking": True}

    def test_non_boolean_var_is_ambiguous_and_skipped(self):
        render = self._render_for("{{ bos_token }}A")
        assert detect_boolean_defaults({"bos_token"}, render) == {}

    def test_inert_var_skipped(self):
        render = self._render_for("{% if foo %}{% endif %}A")
        assert detect_boolean_defaults({"foo"}, render) == {}

    def test_raising_var_skipped_without_crashing(self):
        render = self._render_for("{{ fn() }}")
        assert detect_boolean_defaults({"fn"}, render) == {}

    def test_multiple_toggles_each_detected(self):
        render = self._render_for("{% if a %}A{% endif %}{% if b is not defined or b %}B{% endif %}")
        assert detect_boolean_defaults({"a", "b"}, render) == {"a": False, "b": True}


class TestDetectTemplateToggleDefaults:
    """Must exclude ``apply_chat_template``'s own signature params (e.g.
    ``add_generation_prompt``), which would otherwise collide with vLLM's
    explicit argument at request time."""

    def test_signature_params_are_excluded(self):
        class FakeTokenizer:
            # add_generation_prompt is a real branch AND a signature param -> must not be pinned.
            def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kw):
                thinking = "<think>" if kw.get("enable_thinking") else ""
                gen = "<gen>" if add_generation_prompt else ""
                return f"{gen}{thinking}A"

        src = "{% if add_generation_prompt %}<gen>{% endif %}{% if enable_thinking %}<think>{% endif %}A"
        result = detect_template_toggle_defaults(src, FakeTokenizer())
        assert "add_generation_prompt" not in result
        assert result == {"enable_thinking": False}


_HERMES_MARKERS = "{% if tools %}<tool_call>{% endif %}"


def _unbuildable(tokenizer: Any) -> None:
    raise RuntimeError("tokens missing")


@pytest.fixture
def tool_registry(monkeypatch) -> dict[str, Any]:
    parsers: dict[str, Any] = {}
    monkeypatch.setattr(VllmToolParserManager, "list_registered", lambda: list(parsers))
    monkeypatch.setattr(VllmToolParserManager, "get_tool_parser", parsers.__getitem__)
    return parsers


@pytest.fixture
def probe(monkeypatch):
    def answer(call: str | None, readers: tuple[str, ...] = ()) -> None:
        monkeypatch.setattr(detect, "rendered_tool_call", lambda tokenizer, eos_token_id: call)
        monkeypatch.setattr(detect, "tool_parsers_reading", lambda tokenizer, call: list(readers))

    return answer


class TestResolveToolParsers:
    def test_explicit_parser_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="hermes"))
        assert resolve_tool_parser(cfg, None, None, _hf(), None) == "hermes"

    def test_unknown_explicit_raises(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="not-a-real-parser"))
        with pytest.raises(ValueError, match="not-a-real-parser"):
            resolve_tool_parser(cfg, None, None, _hf(), None)

    def test_opt_out_leaves_none(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_auto_tool_choice=False, tool_call_parser="hermes"))
        assert resolve_tool_parser(cfg, None, None, _hf(), None) is None

    def test_explicit_wins_over_the_probe(self, probe):
        probe("call", ("glm45",))
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="hermes"))
        assert resolve_tool_parser(cfg, "plain", None, _hf(), None) == "hermes"

    def test_no_template_leaves_none(self, probe):
        probe("call", ("hermes",))
        assert resolve_tool_parser(_make_cfg(), None, None, _hf(), None) is None

    def test_the_parser_that_reads_the_call_is_picked(self, probe):
        probe("call", ("lfm2",))
        assert resolve_tool_parser(_make_cfg(), "plain", None, _hf(), None) == "lfm2"

    def test_tied_readers_go_to_the_name_closest_to_model_type(self, probe):
        probe("call", ("abcd1", "glm47"))
        assert resolve_tool_parser(_make_cfg(), "plain", None, _hf(model_type="glm4_moe"), None) == "glm47"

    def test_the_architecture_counts_for_the_closest_name(self, probe):
        probe("call", ("mimo", "qwen3_coder", "qwen3_xml"))
        hf_config = _hf(architecture="Qwen3ForCausalLM")
        assert resolve_tool_parser(_make_cfg(), "plain", None, hf_config, None) == "qwen3_coder"

    def test_a_shared_run_under_four_characters_does_not_count(self, probe):
        probe("call", ("abc9", "xyz1"))
        assert resolve_tool_parser(_make_cfg(), "plain", None, _hf(model_type="xyz_chat"), None) == "abc9"

    def test_the_names_in_models_yaml_do_not_break_a_tie(self, probe):
        probe("call", ("abcd1", "glm47"))
        cfg = _make_cfg(name="glm47", model="org/GLM-4.7")
        assert resolve_tool_parser(cfg, "plain", None, _hf(), None) == "abcd1"

    def test_a_call_no_parser_reads_leaves_none(self, probe):
        probe("call")
        assert resolve_tool_parser(_make_cfg(), _HERMES_MARKERS, None, _hf(), None) is None

    def test_without_a_written_call_a_name_in_model_type_is_picked(self, probe, tool_registry):
        probe(None)
        tool_registry["acme9"] = lambda tokenizer: None
        assert resolve_tool_parser(_make_cfg(), "plain", None, _hf(model_type="acme9"), None) == "acme9"

    def test_without_a_written_call_the_names_in_models_yaml_are_not_searched(self, probe, tool_registry):
        probe(None)
        tool_registry["acme9"] = lambda tokenizer: None
        cfg = _make_cfg(name="acme9", model="org/Acme-9-Chat")
        assert resolve_tool_parser(cfg, "plain", None, _hf(), None) is None

    def test_a_named_parser_that_cannot_be_built_is_skipped(self, probe, tool_registry):
        probe(None)
        tool_registry.update(acme9=_unbuildable, hermes=_unbuildable)
        hf_config = _hf(model_type="acme9")
        assert resolve_tool_parser(_make_cfg(), _HERMES_MARKERS, None, hf_config, None) == "hermes"

    def test_without_a_written_call_or_a_name_the_markers_decide(self, probe):
        probe(None)
        assert resolve_tool_parser(_make_cfg(), _HERMES_MARKERS, None, _hf(), None) == "hermes"

    def test_unrecognized_markers_leave_none(self, probe):
        probe(None)
        assert resolve_tool_parser(_make_cfg(), "{% if tools %}x{% endif %}", None, _hf(), None) is None
