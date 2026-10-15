from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.infer.vllm import vllm_infer
from modelship.infer.vllm.vllm_infer import VllmInfer, _ChatParsers
from modelship.logging import VLLM_CHILD_ENV

_THINKING_TEMPLATE = "{% if tools %}<tool_call>{% endif %}<think>"
_PLAIN_TEMPLATE = "plain"


def _config(usecase: ModelUsecase = ModelUsecase.generate, **engine_kwargs: Any) -> ModelshipModelConfig:
    config = ModelshipModelConfig(
        name="m", model="org/m", usecase=usecase, loader=ModelLoader.vllm, vllm_engine_kwargs=engine_kwargs
    )
    config._resolved_path = "/models/m"
    return config


class _FakeReasoningConfig:
    """Stands in for vLLM's `ReasoningConfig`, whose `initialize_token_ids` needs a real tokenizer."""

    has_markers = True

    def __init__(self) -> None:
        self.reasoning_parser = ""
        self.enabled = False
        self.initialized_with: Any = None

    def initialize_token_ids(self, model_config: Any) -> None:
        self.initialized_with = model_config
        self.enabled = self.has_markers


def _build(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelshipModelConfig,
    template: str | None,
    *,
    has_markers: bool = True,
) -> tuple[VllmInfer, dict[str, Any]]:
    started: dict[str, Any] = {}

    class EngineArgs:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def create_engine_config(self, usage_context: Any) -> Any:
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type=None, architectures=None),
                    try_get_generation_config=dict,
                ),
                structured_outputs_config=SimpleNamespace(reasoning_parser=""),
                reasoning_config=None,
            )

    class Engine:
        @classmethod
        def from_vllm_config(cls, vllm_config: Any, **kwargs: Any) -> Any:
            started["reasoning_parser"] = vllm_config.structured_outputs_config.reasoning_parser
            started["reasoning_config"] = vllm_config.reasoning_config
            return SimpleNamespace(shutdown=lambda: None)

    def tokenizer(model_config: Any) -> Any:
        assert template is not None
        return SimpleNamespace(get_chat_template=lambda: template, get_vocab=lambda: {"<think>": 1, "</think>": 2})

    monkeypatch.setattr(vllm_infer, "run_preflight", lambda config, hw: {})
    monkeypatch.setattr(vllm_infer, "discover_hardware", lambda: None)
    monkeypatch.setattr(vllm_infer, "resolve_gpu_memory_utilization", lambda config, recommended: 0.9)
    monkeypatch.setattr(vllm_infer, "VllmAsyncEngineArgs", EngineArgs)
    monkeypatch.setattr(vllm_infer, "VllmAsyncLLM", Engine)
    monkeypatch.setattr(
        vllm_infer, "VllmReasoningConfig", type("Fake", (_FakeReasoningConfig,), {"has_markers": has_markers})
    )
    monkeypatch.setattr(vllm_infer, "vllm_cached_tokenizer_from_config", tokenizer)
    monkeypatch.setattr(vllm_infer, "_METRICS_ENABLED", False)
    monkeypatch.delenv(VLLM_CHILD_ENV, raising=False)
    return VllmInfer(config), started


class TestEngineStartsWithTheReasoningParser:
    def test_detected_from_the_chat_template(self, monkeypatch):
        infer, started = _build(monkeypatch, _config(), _THINKING_TEMPLATE)

        assert started["reasoning_parser"] == "deepseek_r1"
        assert infer._chat_parsers == _ChatParsers(_THINKING_TEMPLATE, "hermes", "deepseek_r1")

    def test_named_in_the_config(self, monkeypatch):
        _, started = _build(monkeypatch, _config(reasoning_parser="deepseek_r1"), _PLAIN_TEMPLATE)

        assert started["reasoning_parser"] == "deepseek_r1"

    def test_unset_for_a_template_without_reasoning(self, monkeypatch):
        infer, started = _build(monkeypatch, _config(), _PLAIN_TEMPLATE)

        assert started["reasoning_parser"] == ""
        assert infer._chat_parsers == _ChatParsers(_PLAIN_TEMPLATE, None, None)

    def test_unset_when_reasoning_is_switched_off(self, monkeypatch):
        _, started = _build(monkeypatch, _config(enable_reasoning=False), _THINKING_TEMPLATE)

        assert started["reasoning_parser"] == ""

    def test_a_non_chat_model_is_not_resolved(self, monkeypatch):
        infer, started = _build(monkeypatch, _config(ModelUsecase.embed), None)

        assert started["reasoning_parser"] == ""
        assert infer._chat_parsers is None


class TestThinkingBudget:
    @pytest.fixture
    def warnings(self, monkeypatch) -> list[str]:
        logged: list[str] = []
        monkeypatch.setattr(vllm_infer.logger, "warning", lambda msg, *args, **kwargs: logged.append(msg % args))
        return logged

    def test_reasoning_config_is_enabled_for_the_reasoning_parser(self, monkeypatch):
        infer, started = _build(monkeypatch, _config(thinking_token_budget=512), _THINKING_TEMPLATE)

        reasoning_config = started["reasoning_config"]
        assert reasoning_config.reasoning_parser == "deepseek_r1"
        assert reasoning_config.enabled
        assert reasoning_config.initialized_with is not None
        assert infer._thinking_budget_cap == 512

    def test_requests_may_set_a_budget_without_an_operator_one(self, monkeypatch):
        infer, started = _build(monkeypatch, _config(), _THINKING_TEMPLATE)

        assert started["reasoning_config"].enabled
        assert infer._thinking_budget_cap is None

    def test_the_markers_in_the_config_reach_the_engine(self, monkeypatch):
        config = _config(reasoning_config={"reasoning_end_str": "stop</think>"})

        _, started = _build(monkeypatch, config, _THINKING_TEMPLATE)

        assert started["reasoning_config"].reasoning_end_str == "stop</think>"
        assert not hasattr(started["reasoning_config"], "reasoning_start_str")

    def test_a_model_without_a_reasoning_parser_ignores_both_keys(self, monkeypatch, warnings):
        config = _config(thinking_token_budget=512, reasoning_config={"reasoning_end_str": "</think>"})

        infer, started = _build(monkeypatch, config, _PLAIN_TEMPLATE)

        assert started["reasoning_config"] is None
        assert infer._thinking_budget_cap is None
        assert any("no reasoning parser" in message for message in warnings)

    def test_a_parser_without_markers_leaves_the_budget_off(self, monkeypatch, warnings):
        infer, _ = _build(monkeypatch, _config(thinking_token_budget=512), _THINKING_TEMPLATE, has_markers=False)

        assert infer._thinking_budget_cap is None
        assert any("reasoning_start_str and reasoning_end_str" in message for message in warnings)

    @pytest.mark.parametrize(
        ("cap", "requested", "expected"),
        [
            (None, None, None),
            (None, 900, 900),
            (400, None, 400),
            (400, 100, 100),
            (400, 5000, 400),
            (400, 0, 0),
            (0, 100, 0),
        ],
    )
    def test_the_operator_budget_is_the_default_and_the_ceiling(self, cap, requested, expected):
        infer = object.__new__(VllmInfer)
        if cap is not None:
            infer._thinking_budget_cap = cap
        request = SimpleNamespace(thinking_token_budget=requested)

        infer._limit_thinking(request)  # type: ignore[arg-type]

        assert request.thinking_token_budget == expected


def _started_infer(supported_tasks: list[str]) -> VllmInfer:
    infer = object.__new__(VllmInfer)
    infer.model_config = _config()
    infer.vllm_engine_kwargs = VllmEngineConfig()
    infer.supported_tasks = supported_tasks
    infer._chat_parsers = _ChatParsers(_PLAIN_TEMPLATE, "hermes", "deepseek_r1")
    infer.engine = SimpleNamespace(
        model_config=object(),
        renderer=object(),
        get_tokenizer=lambda: SimpleNamespace(apply_chat_template=lambda: ""),
        shutdown=lambda: None,
    )
    return infer


class TestRenderer:
    @pytest.fixture
    def built(self, monkeypatch) -> dict[str, Any]:
        built: dict[str, Any] = {}

        def renderer(**kwargs: Any) -> Any:
            built.update(kwargs)
            return SimpleNamespace(renderer=SimpleNamespace(tokenizer=object()))

        monkeypatch.setattr(vllm_infer, "VllmOnlineRenderer", renderer)
        return built

    @pytest.mark.asyncio
    async def test_built_with_the_resolved_parsers(self, built):
        infer = _started_infer(["generate"])

        await infer.init_serving_chat()

        assert built["tool_parser"] == "hermes"
        assert built["reasoning_parser"] == "deepseek_r1"
        assert built["enable_auto_tools"] is True

    @pytest.mark.asyncio
    async def test_the_config_kwargs_go_over_the_detected_template_defaults(self, built, monkeypatch):
        monkeypatch.setattr(
            vllm_infer,
            "detect_template_toggle_defaults",
            lambda template, tokenizer: {"enable_thinking": True, "clear_thinking": True},
        )
        infer = _started_infer(["generate"])
        infer.vllm_engine_kwargs = VllmEngineConfig(default_chat_template_kwargs={"enable_thinking": False})

        await infer.init_serving_chat()

        assert built["default_chat_template_kwargs"] == {"enable_thinking": False, "clear_thinking": True}
        assert infer.vllm_engine_kwargs.default_chat_template_kwargs == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_not_built_without_the_generate_task(self, built):
        infer = _started_infer(["embed"])

        await infer.init_serving_chat()

        assert built == {}
        assert not hasattr(infer, "openai_serving_render")
