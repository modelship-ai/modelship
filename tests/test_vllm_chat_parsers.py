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


def _build(
    monkeypatch: pytest.MonkeyPatch, config: ModelshipModelConfig, template: str | None
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
            )

    class Engine:
        @classmethod
        def from_vllm_config(cls, vllm_config: Any, **kwargs: Any) -> Any:
            started["reasoning_parser"] = vllm_config.structured_outputs_config.reasoning_parser
            return SimpleNamespace(shutdown=lambda: None)

    def tokenizer(model_config: Any) -> Any:
        assert template is not None
        return SimpleNamespace(get_chat_template=lambda: template, get_vocab=lambda: {"<think>": 1, "</think>": 2})

    monkeypatch.setattr(vllm_infer, "run_preflight", lambda config, hw: {})
    monkeypatch.setattr(vllm_infer, "discover_hardware", lambda: None)
    monkeypatch.setattr(vllm_infer, "resolve_gpu_memory_utilization", lambda config, recommended: 0.9)
    monkeypatch.setattr(vllm_infer, "VllmAsyncEngineArgs", EngineArgs)
    monkeypatch.setattr(vllm_infer, "VllmAsyncLLM", Engine)
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
    async def test_not_built_without_the_generate_task(self, built):
        infer = _started_infer(["embed"])

        await infer.init_serving_chat()

        assert built == {}
        assert not hasattr(infer, "openai_serving_render")
