from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as VllmChatCompletionRequest

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
    async def test_prompt_endings_are_kept_for_a_reasoning_parser(self, built):
        infer = _started_infer(["generate"])

        await infer.init_serving_chat()

        assert infer._prompt_endings == {}

    @pytest.mark.asyncio
    async def test_no_prompt_endings_are_kept_without_a_reasoning_parser(self, built):
        infer = _started_infer(["generate"])
        infer._chat_parsers = _ChatParsers(_PLAIN_TEMPLATE, "hermes", None)

        await infer.init_serving_chat()

        assert infer._prompt_endings is None

    @pytest.mark.asyncio
    async def test_not_built_without_the_generate_task(self, built):
        infer = _started_infer(["embed"])

        await infer.init_serving_chat()

        assert built == {}
        assert not hasattr(infer, "openai_serving_render")


def _request(reasoning: str | None = "r", **overrides: Any) -> VllmChatCompletionRequest:
    assistant: dict[str, Any] = {"role": "assistant", "content": "a"}
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    messages = [{"role": "user", "content": "hi"}, assistant, {"role": "user", "content": "go on"}]
    return VllmChatCompletionRequest(model="m", messages=messages, **overrides)


class TestPromptEndingPerRequest:
    @pytest.fixture
    def learned(self, monkeypatch) -> list[Any]:
        learned: list[Any] = []

        async def render_and_params(render: Any, vllm_req: Any) -> Any:
            return SimpleNamespace(), SimpleNamespace()

        async def learn_prompt_ending(render: Any, tokenizer: Any, vllm_req: Any) -> list[int]:
            learned.append(vllm_req.chat_template_kwargs)
            return [7, 8]

        monkeypatch.setattr(vllm_infer.engine_ops, "render_and_params", render_and_params)
        monkeypatch.setattr(vllm_infer.engine_ops, "learn_prompt_ending", learn_prompt_ending)
        monkeypatch.setattr(vllm_infer.engine_ops, "extract_prompt_token_ids", lambda render, engine_input: [1, 7, 8])
        return learned

    def _infer(self, prompt_endings: dict[str, list[int] | None] | None) -> VllmInfer:
        infer = object.__new__(VllmInfer)
        infer.openai_serving_render = SimpleNamespace()
        infer._tokenizer = SimpleNamespace()
        infer._prompt_endings = prompt_endings
        return infer

    @pytest.mark.asyncio
    async def test_a_prompt_ending_is_learned_once_per_template_kwargs(self, learned):
        infer = self._infer({})

        endings = [
            (await infer._render_bundle(_request(chat_template_kwargs=kwargs))).prompt_ending
            for kwargs in ({"enable_thinking": True}, {"enable_thinking": True}, {"enable_thinking": False})
        ]

        assert endings == [[7, 8]] * 3
        assert learned == [{"enable_thinking": True}, {"enable_thinking": False}]

    @pytest.mark.asyncio
    async def test_no_prompt_ending_without_reasoning_or_a_reasoning_parser(self, learned):
        without_reasoning = await self._infer({})._render_bundle(_request(reasoning=None))
        without_parser = await self._infer(None)._render_bundle(_request())

        assert without_reasoning.prompt_ending is None
        assert without_parser.prompt_ending is None
        assert learned == []

    @pytest.mark.asyncio
    async def test_the_oldest_prompt_ending_is_dropped_past_64_template_kwargs(self, learned):
        infer = self._infer({})

        for i in [*range(65), 0]:
            await infer._render_bundle(_request(chat_template_kwargs={"i": i}))

        assert infer._prompt_endings is not None and len(infer._prompt_endings) == 64
        assert learned.count({"i": 0}) == 2
