from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    RawRequestProxy,
)
from modelship.infer.vllm import vllm_infer
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import ChatCompletionRequest, ErrorResponse

_NO_TEMPLATE = ValueError(
    "Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!"
)


def _make_infer(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> VllmInfer:
    infer = object.__new__(VllmInfer)
    infer.model_config = ModelshipModelConfig(
        name="base-model", model="some/base", usecase=ModelUsecase.generate, loader=ModelLoader.vllm
    )
    infer.supported_tasks = ["generate"]
    # shutdown is a no-op so the actor's __del__ does not log a stub teardown failure.
    infer.engine = SimpleNamespace(shutdown=lambda: None)

    def get_chat_template():
        raise exc

    monkeypatch.setattr(
        vllm_infer,
        "vllm_cached_tokenizer_from_config",
        lambda model_config: SimpleNamespace(get_chat_template=get_chat_template),
    )
    infer._chat_parsers = infer._resolve_chat_parsers(SimpleNamespace(model_config=object()))
    return infer


class TestMissingChatTemplate:
    @pytest.mark.asyncio
    async def test_init_serving_chat_leaves_the_pipeline_unset(self, monkeypatch):
        infer = _make_infer(monkeypatch, _NO_TEMPLATE)
        assert infer._chat_parsers is None
        await infer.init_serving_chat()
        assert not hasattr(infer, "openai_serving_render")

    def test_a_non_value_error_still_propagates(self, monkeypatch):
        with pytest.raises(RuntimeError):
            _make_infer(monkeypatch, RuntimeError("tokenizer is gone"))

    @pytest.mark.asyncio
    async def test_chat_requests_are_rejected_rather_than_crashing(self, monkeypatch):
        infer = _make_infer(monkeypatch, _NO_TEMPLATE)
        await infer.init_serving_chat()
        result = await infer._prepare_chat(
            ChatCompletionRequest(model="base-model", messages=[{"role": "user", "content": "hi"}]),
            RawRequestProxy(None, {}),
        )
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 404
