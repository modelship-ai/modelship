import pytest

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import RawRequestProxy
from modelship.openai.protocol import ChatCompletionRequest, create_error_response

_TOOL = {"type": "function", "function": {"name": "f"}}


class _Infer(BaseInfer[None]):
    def shutdown(self) -> None: ...

    async def start(self) -> None: ...

    async def warmup(self) -> None: ...

    async def _prepare_chat(self, request, raw_request):
        self.prepared = request
        return create_error_response("stop")


async def _prepared(tool_choice: str) -> ChatCompletionRequest:
    infer = object.__new__(_Infer)
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], tools=[_TOOL], tool_choice=tool_choice
    )
    await infer.create_chat_completion(request, RawRequestProxy(None, {}))
    return infer.prepared


class TestToolChoiceNone:
    @pytest.mark.asyncio
    async def test_the_loader_gets_no_tools(self):
        prepared = await _prepared("none")

        assert prepared.tools is None
        assert prepared.tool_choice == "none"

    @pytest.mark.asyncio
    async def test_other_choices_keep_the_tools(self):
        assert (await _prepared("auto")).tools == [_TOOL]
