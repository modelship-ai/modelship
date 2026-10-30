import time
from unittest.mock import MagicMock, patch

import pytest

from modelship.infer import base_infer
from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import RawRequestProxy
from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    DeltaToolCall,
)


class _Infer(BaseInfer):
    def __init__(self):
        super().__init__(MagicMock())
        self.model_config.name = "m"

    def shutdown(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def warmup(self) -> None:
        pass


def _chunk(**delta) -> ChatCompletionStreamResponse:
    choice = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(**delta))
    return ChatCompletionStreamResponse(model="m", choices=[choice])


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


async def _observed(*chunks, raw_request=None) -> list:
    raw_request = raw_request or RawRequestProxy(MagicMock(), {}, "req-1")
    with patch.object(base_infer, "TIME_TO_FIRST_TOKEN_SECONDS") as histogram:
        yielded = [chunk async for chunk in _Infer()._observe_first_token(_stream(*chunks), raw_request)]
    assert yielded == list(chunks)
    return histogram.observe.call_args_list


@pytest.mark.asyncio
async def test_a_role_only_chunk_is_not_the_first_token():
    assert await _observed(_chunk(role="assistant", content="")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delta",
    [{"content": "hi"}, {"reasoning": "hmm"}, {"tool_calls": [DeltaToolCall(index=0)]}],
    ids=["content", "reasoning", "tool_call"],
)
async def test_the_first_generated_chunk_is_observed_once(delta):
    observed = await _observed(_chunk(role="assistant", content=""), _chunk(**delta), _chunk(content="more"))

    assert len(observed) == 1
    assert observed[0].kwargs["tags"] == {"model": "m"}


@pytest.mark.asyncio
async def test_the_time_runs_from_the_requests_arrival():
    raw_request = RawRequestProxy(MagicMock(), {}, "req-1")
    raw_request.received_at = time.monotonic() - 2

    observed = await _observed(_chunk(content="hi"), raw_request=raw_request)

    assert observed[0].args[0] >= 2


@pytest.mark.asyncio
async def test_an_empty_stream_observes_nothing():
    assert await _observed() == []


@pytest.mark.asyncio
async def test_a_warmup_request_observes_nothing():
    assert await _observed(_chunk(content="hi"), raw_request=RawRequestProxy(None, {})) == []
