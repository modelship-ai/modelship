"""Chat completion request/response schemas, including tool calls and logprobs."""

import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from modelship.openai.protocol.base import OpenAIBaseModel, random_uuid
from modelship.openai.protocol.usage import UsageInfo

# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class FunctionCall(OpenAIBaseModel):
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


# ---------------------------------------------------------------------------
# Logprobs
# ---------------------------------------------------------------------------


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    field_names: ClassVar[set[str] | None] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = False
    continuous_usage_stats: bool | None = False


ChatCompletionMessageParam = dict[str, Any]


class ChatCompletionRequest(OpenAIBaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = Field(default_factory=list)
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    parallel_tool_calls: bool | None = True
    user: str | None = None

    @model_validator(mode="after")
    def _validate_tools_response_format_compat(self) -> "ChatCompletionRequest":
        # response_format=json_object/json_schema is enforced via a sampling-time
        # grammar that excludes any token outside the schema — including the
        # markers a model would use to emit a tool call (<tool_call>...,
        # <|python_tag|>..., etc.). The two cannot meaningfully coexist on any
        # loader we support: vLLM passes both through but the grammar dominates.
        # Reject upfront so callers don't discover the conflict by watching
        # tool calls never fire in prod.
        if not self.tools or not self.response_format:
            return self
        fmt_type = self.response_format.get("type")
        if fmt_type in (None, "text"):
            return self
        if self.tool_choice == "none":
            return self
        raise ValueError(
            f"response_format with type={fmt_type!r} cannot be combined with tools unless tool_choice='none'."
        )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = "stop"
    stop_reason: int | str | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


__all__ = [
    "ChatCompletionLogProb",
    "ChatCompletionLogProbs",
    "ChatCompletionLogProbsContent",
    "ChatCompletionMessageParam",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
    "ChatMessage",
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "FunctionCall",
    "StreamOptions",
    "ToolCall",
]
