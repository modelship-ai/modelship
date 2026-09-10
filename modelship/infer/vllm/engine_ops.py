"""Quarantine for every vLLM-internal touchpoint the vllm loader needs.

Modelship types go in; a parsed 3-tuple (via vllm.parser) or a raw engine
stream comes out. Nothing outside this module should import from
`vllm.entrypoints.*`/`vllm.parser`/`vllm.v1.engine.*` directly — that keeps a
vLLM version bump's blast radius confined to one file.
"""

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam as VllmChatCompletionNamedToolChoiceParam,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as VllmDeltaMessage
from vllm.entrypoints.openai.engine.protocol import DeltaToolCall as VllmDeltaToolCall
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as VllmErrorResponse
from vllm.entrypoints.openai.engine.protocol import FunctionCall as VllmFunctionCall
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens as vllm_get_max_tokens
from vllm.inputs import EngineInput as VllmEngineInput
from vllm.logprobs import Logprob as VllmLogprob
from vllm.outputs import CompletionOutput as VllmCompletionOutput
from vllm.outputs import RequestOutput as VllmRequestOutput
from vllm.parser import Parser as VllmParser
from vllm.renderers.inputs.preprocess import extract_prompt_components as vllm_extract_prompt_components
from vllm.renderers.inputs.preprocess import extract_prompt_len as vllm_extract_prompt_len
from vllm.renderers.online_renderer import OnlineRenderer as VllmOnlineRenderer
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.tokenizers import TokenizerLike as VllmTokenizerLike
from vllm.v1.engine.async_llm import AsyncLLM as VllmAsyncLLM

from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionTokenUsageInfo,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    UsageInfo,
    random_uuid,
)
from modelship.openai.utils.chat import ParsedChatOutput

logger = get_logger("infer.vllm.engine_ops")


def build_vllm_request(
    request: ChatCompletionRequest,
    chat_template_kwargs: dict[str, Any] | None,
    cache_salt: str | None = None,
) -> VllmChatCompletionRequest:
    """Shape a modelship chat request into vLLM's own request model.

    Merges the model's default `chat_template_kwargs` under any per-request
    value (request wins) — vLLM renders the chat template internally, so
    unlike llama.cpp-family loaders this can't be patched in after the fact.

    `cache_salt` scopes vLLM's prefix-cache reuse to the caller's identity.
    """
    request_data = request.model_dump()
    if chat_template_kwargs:
        request_data["chat_template_kwargs"] = {
            **chat_template_kwargs,
            **(request_data.get("chat_template_kwargs") or {}),
        }
    if cache_salt is not None:
        request_data["cache_salt"] = cache_salt
    return VllmChatCompletionRequest(**request_data)


async def render_and_params(
    render: VllmOnlineRenderer,
    vllm_req: VllmChatCompletionRequest,
) -> tuple[VllmEngineInput, VllmSamplingParams] | VllmErrorResponse:
    """Render the chat template and derive `VllmSamplingParams`, in that order.

    `render_chat` mutates `vllm_req` in place as a side effect of rendering
    (`ToolParser.adjust_request` sets `structured_outputs` /
    `_grammar_from_parser`), and `to_sampling_params` reads that
    mutation. The order is load-bearing — this function exists specifically
    so callers can't split the two apart or run them against a rebuilt copy
    of the request, which would silently drop the mutation.
    """
    result = await render.render_chat(vllm_req)
    if isinstance(result, VllmErrorResponse):
        return result
    _conversation, engine_inputs = result
    if len(engine_inputs) != 1:
        raise RuntimeError(f"expected exactly 1 rendered engine prompt for a chat request, got {len(engine_inputs)}")
    engine_input = engine_inputs[0]

    # Mirrors OpenAIServingChat.__init__'s own computation off model_config.
    model_config = render.model_config
    default_sampling_params = model_config.get_diff_sampling_param()
    override_max_tokens = (
        default_sampling_params.get("max_tokens")
        if model_config.generation_config not in ("auto", "vllm")
        else getattr(model_config, "override_generation_config", {}).get("max_new_tokens")
    )

    max_tokens = vllm_get_max_tokens(
        model_config.max_model_len,
        vllm_req.max_completion_tokens if vllm_req.max_completion_tokens is not None else vllm_req.max_tokens,
        vllm_extract_prompt_len(model_config, engine_input),
        default_sampling_params,
        override_max_tokens,
        truncate_prompt_tokens=vllm_req.truncate_prompt_tokens,
    )
    sampling_params = vllm_req.to_sampling_params(max_tokens, default_sampling_params)
    return engine_input, sampling_params


def extract_prompt_token_ids(render: VllmOnlineRenderer, engine_input: VllmEngineInput) -> list[int]:
    """Extract the rendered prompt's token IDs, needed for `derive_reasoning_ended`."""
    return list(vllm_extract_prompt_components(render.model_config, engine_input).token_ids or [])


def make_parsers(
    render: VllmOnlineRenderer,
    tokenizer: VllmTokenizerLike,
    vllm_req: VllmChatCompletionRequest,
    chat_template_kwargs: dict[str, Any] | None,
    n: int,
) -> list[VllmParser | None]:
    """Instantiate one parser per choice.

    Parsers carry per-choice streaming state (`VllmParser._stream_state`), so a
    request with `n > 1` needs `n` independent instances — sharing one across
    choices corrupts state on every choice after the first. `render.parser`
    is the same class `render_chat` already resolved internally via
    `ParserManager.get_parser`, so this can't drift out of sync with it.
    """
    if render.parser is None:
        return [None] * n
    parser_cls = render.parser
    return [
        parser_cls(tokenizer, vllm_req.tools, chat_template_kwargs=chat_template_kwargs)  # type: ignore[arg-type]
        for _ in range(n)
    ]


def derive_reasoning_ended(
    vllm_req: VllmChatCompletionRequest,
    parser: VllmParser | None,
    prompt_token_ids: list[int],
) -> bool | None:
    """Replicates the reasoning_ended precedence in vLLM's own chat completion serving.

    Mistral's grammar (when built) already encodes an optional `think?` rule
    covering both reasoning and non-reasoning outputs, so `reasoning_ended`
    is forced True whenever `_grammar_from_parser` is set. But that flag
    is only set on the grammar-building branch of
    `MistralParser.adjust_request` — a request with tools but no
    structured-outputs constraint active takes an early-return branch that
    leaves it False, so this must not assume the flag is reliably True
    whenever a mistral tool parser is in play.
    """
    if not vllm_req.include_reasoning:
        return True
    if vllm_req._grammar_from_parser:
        return True
    if parser is not None and parser.reasoning_parser is not None:
        return parser.is_reasoning_end(prompt_token_ids)
    return None


def generate(
    engine: VllmAsyncLLM,
    engine_input: VllmEngineInput,
    sampling_params: VllmSamplingParams,
    request_id: str,
    *,
    reasoning_ended: bool | None,
    parser: VllmParser | None,
    chat_template_kwargs: dict[str, Any] | None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> AsyncGenerator[VllmRequestOutput, None]:
    """Thin wrapper over `VllmAsyncLLM.generate` — the only place this loader touches the engine directly."""
    reasoning_parser_kwargs = None
    if parser is not None and parser.reasoning_parser is not None:
        reasoning_parser_kwargs = {"chat_template_kwargs": chat_template_kwargs}
    return engine.generate(
        engine_input,
        sampling_params,
        request_id,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        reasoning_ended=reasoning_ended,
        reasoning_parser_kwargs=reasoning_parser_kwargs,
    )


def project_tool_calls(vllm_tool_calls: list[VllmFunctionCall] | None) -> list[ToolCall]:
    """Project a parser's vLLM-shaped tool calls onto modelship's OpenAI `ToolCall`.

    `vllm.parser.Parser.parse()`'s `FunctionCall` has the same `id`/`name`/`arguments`
    shape as modelship's own; `id` is only set when the tool_call_id_type config
    minted one (e.g. kimi_k2), so most calls need one generated here.
    """
    return [
        ToolCall(
            id=tc.id or f"chatcmpl-tool-{random_uuid()}",
            function=FunctionCall(name=tc.name, arguments=tc.arguments),
        )
        for tc in (vllm_tool_calls or [])
    ]


def project_delta_tool_calls(vllm_tool_calls: list[VllmDeltaToolCall]) -> list[DeltaToolCall] | None:
    """Project one streaming delta's vLLM tool-call fragments onto modelship's `DeltaToolCall`.

    Unlike `project_tool_calls`, nothing is synthesized here: only the first
    delta for a given tool call carries `id`/`type`/`function.name` (per
    `VllmParser.parse_delta`'s own streaming protocol) — later deltas for the same
    `index` carry only incremental `function.arguments`, which must pass
    through as-is for the client to accumulate correctly.
    """
    if not vllm_tool_calls:
        return None
    return [
        DeltaToolCall(
            index=tc.index,
            id=tc.id,
            type=tc.type,
            function=DeltaFunctionCall(name=tc.function.name, arguments=tc.function.arguments)
            if tc.function is not None
            else None,
        )
        for tc in vllm_tool_calls
    ]


def build_chat_logprobs(
    token_ids: Sequence[int],
    top_logprobs: Sequence[dict[int, VllmLogprob] | None],
    tokenizer: VllmTokenizerLike,
    num_output_top_logprobs: int | None,
) -> ChatCompletionLogProbs:
    """Project a choice's per-token logprobs onto modelship's OpenAI logprobs shape.

    Mirrors `OpenAIServingChat._create_chat_logprobs`/`_get_top_logprobs`, minus the
    `return_tokens_as_token_ids` branch — modelship's request has no such field, so
    tokens are always decoded to text.
    """
    content: list[ChatCompletionLogProbsContent] = []
    for i, token_id in enumerate(token_ids):
        # Defensive: token_ids and top_logprobs should be the same length, but
        # guard against a mismatched pair rather than risk an IndexError.
        step_top_logprobs = top_logprobs[i] if i < len(top_logprobs) else None
        chosen = step_top_logprobs.get(token_id) if step_top_logprobs else None
        if chosen is None:
            token = tokenizer.decode(token_id)
            content.append(
                ChatCompletionLogProbsContent(token=token, bytes=list(token.encode("utf-8", errors="replace")))
            )
            continue
        decoded = chosen.decoded_token if chosen.decoded_token is not None else tokenizer.decode(token_id)
        content.append(
            ChatCompletionLogProbsContent(
                token=decoded,
                logprob=max(chosen.logprob, -9999.0),
                bytes=list(decoded.encode("utf-8", errors="replace")),
                top_logprobs=[
                    ChatCompletionLogProb(
                        token=(tok := lp.decoded_token if lp.decoded_token is not None else tokenizer.decode(tid)),
                        logprob=max(lp.logprob, -9999.0),
                        bytes=list(tok.encode("utf-8", errors="replace")),
                    )
                    for idx, (tid, lp) in enumerate(step_top_logprobs.items())
                    if (num_output_top_logprobs and idx < num_output_top_logprobs) or num_output_top_logprobs == -1
                ]
                if step_top_logprobs
                else [],
            )
        )
    return ChatCompletionLogProbs(content=content)


async def consume_final_output(
    engine: VllmAsyncLLM,
    engine_input: VllmEngineInput,
    sampling_params: VllmSamplingParams,
    request_id: str,
    *,
    reasoning_ended: bool | None,
    parser: VllmParser | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> VllmRequestOutput:
    """Drive `generate()` to completion and return the final `VllmRequestOutput`.

    Non-streaming only needs the last output (it carries every choice's full
    text). Cancelling the task awaiting this coroutine (e.g. on client
    disconnect) propagates into the `async for` below and into `VllmAsyncLLM.generate`'s
    own `except (CancelledError, GeneratorExit): abort(...)` — no separate abort
    call is needed here.
    """
    final: VllmRequestOutput | None = None
    async for res in generate(
        engine,
        engine_input,
        sampling_params,
        request_id,
        reasoning_ended=reasoning_ended,
        parser=parser,
        chat_template_kwargs=chat_template_kwargs,
    ):
        final = res
    if final is None:
        raise RuntimeError(f"engine produced no output for request {request_id}")
    return final


def _finish_reason_for_choice(
    vllm_req: VllmChatCompletionRequest,
    has_tool_calls: bool,
    engine_finish_reason: str | None,
) -> str:
    """OpenAI `finish_reason` for one choice, mirroring `OpenAIServingChat`'s precedence.

    A parsed tool call reports finish_reason="tool_calls" for auto/required
    tool_choice, but the engine's own reason (usually "stop") for a named-function
    tool_choice — the client already knows which function was called, so the turn
    just "stopped" rather than the model "deciding" to call a tool.
    """
    if not has_tool_calls:
        return engine_finish_reason or "stop"
    if isinstance(vllm_req.tool_choice, VllmChatCompletionNamedToolChoiceParam):
        return engine_finish_reason or "stop"
    return "tool_calls"


def total_reasoning_tokens(outputs: Sequence[VllmCompletionOutput], parser: VllmParser | None) -> int | None:
    """Sum reasoning-classified tokens across every choice's completed `token_ids`,
    mirroring vLLM's own `entrypoints.openai.responses.serving` usage accounting.

    Returns `None` when no reasoning parser is active for this request, so callers can
    leave `completion_tokens_details` unset rather than reporting a misleading zero.
    """
    if parser is None or parser.reasoning_parser is None:
        return None
    return sum(parser.reasoning_parser.count_reasoning_tokens(list(o.token_ids)) for o in outputs)


def build_choices(
    final_res: VllmRequestOutput,
    vllm_req: VllmChatCompletionRequest,
    parser: VllmParser | None,
    tokenizer: VllmTokenizerLike,
    *,
    enable_auto_tools: bool,
    want_logprobs: bool,
    num_output_top_logprobs: int | None,
) -> tuple[list[ParsedChatOutput], list[str | None], list[ChatCompletionLogProbs | None]]:
    """Parse every choice in a finished `VllmRequestOutput` into modelship's response DTOs.

    Non-streaming reuses one shared `parser` instance across every choice —
    `.parse()` is stateless per full-text call, unlike the streaming path's
    per-choice `VllmParser._stream_state` (see `make_parsers`).
    """
    choices: list[ParsedChatOutput] = []
    finish_reasons: list[str | None] = []
    logprobs_list: list[ChatCompletionLogProbs | None] = []

    for output in final_res.outputs:
        if parser is not None:
            reasoning, content, raw_tool_calls = parser.parse(
                output.text,
                vllm_req,
                enable_auto_tools=enable_auto_tools,
                model_output_token_ids=output.token_ids,
            )
        else:
            reasoning, content, raw_tool_calls = None, output.text, None

        dto = ParsedChatOutput(content=content, reasoning=reasoning, tool_calls=project_tool_calls(raw_tool_calls))
        choices.append(dto)
        finish_reasons.append(_finish_reason_for_choice(vllm_req, dto.has_tool_calls, output.finish_reason))

        if want_logprobs and output.logprobs is not None:
            logprobs_list.append(
                build_chat_logprobs(output.token_ids, output.logprobs, tokenizer, num_output_top_logprobs)
            )
        else:
            logprobs_list.append(None)

    return choices, finish_reasons, logprobs_list


def _reconcile_trapped_content(
    render: VllmOnlineRenderer,
    tokenizer: VllmTokenizerLike,
    vllm_req: VllmChatCompletionRequest,
    full_text: str,
    token_ids: list[int],
    *,
    enable_auto_tools: bool,
) -> str | None:
    """Recover an answer the streaming parser left stranded in `reasoning`.

    vLLM's streaming parser engine starts in `REASONING` whenever the prompt primes
    thinking, and its `finish()` only marks `REASONING_END` — it never reclassifies
    chunks it already emitted. So a model that ends a turn without its reasoning-close
    marker leaves the whole reply in `reasoning` with empty `content`, while the
    full-text `parse()` the non-streaming path uses (`build_choices`) reads the same
    tokens as `content`. This re-runs that authoritative parse to settle the
    disagreement in favour of the non-streaming answer.

    `full_text` must be the engine's own detokenized text (the streamed deltas joined),
    not a re-decode of `token_ids`: the engine drops the stop token from its text while
    `token_ids` retains it, and this `parse()` — unlike the offline `parse_thinking_output`
    helper — does not strip trailing sentinels, so a re-decode leaks e.g. `<turn|>` into
    the recovered answer.

    A fresh parser is required: the streamed instance carries mid-stream state and
    would not reproduce the full-text result. Returns the recovered content, or None
    when the parse agrees there is none (e.g. reasoning that closed with no answer)
    or when the re-parse itself fails — this runs after reasoning has already been
    streamed out, so a parser error here must not crash the generator mid-stream.
    """
    try:
        parser = make_parsers(render, tokenizer, vllm_req, vllm_req.chat_template_kwargs, n=1)[0]
        if parser is None:
            return None
        _reasoning, content, _tool_calls = parser.parse(
            full_text,
            vllm_req,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=token_ids,
        )
    except Exception:
        logger.exception("Reconcile re-parse failed; leaving trapped content in reasoning.")
        return None
    return content or None


async def stream_chat_completion(
    engine: VllmAsyncLLM,
    render: VllmOnlineRenderer,
    vllm_req: VllmChatCompletionRequest,
    engine_input: VllmEngineInput,
    sampling_params: VllmSamplingParams,
    request_id: str,
    model_name: str,
    tokenizer: VllmTokenizerLike,
    *,
    enable_auto_tools: bool,
    want_logprobs: bool,
    num_output_top_logprobs: int | None,
) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
    """Drive one streaming chat completion end to end: per-choice parsers,
    per-delta parsing via `VllmParser.parse_delta`, and the OpenAI streaming
    chunk lifecycle (role chunk, content/tool/reasoning deltas, finish
    chunk, optional usage chunk) — the streaming counterpart of `build_choices`.

    Yields fully-formed modelship chunks; the caller owns SSE encoding and
    the trailing `[DONE]` line (symmetric with how `build_choices` leaves
    `ChatCompletionResponse` assembly to `utils.chat.build_from_parsed`).
    """
    num_choices = vllm_req.n or 1
    parsers = make_parsers(render, tokenizer, vllm_req, vllm_req.chat_template_kwargs, n=num_choices)
    prompt_token_ids = extract_prompt_token_ids(render, engine_input)
    reasoning_ended = derive_reasoning_ended(vllm_req, parsers[0], prompt_token_ids)

    stream_options = vllm_req.stream_options
    include_usage = bool(stream_options and stream_options.include_usage)
    include_continuous_usage = include_usage and bool(stream_options and stream_options.continuous_usage_stats)

    previous_num_tokens = [0] * num_choices
    accumulated_token_ids: list[list[int]] = [[] for _ in range(num_choices)]
    finish_reason_sent = [False] * num_choices
    tools_streamed = [False] * num_choices
    # Per-choice record of what actually reached the client, so the finish branch can
    # spot a reasoning-only stream and reconcile it (see `_reconcile_trapped_content`).
    content_streamed = [False] * num_choices
    reasoning_streamed = [False] * num_choices
    # The engine's own detokenized text, kept only until this choice streams content —
    # at that point it can never be reconciled, so the buffer is dropped rather than
    # carried for the rest of the request.
    accumulated_text: list[list[str]] = [[] for _ in range(num_choices)]
    first_iteration = True
    num_prompt_tokens = 0

    async for res in generate(
        engine,
        engine_input,
        sampling_params,
        request_id,
        reasoning_ended=reasoning_ended,
        parser=parsers[0],
        chat_template_kwargs=vllm_req.chat_template_kwargs,
    ):
        if res.prompt_token_ids is not None:
            num_prompt_tokens = len(res.prompt_token_ids)

        if first_iteration:
            first_iteration = False
            role_choices = [
                ChatCompletionResponseStreamChoice(index=i, delta=DeltaMessage(role="assistant", content=""))
                for i in range(num_choices)
            ]
            yield ChatCompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=role_choices,
                usage=_continuous_usage(num_prompt_tokens, 0) if include_continuous_usage else None,
            )

        for output in res.outputs:
            i = output.index
            if finish_reason_sent[i]:
                continue

            delta_text = output.text
            if not delta_text and not output.token_ids and not previous_num_tokens[i]:
                # Chunked prefill: nothing new to emit yet.
                continue

            parser = parsers[i]
            if parser is not None:
                vllm_delta = parser.parse_delta(
                    delta_text=delta_text,
                    delta_token_ids=list(output.token_ids),
                    request=vllm_req,
                    prompt_token_ids=res.prompt_token_ids,
                    finished=output.finish_reason is not None,
                )
                if vllm_delta is not None and vllm_delta.tool_calls:
                    tools_streamed[i] = True
            else:
                vllm_delta = VllmDeltaMessage(content=delta_text)

            previous_num_tokens[i] += len(output.token_ids)
            accumulated_token_ids[i].extend(output.token_ids)
            # Mirror the raw engine text alongside the ids: the parser returns None
            # while it defers text it hasn't decided on yet, and those deltas must
            # still reach the reconcile's full-text parse.
            if not content_streamed[i]:
                accumulated_text[i].append(delta_text)

            if vllm_delta is None:
                # VllmParser swallowed a control token (e.g. a `<think>` marker) with
                # nothing yet emittable — skip unless this is the final delta,
                # which still needs a (possibly empty) delta to carry finish_reason.
                if output.finish_reason is None:
                    continue
                vllm_delta = VllmDeltaMessage()

            delta_message = DeltaMessage(
                role=vllm_delta.role,
                content=vllm_delta.content,
                reasoning=vllm_delta.reasoning,
                tool_calls=project_delta_tool_calls(vllm_delta.tool_calls),
            )
            if delta_message.content:
                content_streamed[i] = True
                accumulated_text[i].clear()
            if delta_message.reasoning:
                reasoning_streamed[i] = True

            logprobs = None
            if want_logprobs and output.logprobs is not None:
                logprobs = build_chat_logprobs(output.token_ids, output.logprobs, tokenizer, num_output_top_logprobs)

            if output.finish_reason is None:
                choice = ChatCompletionResponseStreamChoice(index=i, delta=delta_message, logprobs=logprobs)
            else:
                # Reasoning-only stream: the answer may be trapped in `reasoning`. The
                # flags are a cheap pre-filter; the re-parse is what decides, so a
                # genuinely answer-less turn stays untouched.
                if parser is not None and reasoning_streamed[i] and not content_streamed[i] and not tools_streamed[i]:
                    recovered = _reconcile_trapped_content(
                        render,
                        tokenizer,
                        vllm_req,
                        "".join(accumulated_text[i]),
                        accumulated_token_ids[i],
                        enable_auto_tools=enable_auto_tools,
                    )
                    if recovered:
                        logger.debug(
                            "request %s choice %d: recovered %d chars trapped in reasoning",
                            request_id,
                            i,
                            len(recovered),
                        )
                        delta_message.content = recovered
                finish_reason_sent[i] = True
                choice = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=delta_message,
                    logprobs=logprobs,
                    finish_reason=_finish_reason_for_choice(vllm_req, tools_streamed[i], output.finish_reason),
                    stop_reason=output.stop_reason,
                )

            yield ChatCompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[choice],
                usage=_continuous_usage(num_prompt_tokens, previous_num_tokens[i])
                if include_continuous_usage
                else None,
            )

    if include_usage:
        completion_tokens = sum(previous_num_tokens)
        reasoning_tokens = None
        if parsers[0] is not None and parsers[0].reasoning_parser is not None:
            reasoning_tokens = 0
            for i, choice_parser in enumerate(parsers):
                if choice_parser is not None and choice_parser.reasoning_parser is not None:
                    reasoning_tokens += choice_parser.reasoning_parser.count_reasoning_tokens(accumulated_token_ids[i])
        yield ChatCompletionStreamResponse(
            id=request_id,
            model=model_name,
            choices=[],
            usage=UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
                completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
                if reasoning_tokens is not None
                else None,
            ),
        )


def _continuous_usage(prompt_tokens: int, completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
