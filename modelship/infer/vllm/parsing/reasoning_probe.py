"""Tests vLLM's reasoning parsers against the reasoning a model's chat template writes."""

from __future__ import annotations

import itertools
import json
from typing import Any

from modelship.infer.vllm.parsing.tool_probe import _ARGUMENTS, _CALL_IDS, _CONTENT, _NAME, _TOOL, _USER, _silenced

_REASONING = "PROBE-REASONING"


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    thinking: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        enable_thinking=thinking,
        thinking=thinking,
    )


def _reply(tokenizer: Any, assistant: dict[str, Any], tools: list[dict[str, Any]] | None) -> tuple[str, str] | None:
    planted = {**assistant, "reasoning_content": _REASONING, "reasoning": _REASONING, "thinking": _REASONING}
    try:
        prompt = _render(tokenizer, [_USER], True, tools)
        turn = _render(tokenizer, [_USER, planted], False, tools)
    except Exception:
        return None
    if _REASONING not in turn or not turn.startswith(prompt):
        return None
    return prompt, turn[len(prompt) :]


def rendered_reasoning_reply(tokenizer: Any) -> tuple[str, str] | None:
    """The prompt and the probe answer with its reasoning as the chat template writes them, or None."""
    answer = {"role": "assistant", "content": _CONTENT}
    try:
        plain = _render(tokenizer, [_USER, answer], False)
    except Exception:
        return None
    reply = _reply(tokenizer, answer, None)
    if reply is None:
        return None
    prompt, text = reply
    return prompt, text.removesuffix(plain.rpartition(_CONTENT)[2])


def rendered_reasoning_tool_call(tokenizer: Any, eos_token_id: int | list[int] | None) -> tuple[str, str] | None:
    """The prompt and the probe tool call with its reasoning as the chat template writes them, or None."""
    for content, arguments, call_id in itertools.product(("", None), (_ARGUMENTS, json.dumps(_ARGUMENTS)), _CALL_IDS):
        call = {"id": call_id, "type": "function", "function": {"name": _NAME, "arguments": arguments}}
        reply = _reply(tokenizer, {"role": "assistant", "content": content, "tool_calls": [call]}, [_TOOL])
        if reply is None or _NAME not in reply[1]:
            continue
        prompt, text = reply
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id or []
        for stop in (tokenizer.eos_token, *tokenizer.convert_ids_to_tokens(eos_ids)):
            if stop:
                text = text.split(stop)[0] or text
        return prompt, text.rstrip()
    return None


def thinking_switch(tokenizer: Any) -> bool:
    """Whether the chat template's generation prompt changes when thinking is turned off."""
    try:
        return _render(tokenizer, [_USER], True) != _render(tokenizer, [_USER], True, thinking=False)
    except Exception:
        return False


def _streamed(parser: Any, tokenizer: Any, request: Any, prompt: str, ids: list[int]) -> tuple[str, str, list[Any]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    reasoning, content, calls, previous = "", "", {}, ""
    for i, token_id in enumerate(ids):
        text = tokenizer.decode(ids[: i + 1])
        delta = parser.parse_delta(
            text[len(previous) :], [token_id], request, prompt_token_ids=prompt_ids, finished=i == len(ids) - 1
        )
        previous = text
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            name, arguments = calls.get(call.index, (None, ""))
            if call.function is not None:
                name, arguments = call.function.name or name, arguments + (call.function.arguments or "")
            calls[call.index] = (name, arguments)
    return reasoning, content, [(name, json.loads(arguments)) for name, arguments in calls.values()]


def _reads(parser_cls: Any, tokenizer: Any, prompt: str, reply: str, tools: list[dict[str, Any]] | None) -> bool:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as VllmChatCompletionRequest

    thinking = {"enable_thinking": True, "thinking": True}
    body: dict[str, Any] = {"model": "probe", "messages": [_USER], "chat_template_kwargs": thinking}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    request = VllmChatCompletionRequest.model_validate(body)
    expected = (_REASONING, "", [(_NAME, _ARGUMENTS)]) if tools else (_REASONING, _CONTENT, [])
    ids = tokenizer.encode(reply, add_special_tokens=False)

    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs=thinking)
    reasoning, content, calls = parser.parse(reply, request, enable_auto_tools=bool(tools), model_output_token_ids=ids)
    parsed = (reasoning, content, [(call.name, json.loads(call.arguments)) for call in calls or []])
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs=thinking)
    results = (parsed, _streamed(parser, tokenizer, request, prompt, ids))
    return all(((r or "").strip(), (c or "").strip(), k) == expected for r, c, k in results)


def reasoning_parsers_reading(
    tokenizer: Any,
    answer: tuple[str, str],
    tool_call: tuple[str, str] | None,
    tool_parser: str | None,
    harmony: bool,
) -> list[str]:
    """Registered parsers that read the replies."""
    from vllm.parser.parser_manager import ParserManager as VllmParserManager
    from vllm.reasoning import ReasoningParser as VllmReasoningParser
    from vllm.reasoning import ReasoningParserManager as VllmReasoningParserManager

    replies = [(*answer, None)] + ([(*tool_call, [_TOOL])] if tool_call is not None else [])
    readers = []
    with _silenced("vllm.reasoning", "vllm.parser", "vllm.tool_parsers"):
        for name in sorted(VllmReasoningParserManager.list_registered()):
            try:
                reasoning_cls = VllmReasoningParserManager.get_reasoning_parser(name)
                if reasoning_cls.is_reasoning_end is VllmReasoningParser.is_reasoning_end:
                    continue
                parser_cls = VllmParserManager.get_parser(
                    tool_parser_name=tool_parser,
                    reasoning_parser_name=name,
                    enable_auto_tools=tool_parser is not None,
                    is_harmony=harmony,
                )
                if all(_reads(parser_cls, tokenizer, *reply) for reply in replies):
                    readers.append(name)
            except Exception:
                continue
    return readers
