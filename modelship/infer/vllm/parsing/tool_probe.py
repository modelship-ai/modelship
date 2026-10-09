"""Tests vLLM's tool-call parsers against the tool call a model's chat template writes."""

from __future__ import annotations

import itertools
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_NAME = "get_weather"
_ARGUMENTS = {"city": "Paris"}
_TOOL = {
    "type": "function",
    "function": {
        "name": _NAME,
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_USER = {"role": "user", "content": "What is the weather in Paris?"}
_CONTENT = "PROBE-CONTENT"
# Mistral templates accept only 9-character alphanumeric call ids.
_CALL_IDS = ("call_1", "probe0001")


@contextmanager
def _silenced(*names: str) -> Iterator[None]:
    loggers = [logging.getLogger(name) for name in names]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def _render(tokenizer: Any, assistant: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template([_USER, assistant], tools=[_TOOL], tokenize=False, add_generation_prompt=False)


def _assistant_text(plain: str, turn: str, stops: list[str]) -> str:
    start = len(os.path.commonprefix([plain, turn]))
    # A cut inside a tag moves back to the tag's opening bracket.
    opened = turn.rfind("<", 0, start)
    if opened > turn.rfind(">", 0, start):
        start = opened
    text = turn[start:].removesuffix(plain.rpartition(_CONTENT)[2])
    for stop in stops:
        text = text.split(stop)[0] or text
    return text.strip()


def rendered_tool_call(tokenizer: Any, eos_token_id: int | list[int] | None) -> str | None:
    """The probe call as the chat template writes it, or None when it writes none."""
    try:
        plain = _render(tokenizer, {"role": "assistant", "content": _CONTENT})
    except Exception:
        return None
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id or []
    stops = [stop for stop in (tokenizer.eos_token, *tokenizer.convert_ids_to_tokens(eos_ids)) if stop]
    for content, arguments, call_id in itertools.product(("", None), (_ARGUMENTS, json.dumps(_ARGUMENTS)), _CALL_IDS):
        call = {"id": call_id, "type": "function", "function": {"name": _NAME, "arguments": arguments}}
        try:
            turn = _render(tokenizer, {"role": "assistant", "content": content, "tool_calls": [call]})
        except Exception:
            continue
        text = _assistant_text(plain, turn, stops)
        if _NAME in text:
            return text
    return None


def _grammar_accepting(tokenizer: Any, call: str, readers: dict[str, Any], request: Any) -> list[str]:
    import xgrammar as xgr

    try:
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer)))
    except Exception:
        return []
    accepting = []
    for name, parser in readers.items():
        try:
            tag = parser.get_structural_tag(request, reasoning=False)
            if tag is None:
                continue
            grammar = compiler.compile_structural_tag(json.dumps(tag.model_dump()))
            if xgr.GrammarMatcher(grammar).accept_string(call):
                accepting.append(name)
        except Exception:
            continue
    return accepting


def tool_parsers_reading(tokenizer: Any, call: str) -> list[str]:
    """Registered parsers that read ``call``, narrowed to those whose grammar accepts it."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as VllmChatCompletionRequest
    from vllm.tool_parsers import ToolParserManager as VllmToolParserManager

    request = VllmChatCompletionRequest.model_validate(
        {"model": "probe", "messages": [_USER], "tools": [_TOOL], "tool_choice": "required"}
    )
    readers = {}
    # vLLM logs a traceback for each parser that cannot load or read the text.
    with _silenced("vllm.tool_parsers", "vllm.parser"):
        for name in sorted(VllmToolParserManager.list_registered()):
            try:
                parser = VllmToolParserManager.get_tool_parser(name)(tokenizer, request.tools)  # type: ignore[arg-type]
                parsed = parser.extract_tool_calls(call, request)
                calls = [(found.function.name, json.loads(found.function.arguments)) for found in parsed.tool_calls]
            except Exception:
                continue
            if calls == [(_NAME, _ARGUMENTS)] and not (parsed.content or "").strip():
                readers[name] = parser
        if len(readers) < 2:
            return list(readers)
        return _grammar_accepting(tokenizer, call, readers, request) or list(readers)
