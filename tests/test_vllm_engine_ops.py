"""Guard tests for modelship/infer/vllm/engine_ops.py, the vLLM-internal quarantine layer.

Fast mocked unit tests for the branching logic, plus
`TestVllmParserAcceptsOurRequest`, which runs a real (GPU-free) vLLM render
pipeline against cached tokenizers to catch call-signature drift.
"""

import inspect
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall as VllmDeltaFunctionCall
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as VllmDeltaMessage
from vllm.entrypoints.openai.engine.protocol import DeltaToolCall as VllmDeltaToolCall
from vllm.parser import Parser as VllmParser
from vllm.renderers.online_renderer import OnlineRenderer as VllmOnlineRenderer
from vllm.tokenizers import TokenizerLike as VllmTokenizerLike

from modelship.infer.vllm import engine_ops
from modelship.openai.protocol import ChatCompletionRequest


def _vllm_req(**overrides: Any) -> VllmChatCompletionRequest:
    base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    base.update(overrides)
    return VllmChatCompletionRequest(**base)


class TestBuildVllmRequest:
    def test_request_chat_template_kwargs_wins_over_model_default(self):
        # chat_template_kwargs is a client-supplied extra (extra="allow"), not a
        # declared field, so it must be built via model_validate, not kwargs.
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs={"enable_thinking": True, "x": 1})
        assert vllm_req.chat_template_kwargs == {"enable_thinking": False, "x": 1}

    def test_no_model_defaults_leaves_request_value_untouched(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"a": 1},
            }
        )
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs=None)
        assert vllm_req.chat_template_kwargs == {"a": 1}

    def test_cache_salt_set_from_identity(self):
        # cache_salt scopes vLLM's prefix-cache reuse per caller identity.
        request = ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs=None, cache_salt="identity-a")
        assert vllm_req.cache_salt == "identity-a"

    def test_no_cache_salt_leaves_it_unset(self):
        request = ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs=None)
        assert vllm_req.cache_salt is None


class TestDeriveReasoningEnded:
    def test_include_reasoning_false_forces_true(self):
        vllm_req = _vllm_req(include_reasoning=False)
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is True

    def test_grammar_from_parser_forces_true(self):
        vllm_req = _vllm_req()
        vllm_req._grammar_from_parser = True
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is True

    def test_reasoning_parser_present_defers_to_is_reasoning_end(self):
        vllm_req = _vllm_req()
        parser = Mock(spec=VllmParser)
        parser.reasoning_parser = Mock()
        parser.is_reasoning_end.return_value = True
        result = engine_ops.derive_reasoning_ended(vllm_req, parser=parser, prompt_token_ids=[1, 2, 3])
        assert result is True
        parser.is_reasoning_end.assert_called_once_with([1, 2, 3])

    def test_no_reasoning_parser_and_no_grammar_flag_is_none(self):
        vllm_req = _vllm_req()
        parser = Mock(spec=VllmParser)
        parser.reasoning_parser = None
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=parser, prompt_token_ids=[]) is None

    def test_no_parser_at_all_is_none(self):
        vllm_req = _vllm_req()
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is None


class TestTotalReasoningTokens:
    """`usage.completion_tokens_details.reasoning_tokens` sums vLLM's own
    `count_reasoning_tokens` across choices."""

    def test_no_parser_returns_none(self):
        outputs = [Mock(token_ids=[1, 2, 3])]
        assert engine_ops.total_reasoning_tokens(outputs, parser=None) is None

    def test_parser_without_reasoning_returns_none(self):
        parser = Mock(spec=VllmParser)
        parser.reasoning_parser = None
        outputs = [Mock(token_ids=[1, 2, 3])]
        assert engine_ops.total_reasoning_tokens(outputs, parser=parser) is None

    def test_sums_across_choices(self):
        parser = Mock(spec=VllmParser)
        parser.reasoning_parser = Mock()
        parser.reasoning_parser.count_reasoning_tokens.side_effect = [5, 2]
        outputs = [Mock(token_ids=[1, 2, 3, 4, 5]), Mock(token_ids=[6, 7])]

        result = engine_ops.total_reasoning_tokens(outputs, parser=parser)

        assert result == 7
        assert parser.reasoning_parser.count_reasoning_tokens.call_args_list == [
            ((list(outputs[0].token_ids),),),
            ((list(outputs[1].token_ids),),),
        ]


class TestMakeParsers:
    def test_no_parser_class_returns_n_nones(self):
        render = Mock(spec=VllmOnlineRenderer)
        render.parser = None
        result = engine_ops.make_parsers(render, tokenizer=Mock(), vllm_req=_vllm_req(), chat_template_kwargs=None, n=3)
        assert result == [None, None, None]

    def test_instantiates_one_independent_parser_per_choice(self):
        render = Mock(spec=VllmOnlineRenderer)
        parser_cls = Mock()
        instances = [Mock(), Mock(), Mock()]
        parser_cls.side_effect = instances
        render.parser = parser_cls
        vllm_req = _vllm_req(tools=None)
        tokenizer = Mock()

        result = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs={"k": "v"}, n=3)

        assert result == instances
        assert parser_cls.call_count == 3
        for call in parser_cls.call_args_list:
            args, kwargs = call
            assert args == (tokenizer, vllm_req.tools)
            assert kwargs == {"chat_template_kwargs": {"k": "v"}}


class _TrapParser:
    """Mirrors vLLM's parse/parse_delta split: parse_delta streams everything as
    reasoning while parse reads the same tokens as content."""

    full_parse_result: ClassVar[tuple[str | None, str | None, Any]] = (None, "the answer", None)
    delta_tool_calls: ClassVar[list[VllmDeltaToolCall]] = []
    delta_as_content = False
    parsed_text: ClassVar[str | None] = None

    def __init__(self, tokenizer, tools=None, chat_template_kwargs=None):
        self.reasoning_parser = None

    def parse_delta(self, *, delta_text, delta_token_ids, request, prompt_token_ids, finished):
        if self.delta_as_content:
            return VllmDeltaMessage(content=delta_text)
        return VllmDeltaMessage(reasoning=delta_text, tool_calls=self.delta_tool_calls)

    def parse(self, model_output, request, enable_auto_tools=False, model_output_token_ids=()):
        # Record what the reconcile handed us: it must be the engine's own streamed text,
        # not a re-decode of the token ids (which would carry the stop token).
        type(self).parsed_text = model_output
        return self.full_parse_result


def _res(text: str, token_ids: list[int], finish_reason: str | None):
    output = SimpleNamespace(
        index=0, text=text, token_ids=token_ids, finish_reason=finish_reason, logprobs=None, stop_reason=None
    )
    return SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])


async def _drive_stream(monkeypatch, parser_cls, *, results=None) -> list[Any]:
    """Run `stream_chat_completion` over a scripted engine stream, returning the chunks."""
    monkeypatch.setattr(engine_ops, "extract_prompt_token_ids", lambda render, engine_input: [1, 2])

    async def fake_generate(*args, **kwargs):
        for r in results or [_res("Hello there", [10], None), _res("", [], "stop")]:
            yield r

    engine = Mock()
    engine.generate = fake_generate
    render = Mock(spec=VllmOnlineRenderer)
    render.parser = parser_cls

    chunks = []
    async for chunk in engine_ops.stream_chat_completion(
        engine,
        render,
        _vllm_req(tools=None),
        engine_input=Mock(),
        sampling_params=Mock(),
        request_id="req-1",
        model_name="m",
        tokenizer=Mock(decode=Mock(return_value="a token re-decode, not the engine's text")),
        enable_auto_tools=False,
        want_logprobs=False,
        num_output_top_logprobs=None,
    ):
        chunks.append(chunk)
    return chunks


class TestStreamReconcilesTrappedContent:
    """A model that never closes its reasoning leaves the whole streamed reply in
    `reasoning` with empty `content`; the finish branch reconciles against the
    non-streaming `parse()`, which reads the same tokens as `content`."""

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_recovers_content_on_finish(self, monkeypatch):
        chunks = await _drive_stream(monkeypatch, _TrapParser)

        final = chunks[-1]
        assert final.choices[0].finish_reason == "stop"
        assert final.choices[0].delta.content == "the answer"

    @pytest.mark.asyncio
    async def test_deferred_deltas_still_reach_the_parse(self, monkeypatch):
        # Deferred deltas (parse_delta returns None) must still be accumulated
        # into the reconciled text, not dropped.
        class DeferringParser(_TrapParser):
            def parse_delta(self, *, delta_text, delta_token_ids, request, prompt_token_ids, finished):
                if delta_text == " + ":  # deferred: consumed but not emitted
                    return None
                return VllmDeltaMessage(reasoning=delta_text)

        _TrapParser.parsed_text = None
        results = [
            _res("2", [1], None),
            _res(" + ", [2], None),
            _res("2 equals 4.", [3], None),
            _res("", [], "stop"),
        ]
        await _drive_stream(monkeypatch, DeferringParser, results=results)

        assert DeferringParser.parsed_text == "2 + 2 equals 4."

    @pytest.mark.asyncio
    async def test_streamed_content_is_not_reconciled(self, monkeypatch):
        class ContentParser(_TrapParser):
            delta_as_content = True
            full_parse_result = (None, "SHOULD NOT APPEAR", None)

        chunks = await _drive_stream(monkeypatch, ContentParser)

        streamed = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert "SHOULD NOT APPEAR" not in streamed
        assert "Hello there" in streamed

    @pytest.mark.asyncio
    async def test_reasoning_with_no_answer_stays_empty(self, monkeypatch):
        # Reasoning that genuinely closed with no answer: parse() agrees there is
        # no content, so nothing is attached.
        class NoAnswerParser(_TrapParser):
            full_parse_result = ("some reasoning", None, None)

        chunks = await _drive_stream(monkeypatch, NoAnswerParser)

        assert chunks[-1].choices[0].delta.content is None

    @pytest.mark.asyncio
    async def test_tool_call_stream_is_not_reconciled(self, monkeypatch):
        class ToolParser(_TrapParser):
            delta_tool_calls: ClassVar[list[VllmDeltaToolCall]] = [
                VllmDeltaToolCall(
                    index=0, id="c1", type="function", function=VllmDeltaFunctionCall(name="f", arguments="{}")
                )
            ]
            full_parse_result: ClassVar[tuple[str | None, str | None, Any]] = (None, "SHOULD NOT APPEAR", None)

        chunks = await _drive_stream(monkeypatch, ToolParser)

        assert chunks[-1].choices[0].delta.content is None


class TestSignaturesGuardVllmBump:
    """Pure import-time signature checks (no engine/tokenizer) that fail loudly
    if a vLLM bump renames or drops a kwarg engine_ops relies on."""

    def test_async_llm_generate_accepts_reasoning_kwargs(self):
        from vllm.v1.engine.async_llm import AsyncLLM as VllmAsyncLLM

        params = inspect.signature(VllmAsyncLLM.generate).parameters
        assert "reasoning_ended" in params
        assert "reasoning_parser_kwargs" in params

    def test_chat_completion_request_has_grammar_from_parser_private_attr(self):
        vllm_req = _vllm_req()
        assert hasattr(vllm_req, "_grammar_from_parser")
        assert vllm_req._grammar_from_parser is False

    def test_to_sampling_params_signature_unchanged(self):
        params = inspect.signature(VllmChatCompletionRequest.to_sampling_params).parameters
        assert list(params)[1:] == ["max_tokens", "default_sampling_params"]

    def test_render_chat_signature_unchanged(self):
        params = inspect.signature(VllmOnlineRenderer.render_chat).parameters
        assert "request" in params


class TestVllmParserAcceptsOurRequest:
    """Real vLLM render pipeline and cached tokenizers, no engine/GPU. Skips
    cleanly if the tokenizer can't be fetched."""

    def _build_render(
        self, model: str, *, tokenizer_mode: str = "auto", **tool_reasoning_kwargs: Any
    ) -> tuple[VllmOnlineRenderer, VllmTokenizerLike]:
        from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs
        from vllm.entrypoints.serve.utils.request_logger import RequestLogger as VllmRequestLogger
        from vllm.renderers import renderer_from_config as vllm_renderer_from_config
        from vllm.usage.usage_lib import UsageContext as VllmUsageContext

        try:
            engine_args = VllmAsyncEngineArgs(
                model=model, tokenizer_mode=tokenizer_mode, max_model_len=4096, enforce_eager=True
            )
            vllm_config = engine_args.create_engine_config(usage_context=VllmUsageContext.OPENAI_API_SERVER)
            renderer = vllm_renderer_from_config(vllm_config)
        except Exception as e:
            pytest.skip(f"could not build a GPU-free render pipeline for {model!r}: {e}")

        render = VllmOnlineRenderer(
            model_config=vllm_config.model_config,
            renderer=renderer,
            request_logger=VllmRequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            **tool_reasoning_kwargs,
        )
        assert renderer.tokenizer is not None
        return render, renderer.tokenizer

    def _tool_request(self, **overrides: Any) -> VllmChatCompletionRequest:
        base: dict[str, Any] = dict(
            model="test-model",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=64,
        )
        base.update(overrides)
        return VllmChatCompletionRequest(**base)

    @pytest.mark.asyncio
    async def test_hermes_qwen3_reasoning_and_tools(self):
        render, tokenizer = self._build_render("Qwen/Qwen3-0.6B", tool_parser="hermes", reasoning_parser="qwen3")
        vllm_req = self._tool_request()

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, sampling_params = result
        assert sampling_params.max_tokens == 64

        prompt_ids = engine_ops.extract_prompt_token_ids(render, engine_input)
        assert prompt_ids

        n = 2
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs=None, n=n)
        assert len(parsers) == n
        assert all(p is not None for p in parsers)

        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_req, parsers[0], prompt_ids)
        assert reasoning_ended is False  # prompt has no prior reasoning to have already ended

        fake_output = (
            "<think>I should check the weather.</think>\n"
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
        )
        for parser in parsers:
            assert parser is not None
            reasoning, content, tool_calls = parser.parse(
                fake_output, vllm_req, enable_auto_tools=True, model_output_token_ids=[]
            )
            assert reasoning == "I should check the weather."
            assert content is None
            assert tool_calls is not None and len(tool_calls) == 1
            assert tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_mistral_tool_only_no_grammar_flag(self):
        # A tool-only request takes MistralParser.adjust_request's early-return
        # branch and never sets _grammar_from_parser.
        render, tokenizer = self._build_render(
            "mistralai/Mistral-7B-Instruct-v0.3", tokenizer_mode="mistral", tool_parser="mistral"
        )
        vllm_req = self._tool_request()

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, _sampling_params = result

        assert vllm_req._grammar_from_parser is False

        prompt_ids = engine_ops.extract_prompt_token_ids(render, engine_input)
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs=None, n=1)
        assert len(parsers) == 1
        parser = parsers[0]
        assert parser is not None

        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_req, parser, prompt_ids)
        assert reasoning_ended is None  # no reasoning parser configured for this model

        # No AttributeError on a plain-text (non-tool-call) output.
        reasoning, content, tool_calls = parser.parse(
            "Just a plain text response.", vllm_req, enable_auto_tools=True, model_output_token_ids=[]
        )
        assert reasoning is None
        assert content == "Just a plain text response."
        assert tool_calls in (None, [])
