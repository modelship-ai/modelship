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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.generate.base.protocol import DeltaFunctionCall as VllmDeltaFunctionCall
from vllm.entrypoints.generate.base.protocol import DeltaMessage as VllmDeltaMessage
from vllm.entrypoints.generate.base.protocol import DeltaToolCall as VllmDeltaToolCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.parser import Parser as VllmParser
from vllm.parser import ParserManager as VllmParserManager
from vllm.renderers.online_renderer import OnlineRenderer as VllmOnlineRenderer
from vllm.tokenizers import TokenizerLike as VllmTokenizerLike

from modelship.infer.vllm import engine_ops
from modelship.openai.protocol import ChatCompletionRequest


def _vllm_req(**overrides: Any) -> VllmChatCompletionRequest:
    base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    base.update(overrides)
    return VllmChatCompletionRequest(**base)


def _render(parser_cls: Any, **default_chat_template_kwargs: Any) -> Mock:
    return Mock(
        spec=VllmOnlineRenderer,
        parser=parser_cls,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs=default_chat_template_kwargs,
    )


class TestBuildVllmRequest:
    def test_request_chat_template_kwargs_are_kept(self):
        # chat_template_kwargs is a client-supplied extra (extra="allow"), not a
        # declared field, so it must be built via model_validate, not kwargs.
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"a": 1},
            }
        )
        vllm_req = engine_ops.build_vllm_request(request)
        assert vllm_req.chat_template_kwargs == {"a": 1}

    def test_cache_salt_set_from_identity(self):
        # cache_salt scopes vLLM's prefix-cache reuse per caller identity.
        request = ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        vllm_req = engine_ops.build_vllm_request(request, cache_salt="identity-a")
        assert vllm_req.cache_salt == "identity-a"

    def test_no_cache_salt_leaves_it_unset(self):
        request = ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        vllm_req = engine_ops.build_vllm_request(request)
        assert vllm_req.cache_salt is None

    def test_no_tools_and_no_tool_choice_gives_none(self):
        request = ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        vllm_req = engine_ops.build_vllm_request(request)
        assert vllm_req.tool_choice == "none"

    def test_tools_and_no_tool_choice_gives_auto(self):
        tool = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        request = ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}
        )
        vllm_req = engine_ops.build_vllm_request(request)
        assert vllm_req.tool_choice == "auto"

    def test_a_tool_choice_from_the_request_is_kept(self):
        tool = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        request = ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [tool], "tool_choice": "required"}
        )
        vllm_req = engine_ops.build_vllm_request(request)
        assert vllm_req.tool_choice == "required"


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


class TestResolvedChatTemplateKwargs:
    @pytest.mark.parametrize(
        ("defaults", "request_kwargs", "effort", "expected"),
        [
            ({"enable_thinking": True}, None, None, True),
            ({"enable_thinking": True}, None, "none", False),
            ({"enable_thinking": False}, None, "low", True),
            ({"enable_thinking": True}, {"enable_thinking": False}, "high", False),
            ({"enable_thinking": False}, {"enable_thinking": True}, "none", True),
        ],
    )
    def test_enable_thinking_is_the_requests_then_the_efforts_then_the_default(
        self, defaults, request_kwargs, effort, expected
    ):
        vllm_req = _vllm_req(chat_template_kwargs=request_kwargs, reasoning_effort=effort)

        resolved = engine_ops.resolved_chat_template_kwargs(_render(None, **defaults), vllm_req)

        assert resolved["enable_thinking"] is expected

    def test_an_effort_leaves_the_other_defaults_alone(self):
        vllm_req = _vllm_req(reasoning_effort="none")

        resolved = engine_ops.resolved_chat_template_kwargs(_render(None, clear_thinking=False), vllm_req)

        assert resolved["clear_thinking"] is False


class TestMakeParsers:
    def test_no_parser_class_returns_n_nones(self):
        result = engine_ops.make_parsers(_render(None), tokenizer=Mock(), vllm_req=_vllm_req(), n=3)
        assert result == [None, None, None]

    def test_instantiates_one_independent_parser_per_choice(self):
        parser_cls = Mock()
        instances = [Mock(), Mock(), Mock()]
        parser_cls.side_effect = instances
        vllm_req = _vllm_req(tools=None)
        tokenizer = Mock()

        result = engine_ops.make_parsers(_render(parser_cls, k="v"), tokenizer, vllm_req, n=3)

        assert result == instances
        assert parser_cls.call_count == 3
        for call in parser_cls.call_args_list:
            args, kwargs = call
            assert args == (tokenizer, vllm_req.tools)
            assert list(kwargs) == ["chat_template_kwargs"]
            assert kwargs["chat_template_kwargs"]["k"] == "v"

    @pytest.mark.parametrize("tool_parser", ["qwen3_xml", "hermes"])
    def test_effort_none_tells_the_parser_thinking_is_off(self, tool_parser):
        parser_cls = VllmParserManager.get_parser(
            tool_parser_name=tool_parser, reasoning_parser_name="qwen3", enable_auto_tools=True
        )
        tokenizer = _word_tokenizer()
        vllm_req = _vllm_req(reasoning_effort="none")

        parser = engine_ops.make_parsers(_render(parser_cls, enable_thinking=True), tokenizer, vllm_req, n=1)[0]
        reasoning, content, _ = parser.parse(
            "ok", vllm_req, enable_auto_tools=True, model_output_token_ids=tokenizer.encode("ok")
        )

        assert (reasoning, content) == (None, "ok")


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


async def _drive_stream(monkeypatch, parser_cls, *, results=None, vllm_req=None, tokenizer=None) -> list[Any]:
    """Run `stream_chat_completion` over a scripted engine stream, returning the chunks."""
    monkeypatch.setattr(engine_ops, "extract_prompt_token_ids", lambda render, engine_input: [1, 2])

    async def fake_generate(*args, **kwargs):
        for r in results or [_res("Hello there", [10], None), _res("", [], "stop")]:
            yield r

    engine = Mock()
    engine.generate = fake_generate
    render = _render(parser_cls)

    chunks = []
    async for chunk in engine_ops.stream_chat_completion(
        engine,
        render,
        vllm_req or _vllm_req(tools=None),
        engine_input=Mock(),
        sampling_params=Mock(),
        request_id="req-1",
        model_name="m",
        tokenizer=tokenizer or Mock(decode=Mock(return_value="a token re-decode, not the engine's text")),
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


def _word_tokenizer() -> PreTrainedTokenizerFast:
    vocab = ["<unk>", "a", "b", "c", "ok", "<think>", "</think>", "<tool_call>", "</tool_call>"]
    tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(vocab)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")


class TestReasoningTokens:
    @pytest.mark.parametrize(
        ("tool_parser", "reasoning_parser", "expected"),
        [("qwen3_xml", "qwen3", 4), ("hermes", "qwen3", 4), ("hermes", None, None)],
    )
    def test_build_choices_sums_the_reasoning_of_every_choice(self, tool_parser, reasoning_parser, expected):
        tokenizer = _word_tokenizer()
        parser_cls = VllmParserManager.get_parser(
            tool_parser_name=tool_parser, reasoning_parser_name=reasoning_parser, enable_auto_tools=True
        )
        outputs = [
            SimpleNamespace(text=text, token_ids=tokenizer.encode(text), finish_reason="stop", logprobs=None)
            for text in ("<think> a b c </think> ok", "<think> a </think> ok")
        ]

        *_, reasoning_tokens = engine_ops.build_choices(
            SimpleNamespace(outputs=outputs),
            _vllm_req(),
            parser_cls(tokenizer),
            tokenizer,
            enable_auto_tools=True,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )

        assert reasoning_tokens == expected

    def test_build_choices_without_a_parser_counts_nothing(self):
        outputs = [SimpleNamespace(text="ok", token_ids=[4], finish_reason="stop", logprobs=None)]

        *_, reasoning_tokens = engine_ops.build_choices(
            SimpleNamespace(outputs=outputs),
            _vllm_req(),
            None,
            Mock(),
            enable_auto_tools=False,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )

        assert reasoning_tokens is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_parser", ["qwen3_xml", "hermes"])
    async def test_the_stream_usage_chunk_counts_the_reasoning(self, monkeypatch, tool_parser):
        tokenizer = _word_tokenizer()
        parser_cls = VllmParserManager.get_parser(
            tool_parser_name=tool_parser, reasoning_parser_name="qwen3", enable_auto_tools=True
        )
        words = ["<think>", "a", "b", "</think>", "ok"]
        results = [_res(f"{word} ", tokenizer.encode(word), None) for word in words]
        results.append(_res("", [], "stop"))

        chunks = await _drive_stream(
            monkeypatch,
            parser_cls,
            results=results,
            vllm_req=_vllm_req(stream=True, stream_options={"include_usage": True}),
            tokenizer=tokenizer,
        )

        assert chunks[-1].usage.completion_tokens_details.reasoning_tokens == 2


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

    def test_online_renderer_takes_default_chat_template_kwargs(self):
        assert "default_chat_template_kwargs" in inspect.signature(VllmOnlineRenderer.__init__).parameters

    def test_engine_args_take_a_served_model_name(self):
        from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs

        assert "served_model_name" in inspect.signature(VllmAsyncEngineArgs).parameters

    def test_structured_outputs_config_has_a_reasoning_parser(self):
        from vllm.config import StructuredOutputsConfig as VllmStructuredOutputsConfig

        assert VllmStructuredOutputsConfig().reasoning_parser == ""

    def test_cached_tokenizer_from_config_takes_the_model_config(self):
        from vllm.tokenizers import cached_tokenizer_from_config as vllm_cached_tokenizer_from_config

        assert next(iter(inspect.signature(vllm_cached_tokenizer_from_config).parameters)) == "model_config"


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
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, n=n)
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
    async def test_a_message_reasoning_is_in_the_prompt_while_its_tool_call_is_answered(self):
        render, tokenizer = self._build_render("Qwen/Qwen3-0.6B", tool_parser="hermes", reasoning_parser="qwen3")
        vllm_req = self._tool_request(
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "I should check the weather.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, _sampling_params = result
        prompt = tokenizer.decode(engine_ops.extract_prompt_token_ids(render, engine_input))
        assert "I should check the weather." in prompt

    @pytest.mark.asyncio
    async def test_effort_none_turns_thinking_off_in_the_prompt_and_the_parser(self):
        render, tokenizer = self._build_render(
            "Qwen/Qwen3-0.6B",
            tool_parser="hermes",
            reasoning_parser="qwen3",
            default_chat_template_kwargs={"enable_thinking": True},
        )
        vllm_req = _vllm_req(reasoning_effort="none")

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, _sampling_params = result
        prompt = tokenizer.decode(engine_ops.extract_prompt_token_ids(render, engine_input))
        assert prompt.endswith("<think>\n\n</think>\n\n")

        parser = engine_ops.make_parsers(render, tokenizer, vllm_req, n=1)[0]
        assert parser is not None
        reasoning, content, _tool_calls = parser.parse(
            "56", vllm_req, enable_auto_tools=True, model_output_token_ids=[]
        )
        assert (reasoning, content) == (None, "56")

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
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, n=1)
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

    def test_structured_output_manager_takes_a_reasoning_parser_set_on_the_built_config(self):
        from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs
        from vllm.reasoning import ReasoningParserManager as VllmReasoningParserManager
        from vllm.tokenizers import cached_tokenizer_from_config as vllm_cached_tokenizer_from_config
        from vllm.usage.usage_lib import UsageContext as VllmUsageContext
        from vllm.v1.structured_output import StructuredOutputManager as VllmStructuredOutputManager

        try:
            engine_args = VllmAsyncEngineArgs(model="Qwen/Qwen3-0.6B", max_model_len=4096, enforce_eager=True)
            vllm_config = engine_args.create_engine_config(usage_context=VllmUsageContext.OPENAI_API_SERVER)
            vllm_cached_tokenizer_from_config(vllm_config.model_config)
        except Exception as e:
            pytest.skip(f"could not build a GPU-free engine config: {e}")

        vllm_config.structured_outputs_config.reasoning_parser = "deepseek_r1"
        manager = VllmStructuredOutputManager(vllm_config)

        assert manager.reasoner_cls is VllmReasoningParserManager.get_reasoning_parser("deepseek_r1")
