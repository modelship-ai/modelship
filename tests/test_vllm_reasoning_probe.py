from __future__ import annotations

from types import SimpleNamespace

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from vllm.logger import init_logger as vllm_init_logger

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.vllm.parsing.detect import resolve_reasoning_parser
from modelship.infer.vllm.parsing.reasoning_probe import (
    reasoning_parsers_reading,
    rendered_reasoning_reply,
    rendered_reasoning_tool_call,
    thinking_switch,
)

_PROMPT = "<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n"
_ANSWER = "<think>\nPROBE-REASONING\n</think>\n\nPROBE-CONTENT"
_CALL = (
    "<think>\nPROBE-REASONING\n</think>\n\n"
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)
_CALLS = (
    "{% for c in m.tool_calls %}<tool_call>\n"
    '{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}\n</tool_call>{% endfor %}'
)
_REASONING = "{% if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}"
_REPLY = _REASONING + "{% if m.tool_calls %}" + _CALLS + "{% else %}{{ m.content }}{% endif %}"
_SWITCH = "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
_READERS = [
    "deepseek_r1",
    "deepseek_v3",
    "deepseek_v4",
    "deepseek_v41",
    "ernie45",
    "glm45",
    "glm47",
    "granite_thinking_parser",
    "holo2",
    "hy_v4",
    "kimi_k2",
    "ling3",
    "mimo",
    "minimax_m2",
    "nemotron_v3",
    "qwen3",
    "step3p5",
]


def _tokenizer(reply: str = _REPLY, opening: str = _SWITCH) -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={byte: i for i, byte in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.role == 'assistant' %}"
        + reply
        + "{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n" + opening + "{% endif %}"
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|im_end|>", chat_template=template)
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    fast.add_tokens(["<think>", "</think>", "<tool_call>", "</tool_call>"])
    return fast


class TestRenderedReasoningReply:
    def test_the_reply_is_cut_out_of_the_assistant_turn(self):
        assert rendered_reasoning_reply(_tokenizer()) == (_PROMPT, _ANSWER)

    def test_a_template_that_drops_the_reasoning_gives_none(self):
        assert rendered_reasoning_reply(_tokenizer("{{ m.content }}")) is None

    def test_a_template_that_raises_gives_none(self):
        assert rendered_reasoning_reply(_tokenizer("{{ raise_exception('no') }}")) is None

    def test_a_turn_that_does_not_start_with_the_prompt_gives_none(self):
        assert rendered_reasoning_reply(_tokenizer(opening="<|thought|>")) is None


class TestRenderedReasoningToolCall:
    def test_the_call_is_cut_out_of_the_assistant_turn(self):
        assert rendered_reasoning_tool_call(_tokenizer(), None) == (_PROMPT, _CALL)

    def test_a_template_that_writes_no_call_gives_none(self):
        assert rendered_reasoning_tool_call(_tokenizer(_REASONING + "{{ m.content }}"), None) is None


class TestThinkingSwitch:
    def test_a_prompt_that_changes_with_thinking_off_is_a_switch(self):
        assert thinking_switch(_tokenizer()) is True

    def test_a_prompt_that_stays_the_same_is_no_switch(self):
        assert thinking_switch(_tokenizer(opening="")) is False

    def test_a_template_that_raises_is_no_switch(self):
        assert thinking_switch(_tokenizer(opening="{{ raise_exception('no') }}")) is False


class TestReasoningParsersReading:
    def test_the_parsers_that_read_the_reply(self):
        assert reasoning_parsers_reading(_tokenizer(), (_PROMPT, _ANSWER), None, None, False) == _READERS

    def test_the_parsers_that_also_read_the_tool_call(self):
        assert (
            reasoning_parsers_reading(_tokenizer(), (_PROMPT, _ANSWER), (_PROMPT, _CALL), "hermes", False) == _READERS
        )

    def test_a_reply_no_parser_reads_gives_nothing(self):
        answer = (_PROMPT, "[[PROBE-REASONING]]PROBE-CONTENT")
        assert reasoning_parsers_reading(_tokenizer(), answer, None, None, False) == []

    def test_the_parsers_tried_log_nothing(self, caplog, monkeypatch):
        monkeypatch.setattr(vllm_init_logger("vllm"), "propagate", True)
        reasoning_parsers_reading(_tokenizer(), (_PROMPT, _ANSWER), (_PROMPT, _CALL), "hermes", False)
        assert [record for record in caplog.records if record.name.startswith("vllm.")] == []


def test_the_resolver_picks_the_parser_that_reads_the_template() -> None:
    cfg = ModelshipModelConfig(name="m", model="some/model", usecase=ModelUsecase.generate, loader=ModelLoader.vllm)
    tokenizer = _tokenizer()
    hf_config = SimpleNamespace(model_type="qwen3", architectures=None)

    assert resolve_reasoning_parser(cfg, tokenizer.get_chat_template(), tokenizer, hf_config, "hermes", None) == "qwen3"
