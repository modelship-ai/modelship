from __future__ import annotations

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from vllm.logger import init_logger as vllm_init_logger

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.vllm.parsing import tool_probe
from modelship.infer.vllm.parsing.detect import resolve_tool_parser
from modelship.infer.vllm.parsing.tool_probe import rendered_tool_call, tool_parsers_reading

_VOCAB = ["<unk>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<|eom|>"]
_HERMES_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
_CALLS = (
    "{% for c in m.tool_calls %}<tool_call>\n"
    '{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}\n</tool_call>{% endfor %}'
)
_HERMES = "{% if m.tool_calls %}" + _CALLS + "{% else %}{{ m.content }}{% endif %}<|im_end|>"


def _tokenizer(turn: str = _HERMES) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(_VOCAB)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    template = "{% for m in messages %}<|im_start|>{{ m.role }}\n" + turn + "\n{% endfor %}"
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<|im_end|>", chat_template=template
    )


class TestRenderedToolCall:
    def test_the_call_is_cut_out_of_the_assistant_turn(self):
        assert rendered_tool_call(_tokenizer(), None) == _HERMES_CALL

    def test_a_template_that_writes_no_call_gives_none(self):
        assert rendered_tool_call(_tokenizer("{{ m.content }}<|im_end|>"), None) is None

    def test_a_template_that_raises_gives_none(self):
        assert rendered_tool_call(_tokenizer("{{ raise_exception('no') }}"), None) is None

    def test_a_cut_inside_a_tag_moves_back_to_its_start(self):
        turn = "{% if m.tool_calls %}" + _CALLS + "{% else %}<text>{{ m.content }}</text>{% endif %}<|im_end|>"
        assert rendered_tool_call(_tokenizer(turn), None) == _HERMES_CALL

    def test_arguments_are_retried_as_a_string(self):
        turn = _HERMES.replace("c.function.arguments | tojson", 'c.function.arguments + ""')
        assert rendered_tool_call(_tokenizer(turn), None) == _HERMES_CALL

    @pytest.mark.parametrize("eos_token_id", [_VOCAB.index("<|eom|>"), [_VOCAB.index("<|eom|>")]])
    def test_a_generation_stop_token_ends_the_call(self, eos_token_id):
        turn = "{% if m.tool_calls %}" + _CALLS + "<|eom|>{% else %}{{ m.content }}<|im_end|>{% endif %}"
        assert rendered_tool_call(_tokenizer(turn), eos_token_id) == _HERMES_CALL


class TestToolParsersReading:
    def test_the_grammar_narrows_the_parsers_that_read_a_call(self):
        assert tool_parsers_reading(_tokenizer(), _HERMES_CALL) == ["hermes"]

    def test_every_reader_is_kept_when_no_grammar_accepts(self):
        call = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        assert tool_parsers_reading(_tokenizer(), call) == ["granite", "xlam"]

    def test_a_parser_that_leaves_content_is_not_a_reader(self):
        call = "<seed:tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</seed:tool_call>"
        assert tool_parsers_reading(_tokenizer(), call) == ["seed_oss"]

    def test_a_call_no_parser_reads_gives_nothing(self):
        assert tool_parsers_reading(_tokenizer(), "I will call get_weather for Paris.") == []

    def test_a_single_reader_skips_the_grammar(self, monkeypatch):
        monkeypatch.setattr(tool_probe, "_grammar_accepting", lambda *args: pytest.fail("grammar compiled"))
        call = '<function_calls>get_weather(city="Paris")</function_calls>'
        assert tool_parsers_reading(_tokenizer(), call) == ["olmo3"]

    def test_the_parsers_tried_log_nothing(self, caplog, monkeypatch):
        monkeypatch.setattr(vllm_init_logger("vllm"), "propagate", True)
        tool_parsers_reading(_tokenizer(), _HERMES_CALL)
        assert [record for record in caplog.records if record.name.startswith("vllm.")] == []


def test_the_resolver_picks_the_parser_that_reads_the_template() -> None:
    cfg = ModelshipModelConfig(name="m", model="some/model", usecase=ModelUsecase.generate, loader=ModelLoader.vllm)
    tokenizer = _tokenizer()
    hf_config = SimpleNamespace(model_type=None, architectures=None)

    assert resolve_tool_parser(cfg, tokenizer.get_chat_template(), tokenizer, hf_config, None) == "hermes"
