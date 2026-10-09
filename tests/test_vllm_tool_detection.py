from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.vllm.parsing.detect import resolve_tool_parser

_PICKS = [
    ("Qwen/Qwen3-0.6B", "hermes"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "hermes"),
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "qwen3_coder"),
    ("cyankiwi/Qwen3.5-9B-AWQ-4bit", "qwen3_coder"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
    ("zai-org/GLM-4.5-Air", "glm45"),
    ("zai-org/GLM-4.6", "glm45"),
    ("zai-org/GLM-4.7", "glm45"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
    ("allenai/Olmo-3-7B-Instruct", "olmo3"),
    ("allenai/Olmo-3-7B-Think", "olmo3"),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "ernie45"),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "seed_oss"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", None),
    ("ibm-granite/granite-3.3-8b-instruct", "granite"),
    ("ibm-granite/granite-4.0-micro", "hermes"),
    ("tencent/Hunyuan-A13B-Instruct", None),
    ("openai/gpt-oss-20b", None),
    ("google/gemma-4-12B-it-qat-w4a16-ct", "gemma4"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "mimo"),
    ("HuggingFaceTB/SmolLM3-3B", "hermes"),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "hermes"),
    ("unsloth/Llama-3.2-3B-Instruct", "llama3_json"),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct", "llama4_pythonic"),
    ("unsloth/mistral-7b-instruct-v0.3", "mistral"),
    ("microsoft/Phi-4-mini-instruct", None),
    ("Salesforce/xLAM-2-1b-fc-r", "granite"),
    ("LiquidAI/LFM2-1.2B", "lfm2"),
]


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.parametrize(("model", "parser"), _PICKS)
def test_the_tool_parser_detected_for(model: str, parser: str | None) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        with open(hf_hub_download(model, "config.json")) as f:
            config = json.load(f)
        try:
            with open(hf_hub_download(model, "generation_config.json")) as f:
                eos_token_id = json.load(f).get("eos_token_id")
        except EntryNotFoundError:
            eos_token_id = None
    except Exception as e:
        pytest.skip(f"could not fetch the tokenizer and config of {model!r}: {e}")

    cfg = ModelshipModelConfig(name="m", model=model, usecase=ModelUsecase.generate, loader=ModelLoader.vllm)
    hf_config = SimpleNamespace(model_type=config.get("model_type"), architectures=config.get("architectures"))
    template = tokenizer.get_chat_template()

    assert resolve_tool_parser(cfg, template, tokenizer, hf_config, eos_token_id) == parser
