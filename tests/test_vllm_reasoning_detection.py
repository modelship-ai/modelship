from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.vllm.parsing.detect import resolve_reasoning_parser

_PICKS = [
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-4B-Thinking-2507", "qwen3"),
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "qwen3"),
    ("zai-org/GLM-4.5-Air", "qwen3"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
    ("allenai/Olmo-3-7B-Think", "olmo3"),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "ernie45"),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "seed_oss"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v3"),
    ("ibm-granite/granite-3.3-8b-instruct", "granite"),
    ("tencent/Hunyuan-A13B-Instruct", None),
    ("openai/gpt-oss-20b", None),
    ("google/gemma-4-12B-it-qat-w4a16-ct", "gemma4"),
    ("zai-org/GLM-4.6", "qwen3"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "qwen3"),
    ("HuggingFaceTB/SmolLM3-3B", "qwen3"),
    ("Qwen/QwQ-32B", "deepseek_r1"),
    ("Qwen/Qwen3-VL-2B-Instruct", None),
    ("Qwen/Qwen2.5-0.5B-Instruct", None),
]


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.parametrize(("model", "parser"), _PICKS)
def test_the_reasoning_parser_detected_for(model: str, parser: str | None) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        with open(hf_hub_download(model, "config.json")) as f:
            config = json.load(f)
    except Exception as e:
        pytest.skip(f"could not fetch the tokenizer and config of {model!r}: {e}")

    cfg = ModelshipModelConfig(name="m", model=model, usecase=ModelUsecase.generate, loader=ModelLoader.vllm)
    hf_config = SimpleNamespace(model_type=config.get("model_type"), architectures=config.get("architectures"))

    assert resolve_reasoning_parser(cfg, tokenizer.get_chat_template(), tokenizer, hf_config) == parser
