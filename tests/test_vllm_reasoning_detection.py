from __future__ import annotations

import pytest

from tests.test_vllm_tool_detection import detected_parsers

_PICKS = [
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-4B-Thinking-2507", "qwen3"),
    ("cyankiwi/Qwen3.5-9B-AWQ-4bit", "qwen3"),
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "qwen3"),
    ("zai-org/GLM-4.5-Air", "glm45"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
    ("allenai/Olmo-3-7B-Think", "olmo3"),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "ernie45"),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "seed_oss"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1"),
    ("deepseek-ai/DeepSeek-R1-0528", "deepseek_r1"),
    ("deepseek-ai/DeepSeek-V3-0324", None),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v3"),
    ("ibm-granite/granite-3.3-8b-instruct", None),
    ("tencent/Hunyuan-A13B-Instruct", None),
    ("openai/gpt-oss-20b", "openai_gptoss"),
    ("google/gemma-4-12B-it-qat-w4a16-ct", "gemma4"),
    ("zai-org/GLM-4.6", "glm45"),
    ("zai-org/GLM-4.7", "glm45"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_v3"),
    ("HuggingFaceTB/SmolLM3-3B", "deepseek_r1"),
    ("Qwen/QwQ-32B", "deepseek_r1"),
    ("Qwen/Qwen3-VL-2B-Instruct", None),
    ("Qwen/Qwen2.5-0.5B-Instruct", None),
]


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.parametrize(("model", "parser"), _PICKS)
def test_the_reasoning_parser_detected_for(model: str, parser: str | None) -> None:
    assert detected_parsers(model)[1] == parser
