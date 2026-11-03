from __future__ import annotations

import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase

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
    ("Salesforce/xLAM-2-1b-fc-r", "xlam"),
    ("LiquidAI/LFM2-1.2B", "lfm2"),
]


def detected_parsers(model: str) -> tuple[str | None, str | None]:
    from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext as VllmUsageContext

    from modelship.infer.vllm.vllm_infer import VllmInfer

    infer = object.__new__(VllmInfer)
    infer.model_config = ModelshipModelConfig(
        name="m", model=model, usecase=ModelUsecase.generate, loader=ModelLoader.vllm
    )
    try:
        vllm_config = VllmAsyncEngineArgs(model=model, max_model_len=2048).create_engine_config(
            usage_context=VllmUsageContext.OPENAI_API_SERVER
        )
        parsers = infer._resolve_chat_parsers(vllm_config)
    except OSError as e:
        pytest.skip(f"could not fetch {model!r}: {e}")
    assert parsers is not None
    return parsers.tool_parser, parsers.reasoning_parser


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.parametrize(("model", "parser"), _PICKS)
def test_the_tool_parser_detected_for(model: str, parser: str | None) -> None:
    assert detected_parsers(model)[0] == parser
