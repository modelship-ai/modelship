"""What a vLLM model reports as speech to text: language detection comes from its model class."""

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

from modelship.infer.vllm.vllm_infer import VllmInfer


def _infer(serving) -> VllmInfer:
    infer = VllmInfer.__new__(VllmInfer)
    infer.serving_transcription = serving
    return infer


@pytest.mark.asyncio
async def test_whisper_reports_its_language():
    found = await _infer(SimpleNamespace(model_cls=WhisperForConditionalGeneration)).stt_capabilities()
    assert found is not None
    assert (found.streaming, found.reports_language) == (False, True)


@pytest.mark.asyncio
async def test_a_model_class_without_language_detection_reports_none():
    found = await _infer(SimpleNamespace(model_cls=object)).stt_capabilities()
    assert found is not None
    assert found.reports_language is False


@pytest.mark.asyncio
async def test_a_model_that_does_not_transcribe_has_no_capabilities():
    assert await _infer(None).stt_capabilities() is None
