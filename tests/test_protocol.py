"""Strict-OpenAI shape tests for modelship/openai/protocol.py: ErrorInfo.code
retype, verbose response shapes, required `model` fields, and
ImageGenerationRequest without its Diffusers-only knobs."""

from http import HTTPStatus
from typing import get_args

import pytest
from pydantic import ValidationError

from modelship.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingCompletionRequest,
    ErrorInfo,
    ErrorResponse,
    ImageGenerationRequest,
    TranscriptionRequest,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
    TranscriptionWord,
    TranslationRequest,
    TranslationResponseVerbose,
    create_error_response,
)


def test_error_info_code_accepts_string_or_none():
    e = ErrorInfo(message="boom", type="invalid_request_error")
    assert e.code is None

    e2 = ErrorInfo(message="boom", type="invalid_request_error", code="context_length_exceeded")
    assert e2.code == "context_length_exceeded"


def test_error_info_rejects_int_code():
    with pytest.raises(ValidationError):
        ErrorInfo(message="boom", type="invalid_request_error", code=400)  # type: ignore[arg-type]


def test_error_response_http_status_default_is_500():
    resp = ErrorResponse(error=ErrorInfo(message="boom", type="api_error"))
    assert resp._http_status == 500


def test_error_response_http_status_is_not_serialized():
    resp = ErrorResponse(error=ErrorInfo(message="boom", type="api_error"))
    resp._http_status = 418
    dumped = resp.model_dump()
    assert "_http_status" not in dumped
    assert "http_status" not in dumped
    # The OpenAI-spec body is the `error` object only.
    assert set(dumped.keys()) == {"error"}


def test_create_error_response_sets_http_status_from_status_code():
    resp = create_error_response(
        message="overflow",
        err_type="invalid_request_error",
        status_code=HTTPStatus.BAD_REQUEST,
    )
    assert resp._http_status == 400
    assert resp.error.type == "invalid_request_error"
    assert resp.error.code is None


def test_create_error_response_maps_value_error_to_400():
    resp = create_error_response(ValueError("bad input"))
    assert resp._http_status == 400
    assert resp.error.type == "invalid_request_error"


def test_create_error_response_maps_not_implemented_to_501():
    resp = create_error_response(NotImplementedError("nope"))
    assert resp._http_status == 501
    assert resp.error.type == "api_error"


def test_create_error_response_maps_other_exception_to_500():
    resp = create_error_response(RuntimeError("boom"))
    assert resp._http_status == 500
    assert resp.error.type == "api_error"


def test_transcription_verbose_duration_is_float():
    v = TranscriptionResponseVerbose(language="en", duration=8.47, text="hi")
    assert v.duration == 8.47
    assert isinstance(v.duration, float)


def test_transcription_verbose_has_optional_words_segments_usage():
    v = TranscriptionResponseVerbose(
        language="en",
        duration=10.0,
        text="hello world",
        words=[TranscriptionWord(word="hello", start=0.0, end=0.5)],
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="hello world")],
        usage=TranscriptionUsageAudio(seconds=10),
    )
    assert len(v.words) == 1
    assert len(v.segments) == 1
    assert v.usage is not None and v.usage.seconds == 10


def test_transcription_verbose_has_task_field_pinned_to_transcribe():
    # OpenAI emits `"task": "transcribe"` on this response shape — wire-fidelity for
    # clients that don't tolerate unknown field absence.
    v = TranscriptionResponseVerbose(language="en", duration=10.0, text="hi")
    assert v.task == "transcribe"
    assert "task" in v.model_dump()


def test_translation_verbose_has_task_field_pinned_to_translate():
    v = TranslationResponseVerbose(language="english", duration=10.0, text="hi")
    assert v.task == "translate"
    assert "task" in v.model_dump()


def test_translation_verbose_duration_is_float():
    v = TranslationResponseVerbose(language="english", duration=4.2, text="hi")
    assert v.duration == 4.2


def test_translation_verbose_has_segments_no_words():
    v = TranslationResponseVerbose(
        language="english",
        duration=10.0,
        text="hello",
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="hello")],
    )
    assert len(v.segments) == 1
    # No `words` field on translation per OpenAI spec.
    assert not hasattr(v, "words") or "words" not in TranslationResponseVerbose.model_fields


def test_transcription_request_model_is_required():
    with pytest.raises(ValidationError):
        TranscriptionRequest()  # type: ignore[call-arg]


def test_translation_request_model_is_required():
    with pytest.raises(ValidationError):
        TranslationRequest()  # type: ignore[call-arg]


def test_embedding_request_model_is_required():
    with pytest.raises(ValidationError):
        EmbeddingCompletionRequest(input="hello")  # type: ignore[call-arg]


def test_image_generation_request_drops_diffusers_knobs():
    # num_inference_steps and guidance_scale must not be declared model fields;
    # extra="allow" still lets them through as extras, but serving_image no longer reads them.
    assert "num_inference_steps" not in ImageGenerationRequest.model_fields
    assert "guidance_scale" not in ImageGenerationRequest.model_fields

    req = ImageGenerationRequest(model="sd-turbo", prompt="a cat")
    dumped = req.model_dump()
    assert "num_inference_steps" not in dumped
    assert "guidance_scale" not in dumped


def _literal_values(annotation) -> set:
    return {value for arg in get_args(annotation) for value in get_args(arg)}


def test_chat_request_reasoning_effort_matches_openai():
    from openai.types.shared import ReasoningEffort

    ours = ChatCompletionRequest.model_fields["reasoning_effort"].annotation
    assert _literal_values(ours) == _literal_values(ReasoningEffort)


def test_chat_request_rejects_an_unknown_reasoning_effort():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], reasoning_effort="turbo")
