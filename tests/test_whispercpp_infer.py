"""Unit tests for the whispercpp loader: request-param forwarding, real
detected-language plumbing, ggml model-file validation, and streaming — all
against a stubbed `pywhispercpp.model.Model` (no real ggml file needed)."""

from __future__ import annotations

import asyncio
import io
import threading
import time

import pytest
from fastapi import UploadFile

from modelship.infer.base_infer import MINIMAL_WAV
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.whispercpp.whispercpp_infer import WhispercppInfer, _validate_ggml_model_file
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from tests.test_base_infer_run_blocking import wait_for_turn


class _Segment:
    def __init__(self, text: str, t0: int = 0, t1: int = 100):
        self.text = text
        self.t0 = t0
        self.t1 = t1


class _StubModel:
    """Stand-in for a `pywhispercpp.model.Model`. Records decode kwargs per
    `transcribe` call; polls `abort_callback` between segments like real whisper.cpp."""

    def __init__(self, segments: list[_Segment], detected: tuple[str, float] = ("es", 0.9)):
        self._segments = segments
        self._detected = detected
        self.calls: list[dict] = []
        self.aborted_early = False
        self.started = threading.Event()

    def transcribe(self, samples, new_segment_callback=None, abort_callback=None, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        emitted = []
        for seg in self._segments:
            if abort_callback is not None and abort_callback():
                self.aborted_early = True
                break
            if new_segment_callback is not None:
                new_segment_callback(seg)
            emitted.append(seg)
            time.sleep(0.05)  # gives the consumer a window to aclose() mid-loop
        return emitted

    def auto_detect_language(self, samples):
        return self._detected, {self._detected[0]: self._detected[1]}


def _config(**overrides) -> ModelshipModelConfig:
    defaults = {
        "name": "stt",
        "model": "tiny.en",
        "usecase": ModelUsecase.transcription,
        "loader": ModelLoader.whispercpp,
    }
    return ModelshipModelConfig(**{**defaults, **overrides})


def _infer(model: _StubModel) -> WhispercppInfer:
    infer = WhispercppInfer(_config())
    infer.model = model
    return infer


def _transcription_request(**overrides) -> TranscriptionRequest:
    file = UploadFile(file=io.BytesIO(b"x"), filename="a.wav")
    return TranscriptionRequest(**{"file": file, "model": "stt", **overrides})


def _translation_request(**overrides) -> TranslationRequest:
    file = UploadFile(file=io.BytesIO(b"x"), filename="a.wav")
    return TranslationRequest(**{"file": file, "model": "stt", **overrides})


class TestDecodeKwargs:
    """prompt/temperature must reach whisper.cpp's transcribe() call."""

    def test_prompt_and_temperature_forwarded(self):
        infer = _infer(_StubModel([]))
        kwargs = infer._decode_kwargs(_transcription_request(prompt="hello", temperature=0.6), translate=False)
        assert kwargs["initial_prompt"] == "hello"
        assert kwargs["temperature"] == 0.6

    def test_translate_forwards_language_as_source_hint(self):
        infer = _infer(_StubModel([]))
        kwargs = infer._decode_kwargs(_translation_request(language="fr"), translate=True)
        assert kwargs["language"] == "fr"
        assert kwargs["translate"] is True

    def test_to_language_is_ignored(self):
        infer = _infer(_StubModel([]))
        kwargs = infer._decode_kwargs(_translation_request(to_language="de"), translate=True)
        assert "to_language" not in kwargs
        assert "language" not in kwargs


class TestTranscribeOnce:
    @pytest.mark.asyncio
    async def test_plain_json_has_no_language(self):
        infer = _infer(_StubModel([_Segment("hello "), _Segment("world")]))
        result = await infer._transcribe_once(MINIMAL_WAV, _transcription_request(), translate=False, request_id="r1")
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_verbose_json_uses_explicit_language_without_detection(self):
        model = _StubModel([_Segment("hola")], detected=("en", 0.99))
        infer = _infer(model)
        result = await infer._transcribe_once(
            MINIMAL_WAV,
            _transcription_request(response_format="verbose_json", language="es"),
            translate=False,
            request_id="r1",
        )
        assert isinstance(result, TranscriptionResponseVerbose)
        assert result.language == "es"  # user-given language wins, no detection call needed

    @pytest.mark.asyncio
    async def test_verbose_json_detects_real_language_when_unset(self):
        """No language given -> the real detected language, not a fabricated value."""
        model = _StubModel([_Segment("hola")], detected=("es", 0.87))
        infer = _infer(model)
        result = await infer._transcribe_once(
            MINIMAL_WAV, _transcription_request(response_format="verbose_json"), translate=False, request_id="r1"
        )
        assert isinstance(result, TranscriptionResponseVerbose)
        assert result.language == "es"

    @pytest.mark.asyncio
    async def test_english_only_model_skips_detection(self):
        # A real ggml-tiny.en detects as 'sq' here.
        model = _StubModel([_Segment("hello")], detected=("sq", 0.4))
        infer = _infer(model)
        infer._multilingual = False
        result = await infer._transcribe_once(
            MINIMAL_WAV, _transcription_request(response_format="verbose_json"), translate=False, request_id="r1"
        )
        assert isinstance(result, TranscriptionResponseVerbose)
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_translation_verbose_language_is_always_english(self):
        infer = _infer(_StubModel([_Segment("hello")]))
        result = await infer._transcribe_once(
            MINIMAL_WAV,
            _translation_request(response_format="verbose_json", language="fr"),
            translate=True,
            request_id="r1",
        )
        assert isinstance(result, TranslationResponseVerbose)
        assert result.language == "english"

    @pytest.mark.asyncio
    async def test_translation_plain_json(self):
        infer = _infer(_StubModel([_Segment("hello")]))
        result = await infer._transcribe_once(MINIMAL_WAV, _translation_request(), translate=True, request_id="r1")
        assert isinstance(result, TranslationResponse)
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_bad_audio_returns_error_response(self):
        infer = _infer(_StubModel([]))
        result = await infer._transcribe_once(b"not audio", _transcription_request(), translate=False, request_id="r1")
        assert isinstance(result, ErrorResponse)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_yields_delta_per_segment_then_done(self):
        infer = _infer(_StubModel([_Segment("hello"), _Segment("world")]))
        chunks = [
            c async for c in infer._stream(MINIMAL_WAV, _transcription_request(), translate=False, request_id="r1")
        ]
        assert '"type": "transcript.text.delta"' in chunks[0]
        assert '"delta": "hello"' in chunks[0]
        assert chunks[-1].strip().endswith("}")
        assert '"type": "transcript.text.done"' in chunks[-1]
        assert '"text": "hello world"' in chunks[-1]

    @pytest.mark.asyncio
    async def test_early_aclose_sets_abort_before_worker_finishes(self):
        model = _StubModel([_Segment("a"), _Segment("b"), _Segment("c")])
        infer = _infer(model)
        gen = infer._stream(MINIMAL_WAV, _transcription_request(), translate=False, request_id="r1")
        await gen.__anext__()  # first delta
        await gen.aclose()
        await wait_for_turn(infer)
        assert model.aborted_early


class TestCancellation:
    @pytest.mark.asyncio
    async def test_a_cancelled_transcription_stops_its_thread(self):
        model = _StubModel([_Segment("a"), _Segment("b"), _Segment("c")])
        infer = _infer(model)
        task = asyncio.ensure_future(
            infer._transcribe_once(MINIMAL_WAV, _transcription_request(), translate=False, request_id="r1")
        )
        await asyncio.to_thread(model.started.wait, 5)
        task.cancel()
        await wait_for_turn(infer)
        assert model.aborted_early


class TestValidateGgmlModelFile:
    def test_accepts_ggml_magic(self, tmp_path):
        path = tmp_path / "ggml-base.en.bin"
        # Byte order matches a real whisper.cpp model file, not the "ggml" spelling.
        path.write_bytes((0x67676D6C).to_bytes(4, "little") + b"\x00" * 16)
        _validate_ggml_model_file(str(path), "m")  # no raise

    def test_rejects_gguf(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(b"GGUF" + b"\x00" * 16)
        with pytest.raises(ValueError, match="GGUF"):
            _validate_ggml_model_file(str(path), "m")

    def test_rejects_safetensors(self, tmp_path):
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="safetensors"):
            _validate_ggml_model_file(str(path), "m")

    def test_rejects_unknown_format(self, tmp_path):
        path = tmp_path / "weights.bin"
        path.write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="ggml"):
            _validate_ggml_model_file(str(path), "m")
