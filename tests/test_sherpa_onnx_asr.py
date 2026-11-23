"""sherpa_onnx offline speech to text: words and sentences from tokens, cutting long audio, the transcription
responses against a fake recognizer, config validation and the Silero pin. `test_sherpa_onnx_integration.py`
runs the real bundle."""

import asyncio
import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from modelship.deploy import config as deploy_config
from modelship.infer import base_infer
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.sherpa_onnx import asr
from modelship.infer.sherpa_onnx.asr import Decoded, Piece, SherpaOnnxAsrInfer, pieces, segments, words
from modelship.infer.sherpa_onnx.bundle import bundle_paths
from modelship.infer.sherpa_onnx.registry import REGISTRY
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.utils.vad import SpeechStopped

PARAKEET = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def _config(**overrides) -> ModelshipModelConfig:
    raw = {"name": "stt", "model": PARAKEET, "usecase": "transcription", "loader": "sherpa_onnx", **overrides}
    return ModelshipModelConfig.model_validate(raw)


def _decoded(offset: float, *tokens: tuple[str, float, float]) -> Decoded:
    text = "".join(token for token, _, _ in tokens).strip()
    return Decoded(offset, text, [t[0] for t in tokens], [t[1] for t in tokens], [t[2] for t in tokens])


SENTENCES = _decoded(
    0.0,
    (" Al", 0.0, 0.2),
    ("les", 0.2, 0.1),
    (" gut", 0.3, 0.2),
    (".", 0.5, 0.1),
    (" Ja", 1.0, 0.2),
    ("?", 1.2, 0.1),
    (" Nein", 2.0, 0.3),
)


class TestWords:
    def test_a_token_with_a_leading_space_starts_a_word_and_the_rest_extend_it(self):
        assert [(w.word, w.start, w.end) for w in words([SENTENCES])] == [
            ("Alles", 0.0, 0.3),
            ("gut.", 0.3, 0.6),
            ("Ja?", 1.0, 1.3),
            ("Nein", 2.0, 2.3),
        ]

    def test_a_later_piece_is_shifted_by_its_offset_and_starts_a_new_word(self):
        found = words([_decoded(0.0, (" one", 0.0, 0.2)), _decoded(40.0, ("two", 0.1, 0.2))])
        assert [(w.word, w.start, w.end) for w in found] == [("one", 0.0, 0.2), ("two", 40.1, 40.3)]

    def test_sentencepiece_markers_start_a_word_too(self):
        assert [w.word for w in words([_decoded(0.0, ("▁hi", 0.0, 0.1), ("▁you", 0.1, 0.1))])] == ["hi", "you"]


class TestSegments:
    def test_a_segment_closes_after_a_sentence(self):
        found = segments(words([SENTENCES]))
        assert [(s.id, s.start, s.end, s.text) for s in found] == [
            (0, 0.0, 0.6, "Alles gut."),
            (1, 1.0, 1.3, "Ja?"),
            (2, 2.0, 2.3, "Nein"),
        ]

    def test_no_words_no_segments(self):
        assert segments([]) == []


class TestPieces:
    def test_audio_up_to_the_limit_is_one_piece(self):
        samples = np.zeros(int(asr.MAX_SEGMENT_S * asr.SAMPLE_RATE), dtype=np.float32)
        (piece,) = pieces(samples, "silero.onnx")
        assert piece.offset == 0.0
        assert len(piece.samples) == len(samples)

    def test_longer_audio_is_cut_where_speech_stops(self, monkeypatch):
        cut = [
            SpeechStopped(0, 12_000, np.ones(10, dtype=np.float32)),
            SpeechStopped(31_500, 40_000, np.ones(20, dtype=np.float32)),
        ]

        class _Vad:
            def __init__(self, model, settings, max_speech_s):
                assert (model, max_speech_s) == ("silero.onnx", asr.MAX_SEGMENT_S)

            def feed(self, samples):
                return cut[:1]

            def flush(self):
                return cut[1:]

        monkeypatch.setattr(asr, "Vad", _Vad)
        found = pieces(np.zeros(asr.SAMPLE_RATE * 45, dtype=np.float32), "silero.onnx")
        assert [(p.offset, len(p.samples)) for p in found] == [(0.0, 10), (31.5, 20)]


def _wav(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * seconds), dtype=np.int16).tobytes())
    return buffer.getvalue()


class _Recognizer:
    """Decodes each piece to the next scripted result."""

    def __init__(self, *results: Decoded):
        self.results = list(results)

    def create_stream(self):
        recognizer = self

        class _Stream:
            def accept_waveform(self, rate, samples):
                assert rate == 16000

            @property
            def result(self):
                d = recognizer.results.pop(0)
                return SimpleNamespace(text=d.text, tokens=d.tokens, timestamps=d.timestamps, durations=d.durations)

        return _Stream()

    def decode_stream(self, stream):
        pass


def _infer(monkeypatch, *results: Decoded, parts: int = 1) -> SherpaOnnxAsrInfer:
    infer = SherpaOnnxAsrInfer(_config())
    infer.recognizer = _Recognizer(*results)
    monkeypatch.setattr(asr, "pieces", lambda samples, vad: [Piece(float(i) * 40, samples) for i in range(parts)])
    return infer


def _request(**fields) -> TranscriptionRequest:
    base = {
        "file": None,
        "model": "stt",
        "language": None,
        "prompt": "",
        "response_format": "json",
        "timestamp_granularities": ["segment"],
        "stream": False,
        "temperature": 0.0,
    }
    return TranscriptionRequest.model_construct(**(base | fields))


class TestTranscription:
    def test_json_joins_the_pieces(self, monkeypatch):
        infer = _infer(monkeypatch, _decoded(0, (" one", 0, 0.1)), _decoded(40, (" two", 0, 0.1)), parts=2)
        result = asyncio.run(infer.create_transcription(_wav(), _request(), None))
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "one two"
        assert result.usage.seconds == pytest.approx(1.0)

    def test_verbose_json_has_segments_and_words_only_when_asked(self, monkeypatch):
        request = _request(response_format="verbose_json")
        result = asyncio.run(_infer(monkeypatch, SENTENCES).create_transcription(_wav(), request, None))
        assert isinstance(result, TranscriptionResponseVerbose)
        assert [s.text for s in result.segments] == ["Alles gut.", "Ja?", "Nein"]
        assert result.words == []
        assert result.language == "unknown"
        request = _request(response_format="verbose_json", timestamp_granularities=["word"], language="de")
        result = asyncio.run(_infer(monkeypatch, SENTENCES).create_transcription(_wav(), request, None))
        assert isinstance(result, TranscriptionResponseVerbose)
        assert [w.word for w in result.words] == ["Alles", "gut.", "Ja?", "Nein"]
        assert result.language == "de"

    def test_stream_sends_a_delta_per_piece_then_the_text(self, monkeypatch):
        infer = _infer(monkeypatch, _decoded(0, (" one", 0, 0.1)), _decoded(40, (" two", 0, 0.1)), parts=2)

        async def run():
            stream = await infer.create_transcription(_wav(), _request(stream=True), None)
            assert not isinstance(stream, ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose)
            return [json.loads(event.removeprefix("data: ")) async for event in stream]

        assert asyncio.run(run()) == [
            {"type": "transcript.text.delta", "delta": "one"},
            {"type": "transcript.text.delta", "delta": "two"},
            {"type": "transcript.text.done", "text": "one two"},
        ]

    def test_audio_that_does_not_decode_is_an_error(self, monkeypatch):
        result = asyncio.run(_infer(monkeypatch).create_transcription(b"not audio", _request(), None))
        assert isinstance(result, ErrorResponse)
        assert "failed to decode audio" in result.error.message


class TestConfig:
    def test_parakeet_is_a_transcription_model(self):
        assert _config().usecase.value == "transcription"

    def test_parakeet_refuses_another_usecase(self):
        with pytest.raises(ValidationError, match="is a transcription model; set usecase: transcription"):
            _config(usecase="tts")

    def test_kokoro_refuses_transcription(self):
        with pytest.raises(ValidationError, match="is a tts model"):
            _config(model="kokoro-en-v0_19")

    def test_a_default_voice_is_for_tts_models_only(self):
        with pytest.raises(ValidationError, match="default_voice is for a tts model"):
            _config(sherpa_onnx_config={"default_voice": "af"})

    def test_the_bundle_needs_the_recognizer_files(self):
        assert set(bundle_paths(REGISTRY[PARAKEET])) == {
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        }


class TestSileroPin:
    def test_a_transcription_model_pins_silero_and_a_tts_model_does_not(self, monkeypatch):
        monkeypatch.setattr("modelship.infer.sherpa_onnx.bundle.bundle_source", lambda model: f"bundle:{model}")
        monkeypatch.setattr("modelship.utils.vad.silero_vad_source", lambda: "silero")
        monkeypatch.setattr(deploy_config, "_pin_label", lambda pinned: str(pinned))
        stt = _config()
        tts = _config(name="tts", model="kokoro-en-v0_19", usecase="tts")
        deploy_config.resolve_all_model_sources(ModelshipConfig(models=[stt, tts]))
        assert (stt._pinned_source, stt._pinned_vad) == (f"bundle:{PARAKEET}", "silero")
        assert tts._pinned_vad is None

    def test_the_replica_downloads_it(self, monkeypatch):
        fetched = []

        async def download(source, name):
            fetched.append(source)
            return f"/cache/{source}"

        monkeypatch.setattr(base_infer, "locked_download", download)
        config = _config()
        config._pinned_vad = "silero"  # type: ignore[assignment]
        asyncio.run(base_infer.BaseInfer.ensure_downloaded(config))
        assert (fetched, config._resolved_vad) == (["silero"], "/cache/silero")
