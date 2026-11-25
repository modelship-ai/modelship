"""sherpa_onnx streaming speech to text against a fake recognizer: the decode thread's streams (opening, the zeros
around the audio, batching, closing, idle streams, failures), the pushes, `/v1/audio/transcriptions`, the registry
entry and the Silero pin. `test_sherpa_onnx_integration.py` runs the real bundle."""

import io
import json
import threading
import time
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from modelship.deploy import config as deploy_config
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.sherpa_onnx import stream_decoder, streaming_stt_infer
from modelship.infer.sherpa_onnx.bundle import bundle_paths
from modelship.infer.sherpa_onnx.family import nemo_streaming_transducer
from modelship.infer.sherpa_onnx.registry import REGISTRY, StreamFeed, is_streaming
from modelship.infer.sherpa_onnx.stream_decoder import StreamDecoder, quiet_tail_ms
from modelship.infer.sherpa_onnx.streaming_stt_infer import SherpaOnnxStreamingSttInfer
from modelship.infer.sherpa_onnx.stt_infer import SherpaOnnxSttInfer
from modelship.openai.protocol import (
    ErrorResponse,
    SttStreamOptions,
    SttStreamText,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)

NEMOTRON = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11"
RATE = 16000
CHUNK = RATE // 10
LOUD = 0.5


class _Stream:
    def __init__(self):
        self.options: dict[str, str] = {}
        self.audio = np.zeros(0, dtype=np.float32)
        self.rates: set[int] = set()
        self.decoded = 0
        self.finished = False
        self.words: list[tuple[str, float]] = []

    def set_option(self, key: str, value: str) -> None:
        self.options[key] = value

    def accept_waveform(self, rate: int, samples: np.ndarray) -> None:
        self.rates.add(rate)
        self.audio = np.concatenate([self.audio, samples])

    def input_finished(self) -> None:
        self.finished = True


class _Recognizer:
    """Decodes 100 ms per round: a chunk with loud audio adds a word, timed at the chunk's start. With input
    finished, the last partial chunk decodes too."""

    def __init__(self):
        self.streams: list[_Stream] = []
        self.batches: list[int] = []
        self.fail = False
        self.hold: threading.Event | None = None

    def create_stream(self) -> _Stream:
        stream = _Stream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream: _Stream) -> bool:
        left = len(stream.audio) - stream.decoded
        return left >= CHUNK or (stream.finished and left > 0)

    def decode_streams(self, streams: list[_Stream]) -> None:
        if self.hold is not None:
            self.hold.wait(5)
            self.hold = None
        if self.fail:
            self.fail = False
            raise RuntimeError("decoder blew up")
        self.batches.append(len(streams))
        for stream in streams:
            chunk = stream.audio[stream.decoded : stream.decoded + CHUNK]
            if np.abs(chunk).max() >= LOUD / 2:
                stream.words.append((f"w{len(stream.words)}", stream.decoded / RATE))
            stream.decoded += len(chunk)

    def get_result_all(self, stream: _Stream) -> SimpleNamespace:
        return SimpleNamespace(
            text=" ".join(w for w, _ in stream.words),
            tokens=[f" {w}" for w, _ in stream.words],
            timestamps=[t for _, t in stream.words],
        )


def _audio(ms: int, level: float = LOUD) -> np.ndarray:
    return np.full(RATE * ms // 1000, level, dtype=np.float32)


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).astype(np.int16).tobytes()


NO_ZEROS = StreamFeed(lead_ms=0, tail_ms=0)


def _decoder(feed: StreamFeed = NO_ZEROS) -> tuple[StreamDecoder, _Recognizer]:
    recognizer = _Recognizer()
    return StreamDecoder(recognizer, feed, "test"), recognizer


def _wait(decoder: StreamDecoder, key: str, samples: np.ndarray, finish: bool = False, language: str | None = None):
    return decoder.submit(key, samples, RATE, language, finish).result(timeout=5)


class TestQuietTail:
    def test_the_quiet_frames_after_the_last_loud_one(self):
        assert quiet_tail_ms(np.concatenate([_audio(100), _audio(250, 0.0)]), RATE) == 250.0

    def test_audio_ending_loud_has_none(self):
        assert quiet_tail_ms(_audio(100), RATE) == 0.0

    def test_all_quiet_or_too_short_is_none(self):
        assert quiet_tail_ms(_audio(100, 0.001), RATE) is None
        assert quiet_tail_ms(np.zeros(10, dtype=np.float32), RATE) is None


class TestStreamDecoder:
    def test_a_push_returns_the_text_so_far_and_the_audio_the_stream_took(self):
        decoder, _ = _decoder()
        first = _wait(decoder, "s1:t1", _audio(200))
        second = _wait(decoder, "s1:t1", _audio(100))
        assert (first.text, first.audio_ms) == ("w0 w1", 200.0)
        assert (second.text, second.audio_ms) == ("w0 w1 w2", 300.0)
        assert decoder.streams == 1

    def test_a_stream_opens_with_the_lead_zeros_and_its_language(self):
        decoder, recognizer = _decoder(StreamFeed(lead_ms=300, tail_ms=0))
        result = _wait(decoder, "s1:t1", _audio(100), language="de")
        [stream] = recognizer.streams
        assert stream.options == {"language": "de"}
        assert np.abs(stream.audio[: RATE * 300 // 1000]).max() == 0.0
        assert (result.text, result.audio_ms, result.language) == ("w0", 100.0, "de")
        assert result.timestamps == [0.0]

    def test_auto_sets_no_language(self):
        decoder, recognizer = _decoder()
        _wait(decoder, "s1:t1", _audio(100))
        assert recognizer.streams[0].options == {}

    def test_finish_pads_the_tail_ends_the_input_and_drops_the_stream(self):
        decoder, recognizer = _decoder(StreamFeed(lead_ms=0, tail_ms=320))
        result = _wait(decoder, "s1:t1", _audio(150), finish=True)
        [stream] = recognizer.streams
        assert stream.finished
        assert len(stream.audio) == RATE * (150 + 320) // 1000
        assert (result.text, result.audio_ms) == ("w0 w1", 150.0)
        assert decoder.streams == 0

    @pytest.mark.parametrize(("quiet_ms", "padded_ms"), [(100, 220), (320, 0), (600, 0)])
    def test_quiet_audio_at_the_end_counts_toward_the_tail(self, quiet_ms, padded_ms):
        decoder, recognizer = _decoder(StreamFeed(lead_ms=0, tail_ms=320))
        _wait(decoder, "s1:t1", _audio(200))
        _wait(decoder, "s1:t1", _audio(quiet_ms, 0.0), finish=True)
        assert len(recognizer.streams[0].audio) == RATE * (200 + quiet_ms + padded_ms) // 1000

    def test_a_push_that_is_all_quiet_adds_to_the_quiet_the_stream_ends_in(self):
        decoder, recognizer = _decoder(StreamFeed(lead_ms=0, tail_ms=320))
        _wait(decoder, "s1:t1", _audio(200))
        _wait(decoder, "s1:t1", _audio(200, 0.0))
        _wait(decoder, "s1:t1", _audio(100, 0.0), finish=True)
        assert len(recognizer.streams[0].audio) == RATE * (200 + 200 + 100 + 20) // 1000

    def test_close_drops_the_stream_and_the_next_push_opens_another(self):
        decoder, recognizer = _decoder()
        _wait(decoder, "s1:t1", _audio(200))
        decoder.close("s1:t1")
        again = _wait(decoder, "s1:t1", _audio(100))
        assert len(recognizer.streams) == 2
        assert (again.text, again.audio_ms) == ("w0", 100.0)

    def test_streams_with_jobs_waiting_decode_together(self):
        decoder, recognizer = _decoder()
        recognizer.hold = threading.Event()
        first = decoder.submit("s1:t1", _audio(500), RATE, None, False)
        time.sleep(0.05)
        second = decoder.submit("s2:t1", _audio(500), RATE, None, False)
        recognizer.hold.set()
        assert first.result(timeout=5).text == second.result(timeout=5).text == "w0 w1 w2 w3 w4"
        assert 2 in recognizer.batches

    def test_another_rate_on_an_open_stream_fails_and_drops_it(self):
        decoder, _ = _decoder()
        _wait(decoder, "s1:t1", _audio(100))
        with pytest.raises(ValueError, match="takes 16000 Hz audio, not 24000 Hz"):
            decoder.submit("s1:t1", _audio(100), 24000, None, False).result(timeout=5)
        assert decoder.streams == 0

    def test_a_failed_decode_fails_its_jobs_drops_their_streams_and_the_next_works(self):
        decoder, recognizer = _decoder()
        recognizer.fail = True
        with pytest.raises(RuntimeError, match="decoder blew up"):
            _wait(decoder, "s1:t1", _audio(100))
        assert decoder.streams == 0
        assert _wait(decoder, "s2:t1", _audio(100)).text == "w0"

    def test_a_stream_no_job_reached_for_idle_s_is_dropped(self, monkeypatch):
        monkeypatch.setattr(stream_decoder, "IDLE_S", 0.05)
        monkeypatch.setattr(stream_decoder, "SWEEP_S", 0.02)
        decoder, _ = _decoder()
        _wait(decoder, "s1:t1", _audio(100))
        deadline = time.monotonic() + 5
        while decoder.streams and time.monotonic() < deadline:
            time.sleep(0.01)
        assert decoder.streams == 0

    def test_a_close_fails_the_push_waiting_on_it_and_a_later_push_gets_its_own_stream(self):
        decoder, recognizer = _decoder()
        recognizer.hold = threading.Event()
        decoder.submit("s0:t1", _audio(100), RATE, None, False)
        time.sleep(0.05)
        waiting = decoder.submit("s1:t1", _audio(300), RATE, None, False)
        decoder.close("s1:t1")
        later = decoder.submit("s1:t1", _audio(100), RATE, None, False)
        recognizer.hold.set()
        with pytest.raises(KeyError, match="was closed"):
            waiting.result(timeout=5)
        assert (later.result(timeout=5).text, later.result().audio_ms) == ("w0", 100.0)

    def test_stopping_fails_the_jobs_queued_with_the_stop(self):
        decoder, recognizer = _decoder()
        recognizer.hold = threading.Event()
        decoder.submit("s0:t1", _audio(100), RATE, None, False)
        time.sleep(0.05)
        queued = decoder.submit("s1:t1", _audio(100), RATE, None, False)
        decoder.stop()
        recognizer.hold.set()
        with pytest.raises(RuntimeError, match="shut down"):
            queued.result(timeout=5)

    def test_stopping_fails_the_jobs_waiting(self):
        decoder, recognizer = _decoder()
        recognizer.hold = threading.Event()
        waiting = decoder.submit("s1:t1", _audio(500), RATE, None, False)
        time.sleep(0.05)
        decoder.stop()
        recognizer.hold.set()
        with pytest.raises(RuntimeError, match="shut down"):
            waiting.result(timeout=5)


def _config(**overrides) -> ModelshipModelConfig:
    raw = {"name": "stt", "model": NEMOTRON, "usecase": "transcription", "loader": "sherpa_onnx", **overrides}
    return ModelshipModelConfig.model_validate(raw)


def _infer(feed: StreamFeed = NO_ZEROS) -> tuple[SherpaOnnxStreamingSttInfer, _Recognizer]:
    infer = SherpaOnnxStreamingSttInfer(_config())
    recognizer = _Recognizer()
    infer.decoder = StreamDecoder(recognizer, feed, "test")
    return infer, recognizer


def _wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(_pcm(samples))
    return buffer.getvalue()


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


class TestStreamedTranscription:
    @pytest.mark.asyncio
    async def test_pushes_take_pcm16_and_finish_ends_the_turn(self):
        infer, recognizer = _infer()
        options = SttStreamOptions(language="fr")
        pushed = await infer.transcribe_push("s1:t1", _pcm(_audio(200)), RATE, options)
        done = await infer.transcribe_finish("s1:t1", _pcm(_audio(100)), RATE, options)
        assert pushed == SttStreamText(text="w0 w1", language="fr", audio_ms=200.0)
        assert done == SttStreamText(text="w0 w1 w2", language="fr", audio_ms=300.0)
        assert recognizer.streams[0].finished
        assert infer.decoder is not None and infer.decoder.streams == 0

    @pytest.mark.asyncio
    async def test_a_failed_push_is_an_error_response(self):
        infer, recognizer = _infer()
        recognizer.fail = True
        result = await infer.transcribe_push("s1:t1", _pcm(_audio(100)), RATE, SttStreamOptions())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 500

    @pytest.mark.asyncio
    async def test_close_drops_the_turn(self):
        infer, _ = _infer()
        await infer.transcribe_push("s1:t1", _pcm(_audio(100)), RATE, SttStreamOptions())
        await infer.transcribe_close("s1:t1")
        await infer.transcribe_push("s2:t1", _pcm(_audio(100)), RATE, SttStreamOptions())
        assert infer.decoder is not None and infer.decoder.streams == 1

    @pytest.mark.asyncio
    async def test_it_streams_and_reports_no_language(self):
        infer, _ = _infer()
        found = await infer.stt_capabilities()
        assert (found.streaming, found.reports_language) == (True, False)


class TestFileTranscription:
    @pytest.mark.asyncio
    async def test_json_is_the_whole_files_text(self):
        infer, recognizer = _infer(StreamFeed(lead_ms=0, tail_ms=320))
        result = await infer.create_transcription(_wav(_audio(300)), _request(language="de"), None)
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "w0 w1 w2"
        assert result.usage.seconds == pytest.approx(0.3)
        assert recognizer.streams[0].options == {"language": "de"}
        assert infer.decoder is not None and infer.decoder.streams == 0

    @pytest.mark.asyncio
    async def test_verbose_json_words_are_timed_from_the_start_of_the_file(self):
        infer, _ = _infer(StreamFeed(lead_ms=500, tail_ms=0))
        request = _request(response_format="verbose_json", timestamp_granularities=["word"])
        audio = np.concatenate([_audio(200, 0.0), _audio(200)])
        result = await infer.create_transcription(_wav(audio), request, None)
        assert isinstance(result, TranscriptionResponseVerbose)
        assert [(w.word, w.start) for w in result.words] == [("w0", 0.2), ("w1", 0.3)]
        assert result.language == "unknown"

    @pytest.mark.asyncio
    async def test_stream_sends_a_delta_per_second_whose_text_grew_then_the_text(self, monkeypatch):
        monkeypatch.setattr(streaming_stt_infer, "FILE_STEP_S", 0.2)
        infer, _ = _infer()
        audio = np.concatenate([_audio(200), _audio(200, 0.0), _audio(100)])
        stream = await infer.create_transcription(_wav(audio), _request(stream=True), None)
        assert not isinstance(stream, ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose)
        events = [json.loads(event.removeprefix("data: ")) async for event in stream]
        assert events == [
            {"type": "transcript.text.delta", "delta": "w0 w1"},
            {"type": "transcript.text.delta", "delta": " w2"},
            {"type": "transcript.text.done", "text": "w0 w1 w2"},
        ]

    @pytest.mark.asyncio
    async def test_audio_that_does_not_decode_is_an_error(self):
        infer, _ = _infer()
        result = await infer.create_transcription(b"not audio", _request(), None)
        assert isinstance(result, ErrorResponse)
        assert "failed to decode audio" in result.error.message


class TestWords:
    def test_a_word_ends_where_the_next_token_starts(self):
        found = nemo_streaming_transducer.words([" Al", "les", " gut", "."], [0.1, 0.3, 0.5, 0.8], 1.0)
        assert [(w.word, w.start, w.end) for w in found] == [("Alles", 1.1, 1.5), ("gut.", 1.5, 1.88)]

    def test_a_last_word_of_one_token_lasts_a_frame(self):
        [word] = nemo_streaming_transducer.words([" Ja"], [0.4], 0.0)
        assert (word.start, word.end) == (0.4, 0.48)


class TestRegistry:
    def test_nemotron_streams_and_the_others_do_not(self):
        assert is_streaming(NEMOTRON)
        assert is_streaming(f"/models/{NEMOTRON}")
        assert not is_streaming("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
        assert not is_streaming("kokoro-en-v0_19")
        assert not is_streaming("not-a-model")

    def test_nemotron_is_a_transcription_model_with_its_recognizer_files(self):
        assert _config().usecase.value == "transcription"
        assert set(bundle_paths(REGISTRY[NEMOTRON])) == {
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        }

    def test_a_streaming_model_pins_no_silero(self, monkeypatch):
        monkeypatch.setattr("modelship.infer.sherpa_onnx.bundle.bundle_source", lambda model: f"bundle:{model}")
        monkeypatch.setattr("modelship.utils.vad.silero_vad_source", lambda: "silero")
        monkeypatch.setattr(deploy_config, "_pin_label", lambda pinned: str(pinned))
        config = _config()
        deploy_config.resolve_all_model_sources(ModelshipConfig(models=[config]))
        assert (config._pinned_source, config._pinned_vad) == (f"bundle:{NEMOTRON}", None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "loader"),
        [(NEMOTRON, SherpaOnnxStreamingSttInfer), ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", SherpaOnnxSttInfer)],
    )
    async def test_a_replica_runs_a_streaming_model_on_the_streaming_loader(self, monkeypatch, model, loader):
        from modelship.infer.model_deployment import ModelDeployment

        monkeypatch.setenv("MSHIP_VARIANT", "cpu")
        deployment = ModelDeployment.func_or_class
        base = MagicMock(ensure_downloaded=AsyncMock())
        monkeypatch.setattr(loader, "start", AsyncMock())
        monkeypatch.setattr(loader, "warmup", AsyncMock())
        replica = deployment.__new__(deployment)
        globals_ = {"BaseInfer": base, "wait_for_deploy_lease": AsyncMock(), "stamp_gateway": MagicMock()}
        with patch.dict(deployment.__init__.__globals__, globals_ | {"reject_unset_cache_roots": MagicMock()}):
            await deployment.__init__(replica, _config(model=model))
        assert type(replica.infer) is loader


class TestSherpaApi:
    def test_the_online_recognizer_and_stream_keep_what_the_decoder_calls(self):
        import sherpa_onnx

        for name in ("from_transducer", "create_stream", "is_ready", "decode_streams", "get_result_all"):
            assert callable(getattr(sherpa_onnx.OnlineRecognizer, name))
        for name in ("accept_waveform", "input_finished", "set_option"):
            assert callable(getattr(sherpa_onnx.OnlineStream, name))
