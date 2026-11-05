"""Unit tests for the sherpa_onnx loader: voice->sid resolution, provider
derivation, config-builder setattr routing, num_speakers mismatch, frontend
wiring, and streaming — all against a stubbed `sherpa_onnx` module and a stubbed
frontend. Nothing here downloads a bundle or spawns a phonemizer."""

import asyncio
import io
import sys
import threading
import time
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.infer.sherpa_onnx import sherpa_onnx_infer
from modelship.infer.sherpa_onnx.registry import REGISTRY, SherpaOnnxRegistryEntry
from modelship.infer.sherpa_onnx.sherpa_onnx_infer import SherpaOnnxInfer
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechRequest

_ENTRY: SherpaOnnxRegistryEntry = REGISTRY["kokoro-en-v0_19"]


class _FakeKokoroConfig:
    def __init__(self):
        self.model = ""
        self.voices = ""
        self.tokens = ""
        self.data_dir = ""
        self.lexicon = ""


class _FakeModelConfig:
    def __init__(self):
        self.kokoro = _FakeKokoroConfig()
        self.provider = ""
        self.num_threads = 0


class _FakeTtsConfig:
    def __init__(self):
        self.model = _FakeModelConfig()


class _FakeGenerationConfig:
    def __init__(self):
        self.sid = 0
        self.speed = 1.0
        self.phoneme_codepoints: list[list[int]] = []


class _FakeOfflineTts:
    """`chunks`, if given, is emitted one per `callback=` call, gated on the
    previous call's return value (0 aborts)."""

    def __init__(self, cfg, num_speakers, chunks: list[np.ndarray] | None = None):
        self.cfg = cfg
        self.num_speakers = num_speakers
        self.sample_rate = 24000
        self.generate_calls: list[tuple] = []
        self.chunks = chunks or [np.array([0.0, 0.1, -0.1], dtype=np.float32)]
        self.aborted_early = False
        self.started = threading.Event()

    def generate(self, text, gen_config=None, callback=None):
        self.generate_calls.append((text, gen_config.sid, gen_config.speed, gen_config.phoneme_codepoints))
        self.started.set()
        generated = []
        for i, chunk in enumerate(self.chunks):
            generated.append(chunk)
            if callback is not None and not callback(chunk, (i + 1) / len(self.chunks)):
                self.aborted_early = True
                break
            time.sleep(0.05)  # gives the consumer a window to aclose() mid-loop
        return SimpleNamespace(samples=np.concatenate(generated).tolist(), sample_rate=self.sample_rate)


class _FakeFrontend:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def chunks(self, text: str, voice: str) -> list[list[int]]:
        self.calls.append((text, voice))
        return [[ord(c) for c in text]]

    def close(self):
        self.closed = True


def _fake_sherpa_onnx_module(num_speakers: int) -> types.ModuleType:
    mod = types.ModuleType("sherpa_onnx")
    mod.OfflineTtsConfig = _FakeTtsConfig
    mod.GenerationConfig = _FakeGenerationConfig
    mod.OfflineTts = lambda cfg: _FakeOfflineTts(cfg, num_speakers)
    return mod


@pytest.fixture(autouse=True)
def _stub_sherpa_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(_ENTRY.voice_names)))


def _config(**overrides) -> ModelshipModelConfig:
    defaults = {
        "name": "tts",
        "model": "kokoro-en-v0_19",
        "usecase": ModelUsecase.tts,
        "loader": ModelLoader.sherpa_onnx,
        "num_cpus": 2,
    }
    return ModelshipModelConfig(**{**defaults, **overrides})


def _infer(entry: SherpaOnnxRegistryEntry = _ENTRY, tts: _FakeOfflineTts | None = None) -> SherpaOnnxInfer:
    infer = SherpaOnnxInfer(_config())
    infer.entry = entry
    infer.tts = tts or _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(entry.voice_names))
    infer.frontend = _FakeFrontend()
    return infer


class TestResolveSid:
    def test_known_voice_name(self):
        infer = _infer()
        assert infer._resolve_sid("af_bella") == _ENTRY.voice_names.index("af_bella")

    def test_numeric_sid_in_range(self):
        infer = _infer()
        assert infer._resolve_sid("3") == 3

    def test_numeric_sid_out_of_range(self):
        infer = _infer()
        assert infer._resolve_sid(str(len(_ENTRY.voice_names))) is None

    def test_unknown_name(self):
        infer = _infer()
        assert infer._resolve_sid("not_a_voice") is None


def _resolved(config: ModelshipModelConfig, path: str = "/bundle") -> ModelshipModelConfig:
    config._resolved_path = path
    return config


def _stub_frontend(monkeypatch) -> list[tuple]:
    built: list[tuple] = []
    monkeypatch.setattr(sherpa_onnx_infer.TokenTable, "load", staticmethod(lambda path: f"tokens@{path}"))
    monkeypatch.setattr(
        sherpa_onnx_infer,
        "KokoroFrontend",
        lambda tokens, chinese_lexicon: built.append((tokens, chinese_lexicon)) or _FakeFrontend(),
    )
    return built


class TestLoad:
    def test_provider_threads_paths_and_no_lexicon_to_sherpa(self, monkeypatch):
        _stub_frontend(monkeypatch)
        infer = SherpaOnnxInfer(_resolved(_config(num_cpus=3)))
        tts, entry, _frontend = infer._load()
        assert entry is _ENTRY
        assert tts.cfg.model.provider == "cpu"
        assert tts.cfg.model.num_threads == 3
        assert tts.cfg.model.kokoro.model == "/bundle/model.onnx"
        assert tts.cfg.model.kokoro.voices == "/bundle/voices.bin"
        assert tts.cfg.model.kokoro.tokens == "/bundle/tokens.txt"
        assert tts.cfg.model.kokoro.data_dir == ""
        assert tts.cfg.model.kokoro.lexicon == ""

    def test_english_bundle_frontend_has_no_lexicon(self, monkeypatch):
        built = _stub_frontend(monkeypatch)
        SherpaOnnxInfer(_resolved(_config()))._load()
        assert built == [("tokens@/bundle/tokens.txt", None)]

    def test_multilang_frontend_gets_the_chinese_lexicon(self, monkeypatch):
        built = _stub_frontend(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(REGISTRY["kokoro-multi-lang-v1_0"].voice_names))
        )
        SherpaOnnxInfer(_resolved(_config(model="kokoro-multi-lang-v1_0")))._load()
        assert built == [("tokens@/bundle/tokens.txt", "/bundle/lexicon-zh.txt")]

    def test_local_dir_uses_the_entry_its_basename_names(self, monkeypatch, tmp_path):
        _stub_frontend(monkeypatch)
        bundle = str(tmp_path / "kokoro-en-v0_19")
        infer = SherpaOnnxInfer(_resolved(_config(model=f"{bundle}/"), bundle))
        tts, entry, _frontend = infer._load()
        assert entry is _ENTRY
        assert tts.cfg.model.kokoro.model == f"{bundle}/model.onnx"

    def test_num_speakers_mismatch_raises(self, monkeypatch):
        _stub_frontend(monkeypatch)
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(num_speakers=999))
        infer = SherpaOnnxInfer(_resolved(_config()))
        with pytest.raises(ValueError, match="999"):
            infer._load()

    def test_rejects_a_config_without_a_resolved_path(self, monkeypatch):
        _stub_frontend(monkeypatch)
        infer = SherpaOnnxInfer(_config())
        with pytest.raises(ValueError, match="no resolved bundle directory"):
            infer._load()


class TestShutdown:
    def test_shutdown_closes_the_frontend(self):
        infer = _infer()
        frontend = infer.frontend
        infer.shutdown()
        assert frontend.closed
        assert infer.frontend is None


class TestCreateSpeech:
    @pytest.mark.asyncio
    async def test_unknown_voice_returns_400(self):
        infer = _infer()
        req = SpeechRequest(input="hi", model="tts", voice="not_a_voice")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)
        assert "not_a_voice" in result.error.message

    @pytest.mark.asyncio
    async def test_generates_wav_and_passes_codepoints_sid_speed(self):
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names))
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hello world", model="tts", voice="af_bella", speed=1.5)
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, RawSpeechResponse)
        with wave.open(io.BytesIO(result.audio)) as wav:
            assert wav.getnframes() == 3
        assert infer.frontend.calls == [("hello world", "af_bella")]
        text, sid, speed, codepoints = tts.generate_calls[0]
        assert text == ""
        assert sid == _ENTRY.voice_names.index("af_bella")
        assert speed == 1.5
        assert codepoints == [[ord(c) for c in "hello world"]]

    @pytest.mark.asyncio
    async def test_numeric_voice_resolves_to_sid(self):
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names))
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi", model="tts", voice="2")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, RawSpeechResponse)
        assert tts.generate_calls[0][1] == 2
        assert infer.frontend.calls == [("hi", _ENTRY.voice_names[2])]

    @pytest.mark.asyncio
    async def test_unknown_voice_returns_400_even_when_streaming(self):
        infer = _infer()
        req = SpeechRequest(input="hi", model="tts", voice="not_a_voice", stream_format="sse")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_yields_delta_per_chunk_then_done(self):
        chunks = [np.array([0.0, 0.1], dtype=np.float32), np.array([0.2, -0.2], dtype=np.float32)]
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names), chunks=chunks)
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi there", model="tts", voice="af_bella", stream_format="sse")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        events = [e async for e in result]
        assert len(events) == 3  # 2 deltas + 1 done
        assert '"type":"speech.audio.delta"' in events[0]
        assert '"type":"speech.audio.delta"' in events[1]
        assert '"type":"speech.audio.done"' in events[2]

    @pytest.mark.asyncio
    async def test_early_aclose_stops_generation_before_all_chunks_sent(self):
        chunks = [np.zeros(2, dtype=np.float32) for _ in range(5)]
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names), chunks=chunks)
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi", model="tts", voice="af_bella", stream_format="sse")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        await result.__anext__()  # first delta
        await result.aclose()
        await _until(lambda: tts.aborted_early)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_a_cancelled_speech_stops_its_thread(self):
        chunks = [np.zeros(2, dtype=np.float32) for _ in range(5)]
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names), chunks=chunks)
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi", model="tts", voice="af_bella")
        task = asyncio.ensure_future(infer.create_speech(req, RawRequestProxy(None, {})))
        await asyncio.to_thread(tts.started.wait, 5)
        task.cancel()
        await _until(lambda: tts.aborted_early)

    @pytest.mark.asyncio
    async def test_speeches_overlap(self):
        both = threading.Barrier(2, timeout=5)

        def generate(*_args, **_kwargs):
            both.wait()
            return SimpleNamespace(samples=[], sample_rate=24000)

        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names))
        tts.generate = generate
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi", model="tts", voice="af_bella")
        await asyncio.gather(*(infer.create_speech(req, RawRequestProxy(None, {})) for _ in range(2)))


async def _until(condition) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never held")
