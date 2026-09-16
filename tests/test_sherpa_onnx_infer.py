"""Unit tests for the sherpa_onnx loader: voice->sid resolution, provider
derivation, config-builder setattr routing, num_speakers mismatch, and
streaming — all against a stubbed `sherpa_onnx` module. Nothing here
downloads a bundle."""

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, RawRequestProxy
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


class _FakeOfflineTts:
    """`chunks`, if given, is emitted one per `callback=` call, gated on the
    previous call's return value (0 aborts)."""

    def __init__(self, cfg, num_speakers, chunks: list[np.ndarray] | None = None):
        self.cfg = cfg
        self.num_speakers = num_speakers
        self.sample_rate = 24000
        self.generate_calls: list[tuple] = []
        self.chunks = chunks or []
        self.aborted_early = False

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.generate_calls.append((text, sid, speed))
        if callback is None:
            return SimpleNamespace(samples=[0.0, 0.1, -0.1], sample_rate=self.sample_rate)
        for i, chunk in enumerate(self.chunks):
            if not callback(chunk, (i + 1) / len(self.chunks)):
                self.aborted_early = True
                break
            time.sleep(0.05)  # gives the consumer a window to aclose() mid-loop
        return SimpleNamespace(samples=[], sample_rate=self.sample_rate)


def _fake_sherpa_onnx_module(num_speakers: int) -> types.ModuleType:
    mod = types.ModuleType("sherpa_onnx")
    mod.OfflineTtsConfig = _FakeTtsConfig
    mod.OfflineTts = lambda cfg: _FakeOfflineTts(cfg, num_speakers)
    return mod


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


class TestLoad:
    def test_provider_and_threads_and_path_routing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(_ENTRY.voice_names)))
        infer = SherpaOnnxInfer(_resolved(_config(num_cpus=3)))
        tts, entry = infer._load()
        assert entry is _ENTRY
        assert tts.cfg.model.provider == "cpu"
        assert tts.cfg.model.num_threads == 3
        assert tts.cfg.model.kokoro.model == "/bundle/model.onnx"
        assert tts.cfg.model.kokoro.voices == "/bundle/voices.bin"
        assert tts.cfg.model.kokoro.tokens == "/bundle/tokens.txt"
        assert tts.cfg.model.kokoro.data_dir == "/bundle/espeak-ng-data"

    def test_lexicon_joined_with_commas(self, monkeypatch):
        entry = REGISTRY["kokoro-multi-lang-v1_0"]
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(entry.voice_names)))
        infer = SherpaOnnxInfer(_resolved(_config(model="kokoro-multi-lang-v1_0")))
        tts, entry2 = infer._load()
        assert entry2 is entry
        assert tts.cfg.model.kokoro.lexicon == "/bundle/lexicon-us-en.txt,/bundle/lexicon-zh.txt"

    def test_local_dir_uses_the_entry_its_basename_names(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(_ENTRY.voice_names)))
        bundle = str(tmp_path / "kokoro-en-v0_19")
        infer = SherpaOnnxInfer(_resolved(_config(model=f"{bundle}/"), bundle))
        tts, entry = infer._load()
        assert entry is _ENTRY
        assert tts.cfg.model.kokoro.model == f"{bundle}/model.onnx"

    def test_num_speakers_mismatch_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(num_speakers=999))
        infer = SherpaOnnxInfer(_resolved(_config()))
        with pytest.raises(ValueError, match="999"):
            infer._load()

    def test_rejects_a_config_without_a_resolved_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx_module(len(_ENTRY.voice_names)))
        infer = SherpaOnnxInfer(_config())
        with pytest.raises(ValueError, match="no resolved bundle directory"):
            infer._load()


class TestCreateSpeech:
    @pytest.mark.asyncio
    async def test_unknown_voice_returns_400(self):
        infer = _infer()
        req = SpeechRequest(input="hi", model="tts", voice="not_a_voice")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, ErrorResponse)
        assert "not_a_voice" in result.error.message

    @pytest.mark.asyncio
    async def test_generates_wav_for_known_voice(self):
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names))
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hello world", model="tts", voice="af_bella", speed=1.5)
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, RawSpeechResponse)
        assert result.audio[:4] == b"RIFF"
        assert tts.generate_calls == [("hello world", _ENTRY.voice_names.index("af_bella"), 1.5)]

    @pytest.mark.asyncio
    async def test_numeric_voice_resolves_to_sid(self):
        tts = _FakeOfflineTts(_FakeTtsConfig(), num_speakers=len(_ENTRY.voice_names))
        infer = _infer(tts=tts)
        req = SpeechRequest(input="hi", model="tts", voice="2")
        result = await infer.create_speech(req, RawRequestProxy(None, {}))
        assert isinstance(result, RawSpeechResponse)
        assert tts.generate_calls[0][1] == 2

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
        assert tts.aborted_early
