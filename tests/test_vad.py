"""Vad: OpenAI server_vad timing over a scripted detector, plus the stream
resampler and PCM16 decoding it's fed through."""

import sys
import types

import numpy as np
import pytest

from modelship.utils.audio import StreamResampler, from_pcm16
from modelship.utils.vad import SAMPLE_RATE, SpeechStarted, SpeechStopped, Vad, VadSettings

_WINDOW = 512


def _s(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE)


class _Segment:
    def __init__(self, start: int, samples: np.ndarray) -> None:
        self.start = start
        self.samples = samples


def _fake_sherpa(speech: list[tuple[float, float]], configs: list) -> types.ModuleType:
    """A detector that hears speech in the given (start, end) seconds: it reports
    speech once it has lasted min_speech_duration, and a segment once
    min_silence_duration of silence follows it."""

    class Config:
        def __init__(self) -> None:
            self.silero_vad = types.SimpleNamespace(window_size=_WINDOW)
            self.sample_rate = 0
            self.num_threads = 0

    class Detector:
        def __init__(self, config, buffer_size_in_seconds) -> None:
            configs.append((config, buffer_size_in_seconds))
            self.cfg = config.silero_vad
            self.pos = 0
            self.audio = np.empty(0, dtype=np.float32)
            self.queue: list[_Segment] = []
            self.pending = [(_s(a), _s(b)) for a, b in speech]

        def accept_waveform(self, samples) -> None:
            self.audio = np.concatenate([self.audio, samples])
            self.pos += len(samples)
            while self.pending and self.pos >= self.pending[0][1] + _s(self.cfg.min_silence_duration):
                start, end = self.pending.pop(0)
                self.queue.append(_Segment(start, self.audio[start:end]))

        def is_speech_detected(self) -> bool:
            return any(start + _s(self.cfg.min_speech_duration) <= self.pos for start, _ in self.pending)

        def empty(self) -> bool:
            return not self.queue

        @property
        def front(self) -> _Segment:
            return self.queue[0]

        def pop(self) -> None:
            self.queue.pop(0)

        def flush(self) -> None:
            for start, _ in self.pending:
                self.queue.append(_Segment(start, self.audio[start:]))
            self.pending = []

    return types.SimpleNamespace(VadModelConfig=Config, VoiceActivityDetector=Detector)


@pytest.fixture
def vad_with(monkeypatch):
    configs: list = []

    def make(speech, settings=None) -> Vad:
        monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa(speech, configs))
        return Vad("/models/silero.onnx", settings)

    make.configs = configs  # type: ignore[attr-defined]
    return make


def _feed(vad: Vad, seconds: float, chunk: float = 0.1) -> list:
    audio = np.arange(_s(seconds), dtype=np.float32)
    events = []
    for i in range(0, len(audio), _s(chunk)):
        events += vad.feed(audio[i : i + _s(chunk)])
    return events


class TestSettings:
    def test_openai_parameters_configure_the_detector(self, vad_with):
        vad_with([], VadSettings(threshold=0.7, prefix_padding_ms=200, silence_duration_ms=800))
        config, buffer = vad_with.configs[0]
        assert config.silero_vad.model == "/models/silero.onnx"
        assert config.silero_vad.threshold == 0.7
        assert config.silero_vad.min_silence_duration == 0.8
        assert config.silero_vad.min_speech_duration == 0.25
        assert config.sample_rate == SAMPLE_RATE
        assert buffer > config.silero_vad.max_speech_duration

    def test_defaults_are_openais(self):
        assert VadSettings() == VadSettings(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)


class TestEvents:
    def test_a_turn_is_started_then_stopped(self, vad_with):
        events = _feed(vad_with([(1.0, 2.0)]), 4.0)
        assert [type(e) for e in events] == [SpeechStarted, SpeechStopped]

    def test_the_stop_spans_prefix_padding_to_the_silence_after_speech(self, vad_with):
        stopped = _feed(vad_with([(1.0, 2.0)]), 4.0)[-1]
        assert stopped.audio_start_ms == 700  # 1.0 s less 300 ms prefix padding
        assert 2500 <= stopped.audio_end_ms < 2500 + 1000 * _WINDOW // SAMPLE_RATE + 1
        start, end = _s(stopped.audio_start_ms / 1000), _s(stopped.audio_end_ms / 1000)
        assert np.array_equal(stopped.audio, np.arange(start, end, dtype=np.float32))

    def test_the_start_is_reported_once_speech_has_lasted_long_enough(self, vad_with):
        started = _feed(vad_with([(1.0, 2.0)]), 4.0)[0]
        # an estimate from where the detector reported it, within one window of the padded start
        assert abs(started.audio_start_ms - 700) <= 1000 * _WINDOW // SAMPLE_RATE

    def test_each_utterance_is_its_own_turn(self, vad_with):
        events = _feed(vad_with([(1.0, 2.0), (3.0, 3.6), (5.0, 6.0)]), 8.0)
        stops = [e for e in events if isinstance(e, SpeechStopped)]
        assert [e.audio_start_ms for e in stops] == [700, 2700, 4700]
        assert [type(e) for e in events] == [SpeechStarted, SpeechStopped] * 3

    def test_prefix_padding_survives_a_long_silence_before_speech(self, vad_with):
        stopped = _feed(vad_with([(30.0, 31.0)]), 33.0)[-1]
        assert stopped.audio[0] == _s(29.7)

    def test_chunk_size_does_not_change_the_turns(self, vad_with):
        def stops(chunk):
            return [
                (e.audio_start_ms, e.audio_end_ms)
                for e in _feed(vad_with([(1.0, 2.0), (3.0, 4.0)]), 6.0, chunk)
                if isinstance(e, SpeechStopped)
            ]

        assert stops(0.02) == stops(0.16) == stops(1.0)

    def test_flush_stops_a_turn_still_in_progress(self, vad_with):
        vad = vad_with([(1.0, 10.0)])
        assert [type(e) for e in _feed(vad, 3.0)] == [SpeechStarted]
        stopped = vad.flush()[-1]
        assert isinstance(stopped, SpeechStopped)
        assert stopped.audio_start_ms == 700
        assert stopped.audio_end_ms == 3000


class TestStreamResampler:
    def test_chunks_match_resampling_the_whole_stream(self):
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.5, 0.5, 24000 * 2).astype(np.float32)
        whole = StreamResampler(24000, 16000)(audio, last=True)
        stream = StreamResampler(24000, 16000)
        sizes = [240, 1000, 4800, 960]
        chunks, i = [], 0
        while i < len(audio):
            size = sizes[len(chunks) % len(sizes)]
            chunks.append(stream(audio[i : i + size], last=i + size >= len(audio)))
            i += size
        assert np.allclose(np.concatenate(chunks), whole, atol=1e-6)
        assert abs(len(whole) - 32000) <= 1

    def test_the_same_rate_passes_through(self):
        audio = np.ones(100, dtype=np.float32)
        assert StreamResampler(16000, 16000)(audio) is not None
        assert np.array_equal(StreamResampler(16000, 16000)(audio), audio)


def test_from_pcm16_scales_to_unit_range():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    assert from_pcm16(pcm).tolist() == [0.0, 0.5, -1.0, 32767 / 32768]
