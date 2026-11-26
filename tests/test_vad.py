"""Vad: OpenAI server_vad timing over a scripted detector; SpeechStream's segments over
a scripted Silero, through a VoiceDetector; plus the stream resampler and PCM16 decoding
they're fed through."""

import sys
import types

import numpy as np
import pytest

from modelship.infer.sherpa_onnx.vad import SpeechStarted, SpeechStopped, Vad, VadSettings
from modelship.utils.audio import StreamResampler, from_pcm16
from modelship.utils.vad import CONTEXT, SAMPLE_RATE, WINDOW, SpeechSegment
from modelship.utils.voice import VoiceDetector

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


class _Silero:
    """Silero as `SpeechStream` runs it: a window's speech probability is twice the level of its samples. Records
    each window it ran and the state it was given, and hands back that state plus one."""

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []
        self.states: list[float] = []

    def run(self, window: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        self.windows.append(window.copy())
        self.states.append(float(state.flat[0]))
        return min(1.0, 2 * float(np.abs(window[CONTEXT:]).mean())), state + 1


def _windows(*runs: tuple[int, float]) -> bytes:
    """16 kHz PCM of (count, probability) runs of windows."""
    level = np.concatenate([np.full(count * WINDOW, probability / 2) for count, probability in runs])
    return (level * 32768).astype("<i2").tobytes()


def _stream(model: _Silero | None = None, start_ms: float = 0.0) -> VoiceDetector:
    return VoiceDetector(model or _Silero(), SAMPLE_RATE, start_ms)  # type: ignore[arg-type]


# one window, in ms
_W = WINDOW * 1000 / SAMPLE_RATE


class TestSpeechStream:
    def test_speech_opens_a_segment_once_it_lasts_dated_back_to_its_start(self):
        stream = _stream(start_ms=1000)
        # 250 ms is 7.8 windows
        assert stream.detect(_windows((10, 0.0), (7, 0.9)), 0.5, 500) == []
        assert stream.detect(_windows((1, 0.9)), 0.5, 500) == [SpeechSegment(1000 + 10 * _W, None, 1000 + 18 * _W)]

    def test_a_segment_closes_once_its_silence_lasts_and_ends_where_the_silence_began(self):
        stream = _stream()
        # 500 ms is 15.6 windows
        assert stream.detect(_windows((10, 0.0), (20, 0.9), (15, 0.0)), 0.5, 500) == [
            SpeechSegment(10 * _W, None, 30 * _W)
        ]
        assert stream.detect(_windows((1, 0.0)), 0.5, 500) == [SpeechSegment(10 * _W, 30 * _W, 30 * _W)]
        assert stream.detect(_windows((20, 0.0)), 0.5, 500) == []

    def test_speech_before_the_silence_lasts_keeps_the_segment(self):
        stream = _stream()
        found = stream.detect(_windows((10, 0.9), (10, 0.0), (10, 0.9), (16, 0.0)), 0.5, 500)
        assert found == [SpeechSegment(0, 30 * _W, 30 * _W)]

    def test_a_window_between_the_thresholds_keeps_speech_going_but_starts_none(self):
        stream = _stream()
        # 0.35 is the threshold less the hysteresis
        assert stream.detect(_windows((20, 0.4)), 0.5, 500) == []
        found = stream.detect(_windows((10, 0.9), (5, 0.4), (16, 0.0)), 0.5, 500)
        assert found == [SpeechSegment(20 * _W, 35 * _W, 35 * _W)]

    def test_once_the_silence_began_a_window_between_the_thresholds_counts_as_silence(self):
        found = _stream().detect(_windows((10, 0.9), (1, 0.0), (15, 0.4)), 0.5, 500)
        assert found == [SpeechSegment(0, 10 * _W, 10 * _W)]

    def test_silence_before_speech_lasts_starts_it_over(self):
        assert _stream().detect(_windows((5, 0.9), (1, 0.0), (5, 0.9), (20, 0.0)), 0.5, 500) == []

    def test_each_feed_takes_the_settings_it_is_given(self):
        stream = _stream()
        assert stream.detect(_windows((10, 0.9), (7, 0.0)), 0.5, 500)[-1].end_ms is None
        assert stream.detect(_windows((1, 0.0)), 0.5, 200) == [SpeechSegment(0, 10 * _W, 10 * _W)]
        # a threshold above the speech hears none
        assert stream.detect(_windows((20, 0.6)), 0.7, 200) == []

    def test_chunk_size_changes_nothing(self):
        audio = _windows((10, 0.0), (20, 0.9), (20, 0.0), (12, 0.9), (20, 0.0))
        whole = _stream().detect(audio, 0.5, 500)
        stream, found = _stream(), []
        for at in range(0, len(audio), 734):
            found += [
                segment for segment in stream.detect(audio[at : at + 734], 0.5, 500) if segment.end_ms is not None
            ]
        assert found == whole == [SpeechSegment(10 * _W, 30 * _W, 30 * _W), SpeechSegment(50 * _W, 62 * _W, 62 * _W)]

    def test_each_window_runs_with_the_64_samples_before_it_and_the_state_after_the_last(self):
        model = _Silero()
        stream = _stream(model)
        stream.detect(_windows((1, 0.2), (1, 0.6)), 0.5, 500)
        stream.detect(_windows((1, 0.4)), 0.5, 500)
        first, second, third = model.windows
        assert len(first) == CONTEXT + WINDOW
        assert not first[:CONTEXT].any()
        assert np.array_equal(second[:CONTEXT], first[-CONTEXT:])
        assert np.array_equal(third[:CONTEXT], second[-CONTEXT:])
        assert model.states == [0.0, 1.0, 2.0]

    def test_audio_at_another_rate_is_timed_on_the_streams_timeline(self):
        rate = 24000
        level = np.concatenate([np.zeros(rate), np.full(rate, 0.45), np.zeros(rate)])
        stream = VoiceDetector(_Silero(), rate, 500.0)  # type: ignore[arg-type]
        (segment,) = stream.detect((level * 32768).astype("<i2").tobytes(), 0.5, 500)
        assert segment.start_ms == pytest.approx(1500, abs=_W)
        assert segment.end_ms == pytest.approx(2500, abs=_W)


def test_from_pcm16_scales_to_unit_range():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    assert from_pcm16(pcm).tolist() == [0.0, 0.5, -1.0, 32767 / 32768]
