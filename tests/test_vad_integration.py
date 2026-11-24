"""The Silero VAD library against the real model, fetched from model-bundles, on
speech the cluster's kokoro model synthesizes. `test_vad.py` covers the event
logic against a stub."""

import io

import numpy as np
import pytest
import soundfile as sf

from modelship.utils.audio import StreamResampler
from modelship.utils.sources import download_model_source, file
from modelship.utils.vad import (
    SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    Vad,
    VadScanner,
    VadSettings,
    silero_vad_source,
)

_PHRASES = ("Turn on the kitchen lights.", "What is the weather tomorrow?", "Play some music.")


@pytest.fixture(scope="module")
def silero(tmp_path_factory) -> str:
    cache = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(file, "cache_dir", lambda: str(cache))
        return download_model_source(silero_vad_source())


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSileroVad:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model")

    @pytest.fixture(scope="class")
    def timeline(self, client) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """The phrases at 16 kHz with 1.2 s of silence before, between and after them,
        and where each phrase is, in seconds."""
        parts, spans, at = [], [], 0
        for phrase in _PHRASES:
            wav = client.audio.speech.create(model="tts-model", voice="af_bella", input=phrase, response_format="wav")
            audio, rate = sf.read(io.BytesIO(wav.content), dtype="float32")
            speech = StreamResampler(rate, SAMPLE_RATE)(audio, last=True)
            gap = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)
            parts += [gap, speech]
            spans.append(((at + len(gap)) / SAMPLE_RATE, (at + len(gap) + len(speech)) / SAMPLE_RATE))
            at += len(gap) + len(speech)
        parts.append(np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32))
        return np.concatenate(parts), spans

    def _events(self, silero, audio, settings=None, chunk_s=0.1):
        vad = Vad(silero, settings)
        step = int(chunk_s * SAMPLE_RATE)
        events = [event for i in range(0, len(audio), step) for event in vad.feed(audio[i : i + step])]
        return events + vad.flush()

    def test_each_phrase_is_one_turn(self, silero, timeline):
        audio, _ = timeline
        events = self._events(silero, audio)
        assert [type(e) for e in events] == [SpeechStarted, SpeechStopped] * len(_PHRASES)

    def test_a_turn_covers_its_phrase_with_padding_and_trailing_silence(self, silero, timeline):
        audio, spans = timeline
        stops = [e for e in self._events(silero, audio) if isinstance(e, SpeechStopped)]
        for stop, (start, end) in zip(stops, spans, strict=True):
            # prefix padding (300 ms) before speech, the 500 ms silence after it
            assert start - 0.45 <= stop.audio_start_ms / 1000 <= start - 0.15
            assert end + 0.3 <= stop.audio_end_ms / 1000 <= end + 0.8
            assert len(stop.audio) == pytest.approx(
                (stop.audio_end_ms - stop.audio_start_ms) * SAMPLE_RATE / 1000, abs=32
            )

    def test_a_longer_silence_merges_nothing_and_ends_turns_later(self, silero, timeline):
        audio, _ = timeline
        default = [e for e in self._events(silero, audio) if isinstance(e, SpeechStopped)]
        patient = [
            e
            for e in self._events(silero, audio, VadSettings(silence_duration_ms=1000))
            if isinstance(e, SpeechStopped)
        ]
        assert len(patient) == len(default)
        assert all(p.audio_end_ms - d.audio_end_ms >= 400 for p, d in zip(patient, default, strict=True))

    def test_silence_alone_is_no_turn(self, silero):
        noise = np.random.default_rng(0).normal(0, 0.003, 5 * SAMPLE_RATE).astype(np.float32)
        assert self._events(silero, noise) == []
        assert VadScanner(silero).scan(noise, 0.5, 500) == []

    def test_a_scan_finds_each_phrase_and_leaves_open_the_one_its_window_cuts(self, silero, timeline):
        audio, spans = timeline
        scanner = VadScanner(silero)
        found = scanner.scan(audio, 0.5, 500)
        assert len(found) == len(_PHRASES)
        for segment, (start, end) in zip(found, spans, strict=True):
            assert segment.end_ms is not None
            assert start - 0.2 <= segment.start_ms / 1000 <= start + 0.25
            assert end - 0.2 <= segment.end_ms / 1000 <= end + 0.15
        cut = scanner.scan(audio[: int((spans[-1][0] + 0.5) * SAMPLE_RATE)], 0.5, 500)
        assert len(cut) == len(_PHRASES)
        assert cut[-1].end_ms is None
        # a window starting inside a phrase, as the next scan of a turn does
        inside = scanner.scan(audio[int((spans[1][0] + 0.3) * SAMPLE_RATE) :], 0.5, 500)
        assert inside[0].start_ms <= 100
        # its detector, pooled and reset, answers the same again
        assert scanner.scan(audio, 0.5, 500) == found
