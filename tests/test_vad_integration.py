"""The Silero VAD library against the real model, fetched from model-bundles, on
speech the cluster's kokoro model synthesizes: sherpa-onnx's `Vad`, and the
onnxruntime `SpeechStream` against it. `test_vad.py` covers the logic against
stubs."""

import io

import numpy as np
import pytest
import soundfile as sf

from modelship.utils.audio import StreamResampler, to_pcm16
from modelship.utils.sources import download_model_source, file
from modelship.utils.vad import (
    SAMPLE_RATE,
    STATE_SHAPE,
    SileroModel,
    SpeechSegment,
    SpeechStarted,
    SpeechStopped,
    SpeechStream,
    Vad,
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

    def _stream(self, model, audio, rate=SAMPLE_RATE, chunk_s=0.1, silence_ms=500) -> list[SpeechSegment]:
        """The segments a stream finds in *audio* (float32 at *rate*) fed *chunk_s* at a time; the last one may be
        open."""
        stream = SpeechStream(model, rate, 0.0)
        pcm, step = to_pcm16(audio), int(chunk_s * rate) * 2
        found: list[SpeechSegment] = []
        for at in range(0, len(pcm), step):
            segments = stream.feed(pcm[at : at + step], 0.5, silence_ms)
            found = [*(segment for segment in found if segment.end_ms is not None), *segments]
        return found

    def test_silence_alone_is_no_turn(self, silero):
        noise = np.random.default_rng(0).normal(0, 0.003, 5 * SAMPLE_RATE).astype(np.float32)
        assert self._events(silero, noise) == []
        assert self._stream(SileroModel(silero), noise) == []

    def test_a_stream_finds_each_phrase_where_sherpas_detector_does(self, silero, timeline):
        audio, spans = timeline
        found = self._stream(SileroModel(silero), audio)
        stops = [e for e in self._events(silero, audio) if isinstance(e, SpeechStopped)]
        assert len(found) == len(stops) == len(_PHRASES)
        for segment, stop, (start, end) in zip(found, stops, spans, strict=True):
            assert segment.end_ms is not None
            assert start - 0.2 <= segment.start_ms / 1000 <= start + 0.25
            assert end - 0.2 <= segment.end_ms / 1000 <= end + 0.15
            # sherpa's turn: the prefix padding before the speech, the silence after it
            assert segment.start_ms - 300 == pytest.approx(stop.audio_start_ms, abs=100)
            assert segment.end_ms + 500 == pytest.approx(stop.audio_end_ms, abs=100)

    def test_an_open_segments_speech_ends_where_its_phrase_does(self, silero, timeline):
        audio, spans = timeline
        model = SileroModel(silero)
        # inside the last phrase, the speech goes on to the audio's end
        cut = self._stream(model, audio[: int((spans[-1][0] + 0.5) * SAMPLE_RATE)])
        assert cut[-1].end_ms is None
        assert cut[-1].speech_end_ms / 1000 >= spans[-1][0] + 0.4
        # 300 ms after a phrase, its segment is still open, and its speech ends where the phrase does
        (_, end) = spans[1]
        trailing = self._stream(model, audio[: int((end + 0.3) * SAMPLE_RATE)])
        assert trailing[-1].end_ms is None
        assert end - 0.25 <= trailing[-1].speech_end_ms / 1000 <= end + 0.15

    def test_audio_at_24_khz_finds_the_same_segments(self, silero, timeline):
        audio, _ = timeline
        model = SileroModel(silero)
        at_24k = StreamResampler(SAMPLE_RATE, 24000)(audio, last=True)
        found, resampled = self._stream(model, audio), self._stream(model, at_24k, rate=24000)
        assert len(resampled) == len(found)
        for segment, other in zip(found, resampled, strict=True):
            assert other.start_ms == pytest.approx(segment.start_ms, abs=70)
            assert other.end_ms == pytest.approx(segment.end_ms, abs=70)


@pytest.mark.integration
@pytest.mark.sherpa_onnx
def test_the_silero_file_takes_the_inputs_silero_model_feeds_it(silero):
    inputs = {i.name: i.shape for i in SileroModel(silero)._session.get_inputs()}
    assert inputs.keys() == {"input", "state", "sr"}
    assert [inputs["state"][0], inputs["state"][2]] == [STATE_SHAPE[0], STATE_SHAPE[2]]
