"""End-to-end coverage for the sherpa_onnx loader against a real sherpa-onnx
binding and a real downloaded kokoro bundle. `test_sherpa_onnx_infer.py` covers
the same logic against a stub."""

import io
import json
import wave

import httpx
import pytest

import openai
from modelship.infer.sherpa_onnx.registry import REGISTRY

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"
_VOICE_NAMES = REGISTRY["kokoro-en-v0_19"].voice_names


def _post_speech(**fields) -> httpx.Response:
    return httpx.post(f"{OPENAI_API_BASE}/audio/speech", json={"model": "tts-model", **fields}, timeout=120)


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line[len("data: ") :].strip() not in ("", "[DONE]")
    ]


def _wav_frames_and_rate(audio: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(audio)) as w:
        return w.getnframes(), w.getframerate()


def _transcribe(client, audio: bytes, tmp_path) -> str:
    audio_file = tmp_path / "speech.wav"
    audio_file.write_bytes(audio)
    with open(audio_file, "rb") as f:
        return client.audio.transcriptions.create(model="stt-cpp-model", file=f).text.lower()


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSherpaOnnx:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-cpp-model")

    def test_multiple_voices_produce_distinct_audio(self, client):
        # A sample across the registry, not all 11 — enough to prove sid
        # resolution actually varies the speaker, not exhaustive coverage.
        sample = (_VOICE_NAMES[0], _VOICE_NAMES[len(_VOICE_NAMES) // 2], _VOICE_NAMES[-1])
        clips = [
            client.audio.speech.create(model="tts-model", voice=voice, input="The quick brown fox.").content
            for voice in sample
        ]
        assert len({clips[0], clips[1], clips[2]}) == len(sample)

    def test_speech_is_intelligible_via_transcription(self, client, tmp_path):
        # Round-trips the audio through a real STT model (whispercpp, CPU-only) to
        # confirm it's clean speech, not silence/noise shaped like a WAV file.
        audio = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="The wizard counted seventeen purple bananas."
        ).content
        text = _transcribe(client, audio, tmp_path)
        assert "wizard" in text
        assert "banana" in text

    def test_different_voices_are_all_intelligible(self, client, tmp_path):
        sample = (_VOICE_NAMES[0], _VOICE_NAMES[len(_VOICE_NAMES) // 2], _VOICE_NAMES[-1])
        for voice in sample:
            audio = client.audio.speech.create(
                model="tts-model", voice=voice, input="Open the door and turn on the light."
            ).content
            text = _transcribe(client, audio, tmp_path)
            assert "door" in text or "light" in text, f"voice {voice!r} transcribed to unintelligible text: {text!r}"

    def test_numeric_voice_is_accepted(self, client, tmp_path):
        # The kokoro vocoder isn't byte-deterministic, so this only checks the digit
        # path round-trips to intelligible audio, not byte equality with the named voice.
        audio = client.audio.speech.create(model="tts-model", voice="0", input="Numeric voice selection.").content
        assert "numeric" in _transcribe(client, audio, tmp_path)

    def test_unknown_voice_speaks_with_the_default_voice(self, client, tmp_path):
        audio = client.audio.speech.create(model="tts-model", voice="alloy", input="Default voice selection.").content
        assert "default" in _transcribe(client, audio, tmp_path)

    def test_speed_parameter_changes_duration(self, client):
        normal = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This sentence has a normal speaking speed.", speed=1.0
        ).content
        fast = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This sentence has a normal speaking speed.", speed=2.0
        ).content
        normal_frames, normal_rate = _wav_frames_and_rate(normal)
        fast_frames, fast_rate = _wav_frames_and_rate(fast)
        assert (fast_frames / fast_rate) < (normal_frames / normal_rate)

    def test_streaming_sse_speech(self):
        response = _post_speech(voice="af_bella", input="Streaming synthesis test.", stream_format="sse")
        assert response.status_code == 200
        events = _sse_events(response.text)
        assert events, "expected at least one SSE event"
        assert events[-1]["type"] == "speech.audio.done"
        deltas = [e for e in events[:-1] if e["type"] == "speech.audio.delta"]
        assert deltas, "expected at least one speech.audio.delta event"
        assert all(e["audio"] for e in deltas)

    def test_response_format_is_accepted_but_audio_is_always_wav(self):
        # RawSpeechResponse always carries media_type audio/wav; response_format is
        # accepted by the schema but not used to transcode the output.
        response = _post_speech(voice="af_bella", input="Format check.", response_format="mp3")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")


_RUN_ON = (
    "the river ran wide and slow past the mill and the orchard and the long line of willows that leaned over the "
    "water where the children of the village used to swim on summer afternoons while their parents worked the fields "
    "and the old men sat on the bench outside the inn telling the same stories they had told for fifty years about "
    "the winter the river froze solid and the year the bridge was washed away by the spring floods and rebuilt by hand"
)


def _frames(audio: bytes) -> int:
    with wave.open(io.BytesIO(audio)) as w:
        return w.getnframes()


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSherpaOnnxMultiLang:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-multi-model", "stt-cpp-model")

    def _speak(self, client, voice, text):
        return client.audio.speech.create(model="tts-multi-model", voice=voice, input=text).content

    def test_chinese_produces_non_trivial_audio(self, client):
        audio = self._speak(client, "zf_xiaobei", "今天天气很好，我们去公园散步吧。")
        assert audio.startswith(b"RIFF")
        assert _frames(audio) > 16000

    def test_japanese_produces_non_trivial_audio(self, client):
        audio = self._speak(client, "jf_alpha", "東京駅で友達と待ち合わせをしました。")
        assert audio.startswith(b"RIFF")
        assert _frames(audio) > 16000

    def test_spanish_and_english_voices_differ(self, client):
        spanish = self._speak(client, "ef_dora", "Hola, ¿cómo estás? Hoy hace mucho sol.")
        english = self._speak(client, "af_heart", "Hola, ¿cómo estás? Hoy hace mucho sol.")
        assert spanish != english

    def test_long_run_on_sentence_is_intelligible_end_to_end(self, client, tmp_path):
        # >400 phoneme tokens: the frontend splits at a word boundary, so the
        # whole sentence should still transcribe, late words included.
        audio = self._speak(client, "af_heart", _RUN_ON)
        text = _transcribe(client, audio, tmp_path)
        assert "washed" in text
        assert "hand" in text

    def test_hostile_input_does_not_kill_the_replica(self, client):
        # A short string that segfaults espeak: the helper process dies, the
        # request fails, and the replica keeps serving.
        with pytest.raises(openai.APIError):
            self._speak(client, "ef_dora", ":-ॠ")
        recovered = self._speak(client, "af_heart", "Still serving after the crash.")
        assert recovered.startswith(b"RIFF")


def _silence(seconds: float, rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


def _wav(frames: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buffer.getvalue()


def _speech_frames(client, text: str) -> tuple[bytes, int]:
    audio = client.audio.speech.create(model="tts-model", voice="af_bella", input=text, response_format="wav").content
    with wave.open(io.BytesIO(audio)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestParakeet:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-parakeet")

    def _transcribe(self, client, audio: bytes, tmp_path, **fields):
        path = tmp_path / "speech.wav"
        path.write_bytes(audio)
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(model="stt-parakeet", file=f, **fields)

    def test_speech_round_trips_through_parakeet(self, client, tmp_path):
        frames, rate = _speech_frames(client, "Please turn on the kitchen lights.")
        text = self._transcribe(client, _wav(frames, rate), tmp_path).text.lower()
        assert "kitchen" in text and "lights" in text, text

    def test_long_audio_is_cut_at_silences_and_keeps_its_timeline(self, client, tmp_path):
        frames, rate = _speech_frames(client, "Open the garage door.")
        clip = frames + _silence(2.0, rate)
        seconds = len(clip) * 12 / 2 / rate
        # past the 30 s a piece may hold
        assert seconds > 35
        result = self._transcribe(
            client,
            _wav(clip * 12, rate),
            tmp_path,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        assert result.text.lower().count("garage") == 12, result.text
        assert result.duration == pytest.approx(seconds, abs=0.05)
        assert result.segments[-1].end > 30
        assert result.words[0].start < result.words[-1].start

    def test_stream_sends_deltas_then_the_text(self, tmp_path, client):
        frames, rate = _speech_frames(client, "Close the window.")
        path = tmp_path / "speech.wav"
        path.write_bytes(_wav(frames, rate))
        with open(path, "rb") as f:
            response = httpx.post(
                f"{OPENAI_API_BASE}/audio/transcriptions",
                files={"file": ("speech.wav", f, "audio/wav")},
                data={"model": "stt-parakeet", "stream": "true"},
                timeout=120,
            )
        events = _sse_events(response.text)
        assert events[0]["type"] == "transcript.text.delta"
        assert events[-1]["type"] == "transcript.text.done"
        assert "window" in events[-1]["text"].lower()
