"""Cross-loader audio round trip: sherpa_onnx TTS output fed back in as
vllm-whisper STT input, through the modelship gateway."""

import httpx
import pytest

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"


def _post_transcription(model: str, audio: bytes, **fields) -> httpx.Response:
    data = {"model": model, **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        f"{OPENAI_API_BASE}/audio/transcriptions",
        data=data,
        files={"file": ("audio.mp3", audio, "audio/mpeg")},
        timeout=180,
    )


@pytest.mark.integration
class TestAudio:
    """tts-model and stt-model are both small enough to share a class;
    test_audio_transcription re-uses the TTS output as its input."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-model")

    @pytest.fixture(scope="class")
    def speech(self, client) -> bytes:
        return client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This is a test transcription."
        ).content

    def test_audio_speech(self, client):
        response = client.audio.speech.create(model="tts-model", voice="af_bella", input="Hello from integration test")
        assert len(response.content) > 1000

    def test_audio_transcription(self, client, tmp_path):
        audio_data = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This is a test transcription."
        ).content

        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(audio_data)

        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(model="stt-model", file=f)
        assert "test" in transcription.text.lower()

    def test_transcription_text_format(self, speech):
        response = _post_transcription("stt-model", speech, response_format="text")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "test" in response.text.lower()
        assert not response.text.lstrip().startswith("{")

    def test_transcription_srt_format(self, speech):
        response = _post_transcription("stt-model", speech, response_format="srt")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "-->" in response.text
        assert response.text.lstrip().startswith("1")

    def test_transcription_vtt_format(self, speech):
        response = _post_transcription("stt-model", speech, response_format="vtt")
        assert response.status_code == 200
        assert response.text.startswith("WEBVTT")
        assert "-->" in response.text

    def test_subtitle_format_cannot_stream(self, speech):
        response = _post_transcription("stt-model", speech, response_format="srt", stream="true")
        assert response.status_code == 400
