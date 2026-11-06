"""End-to-end coverage for the whispercpp loader against a real pywhispercpp
binding. `test_whispercpp_infer.py` covers the same logic against a stub."""

import json

import httpx
import pytest

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"

# `tiny.en` has no language tokens; only `tiny` can auto-detect or translate.
ENGLISH_ONLY_MODEL = "stt-cpp-model"
MULTILINGUAL_MODEL = "stt-cpp-multilingual"


def _post_audio(endpoint: str, model: str, audio: bytes, **fields) -> httpx.Response:
    data = {"model": model, **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        f"{OPENAI_API_BASE}/audio/{endpoint}",
        data=data,
        files={"file": ("audio.mp3", audio, "audio/mpeg")},
        timeout=180,
    )


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line[len("data: ") :].strip() not in ("", "[DONE]")
    ]


@pytest.mark.integration
@pytest.mark.whispercpp
class TestWhispercpp:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", ENGLISH_ONLY_MODEL, MULTILINGUAL_MODEL)

    @pytest.fixture(scope="class")
    def speech(self, client) -> bytes:
        return client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This is a test transcription."
        ).content

    # -- transcription -----------------------------------------------------

    def test_transcription_json(self, client, speech, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(speech)
        with open(audio, "rb") as f:
            result = client.audio.transcriptions.create(model=ENGLISH_ONLY_MODEL, file=f)
        assert "test" in result.text.lower()

    def test_transcription_verbose_json(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="verbose_json")
        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "transcribe"
        assert "test" in body["text"].lower()
        assert body["duration"] > 0
        assert body["segments"], "verbose_json must carry segments"
        first = body["segments"][0]
        assert first["end"] >= first["start"]
        assert body["usage"]["seconds"] >= 0

    def test_transcription_text_format(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="text")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "test" in response.text.lower()
        assert not response.text.lstrip().startswith("{")

    def test_transcription_srt_format(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="srt")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "-->" in response.text and "," in response.text.split("-->")[1]
        assert response.text.lstrip().startswith("1")

    def test_transcription_vtt_format(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="vtt")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text.startswith("WEBVTT")
        assert "-->" in response.text

    def test_subtitle_format_cannot_stream(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="srt", stream="true")
        assert response.status_code == 400

    def test_explicit_language_is_accepted(self, speech):
        response = _post_audio(
            "transcriptions", MULTILINGUAL_MODEL, speech, response_format="verbose_json", language="en"
        )
        assert response.status_code == 200
        assert response.json()["language"] == "en"

    def test_prompt_and_temperature_are_accepted(self, speech):
        # A stubbed Model accepts any kwarg; whisper.cpp rejects ones it doesn't define.
        response = _post_audio(
            "transcriptions",
            ENGLISH_ONLY_MODEL,
            speech,
            prompt="This is a transcription test.",
            temperature=0.2,
        )
        assert response.status_code == 200
        assert response.json()["text"].strip()

    def test_english_only_model_reports_en(self, speech):
        # .en (English-only) models must report 'en' regardless of what detection returns.
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, response_format="verbose_json")
        assert response.status_code == 200
        assert response.json()["language"] == "en"

    def test_multilingual_model_detects_language(self, speech):
        response = _post_audio("transcriptions", MULTILINGUAL_MODEL, speech, response_format="verbose_json")
        assert response.status_code == 200
        assert response.json()["language"] == "en"

    def test_transcription_streaming(self, speech):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, speech, stream=True)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _sse_events(response.text)
        deltas = [e for e in events if e["type"] == "transcript.text.delta"]
        done = [e for e in events if e["type"] == "transcript.text.done"]
        assert deltas
        assert len(done) == 1
        assert done[0] is events[-1]
        assert "test" in done[0]["text"].lower()
        assert done[0]["text"] == " ".join(d["delta"] for d in deltas)

    # -- translation -------------------------------------------------------

    def test_translation_json(self, speech):
        response = _post_audio("translations", MULTILINGUAL_MODEL, speech)
        assert response.status_code == 200
        assert response.json()["text"].strip()

    def test_translation_verbose_json(self, speech):
        response = _post_audio("translations", MULTILINGUAL_MODEL, speech, response_format="verbose_json")
        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "translate"
        assert body["language"] == "english"
        assert body["duration"] > 0
        assert body["segments"]

    def test_translation_srt_format(self, speech):
        response = _post_audio("translations", MULTILINGUAL_MODEL, speech, response_format="srt")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "-->" in response.text
        assert response.text.lstrip().startswith("1")

    def test_source_language_does_not_change_output_language(self, speech):
        response = _post_audio(
            "translations", MULTILINGUAL_MODEL, speech, response_format="verbose_json", language="en"
        )
        assert response.status_code == 200
        assert response.json()["language"] == "english"

    def test_to_language_is_ignored(self, speech):
        response = _post_audio(
            "translations", MULTILINGUAL_MODEL, speech, response_format="verbose_json", to_language="de"
        )
        assert response.status_code == 200
        assert response.json()["language"] == "english"

    def test_translation_streaming(self, speech):
        response = _post_audio("translations", MULTILINGUAL_MODEL, speech, stream=True)
        assert response.status_code == 200
        events = _sse_events(response.text)
        assert [e for e in events if e["type"] == "transcript.text.delta"]
        assert events[-1]["type"] == "transcript.text.done"

    # -- errors ------------------------------------------------------------

    def test_undecodable_audio_returns_400(self):
        response = _post_audio("transcriptions", ENGLISH_ONLY_MODEL, b"this is not audio")
        assert response.status_code == 400
        assert "decode" in response.json()["error"]["message"].lower()
