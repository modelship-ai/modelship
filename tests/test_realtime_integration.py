"""Pipelines on a real cluster: deployed with their models, listed in /v1/models, opening a /v1/realtime session,
a text conversation the pipeline's LLM answers, a spoken turn its stt model transcribes, and spoken answers its tts
model says. `test_realtime_session.py` and the deploy unit tests cover the same logic against fakes."""

import base64
import difflib
import io
import json
import re
import time
import wave

import httpx
import pytest
import yaml
from websockets.sync.client import connect

from modelship.infer.pipelines.assistant.audio import to_pcm16
from modelship.utils.config_schema import parse_pipeline
from tests.conftest import MODEL_CONFIGS, serve_apps

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"
REALTIME = "ws://localhost:8000/modelship/v1/realtime"

STT = MODEL_CONFIGS["stt-cpp-model"]
VOICE = {
    "name": "voice",
    "type": "assistant",
    "assistant_config": {"stt": STT["name"], "session_defaults": {"instructions": "Be brief."}},
}
GHOST = {"name": "ghost", "type": "assistant", "assistant_config": {"stt": "not-deployed"}}
LLM = MODEL_CONFIGS["chat-llama-server-plain"]
CHAT = {
    "name": "chat",
    "type": "assistant",
    "assistant_config": {"llm": LLM["name"], "session_defaults": {"instructions": "Answer in one short sentence."}},
}
TTS = MODEL_CONFIGS["tts-model"]


def _app(raw: dict) -> str:
    return parse_pipeline(raw).deployment_name("modelship")


def _listed() -> dict[str, str]:
    models = httpx.get(f"{OPENAI_API_BASE}/models", timeout=10).json()["data"]
    return {card["id"]: card["owned_by"] for card in models}


def _first_event(query: str) -> dict:
    with connect(f"{REALTIME}{query}", open_timeout=30) as ws:
        return json.loads(ws.recv(timeout=30))


@pytest.mark.integration
class TestPipelines:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime") / "models.yaml"
        config.write_text(yaml.dump({"models": [STT], "pipelines": [VOICE, GHOST]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-pipelines")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    def test_each_pipeline_has_its_app(self):
        apps = serve_apps()
        assert apps[_app(VOICE)]["status"] == "RUNNING"
        assert apps[_app(GHOST)]["status"] == "RUNNING"

    def test_models_lists_the_pipelines_beside_the_models(self):
        listed = _listed()
        assert listed[STT["name"]] == "modelship"
        assert listed["voice"] == "modelship-pipeline"

    def test_a_session_opens_on_a_pipeline_with_its_defaults(self):
        event = _first_event("?model=voice")
        assert event["type"] == "session.created"
        assert event["session"]["type"] == "transcription"
        assert event["session"]["instructions"] == "Be brief."

    def test_a_session_updates(self):
        with connect(f"{REALTIME}?model=voice", open_timeout=30) as ws:
            ws.recv(timeout=30)
            ws.send(json.dumps({"type": "session.update", "session": {"instructions": "Be terse."}}))
            event = json.loads(ws.recv(timeout=30))
        assert event["type"] == "session.updated"
        assert event["session"]["instructions"] == "Be terse."

    def test_a_pipeline_whose_model_is_missing_deploys_and_its_session_is_refused(self):
        event = _first_event("?model=ghost")
        assert event["type"] == "error"
        assert "uses model 'not-deployed'" in event["error"]["message"]

    def test_a_reconcile_without_the_pipelines_removes_them(self, model_deployer):
        model_deployer.deploy(STT["name"])
        assert _app(VOICE) not in serve_apps()
        assert "voice" not in _listed()
        assert _first_event("?model=voice")["error"]["code"] == "model_not_found"


def _until(ws, kind: str) -> list[dict]:
    """Events up to and including the first of *kind*."""
    events = []
    while True:
        events.append(json.loads(ws.recv(timeout=120)))
        if events[-1]["type"] == kind:
            return events


def _send(ws, event: dict) -> None:
    ws.send(json.dumps(event))


def _say(ws, text: str) -> None:
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    _send(ws, {"type": "conversation.item.create", "item": item})
    _until(ws, "conversation.item.done")


def _text(response: dict) -> str:
    messages = [item for item in response["output"] if item["type"] == "message"]
    return "".join(part["text"] for item in messages for part in item["content"])


LIGHTS = {
    "type": "function",
    "name": "turn_on",
    "description": "Turns a room's lights on.",
    "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]},
}


@pytest.mark.integration
class TestTextConversation:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime-chat") / "models.yaml"
        config.write_text(yaml.dump({"models": [LLM], "pipelines": [CHAT]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-chat")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    @pytest.fixture
    def ws(self):
        with connect(f"{REALTIME}?model=chat", open_timeout=30) as ws:
            _until(ws, "conversation.created")
            yield ws

    def test_the_answer_streams_as_text_and_the_next_turn_sees_it(self, ws):
        _say(ws, "My name is Ada.")
        _send(ws, {"type": "response.create"})
        events = _until(ws, "response.done")
        first = events[-1]["response"]
        deltas = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        _say(ws, "What is my name?")
        _send(ws, {"type": "response.create"})
        second = _until(ws, "response.done")[-1]["response"]
        assert (first["status"], second["status"]) == ("completed", "completed")
        assert deltas and deltas == _text(first)
        assert "ada" in _text(second).lower(), _text(second)
        assert second["usage"]["input_tokens"] > first["usage"]["input_tokens"]

    def test_an_out_of_band_answer_stays_out_of_the_conversation(self, ws):
        hello = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}
        _send(ws, {"type": "response.create", "response": {"conversation": "none", "input": [hello]}})
        events = _until(ws, "response.done")
        assert not [event for event in events if event["type"].startswith("conversation.")]
        assert events[-1]["response"]["conversation_id"] is None
        assert _text(events[-1]["response"])

    def test_a_cancelled_answer_ends_cancelled_and_the_session_answers_again(self, ws):
        _say(ws, "Count from 1 to 300, one number per line.")
        _send(ws, {"type": "response.create", "response": {"instructions": "Do exactly as asked."}})
        _until(ws, "response.output_text.delta")
        _send(ws, {"type": "response.cancel"})
        cancelled = _until(ws, "response.done")[-1]["response"]
        _send(ws, {"type": "response.create", "response": {"max_output_tokens": 16}})
        again = _until(ws, "response.done")[-1]["response"]
        assert cancelled["status"] == "cancelled"
        assert again["status"] in ("completed", "incomplete")
        assert _text(again)

    def test_a_forced_tool_call_comes_back_and_its_output_is_answered(self, ws):
        _send(ws, {"type": "session.update", "session": {"tools": [LIGHTS], "tool_choice": "required"}})
        _until(ws, "session.updated")
        _say(ws, "Turn on the kitchen lights.")
        _send(ws, {"type": "response.create"})
        called = _until(ws, "response.done")[-1]["response"]
        calls = [item for item in called["output"] if item["type"] == "function_call"]
        assert calls, called
        assert calls[0]["name"] == "turn_on"
        assert isinstance(json.loads(calls[0]["arguments"]), dict)
        result = {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": '{"ok": true}'}
        _send(ws, {"type": "conversation.item.create", "item": result})
        _until(ws, "conversation.item.done")
        _send(ws, {"type": "response.create", "response": {"tool_choice": "none"}})
        answered = _until(ws, "response.done")[-1]["response"]
        assert answered["status"] == "completed"
        assert _text(answered)


@pytest.mark.integration
class TestAudioInput:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime-audio") / "models.yaml"
        config.write_text(yaml.dump({"models": [STT, TTS], "pipelines": [VOICE]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-audio")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    def test_a_spoken_turn_is_streamed_committed_and_transcribed(self):
        rate, pcm = _speech("Turn on the kitchen lights.")
        with connect(f"{REALTIME}?model=voice", open_timeout=30) as ws:
            _until(ws, "session.created")
            audio_input = {"format": {"type": "audio/pcm", "rate": rate}, "turn_detection": None}
            _send(ws, {"type": "session.update", "session": {"audio": {"input": audio_input}}})
            _until(ws, "session.updated")
            events = _speak_turn(ws, pcm, rate)
        done = [event for event in events if event["type"].startswith("conversation.item.input_audio_transcription")]
        assert [event["type"] for event in done] == ["conversation.item.input_audio_transcription.completed"], done
        assert "kitchen" in done[0]["transcript"].lower(), done[0]["transcript"]


def _speech(text: str, voice: str = "af_bella") -> tuple[int, bytes]:
    """Kokoro saying *text*: its rate and 16-bit PCM."""
    body = {"model": TTS["name"], "voice": voice, "input": text, "response_format": "wav"}
    speech = httpx.post(f"{OPENAI_API_BASE}/audio/speech", json=body, timeout=120).content
    with wave.open(io.BytesIO(speech)) as wav:
        return wav.getframerate(), wav.readframes(wav.getnframes())


def _speak_turn(ws, pcm: bytes, rate: int) -> list[dict]:
    """Streams *pcm* in 100 ms appends, as a client streaming its microphone does, and commits it."""
    step = rate // 10 * 2
    for start in range(0, len(pcm), step):
        _send(ws, {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[start : start + step]).decode()})
    _send(ws, {"type": "input_audio_buffer.commit"})
    return _until(ws, "conversation.item.done")


def _append_all(ws, pcm: bytes, rate: int) -> None:
    """Appends *pcm* in 100 ms pieces, as fast as the socket takes them."""
    step = rate // 10 * 2
    for start in range(0, len(pcm), step):
        _send(ws, {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[start : start + step]).decode()})


def _wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _transcribe(wav: bytes) -> str:
    response = httpx.post(
        f"{OPENAI_API_BASE}/audio/transcriptions",
        data={"model": STT["name"]},
        files={"file": ("answer.wav", wav, "audio/wav")},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["text"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


SPOKEN = {
    "name": "spoken",
    "type": "assistant",
    "assistant_config": {
        "stt": STT["name"],
        "llm": LLM["name"],
        "tts": TTS["name"],
        "session_defaults": {"instructions": "You are a voice assistant. Answer in plain sentences, without lists."},
    },
}
PCM_24K = {"format": {"type": "audio/pcm", "rate": 24000}}
PCM_16K = {"format": {"type": "audio/pcm", "rate": 16000}, "voice": "am_michael", "speed": 1.2}
MU_LAW = {"format": {"type": "audio/pcmu"}, "voice": {"id": "bm_george"}}


@pytest.mark.integration
class TestAudioOutput:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer, tmp_path_factory):
        config = tmp_path_factory.mktemp("realtime-spoken") / "models.yaml"
        config.write_text(yaml.dump({"models": [STT, LLM, TTS], "pipelines": [SPOKEN]}))
        model_deployer.run("--config", str(config), "--reconcile", log_name="realtime-spoken")
        model_deployer.forget()
        yield
        model_deployer.deploy()

    @pytest.mark.parametrize(
        ("question", "voice", "instructions", "output"),
        [
            ("Hello!", "af_bella", "Answer with one short greeting.", PCM_24K),
            ("What colour is the sky on a clear day?", "am_adam", "Answer in one short sentence.", PCM_16K),
            (
                "I'm cooking dinner for six people tonight, two of them are vegetarian, and I only have an hour,"
                " so what could I make?",
                "bf_emma",
                "Answer in four or five full sentences.",
                PCM_24K,
            ),
            (
                "Tell me about the moon: how far away it is, why it has phases, and what the tides have to do with it.",
                "af_bella",
                "Answer in six full sentences.",
                MU_LAW,
            ),
        ],
        ids=["one-word-question", "short-question", "long-question", "long-answer-g711"],
    )
    def test_a_spoken_question_gets_a_spoken_answer_that_says_its_transcript(
        self, question, voice, instructions, output
    ):
        rate, pcm = _speech(question, voice)
        with connect(f"{REALTIME}?model=spoken", open_timeout=30) as ws:
            session = _until(ws, "session.created")[-1]["session"]
            audio_input = {"format": {"type": "audio/pcm", "rate": rate}, "turn_detection": None}
            audio = {"input": audio_input, "output": output}
            _send(ws, {"type": "session.update", "session": {"audio": audio}})
            _until(ws, "session.updated")
            heard = [e for e in _speak_turn(ws, pcm, rate) if e["type"].endswith("transcription.completed")]
            asked = time.monotonic()
            _send(ws, {"type": "response.create", "response": {"instructions": instructions}})
            events, arrived = [], []
            while not events or events[-1]["type"] != "response.done":
                events.append(json.loads(ws.recv(timeout=180)))
                arrived.append(time.monotonic())
        first_audio = next(i for i, e in enumerate(events) if e["type"] == "response.output_audio.delta")
        print(f"\n{question[:30]!r}: first audio after {arrived[first_audio] - asked:.2f} s")
        assert session["output_modalities"] == ["audio"]
        assert heard and heard[0]["transcript"].strip(), heard
        done = events[-1]["response"]
        assert done["status"] == "completed", done
        (part,) = done["output"][0]["content"]
        transcript = part["transcript"]
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_audio_transcript.delta")
        assert (part["type"], deltas) == ("output_audio", transcript)
        spoken = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
        audio_format = output["format"]
        if audio_format["type"] == "audio/pcmu":
            spoken, out_rate = to_pcm16(spoken, "audio/pcmu"), 8000
        else:
            out_rate = audio_format["rate"]
        seconds = len(spoken) / 2 / out_rate
        # Kokoro speaks about 15 characters a second
        assert len(transcript) / 40 < seconds < len(transcript) / 5 + 1, (seconds, transcript)
        said = _transcribe(_wav(spoken, out_rate))
        expected, got = _words(transcript), _words(said)
        if len(expected) <= 3:
            assert set(expected) & set(got), (transcript, said)
        else:
            similarity = difflib.SequenceMatcher(None, expected, got).ratio()
            assert similarity > 0.6, (similarity, transcript, said)

    def test_server_vad_finds_each_spoken_turn_and_answers_it(self):
        questions = (
            "Hello there.",
            "What colour is the sky on a clear day?",
            "I'm cooking dinner for six people tonight, two of them are vegetarian, so what could I make in an hour?",
        )
        with connect(f"{REALTIME}?model=spoken", open_timeout=30) as ws:
            session = _until(ws, "session.created")[-1]["session"]
            starts = []
            for question in questions:
                rate, pcm = _speech(question)
                assert rate == 24000
                # half a second before the speech and a second after it, as fast as the socket takes it
                audio = bytes(rate) + pcm + bytes(rate * 2)
                step = rate // 10 * 2
                for start in range(0, len(audio), step):
                    chunk = base64.b64encode(audio[start : start + step]).decode()
                    _send(ws, {"type": "input_audio_buffer.append", "audio": chunk})
                events = _until(ws, "response.done")
                kinds = [event["type"] for event in events]
                assert kinds.count("input_audio_buffer.speech_started") == 1, (question, kinds)
                assert kinds.count("input_audio_buffer.speech_stopped") == 1, (question, kinds)
                started = next(e for e in events if e["type"] == "input_audio_buffer.speech_started")
                committed = next(e for e in events if e["type"] == "input_audio_buffer.committed")
                assert committed["item_id"] == started["item_id"]
                starts.append(started["audio_start_ms"])
                heard = next(e for e in events if e["type"].endswith("transcription.completed"))["transcript"]
                overlap = set(_words(question)) & set(_words(heard))
                assert len(overlap) >= len(set(_words(question))) / 2, (question, heard)
                done = events[-1]["response"]
                assert done["status"] == "completed", done
                assert done["output"][0]["content"][0]["transcript"].strip()
                assert any(e["type"] == "response.output_audio.delta" for e in events)
        assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"
        # each turn starts later on the session's timeline, its padding before its speech
        assert starts == sorted(starts) and starts[0] >= 200

    def test_talking_over_the_answer_cuts_it_off_and_the_new_question_is_answered(self):
        with connect(f"{REALTIME}?model=spoken", open_timeout=30) as ws:
            _until(ws, "session.created")
            _send(ws, {"type": "session.update", "session": {"instructions": "Answer in six full sentences."}})
            _until(ws, "session.updated")
            rate, question = _speech("Tell me about the moon.")
            _append_all(ws, bytes(rate) + question + bytes(rate * 2), rate)
            before = _until(ws, "response.output_audio.delta")
            # the user talks over the answer
            rate, interruption = _speech("Actually, what colour is the sky on a clear day?")
            _append_all(ws, bytes(rate) + interruption + bytes(rate * 2), rate)
            cut = _until(ws, "response.done")
            answered = _until(ws, "response.done")
            item_id = cut[-1]["response"]["output"][0]["id"]
            received = b"".join(
                base64.b64decode(e["delta"]) for e in before + cut if e["type"] == "response.output_audio.delta"
            )
            # the client played half of what it got, 16-bit at the session's 24 kHz
            played_ms = len(received) // 2 * 1000 // 24000 // 2
            truncate = {"item_id": item_id, "content_index": 0, "audio_end_ms": played_ms}
            _send(ws, {"type": "conversation.item.truncate", **truncate})
            truncated = _until(ws, "conversation.item.truncated")[-1]
            _send(ws, {"type": "conversation.item.retrieve", "item_id": item_id})
            kept = _until(ws, "conversation.item.retrieved")[-1]["item"]["content"][0]["transcript"]
        kinds = [event["type"] for event in cut]
        assert kinds.index("input_audio_buffer.speech_started") < kinds.index("response.done"), kinds
        done = cut[-1]["response"]
        assert (done["status"], done["status_details"]["reason"]) == ("cancelled", "turn_detected"), done
        heard = next(e for e in answered if e["type"].endswith("transcription.completed"))["transcript"]
        assert {"sky", "clear"} & set(_words(heard)), heard
        assert answered[-1]["response"]["status"] == "completed", answered[-1]
        assert truncated["audio_end_ms"] == played_ms
        written = done["output"][0]["content"][0]["transcript"]
        assert written.startswith(kept) and len(kept) < len(written), (kept, written)
