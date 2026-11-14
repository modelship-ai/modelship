import httpx
import pytest

from tests.conftest import MODEL_CONFIGS, OPENAI_API_BASE, serve_apps

LONG_TEXT = " ".join(["The quick brown fox jumps over the lazy dog while the band plays on."] * 20)


def _replica(model: str) -> tuple[str, int]:
    (app,) = (app for name, app in serve_apps().items() if name.startswith(f"modelship.{model}-"))
    (deployment,) = app["deployments"].values()
    (replica,) = (replica for replica in deployment["replicas"] if replica["state"] == "RUNNING")
    return replica["replica_id"], replica["pid"]


def _post(path: str, timeout: float, **kwargs) -> httpx.Response:
    return httpx.post(f"{OPENAI_API_BASE}{path}", timeout=timeout, **kwargs)


def _dropped_post(path: str, after_s: float, **kwargs) -> None:
    with pytest.raises(httpx.TimeoutException):
        _post(path, after_s, **kwargs)


def _dropped_stream(path: str, **kwargs) -> None:
    with httpx.stream("POST", f"{OPENAI_API_BASE}{path}", timeout=120, **kwargs) as resp:
        assert resp.status_code == 200
        next(line for line in resp.iter_lines() if line.startswith("data: "))


def _audio(model: str, audio: bytes, **fields: str) -> dict:
    return {"data": {"model": model, **fields}, "files": {"file": ("speech.wav", audio, "audio/wav")}}


def _assert_served_by_the_same_replica(model: str, drop, send) -> httpx.Response:
    before = _replica(model)
    drop()
    resp = send()
    assert resp.status_code == 200, resp.text
    assert _replica(model) == before
    return resp


def _assert_an_image_after_a_dropped_one(model: str, after_s: float) -> None:
    body = {"model": model, "prompt": "a red bicycle", "size": "512x512"}

    resp = _assert_served_by_the_same_replica(
        model,
        lambda: _dropped_post("/images/generations", after_s, json=body),
        lambda: _post("/images/generations", 600, json=body),
    )
    assert resp.json()["data"][0]["b64_json"]


@pytest.mark.integration
@pytest.mark.whispercpp
@pytest.mark.sherpa_onnx
@pytest.mark.stable_diffusion_cpp
class TestTheRequestAfterADroppedOne:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy_raw(
            [MODEL_CONFIGS["stt-cpp-model"], MODEL_CONFIGS["tts-model"], MODEL_CONFIGS["image-cpu-model"]]
        )

    @pytest.fixture(scope="class")
    def long_speech(self) -> bytes:
        resp = _post("/audio/speech", 600, json={"model": "tts-model", "voice": "af_bella", "input": LONG_TEXT})
        resp.raise_for_status()
        return resp.content

    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
    def test_a_transcription(self, long_speech, stream):
        audio = _audio("stt-cpp-model", long_speech)

        def drop():
            if stream:
                _dropped_stream("/audio/transcriptions", **_audio("stt-cpp-model", long_speech, stream="true"))
            else:
                _dropped_post("/audio/transcriptions", 0.5, **audio)

        resp = _assert_served_by_the_same_replica(
            "stt-cpp-model", drop, lambda: _post("/audio/transcriptions", 300, **audio)
        )
        assert "quick brown fox" in resp.json()["text"].lower()

    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
    def test_a_speech(self, stream):
        long = {"model": "tts-model", "voice": "af_bella", "input": LONG_TEXT}

        def drop():
            if stream:
                _dropped_stream("/audio/speech", json={**long, "stream_format": "sse"})
            else:
                _dropped_post("/audio/speech", 0.5, json=long)

        resp = _assert_served_by_the_same_replica(
            "tts-model",
            drop,
            lambda: _post("/audio/speech", 300, json={"model": "tts-model", "voice": "af_bella", "input": "Hello."}),
        )
        assert resp.content[:4] == b"RIFF"

    def test_an_image(self):
        _assert_an_image_after_a_dropped_one("image-cpu-model", after_s=2.0)


@pytest.mark.integration
@pytest.mark.diffusers
@pytest.mark.cuda
class TestTheDiffusersImageAfterADroppedOne:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        diffusers_config = {"num_inference_steps": 30, "guidance_scale": 0.0}
        model_deployer.deploy_raw([{**MODEL_CONFIGS["image-model"], "diffusers_config": diffusers_config}])

    def test_an_image(self):
        _assert_an_image_after_a_dropped_one("image-model", after_s=0.4)
