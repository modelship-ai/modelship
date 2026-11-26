"""Smart Turn against the real model, fetched from model-bundles, on speech the cluster's kokoro model synthesizes:
finished sentences are heard finished, a sentence cut off at "and" isn't. `test_smart_turn.py` covers its input."""

import io

import numpy as np
import pytest
import soundfile as sf

from modelship.utils.audio import to_pcm16
from modelship.utils.smart_turn import SmartTurnModel, smart_turn_source
from modelship.utils.sources import download_model_source, file

_FINISHED = (
    "Turn on the kitchen lights.",
    "What's the weather like in Paris tomorrow?",
    "Set a timer for ten minutes.",
)
_CUT_OFF = "Turn on the kitchen lights and"


@pytest.fixture(scope="module")
def smart_turn(tmp_path_factory) -> SmartTurnModel:
    cache = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(file, "cache_dir", lambda: str(cache))
        return SmartTurnModel(download_model_source(smart_turn_source()))


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSmartTurn:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model")

    def _turn(self, client, text: str) -> tuple[bytes, int]:
        """*text* spoken, with the 200 ms pause semantic VAD classifies after, as 16-bit PCM and its rate."""
        wav = client.audio.speech.create(model="tts-model", voice="af_bella", input=text, response_format="wav")
        audio, rate = sf.read(io.BytesIO(wav.content), dtype="float32")
        return to_pcm16(np.concatenate([audio, np.zeros(rate // 5, dtype=np.float32)])), rate

    @pytest.mark.parametrize("text", _FINISHED)
    def test_a_finished_sentence_is_heard_finished(self, smart_turn, client, text):
        assert smart_turn.end_probability(*self._turn(client, text)) > 0.5

    def test_a_sentence_cut_off_at_and_is_heard_unfinished(self, smart_turn, client):
        assert smart_turn.end_probability(*self._turn(client, _CUT_OFF)) < 0.5
