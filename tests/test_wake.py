"""The wake word detector on pymicro-wakeword's own built-in models: how the replica checks a pipeline's words, and
what a session's stream hears in audio fed in pieces of any size."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from modelship.utils import wake
from modelship.utils.config_schema import WakeWord
from modelship.utils.wake import WakeWords

RATE = 16000


def _noise(seconds: float, seed: int = 0) -> bytes:
    samples = np.random.default_rng(seed).normal(0, 3000, int(seconds * RATE))
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _hits(words: WakeWords, audio: bytes, chunk: int) -> list[float]:
    stream = words.stream(1000.0)
    found: list[float] = []
    for start in range(0, len(audio), chunk * 2):
        found += stream.feed(audio[start : start + chunk * 2])
    return found


def _custom(tmp_path: Path, **changes) -> Path:
    """A copy of the built-in hey_jarvis as a custom word, its manifest changed by *changes*."""
    import pymicro_wakeword

    models = Path(pymicro_wakeword.__file__).parent / "models"
    manifest = json.loads((models / "hey_jarvis.json").read_text())
    shutil.copy(models / manifest["model"], tmp_path / manifest["model"])
    manifest.update(changes)
    path = tmp_path / "hey_custom.json"
    path.write_text(json.dumps(manifest))
    return path


class TestWakeWords:
    def test_every_built_in_word_loads(self):
        WakeWords([WakeWord(model=name) for name in ("okay_nabu", "hey_jarvis", "hey_mycroft", "alexa")])

    def test_silence_and_noise_wake_nobody(self):
        words = WakeWords([WakeWord(model="okay_nabu"), WakeWord(model="hey_jarvis")])
        assert _hits(words, bytes(RATE * 2) + _noise(3), 1600) == []

    @pytest.mark.parametrize("chunk", [1, 159, 160, 161, 2560, 4000])
    def test_what_it_hears_doesnt_depend_on_how_the_audio_is_cut(self, chunk):
        # a cutoff noise crosses, so there is something to hear
        words = WakeWords([WakeWord(model="hey_jarvis", cutoff=0.0001)])
        audio = _noise(2, seed=1)
        whole = _hits(words, audio, len(audio) // 2)
        assert whole
        assert _hits(words, audio, chunk) == whole

    def test_a_hit_ends_with_the_step_its_audio_ends_on_the_streams_timeline(self):
        words = WakeWords([WakeWord(model="hey_jarvis", cutoff=0.0001)])
        hits = _hits(words, _noise(2, seed=1), 1600)
        # a feature covers 30 ms, one every 10 ms, from the stream's start
        assert all(1000.0 + 30 <= hit <= 1000.0 + 2000 for hit in hits)
        assert all((hit - 1030) % 10 == 0 for hit in hits)

    def test_a_custom_word_is_its_manifest_with_its_model_beside_it(self, tmp_path):
        WakeWords([WakeWord(model=str(_custom(tmp_path)))])

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            ({"version": 1}, "not a microWakeWord v2 manifest"),
            ({"type": "openwakeword"}, "not a microWakeWord v2 manifest"),
            ({"model": "missing.tflite"}, "isn't beside it"),
        ],
    )
    def test_a_custom_word_that_isnt_usable_is_refused(self, tmp_path, changes, problem):
        with pytest.raises(ValueError, match=problem):
            WakeWords([WakeWord(model=str(_custom(tmp_path, **changes)))])

    def test_a_manifest_that_isnt_there_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="can't be read"):
            WakeWords([WakeWord(model=str(tmp_path / "nope.json"))])

    def test_the_bootstrappers_library_is_the_one_loaded(self, monkeypatch):
        loaded: list = []

        class Model:
            probability_cutoff = 0.97

            @staticmethod
            def from_config(manifest, libtensorflowlite_c_path=None):
                loaded.append(libtensorflowlite_c_path)
                return Model()

        import pymicro_wakeword

        monkeypatch.setattr(pymicro_wakeword, "MicroWakeWord", Model)
        monkeypatch.setenv(wake.TFLITE_LIB_ENV, "/opt/tflite/libtensorflowlite_c.dylib")
        words = WakeWords([WakeWord(model="hey_jarvis", cutoff=0.5)])
        assert loaded == ["/opt/tflite/libtensorflowlite_c.dylib"]
        assert words.stream(0.0)._models[0].probability_cutoff == 0.5
