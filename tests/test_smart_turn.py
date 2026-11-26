"""Smart Turn's input: the Whisper log-mel features, checked against transformers' own extractor, and the 8 s of a
turn the model hears. `test_smart_turn_integration.py` runs the real model."""

import numpy as np
import pytest

from modelship.utils.smart_turn import last_8s
from modelship.utils.whisper_features import SAMPLES, log_mel


def _audio() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLES // 2) / 16000
    return {
        "noise": rng.normal(0, 0.1, SAMPLES).astype(np.float32),
        "padded tone": np.concatenate([np.zeros(SAMPLES // 2), 0.3 * np.sin(2 * np.pi * 220 * t)]).astype(np.float32),
        "silence": np.zeros(SAMPLES, dtype=np.float32),
    }


class TestLogMel:
    @pytest.mark.parametrize("name", list(_audio()))
    def test_the_features_match_transformers_whisper_extractor(self, name):
        transformers = pytest.importorskip("transformers")
        audio = _audio()[name]
        extractor = transformers.WhisperFeatureExtractor(chunk_length=8)
        expected = extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=SAMPLES,
            truncation=True,
            do_normalize=True,
        ).input_features[0]
        features = log_mel(audio)
        assert features.shape == expected.shape == (80, 800)
        assert features.dtype == np.float32
        np.testing.assert_allclose(features, expected, atol=1e-4)

    def test_the_features_take_exactly_8_s(self):
        with pytest.raises(ValueError, match="expected 128000 samples"):
            log_mel(np.zeros(SAMPLES - 1, dtype=np.float32))


class TestLast8s:
    def test_a_short_turn_is_padded_with_silence_before_it(self):
        turn = np.ones(16000, dtype=np.float32)
        heard = last_8s(turn)
        assert len(heard) == SAMPLES
        assert not heard[:-16000].any() and heard[-16000:].all()

    def test_a_long_turn_keeps_its_end(self):
        turn = np.arange(SAMPLES + 500, dtype=np.float32)
        heard = last_8s(turn)
        assert len(heard) == SAMPLES and heard[0] == 500 and heard[-1] == SAMPLES + 499
