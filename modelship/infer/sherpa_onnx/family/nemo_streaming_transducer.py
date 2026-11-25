"""NVIDIA NeMo cache-aware streaming transducers (Nemotron ASR streaming) on sherpa-onnx: the `OnlineRecognizer`
for a bundle, and words from its tokens. The encoder's metadata gives sherpa its feature dimension and chunk."""

import itertools
import os
from typing import Any

from modelship.infer.sherpa_onnx.family import nemo_transducer
from modelship.infer.sherpa_onnx.registry import AsrEntry
from modelship.openai.protocol import TranscriptionWord


def recognizer(entry: AsrEntry, bundle_dir: str, threads: int, provider: str) -> Any:
    """The `OnlineRecognizer` for a NeMo streaming transducer bundle in *bundle_dir*."""
    import sherpa_onnx

    def path(slot: str) -> str:
        return os.path.join(bundle_dir, entry.files[slot])

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=path("encoder"),
        decoder=path("decoder"),
        joiner=path("joiner"),
        tokens=path("tokens"),
        num_threads=threads,
        decoding_method="greedy_search",
        provider=provider,
    )


# an encoder frame: 8x subsampled 10 ms features; a token spans at least one
FRAME_S = 0.08


def words(tokens: list[str], timestamps: list[float], offset: float) -> list[TranscriptionWord]:
    """`nemo_transducer.words` for an online result, which has no token durations: a token lasts until the next one
    starts, the last one a frame."""
    gaps = [later - start for start, later in itertools.pairwise(timestamps)]
    durations = [*gaps, FRAME_S] if timestamps else []
    return nemo_transducer.words(tokens, timestamps, durations, offset)
