"""NVIDIA NeMo transducers (Parakeet TDT) on sherpa-onnx: the `OfflineRecognizer` for a bundle, and words from
its tokens. The encoder's metadata gives sherpa its feature dimension."""

import os
from typing import Any

from modelship.infer.sherpa_onnx.registry import AsrEntry
from modelship.openai.protocol import TranscriptionWord


def recognizer(entry: AsrEntry, bundle_dir: str, threads: int, provider: str) -> Any:
    """The `OfflineRecognizer` for a NeMo transducer bundle in *bundle_dir*."""
    import sherpa_onnx

    def path(slot: str) -> str:
        return os.path.join(bundle_dir, entry.files[slot])

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=path("encoder"),
        decoder=path("decoder"),
        joiner=path("joiner"),
        tokens=path("tokens"),
        num_threads=threads,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
        provider=provider,
    )


def words(tokens: list[str], timestamps: list[float], durations: list[float], offset: float) -> list[TranscriptionWord]:
    """Words from SentencePiece tokens, *offset* seconds into the audio: a token that starts with a space starts a
    word, any other extends the last."""
    found: list[TranscriptionWord] = []
    for token, start, length in zip(tokens, timestamps, durations, strict=False):
        end = round(offset + start + length, 3)
        if token.startswith((" ", "▁")) or not found:
            text = token.lstrip(" ▁")
            if text:
                found.append(TranscriptionWord(word=text, start=round(offset + start, 3), end=end))
            continue
        last = found[-1]
        found[-1] = TranscriptionWord(word=last.word + token, start=last.start, end=end)
    return found
