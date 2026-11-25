"""NVIDIA NeMo cache-aware streaming transducers (Nemotron ASR streaming) on sherpa-onnx: the `OnlineRecognizer`
for a bundle, and words from its tokens. The encoder's metadata gives sherpa its feature dimension and chunk."""

import os
from typing import Any

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
    """Words from SentencePiece tokens, *offset* seconds into the audio: a token that starts with a space starts a
    word, any other extends the last. A word ends where the next token starts, the last one a frame after it starts."""
    found: list[TranscriptionWord] = []
    for index, (token, start) in enumerate(zip(tokens, timestamps, strict=False)):
        end = timestamps[index + 1] if index + 1 < len(timestamps) else start + FRAME_S
        if token.startswith((" ", "▁")) or not found:
            text = token.strip(" ▁")
            if text:
                found.append(TranscriptionWord(word=text, start=round(offset + start, 3), end=round(offset + end, 3)))
            continue
        last = found[-1]
        found[-1] = TranscriptionWord(word=last.word + token, start=last.start, end=round(offset + end, 3))
    return found
