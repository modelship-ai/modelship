"""Curated table of supported sherpa_onnx models, keyed by name. `model:` in
config must be a name here, or a local directory whose basename is one.

No network access and no `import sherpa_onnx` — must be importable without the
wheel installed.
"""

import os
from typing import NamedTuple

from modelship.utils import is_pathy


class SherpaOnnxRegistryEntry(NamedTuple):
    """A text-to-speech bundle. `files`/`dirs` keys are sherpa's own `OfflineTtsKokoroModelConfig`
    attribute names, values are paths relative to the bundle root.
    `chinese_lexicon` is the Chinese lexicon the frontend reads, relative to the
    bundle root (None when the bundle has none). `voice_names[i]` is the OpenAI
    voice name for sid i."""

    tarball_url: str
    sha256: str
    family: str
    usecase: str
    files: dict[str, str]
    dirs: dict[str, str]
    chinese_lexicon: str | None
    voice_names: tuple[str, ...]


class StreamFeed(NamedTuple):
    """How a streaming bundle's stream is fed. The zeros it takes, measured: before the first audio, or it drops the
    first words, and after the last, or it drops the last one; and the audio it decodes at a time."""

    lead_ms: int
    tail_ms: int
    chunk_ms: int


class AsrEntry(NamedTuple):
    """A speech-to-text bundle. `family` names its module in `family/`; `files` maps the files that family reads to
    paths relative to the bundle root. `stream` is set on a streaming bundle, None on an offline one."""

    tarball_url: str
    sha256: str
    family: str
    usecase: str
    files: dict[str, str]
    stream: StreamFeed | None = None


# copies of k2-fsa/sherpa-onnx's tts-models and asr-models assets; a file name ends in its sha256 prefix
_BUNDLES = "https://github.com/modelship-ai/model-bundles/releases/download"
_RELEASE_BASE = f"{_BUNDLES}/tts/kokoro"

REGISTRY: dict[str, SherpaOnnxRegistryEntry | AsrEntry] = {
    "kokoro-en-v0_19": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-en-v0_19-91280485.tar.bz2",
        sha256="912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
        family="kokoro",
        usecase="tts",
        files={
            "model": "model.onnx",
            "tokens": "tokens.txt",
            "voices": "voices.bin",
        },
        dirs={},
        chinese_lexicon=None,
        voice_names=(
            "af",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ),
    ),
    "kokoro-multi-lang-v1_0": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-multi-lang-v1_0-c5f7e2d2.tar.bz2",
        sha256="c5f7e2d2caf082bc1d20fb70334a61d99d20b484500aad32e7cf84c128ea3298",
        family="kokoro",
        usecase="tts",
        files={
            "model": "model.onnx",
            "tokens": "tokens.txt",
            "voices": "voices.bin",
        },
        dirs={},
        chinese_lexicon="lexicon-zh.txt",
        voice_names=(
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
            "ef_dora",
            "em_alex",
            "ff_siwis",
            "hf_alpha",
            "hf_beta",
            "hm_omega",
            "hm_psi",
            "if_sara",
            "im_nicola",
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
            "pf_dora",
            "pm_alex",
            "pm_santa",
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
            # sid 53 in the bundle, out of alphabetical order
            "em_santa",
        ),
    ),
    # NVIDIA's parakeet-tdt-0.6b-v3 (CC-BY-4.0): 25 European languages, punctuated; takes and reports no language
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8": AsrEntry(
        tarball_url=f"{_BUNDLES}/asr/parakeet/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8-1495e91b.tar.bz2",
        sha256="1495e91b59d7fc53ddca91f3431758ec99f8fa253fdb0e1e851be0f72b14b0f9",
        family="nemo_transducer",
        usecase="transcription",
        files={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
    ),
    # NVIDIA's nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1), 160 ms chunks: 35 languages, punctuated; takes a
    # language per stream ("auto" by default) and reports none
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11": AsrEntry(
        tarball_url=(
            f"{_BUNDLES}/asr/nemotron/"
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11-a81909a1.tar.bz2"
        ),
        sha256="a81909a1780d84cff16d73c15e13e67d9d81d8839faf14870d507d8499f7a61a",
        family="nemo_streaming_transducer",
        usecase="transcription",
        files={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        # a word cut off at the audio's end needs up to ~420 ms of quiet after it (Kokoro clips, 3 voices), plus
        # up to a chunk for where it ends on the chunk grid
        stream=StreamFeed(lead_ms=500, tail_ms=640, chunk_ms=160),
    ),
}


def is_streaming(model: str) -> bool:
    """Whether *model* names a streaming speech-to-text bundle."""
    entry = REGISTRY.get(registry_name(model))
    return isinstance(entry, AsrEntry) and entry.stream is not None


def registry_names() -> tuple[str, ...]:
    return tuple(REGISTRY)


def registry_name(model: str) -> str:
    """The registry key a `model:` value names: itself, or a local directory's basename."""
    return os.path.basename(model.rstrip("/")) if is_pathy(model) else model
