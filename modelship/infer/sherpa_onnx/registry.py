"""Curated table of supported sherpa_onnx models, keyed by name. `model:` in
config must be a name here, or a local directory whose basename is one.

No network access and no `import sherpa_onnx` — must be importable without the
wheel installed, same as capabilities.py's LOADER_MODULES.
"""

import os
from typing import NamedTuple

from modelship.utils import is_pathy


class SherpaOnnxRegistryEntry(NamedTuple):
    """`files`/`dirs` keys are sherpa's own `OfflineTtsKokoroModelConfig`
    attribute names, values are paths relative to the bundle root. `lexicon`
    is the list form of sherpa's single comma-joined field. `voice_names[i]`
    is the OpenAI voice name for sid i."""

    tarball_url: str
    sha256: str
    family: str
    usecase: str
    files: dict[str, str]
    dirs: dict[str, str]
    lexicon: tuple[str, ...]
    voice_names: tuple[str, ...]


_RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"

REGISTRY: dict[str, SherpaOnnxRegistryEntry] = {
    "kokoro-en-v0_19": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-en-v0_19.tar.bz2",
        sha256="912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
        family="kokoro",
        usecase="tts",
        files={
            "model": "model.onnx",
            "tokens": "tokens.txt",
            "voices": "voices.bin",
        },
        dirs={
            "data_dir": "espeak-ng-data",
        },
        lexicon=(),
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
    # lexicon order/set matches upstream's own example (kokoro multi-lang needs
    # us-en + zh; gb-en is an alternative to us-en, not additive). dict/ and the
    # *-zh.fst rule files are unused optional extras, left out on purpose.
    "kokoro-multi-lang-v1_0": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-multi-lang-v1_0.tar.bz2",
        sha256="c133d26353d776da730870dac7da07dbfc9a5e3bc80cc5e8e83ab6e823be7046",
        family="kokoro",
        usecase="tts",
        files={
            "model": "model.onnx",
            "tokens": "tokens.txt",
            "voices": "voices.bin",
        },
        dirs={
            "data_dir": "espeak-ng-data",
        },
        lexicon=(
            "lexicon-us-en.txt",
            "lexicon-zh.txt",
        ),
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
        ),
    ),
}


def registry_names() -> tuple[str, ...]:
    return tuple(REGISTRY)


def registry_name(model: str) -> str:
    """The registry key a `model:` value names: itself, or a local directory's basename."""
    return os.path.basename(model.rstrip("/")) if is_pathy(model) else model
