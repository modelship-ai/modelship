"""Text splitting and phoneme-chunk shaping shared by sherpa-onnx TTS frontends."""

import re

_PUNCTUATION_REPLACEMENTS = [
    ("，", ","),
    (":", ","),
    ("、", ","),
    ("；", ";"),
    ("：", ":"),
    ("。", "."),
    ("？", "?"),
    ("！", "!"),
]
_WHITESPACE = re.compile(r"\s+")

CHINESE_RUN = re.compile(r"[一-鿿]+")
RUNS = re.compile(r"[一-鿿]+|[^一-鿿]+")
PUNCTUATION = {";", ":", ",", ".", "!", "?", "—", "…", '"', "(", ")", "“", "”"}

_BREAK_AFTER = {",", ";", ":"}


def normalize_punctuation(text: str) -> str:
    for old, new in _PUNCTUATION_REPLACEMENTS:
        text = text.replace(old, new)
    return _WHITESPACE.sub(" ", text)


def split_to_limit(symbols: list[str], limit: int) -> list[list[str]]:
    """Break a phoneme-symbol run into pieces of at most `limit` symbols, cutting
    after the last comma/semicolon/colon within reach, else at the last space,
    else hard at `limit`."""
    out: list[list[str]] = []
    while len(symbols) > limit:
        cut, drop = _last_break(symbols, limit)
        out.append(symbols[:cut])
        symbols = symbols[cut + drop :]
    out.append(symbols)
    return out


def _last_break(symbols: list[str], limit: int) -> tuple[int, int]:
    for i in range(limit - 1, -1, -1):
        if symbols[i] in _BREAK_AFTER:
            return i + 1, 0
    for i in range(limit - 1, 0, -1):
        if symbols[i] == " ":
            return i, 1
    return limit, 0
