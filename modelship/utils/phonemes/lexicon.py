"""Word-to-phoneme lexicon plus greedy longest-match segmentation, for CJK
sherpa-onnx TTS frontends (kokoro multi-lang reads lexicon-zh this way)."""

from modelship.utils.phonemes.tokens import TokenTable

_MAX_WORD_CHARS = 10


def load_lexicon(path: str, tokens: TokenTable) -> dict[str, str]:
    """`word -> phoneme-symbol string`. The first entry for a word wins, and an
    entry naming a symbol absent from `tokens` is dropped whole."""
    words: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *phones = line.split()
            word = word.lower()
            if word in words or not all(p in tokens for p in phones):
                continue
            words[word] = "".join(phones)
    return words


def segment(run: str, words: dict[str, str]) -> list[str]:
    segments: list[str] = []
    i = 0
    while i < len(run):
        for end in range(min(i + _MAX_WORD_CHARS, len(run)), i + 1, -1):
            if run[i:end] in words:
                segments.append(run[i:end])
                i = end
                break
        else:
            segments.append(run[i])
            i += 1
    return segments


def word_phones(word: str, words: dict[str, str]) -> str:
    if word in words:
        return words[word]
    return "".join(words[char] for char in word if char in words)
