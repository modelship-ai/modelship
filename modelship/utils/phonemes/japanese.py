"""Japanese grapheme-to-phoneme via misaki's Cutlet (fugashi + UniDic).

Loaded on first use. Input is fed in small pieces: the tagger's romaji pass is
roughly cubic in piece length, and its number reader rejects runs of ten or more
digits, so long digit runs are grouped first."""

import re
import threading

_SENTENCE = re.compile(r".*?[。．！？!?]|.+", re.S)
_LONG_DIGITS = re.compile(r"\d{10,}")
_MAX_PIECE_CHARS = 100


class Japanese:
    def __init__(self):
        self._lock = threading.Lock()
        self._g2p = None

    def phonemize(self, text: str) -> str:
        with self._lock:
            if self._g2p is None:
                from misaki.cutlet import Cutlet

                self._g2p = Cutlet()
            return " ".join(self._g2p(piece)[0] for piece in _pieces(text))


def _pieces(text: str) -> list[str]:
    text = _LONG_DIGITS.sub(lambda m: " ".join(re.findall(r"\d{1,9}", m.group())), text)
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE.findall(text):
        if len(sentence) > _MAX_PIECE_CHARS:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(sentence[i : i + _MAX_PIECE_CHARS] for i in range(0, len(sentence), _MAX_PIECE_CHARS))
        elif len(current) + len(sentence) > _MAX_PIECE_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces
