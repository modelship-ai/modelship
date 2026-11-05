"""Kokoro text frontend: text to phoneme-codepoint chunks for sherpa-onnx 2.x.

Ports sherpa-onnx 1.13.4's kokoro frontend (removed upstream with espeak in
2.0). Chinese runs read the lexicon; Japanese voices use a dedicated G2P;
everything else is phonemized by espeak in the voice's own language. Chunks are
held at or below sherpa's own 400-codepoint split so the break falls on a word."""

from modelship.utils.phonemes import lexicon
from modelship.utils.phonemes.espeak import Espeak
from modelship.utils.phonemes.japanese import Japanese
from modelship.utils.phonemes.text import CHINESE_RUN, PUNCTUATION, RUNS, normalize_punctuation, split_to_limit
from modelship.utils.phonemes.tokens import TokenTable

_LIMIT = 400

_LANGUAGE = {"a": "en-us", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt-BR", "z": "en-us"}


def espeak_language(voice: str) -> str:
    return _LANGUAGE.get(voice[:1], "en-us")


def is_japanese(voice: str) -> bool:
    return voice[:1] == "j"


class KokoroFrontend:
    def __init__(
        self,
        tokens: TokenTable,
        chinese_lexicon: str | None,
        espeak: Espeak | None = None,
        japanese: Japanese | None = None,
    ):
        self.tokens = tokens
        self.words = lexicon.load_lexicon(chinese_lexicon, tokens) if chinese_lexicon else None
        self.espeak = espeak or Espeak()
        self.japanese = japanese or Japanese()

    def close(self) -> None:
        self.espeak.close()

    def chunks(self, text: str, voice: str) -> list[list[int]]:
        pieces = self._pieces(text, voice)
        chunks = [chunk for piece in pieces for chunk in split_to_limit(piece, _LIMIT)]
        return [[ord(symbol) for symbol in chunk] for chunk in chunks if chunk]

    def _pieces(self, text: str, voice: str) -> list[list[str]]:
        if self.words is None:
            return self._espeak(text, voice)
        if is_japanese(voice):
            return [self._filter(self.japanese.phonemize(text))]
        return self._multilang(text, voice)

    def _espeak(self, text: str, voice: str) -> list[list[str]]:
        return [self._espeak_symbols(sentence) for sentence in self.espeak.phonemize(text, espeak_language(voice))]

    def _espeak_symbols(self, sentence: list[str]) -> list[str]:
        out: list[str] = []
        for phoneme in sentence:
            for symbol in phoneme:
                if symbol not in self.tokens:
                    continue
                out.append(symbol)
                if symbol == ".":
                    out.append(" ")
        return out

    def _multilang(self, text: str, voice: str) -> list[list[str]]:
        assert self.words is not None
        language = espeak_language(voice)
        merged: list[list[str]] = []
        for run in RUNS.findall(normalize_punctuation(text)):
            if CHINESE_RUN.fullmatch(run):
                pieces = self._chinese(run)
            elif run in PUNCTUATION:
                pieces = [[run]]
            else:
                pieces = [self._espeak_symbols(sentence) for sentence in self.espeak.phonemize(run, language)]
            for piece in pieces:
                _merge(merged, piece)
        return merged

    def _chinese(self, run: str) -> list[list[str]]:
        assert self.words is not None
        pieces: list[list[str]] = []
        current: list[str] = []
        for word in lexicon.segment(run, self.words):
            phones = lexicon.word_phones(word, self.words)
            if not phones:
                continue
            if len(current) + len(phones) > _LIMIT:
                pieces.append(current)
                current = []
            current.extend(phones)
        if current:
            pieces.append(current)
        return pieces

    def _filter(self, phonemes: str) -> list[str]:
        return [symbol for symbol in phonemes if symbol in self.tokens]


def _merge(merged: list[list[str]], piece: list[str]) -> None:
    if len(piece) > 10 or not merged:
        merged.append(piece)
    elif len(merged[-1]) + len(piece) < 46 or len(piece) < 3:
        merged[-1].extend(piece)
    else:
        merged.append(piece)
