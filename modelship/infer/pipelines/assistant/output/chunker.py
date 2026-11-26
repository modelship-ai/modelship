"""Cutting an answer's text into the pieces its speech says, as the LLM writes it: the first piece short, at a sentence
or a clause, so audio starts soon, and the pieces after it whole sentences. Pure text functions; `speech.py` speaks
the pieces."""

import re

# the fewest characters a piece holds, unless it ends the answer
MIN_CHARS = 20
# the most the first piece holds, and the most any other does
FIRST_CHARS = 80
MAX_CHARS = 200

_SENTENCE_ENDS = frozenset(".!?…")
_CJK_SENTENCE_ENDS = frozenset("。！？")
_CLAUSE_MARKS = frozenset(",;:")
_EM_DASH = "—"
_DASHES = frozenset(_EM_DASH + "–")  # noqa: RUF001
_CJK_CLAUSE_MARKS = frozenset("，、；：")
# quotes, brackets and guillemets
_CLOSERS = frozenset("\"')]”’»」』")  # noqa: RUF001
_OPENERS = "\"'([“‘«「『"  # noqa: RUF001
# words a period follows without ending the sentence
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "approx", "inc", "ltd", "co"}
)

_MARKDOWN = (
    # code fences, with their language
    (re.compile(r"```[^\n]*"), " "),
    # links and images, as their text
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # rules
    (re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE), " "),
    # headings, quotes and bullets
    (re.compile(r"^\s{0,3}(#{1,6}|>+|[-*+])\s+", re.MULTILINE), ""),
    (re.compile(r"\*+|`+|~~"), ""),
    # underscores around a word, not inside one
    (re.compile(r"(?<!\w)_+(?=\S)|(?<=\S)_+(?!\w)"), ""),
    (re.compile(r"\|"), " "),
)


def speakable(text: str) -> str:
    """*text* without its Markdown; empty when nothing in it can be spoken."""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    return text if any(char.isalnum() for char in text) else ""


def _ends_sentence(text: str, period: int) -> bool:
    """Whether the period at *period* ends a sentence: not after an abbreviation, an initial or a number."""
    start = max(text.rfind(" ", 0, period), text.rfind("\n", 0, period)) + 1
    word = text[start:period].lstrip(_OPENERS).lower()
    if not word or word in _ABBREVIATIONS or word.isdigit():
        return False
    # "J." and "U.S."
    return not (len(word.replace(".", "")) <= 3 and (len(word) == 1 or "." in word))


def separator(piece: str) -> str:
    """What goes between *piece* and the piece after it: nothing after a mark the text runs on from unspaced."""
    end = piece.rstrip("".join(_CLOSERS))[-1:]
    unspaced = end in _CJK_SENTENCE_ENDS or end in _CJK_CLAUSE_MARKS or end == _EM_DASH
    return "" if unspaced else " "


class Chunker:
    """Cuts streamed text into pieces to speak: the first short, at a sentence or clause, the rest at sentences."""

    def __init__(self) -> None:
        self._text = ""
        self._first = True

    def push(self, text: str) -> list[str]:
        """The pieces *text* completes."""
        self._text += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """The pieces left once the text is all written."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        pieces = []
        while self._text:
            cut = self._cut(final)
            if cut is None:
                break
            piece, self._text = self._text[:cut].strip(), self._text[cut:]
            if piece:
                pieces.append(piece)
                self._first = False
        return pieces

    def _boundaries(self, final: bool) -> list[tuple[int, bool]]:
        """Where the text can be cut, and whether each ends a sentence. A boundary at the end of the text counts
        only once the text is final, since more might follow it."""
        text = self._text
        found = []
        for i, char in enumerate(text):
            if char == "\n":
                found.append((i + 1, True))
                continue
            sentence = char in _SENTENCE_ENDS or char in _CJK_SENTENCE_ENDS
            clause = char in _CLAUSE_MARKS or char in _DASHES or char in _CJK_CLAUSE_MARKS
            if not sentence and not clause:
                continue
            end = i + 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            if end == len(text):
                if final:
                    found.append((end, sentence))
                continue
            # Latin punctuation and the en dash end at a space: "3.14", "1,000" and "10-20" don't
            unspaced = char in _CJK_SENTENCE_ENDS or char in _CJK_CLAUSE_MARKS or char == _EM_DASH
            if not unspaced and not text[end].isspace():
                continue
            if char == "." and not _ends_sentence(text, i):
                continue
            found.append((end, sentence))
        return found

    def _cut(self, final: bool) -> int | None:
        text = self._text
        limit = FIRST_CHARS if self._first else MAX_CHARS
        boundaries = [
            (end, sentence) for end, sentence in self._boundaries(final) if len(text[:end].strip()) >= MIN_CHARS
        ]
        for end, sentence in boundaries:
            if end > limit:
                break
            if sentence or self._first:
                return end
        if len(text) > limit:
            # a long run without a sentence end: its last clause, else its last word
            clauses = [end for end, _ in boundaries if end <= limit]
            if clauses:
                return clauses[-1]
            space = max(text.rfind(" ", 0, limit + 1), text.rfind("\n", 0, limit + 1))
            return space if space > 0 and len(text[:space].strip()) >= MIN_CHARS else limit
        if final:
            return len(text)
        return None
