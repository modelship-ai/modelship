"""Unit tests for the shared phoneme primitives: token table, text shaping, and
the CJK lexicon. The espeak helper-process recovery runs only where the native
phonemizer is installed."""

import pytest

from modelship.utils.phonemes import lexicon
from modelship.utils.phonemes.japanese import _pieces
from modelship.utils.phonemes.text import normalize_punctuation, split_to_limit
from modelship.utils.phonemes.tokens import TokenTable


def _tokens(tmp_path, symbols):
    path = tmp_path / "tokens.txt"
    path.write_text("".join(f"{s} {i}\n" for i, s in enumerate(symbols)), encoding="utf-8")
    return TokenTable.load(str(path))


class TestTokenTable:
    def test_blank_symbol_is_a_space(self, tmp_path):
        path = tmp_path / "tokens.txt"
        path.write_text(" 0\na 1\n", encoding="utf-8")
        table = TokenTable.load(str(path))
        assert " " in table
        assert table.symbol_to_id[" "] == 0
        assert table.id_to_symbol[1] == "a"

    def test_multi_codepoint_symbol_rejected(self, tmp_path):
        path = tmp_path / "tokens.txt"
        path.write_text("ab 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="single codepoint"):
            TokenTable.load(str(path))

    def test_membership(self, tmp_path):
        table = _tokens(tmp_path, [" ", "a", "b"])
        assert "a" in table
        assert "z" not in table


class TestNormalizePunctuation:
    def test_fullwidth_and_ascii_mapped(self):
        assert normalize_punctuation("a，b。c") == "a,b.c"
        assert normalize_punctuation("a：b") == "a:b"

    def test_ascii_colon_becomes_comma(self):
        assert normalize_punctuation("a:b") == "a,b"

    def test_whitespace_collapsed(self):
        assert normalize_punctuation("a   b\n\tc") == "a b c"


class TestSplitToLimit:
    def test_short_run_is_one_piece(self):
        assert split_to_limit(list("abc"), 10) == [list("abc")]

    def test_prefers_comma_boundary(self):
        symbols = list("aa,bbbb")
        assert split_to_limit(symbols, 5) == [list("aa,"), list("bbbb")]

    def test_falls_back_to_space_and_drops_it(self):
        symbols = list("aaa bbbb")
        assert split_to_limit(symbols, 5) == [list("aaa"), list("bbbb")]

    def test_hard_cut_without_a_boundary(self):
        symbols = list("aaaaaaaa")
        assert split_to_limit(symbols, 5) == [list("aaaaa"), list("aaa")]

    def test_recurses_past_multiple_limits(self):
        symbols = list("aa bb cc dd")
        assert all(len(p) <= 3 for p in split_to_limit(symbols, 3))


class TestLexicon:
    def _lex(self, tmp_path, lines, symbols):
        table = _tokens(tmp_path, symbols)
        path = tmp_path / "lexicon.txt"
        path.write_text("".join(lines), encoding="utf-8")
        return lexicon.load_lexicon(str(path), table)

    def test_first_entry_wins(self, tmp_path):
        words = self._lex(tmp_path, ["好 a b\n", "好 a\n"], [" ", "a", "b"])
        assert words["好"] == "ab"

    def test_entry_with_unknown_symbol_dropped(self, tmp_path):
        words = self._lex(tmp_path, ["好 a z\n", "行 a\n"], [" ", "a", "b"])
        assert "好" not in words
        assert words["行"] == "a"

    def test_segment_longest_match(self, tmp_path):
        words = self._lex(tmp_path, ["你好 a\n", "你 b\n", "好 b\n"], [" ", "a", "b"])
        assert lexicon.segment("你好", words) == ["你好"]

    def test_segment_per_char_fallback(self, tmp_path):
        words = self._lex(tmp_path, ["你 a\n", "好 b\n"], [" ", "a", "b"])
        assert lexicon.segment("你好吗", words) == ["你", "好", "吗"]

    def test_word_phones_falls_back_per_char(self, tmp_path):
        words = self._lex(tmp_path, ["你 a\n", "好 b\n"], [" ", "a", "b"])
        assert lexicon.word_phones("你好", words) == "ab"
        assert lexicon.word_phones("吗", words) == ""


class TestJapanesePieces:
    def test_short_text_is_one_piece(self):
        assert _pieces("今日は晴れ。") == ["今日は晴れ。"]

    def test_long_digit_runs_are_grouped(self):
        assert _pieces("12345678901234") == ["123456789 01234"]

    def test_splits_at_sentence_boundaries_under_the_limit(self):
        text = "あ" * 60 + "。" + "い" * 60 + "。"
        pieces = _pieces(text)
        assert all(len(p) <= 100 for p in pieces)
        assert "".join(pieces) == text

    def test_hard_splits_an_overlong_sentence(self):
        text = "あ" * 250
        pieces = _pieces(text)
        assert all(len(p) <= 100 for p in pieces)
        assert "".join(pieces) == text


class TestEspeakRecovery:
    def test_crash_raises_then_restarts(self):
        pytest.importorskip("piper_phonemize")
        from modelship.utils.phonemes.espeak import Espeak, EspeakError

        espeak = Espeak()
        try:
            assert "".join(espeak.phonemize("hello", "en-us")[0])
            with pytest.raises(EspeakError):
                espeak.phonemize(":-ॠ", "es")
            assert "".join(espeak.phonemize("hello", "en-us")[0])
        finally:
            espeak.close()
