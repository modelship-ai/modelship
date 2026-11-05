"""Unit tests for the kokoro text frontend against a synthetic token table and
lexicon with a fake espeak/Japanese backend — the run splitting, merge, Chinese
lookup, Japanese routing, per-voice language, and the 400-codepoint cap."""

from modelship.infer.sherpa_onnx.family.kokoro import KokoroFrontend, espeak_language, is_japanese
from modelship.utils.phonemes.tokens import TokenTable

_SYMBOLS = [" ", ",", ".", "!", "?", ";", ":", *"abcdefghijklmnopqrstuvwxyz"]


class _FakeEspeak:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def phonemize(self, text, voice):
        self.calls.append((text, voice))
        sentences, current = [], ""
        for ch in text:
            current += ch
            if ch in ".!?":
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)
        return [list(s) for s in sentences] or [[]]

    def close(self):
        pass


class _FakeJapanese:
    def __init__(self):
        self.calls: list[str] = []

    def phonemize(self, text):
        self.calls.append(text)
        return text.replace("你", "x").replace("好", "y")


def _tokens(tmp_path) -> TokenTable:
    path = tmp_path / "tokens.txt"
    path.write_text("".join(f"{s} {i}\n" for i, s in enumerate(_SYMBOLS)), encoding="utf-8")
    return TokenTable.load(str(path))


def _lexicon(tmp_path) -> str:
    path = tmp_path / "lexicon-zh.txt"
    path.write_text("你好 a b\n你 a\n好 b\n吗 c\n", encoding="utf-8")
    return str(path)


def _symbols(frontend, text, voice):
    return ["".join(chr(c) for c in chunk) for chunk in frontend.chunks(text, voice)]


class TestLanguageSelection:
    def test_prefix_to_language(self):
        assert espeak_language("af_bella") == "en-us"
        assert espeak_language("bf_emma") == "en"
        assert espeak_language("ef_dora") == "es"
        assert espeak_language("ff_siwis") == "fr"
        assert espeak_language("if_sara") == "it"
        assert espeak_language("pf_dora") == "pt-BR"
        assert espeak_language("zf_xiaobei") == "en-us"

    def test_japanese_detection(self):
        assert is_japanese("jf_alpha")
        assert not is_japanese("af_bella")


class TestEnglishBundle:
    def test_espeak_over_whole_text_in_the_voice_language(self, tmp_path):
        espeak = _FakeEspeak()
        fe = KokoroFrontend(_tokens(tmp_path), None, espeak=espeak, japanese=_FakeJapanese())
        assert _symbols(fe, "ab cd", "bf_emma") == ["ab cd"]
        assert espeak.calls == [("ab cd", "en")]

    def test_space_added_after_period_and_unknown_dropped(self, tmp_path):
        espeak = _FakeEspeak()
        fe = KokoroFrontend(_tokens(tmp_path), None, espeak=espeak, japanese=_FakeJapanese())
        # digits are not in the token table -> dropped; '.' ends the sentence
        assert _symbols(fe, "a1b.", "af_bella") == ["ab. "]

    def test_long_run_splits_at_a_space_under_the_limit(self, tmp_path):
        espeak = _FakeEspeak()
        fe = KokoroFrontend(_tokens(tmp_path), None, espeak=espeak, japanese=_FakeJapanese())
        text = "a" * 250 + " " + "b" * 250
        chunks = _symbols(fe, text, "af_bella")
        assert [len(c) for c in chunks] == [250, 250]


class TestMultiLangBundle:
    def _frontend(self, tmp_path):
        self.espeak = _FakeEspeak()
        self.japanese = _FakeJapanese()
        return KokoroFrontend(_tokens(tmp_path), _lexicon(tmp_path), espeak=self.espeak, japanese=self.japanese)

    def test_chinese_run_uses_the_lexicon(self, tmp_path):
        fe = self._frontend(tmp_path)
        assert _symbols(fe, "你好吗", "zf_xiaobei") == ["abc"]
        assert self.espeak.calls == []

    def test_non_chinese_run_uses_espeak(self, tmp_path):
        fe = self._frontend(tmp_path)
        _symbols(fe, "你好 hi", "zf_xiaobei")
        assert self.espeak.calls == [(" hi", "en-us")]

    def test_japanese_voice_bypasses_runs(self, tmp_path):
        fe = self._frontend(tmp_path)
        assert _symbols(fe, "你好 hi", "jf_alpha") == ["xy hi"]
        assert self.japanese.calls == ["你好 hi"]
        assert self.espeak.calls == []

    def test_spanish_voice_phonemizes_in_spanish(self, tmp_path):
        fe = self._frontend(tmp_path)
        _symbols(fe, "hola", "ef_dora")
        assert self.espeak.calls == [("hola", "es")]


class TestMerge:
    def test_short_pieces_join_the_previous_chunk(self, tmp_path):
        # "hi" (len 2 < 3) is a short non-Chinese run that joins the Chinese piece.
        fe = KokoroFrontend(_tokens(tmp_path), _lexicon(tmp_path), espeak=_FakeEspeak(), japanese=_FakeJapanese())
        assert _symbols(fe, "你好hi", "zf_xiaobei") == ["abhi"]


class TestChineseSplit:
    def test_chinese_run_capped_at_the_limit(self, tmp_path):
        path = tmp_path / "lexicon-zh.txt"
        path.write_text("好 a\n", encoding="utf-8")
        fe = KokoroFrontend(_tokens(tmp_path), str(path), espeak=_FakeEspeak(), japanese=_FakeJapanese())
        chunks = fe.chunks("好" * 500, "zf_xiaobei")
        assert all(len(c) <= 400 for c in chunks)
        assert sum(len(c) for c in chunks) == 500
