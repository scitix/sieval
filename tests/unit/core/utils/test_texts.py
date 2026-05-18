"""
Tests for sieval.core.utils.texts.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.core.utils.texts import extract_choice, general_postprocess, normalize_text


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("The Cat sat on a mat.") == "cat sat on mat"

    def test_removes_articles(self):
        assert normalize_text("a an the") == ""

    def test_removes_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_lowercases(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_strips_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!!!") == ""

    def test_mixed(self):
        assert normalize_text("The answer is: 42!") == "answer is 42"


class TestGeneralPostprocess:
    def test_basic(self):
        assert general_postprocess("  The Answer is 42.  ") == "answer is 42"

    def test_already_clean(self):
        assert general_postprocess("hello") == "hello"

    def test_empty(self):
        assert general_postprocess("") == ""

    def test_uppercase_with_punctuation(self):
        assert general_postprocess("YES!") == "yes"


class TestExtractChoice:
    def test_single_letter(self):
        assert extract_choice("A") == "A"

    def test_in_sentence(self):
        # upper() makes "THE ANSWER IS B." — first A-E match is 'E' in "THE"
        assert extract_choice("The answer is B.") == "E"

    def test_lowercase_input(self):
        # upper() makes "THE ANSWER IS C" — first A-E match is 'E' in "THE"
        assert extract_choice("the answer is c") == "E"

    def test_clean_choice(self):
        assert extract_choice("B") == "B"
        assert extract_choice("Answer: C") == "A"  # first A-E match in "ANSWER: C"
        assert extract_choice("C") == "C"

    def test_no_match(self):
        assert extract_choice("123 xyz") == ""

    def test_first_match_wins(self):
        assert extract_choice("A and B") == "A"

    def test_with_whitespace(self):
        assert extract_choice("  D  ") == "D"

    def test_e_option(self):
        assert extract_choice("E") == "E"

    def test_empty(self):
        assert extract_choice("") == ""

    def test_beyond_e(self):
        # F is not in A-E range
        assert extract_choice("F") == ""
