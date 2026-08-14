"""Unit tests for the three repaired IFEval-family checkers.

Every repair is asserted twice: once that it *reduces to upstream* on the inputs
upstream already handled — which is what makes the divergence exactly the defect
and not a rewrite riding along with it — and once that it changes the verdict on
the input that exposed the defect.

`langdetect.detect` is randomized and SiEval never seeds it, so the
`change_case:english_capital` tests stub the detector rather than calling it. A
test that flips with the process RNG would be no evidence at all, and the point
of that repair is precisely that upstream calls the detector on input it cannot
read.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import random

import langdetect
import pytest
from langdetect.lang_detect_exception import ErrorCode, LangDetectException

from sieval.community.instruction_following_eval import (
    instructions as ifeval_upstream,
)
from sieval.community.instruction_following_eval import (
    instructions_registry as ifeval_registry,
)
from sieval.community.instruction_following_eval_fixed import (
    ENGLISH_CAPITAL,
    FIXED_INSTRUCTION_IDS,
    LETTER_FREQUENCY,
    NTH_PARAGRAPH_FIRST_WORD,
    IFEvalCapitalLettersEnglishCheckerFixed,
    IFEvalLetterFrequencyCheckerFixed,
    IFEvalParagraphFirstWordCheckFixed,
    MultiIFCapitalLettersEnglishCheckerFixed,
    MultiIFLetterFrequencyCheckerFixed,
    MultiIFParagraphFirstWordCheckFixed,
    _build,
    fixed_ifeval_registry,
    fixed_multi_if_registry,
)
from sieval.community.multi_if import ifeval as multi_if_upstream

# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "upstream"),
    [
        (fixed_ifeval_registry, ifeval_registry.INSTRUCTION_DICT),
        (fixed_multi_if_registry, multi_if_upstream.INSTRUCTION_DICT),
    ],
    ids=["ifeval", "multi_if"],
)
def test_registry_replaces_exactly_the_three_declared_checkers(build, upstream):
    fixed = build()
    # Same ids: a repair may not add or drop a constraint type, or the two tasks
    # would differ in *what* they grade rather than in how.
    assert set(fixed) == set(upstream)
    differing = {key for key, cls in fixed.items() if cls is not upstream[key]}
    assert differing == set(FIXED_INSTRUCTION_IDS)


@pytest.mark.parametrize(
    ("build", "upstream"),
    [
        (fixed_ifeval_registry, ifeval_registry.INSTRUCTION_DICT),
        (fixed_multi_if_registry, multi_if_upstream.INSTRUCTION_DICT),
    ],
    ids=["ifeval", "multi_if"],
)
def test_registry_is_a_fresh_dict_and_never_the_vendored_global(build, upstream):
    # Samples grade concurrently. A registry that aliased or mutated the global
    # would change how the *unqualified* task grades in the same session, which
    # is the one thing the `_fixed` split exists to prevent.
    before = dict(upstream)
    first, second = build(), build()
    assert first is not upstream
    assert first is not second
    first["punctuation:no_comma"] = object
    assert upstream == before
    assert upstream[NTH_PARAGRAPH_FIRST_WORD] is before[NTH_PARAGRAPH_FIRST_WORD]


def test_build_refuses_to_mount_a_fix_under_an_unknown_id():
    # The failure this guards against is silent: after an upstream rename the
    # overlay would land on a key nothing looks up, and `_fixed` would grade
    # identically to the unqualified task while still claiming a delta.
    with pytest.raises(KeyError, match="renamed"):
        _build({"a": int}, {"b": str})


def test_build_names_the_offending_id():
    with pytest.raises(KeyError) as excinfo:
        _build({"a": int}, {"length_constraints:nth_paragraph_first_word": str})
    assert "length_constraints:nth_paragraph_first_word" in str(excinfo.value)


@pytest.mark.parametrize(
    ("fixed_cls", "upstream_cls"),
    [
        (IFEvalParagraphFirstWordCheckFixed, ifeval_upstream.ParagraphFirstWordCheck),
        (IFEvalLetterFrequencyCheckerFixed, ifeval_upstream.LetterFrequencyChecker),
        (
            IFEvalCapitalLettersEnglishCheckerFixed,
            ifeval_upstream.CapitalLettersEnglishChecker,
        ),
        (
            MultiIFParagraphFirstWordCheckFixed,
            multi_if_upstream.ParagraphFirstWordCheck,
        ),
        (MultiIFLetterFrequencyCheckerFixed, multi_if_upstream.LetterFrequencyChecker),
        (
            MultiIFCapitalLettersEnglishCheckerFixed,
            multi_if_upstream.CapitalLettersEnglishChecker,
        ),
    ],
)
def test_each_fixed_checker_subclasses_the_one_it_replaces(fixed_cls, upstream_cls):
    # Everything not named by a mixin -- `build_description`'s defaults and
    # validation, `get_instruction_args`, the description pattern -- must stay
    # upstream's, and subclassing is what makes that true by construction rather
    # than by review.
    assert issubclass(fixed_cls, upstream_cls)


# ---------------------------------------------------------------------------
# length_constraints:nth_paragraph_first_word
# ---------------------------------------------------------------------------


def _paragraph_pair(num_paragraphs: int, nth: int, first_word: str):
    fixed = IFEvalParagraphFirstWordCheckFixed(NTH_PARAGRAPH_FIRST_WORD)
    upstream = ifeval_upstream.ParagraphFirstWordCheck(NTH_PARAGRAPH_FIRST_WORD)
    for checker in (fixed, upstream):
        checker.build_description(
            num_paragraphs=num_paragraphs, nth_paragraph=nth, first_word=first_word
        )
    return fixed, upstream


# Responses with no blank `\n\n` chunk and a single-token target: exactly the
# inputs upstream already handles, where the repair must be invisible.
_UNAFFECTED_RESPONSES = [
    "alpha one\n\nbeta two\n\ngamma three",
    "alpha one\n\nWRONG two\n\ngamma three",
    'alpha one\n\n"beta" two\n\ngamma three',
    "alpha one\n\nbeta. two\n\ngamma three",
    "alpha one\n\nbeta two",
    "alpha one\n\nbeta two\n\ngamma three\n\ndelta four",
    "  alpha one\n\n  beta two\n\n  gamma three",
    "alpha",
]


@pytest.mark.parametrize("response", _UNAFFECTED_RESPONSES)
@pytest.mark.parametrize("nth", [1, 2, 3])
def test_paragraph_first_word_reduces_to_upstream_when_no_chunk_is_blank(response, nth):
    fixed, upstream = _paragraph_pair(num_paragraphs=3, nth=nth, first_word="beta")
    assert fixed.check_following(response) == upstream.check_following(response)


def test_paragraph_first_word_indexes_the_list_it_counted():
    # The defect in one line: an empty leading chunk is excluded from the count
    # but still occupies index 0, so paragraph 2 is read out of slot 2 of the
    # *unfiltered* list -- which is paragraph 1.
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    fixed, upstream = _paragraph_pair(num_paragraphs=3, nth=2, first_word="beta")
    assert upstream.check_following(response) is False
    assert fixed.check_following(response) is True


@pytest.mark.parametrize(
    "response",
    [
        # A blank line before the nth paragraph, wherever it comes from:
        "alpha one\n\n\n\nbeta two\n\ngamma three",  # a doubled break mid-response
        "alpha one\n\n   \n\nbeta two\n\ngamma three",  # a whitespace-only chunk
        "\n\nalpha one\n\nbeta two\n\ngamma three",  # a leading blank line
    ],
)
def test_paragraph_first_word_is_shifted_by_any_blank_chunk_not_just_a_leading_one(
    response,
):
    # Worth pinning separately: the defect is easy to describe as an artifact of
    # responses that open with a blank line, and it is not -- any blank chunk
    # *before* the nth paragraph does the same thing. (A blank chunk after it is
    # harmless, since both readings then agree on the index and on the count.)
    fixed, upstream = _paragraph_pair(num_paragraphs=3, nth=2, first_word="beta")
    assert upstream.check_following(response) is False
    assert fixed.check_following(response) is True


def test_paragraph_first_word_is_unaffected_by_a_trailing_blank_chunk():
    # The counting loop already discounts it and no index moves, so upstream and
    # the repair agree -- pinned so the repair's scope is bounded from both
    # sides rather than only demonstrated where it bites.
    response = "alpha one\n\nbeta two\n\ngamma three\n\n"
    fixed, upstream = _paragraph_pair(num_paragraphs=3, nth=2, first_word="beta")
    assert upstream.check_following(response) is True
    assert fixed.check_following(response) is True


def test_paragraph_first_word_still_fails_the_wrong_word():
    # The repair is not a blanket pass: it moves *which* paragraph is read, and
    # that paragraph must still start with what was asked for.
    response = "\nalpha one\n\nWRONG two\n\ngamma three"
    fixed, _upstream = _paragraph_pair(num_paragraphs=3, nth=2, first_word="beta")
    assert fixed.check_following(response) is False


def test_paragraph_first_word_still_fails_the_wrong_paragraph_count():
    # `num_paragraphs` is counted the same way it always was; only the indexing
    # was reconciled with it.
    response = "\nalpha one\n\nbeta two"
    fixed, _upstream = _paragraph_pair(num_paragraphs=3, nth=2, first_word="beta")
    assert fixed.check_following(response) is False


def test_paragraph_first_word_compares_a_multi_token_value_in_order():
    # A multi-token value cannot equal `paragraph.split()[0]`, so upstream
    # returns False for *every* possible response -- the slot measures nothing.
    fixed, upstream = _paragraph_pair(num_paragraphs=2, nth=2, first_word="once upon")
    good = "alpha one\n\nonce upon a time"
    assert upstream.check_following(good) is False
    assert fixed.check_following(good) is True

    # Still an opening, still in order: neither a bag of words nor a prefix.
    assert fixed.check_following("alpha one\n\nupon once a time") is False
    assert fixed.check_following("alpha one\n\nonce a time") is False
    assert fixed.check_following("alpha one\n\na once upon time") is False


def test_paragraph_first_word_normalises_every_token_of_a_multi_token_value():
    # `_leading_word` is upstream's own normalisation, applied per token rather
    # than to the first token only -- quotes and trailing punctuation are handled
    # the same way at position 2 as at position 1.
    fixed, _upstream = _paragraph_pair(num_paragraphs=1, nth=1, first_word="once upon")
    assert fixed.check_following('"Once upon" a time') is True


def test_paragraph_first_word_with_a_blank_value_stays_ungradeable():
    # Three Multi-IF slots ship an empty `first_word`. A constraint with no value
    # states nothing to check, so inventing a rule for it would be worse than
    # upstream's behaviour: the repair deliberately leaves these failing, exactly
    # as upstream does, rather than passing everything.
    fixed, upstream = _paragraph_pair(num_paragraphs=1, nth=1, first_word="")
    for response in ("alpha one", "", "   "):
        assert fixed.check_following(response) == upstream.check_following(response)
        assert fixed.check_following(response) is False


# ---------------------------------------------------------------------------
# keywords:letter_frequency
# ---------------------------------------------------------------------------


def _frequency_pair(letter, frequency=2, relation="at least"):
    fixed = IFEvalLetterFrequencyCheckerFixed(LETTER_FREQUENCY)
    upstream = ifeval_upstream.LetterFrequencyChecker(LETTER_FREQUENCY)
    descriptions = [
        checker.build_description(
            letter=letter, let_frequency=frequency, let_relation=relation
        )
        for checker in (fixed, upstream)
    ]
    return fixed, upstream, descriptions


@pytest.mark.parametrize("letter", ["a", "z", "Q"])
@pytest.mark.parametrize("relation", ["at least", "less than"])
def test_letter_frequency_reduces_to_upstream_for_an_ascii_letter(letter, relation):
    fixed, upstream, (fixed_desc, upstream_desc) = _frequency_pair(
        letter, relation=relation
    )
    assert fixed_desc == upstream_desc
    assert fixed._letter == upstream._letter
    for response in ("aardvark", "quiz", "", "ZZZ"):
        assert fixed.check_following(response) == upstream.check_following(response)


@pytest.mark.parametrize("character", ["#", "!", "1"])
def test_letter_frequency_keeps_the_character_the_item_names(character):
    fixed, upstream, (fixed_desc, _upstream_desc) = _frequency_pair(character)
    assert fixed._letter == character
    assert character in fixed_desc
    # Upstream silently graded a different character -- one it drew itself.
    assert upstream._letter != character
    assert fixed.check_following(f"{character}{character} and more") is True
    assert fixed.check_following("no such character here") is False


def test_letter_frequency_grades_the_same_letter_every_time():
    # Upstream draws freshly *per call*, so the same item is graded against a
    # different letter each time it is scored. Seeded so the comparison is a
    # fact about the two implementations rather than about today's RNG.
    random.seed(20260814)
    upstream_letters = set()
    fixed_letters = set()
    for _ in range(20):
        fixed, upstream, _desc = _frequency_pair("#")
        upstream_letters.add(upstream._letter)
        fixed_letters.add(fixed._letter)
    assert fixed_letters == {"#"}
    assert len(upstream_letters) > 1


def test_letter_frequency_consumes_no_random_draw():
    # A grading fix has no business perturbing the global RNG stream: every other
    # checker that defaults an argument draws from it, so consuming a value here
    # would shift verdicts in constraints this module does not touch. (Built
    # alone, not through `_frequency_pair` -- upstream draws, which is the whole
    # point.)
    random.seed(20260814)
    before = random.getstate()
    checker = IFEvalLetterFrequencyCheckerFixed(LETTER_FREQUENCY)
    checker.build_description(letter="#", let_frequency=2, let_relation="at least")
    assert checker._letter == "#"
    assert random.getstate() == before


@pytest.mark.parametrize("letter", [None, "", "ab", "  "])
def test_letter_frequency_defers_to_upstream_for_an_unusable_value(letter):
    # Not one character after stripping: the item states nothing gradable, so
    # upstream's fallback stands rather than this module inventing a rule.
    fixed, _upstream, (fixed_desc, _upstream_desc) = _frequency_pair(letter)
    assert len(fixed._letter) == 1
    assert "a" <= fixed._letter <= "z"
    assert fixed._letter in fixed_desc


def test_letter_frequency_lowercases_a_cased_substitute_like_upstream():
    fixed, upstream, _desc = _frequency_pair("Q")
    assert fixed._letter == "q" == upstream._letter


# ---------------------------------------------------------------------------
# change_case:english_capital
# ---------------------------------------------------------------------------


def _capital_pair():
    fixed = IFEvalCapitalLettersEnglishCheckerFixed(ENGLISH_CAPITAL)
    upstream = ifeval_upstream.CapitalLettersEnglishChecker(ENGLISH_CAPITAL)
    for checker in (fixed, upstream):
        checker.build_description()
    return fixed, upstream


def test_english_capital_detects_on_a_case_folded_copy(monkeypatch):
    seen: list[str] = []

    def _detect(text):
        seen.append(text)
        return "en"

    monkeypatch.setattr(langdetect, "detect", _detect)
    fixed, upstream = _capital_pair()
    response = "THIS ENTIRE RESPONSE IS IN CAPITAL LETTERS."

    assert upstream.check_following(response) is True
    assert fixed.check_following(response) is True
    # Upstream hands the detector ALL-CAPS text, which every profile it ships is
    # off-distribution for; the repair hands it text the profiles were built on.
    assert seen == [response, response.lower()]


def test_english_capital_does_not_modify_the_response(monkeypatch):
    monkeypatch.setattr(langdetect, "detect", lambda _text: "en")
    fixed, _upstream = _capital_pair()
    response = "ALL CAPS HERE."
    fixed.check_following(response)
    assert response == "ALL CAPS HERE."


def test_english_capital_leaves_the_capitals_requirement_to_upstream(monkeypatch):
    # `isupper()` is upstream's, still first, and still short-circuiting: the set
    # of responses that reach the detector is exactly the set upstream sends
    # there, so this repair cannot turn a mixed-case response into a pass.
    calls: list[str] = []
    monkeypatch.setattr(langdetect, "detect", lambda text: calls.append(text) or "en")
    fixed, upstream = _capital_pair()
    for response in ("Mixed Case Text.", "lower case text.", "1234 !!"):
        assert fixed.check_following(response) is False
        assert upstream.check_following(response) is False
    assert calls == []


def test_english_capital_fails_capitals_in_another_language(monkeypatch):
    monkeypatch.setattr(langdetect, "detect", lambda _text: "de")
    fixed, _upstream = _capital_pair()
    assert fixed.check_following("DIES IST EIN DEUTSCHER SATZ.") is False


def test_english_capital_counts_an_undetectable_text_as_following(monkeypatch):
    # Parity with upstream, kept deliberately: `isupper()` above makes it
    # near-unreachable, and a repair that quietly tightened it would be a second
    # divergence riding along with the measured one.
    def _raise(_text):
        raise LangDetectException(ErrorCode.CantDetectError, "no features")

    monkeypatch.setattr(langdetect, "detect", _raise)
    fixed, upstream = _capital_pair()
    response = "ABC DEF."
    assert fixed.check_following(response) is True
    assert upstream.check_following(response) is True


def test_english_capital_matches_upstream_whenever_detection_agrees(monkeypatch):
    # The only input whose handling changes is the *argument* to the detector.
    # Hold the detector constant and the two implementations are the same
    # function -- which is what makes the measured delta attributable to
    # detection quality rather than to a changed rule.
    monkeypatch.setattr(langdetect, "detect", lambda _text: "en")
    fixed, upstream = _capital_pair()
    for response in ("ALL CAPS.", "Mixed.", "", "   ", "123"):
        assert fixed.check_following(response) == upstream.check_following(response)
