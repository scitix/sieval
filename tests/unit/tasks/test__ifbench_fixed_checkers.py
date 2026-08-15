"""The four repaired IFBench checkers.

Each repair is asserted twice, and the pairing is the point:

* **reduces to upstream** — on an input upstream already handled, the repaired
  checker returns what upstream returns. A repair that quietly changed unrelated
  verdicts would be a second, unmeasured divergence riding along with the fix.
* **changes the verdict** — on the input that exposes the defect, it does not.
  Without this direction a "repair" that no-ops everywhere would pass the suite
  while the measured delta silently came from somewhere else.

The defect inputs are written out rather than lifted from a run: a test that only
replays responses a particular model produced stops testing the checker the day
the model changes, and two of these four are repairs whose defect that model
never happened to exercise.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest

from sieval.community.ifbench import instructions as upstream
from sieval.community.ifbench.instructions_registry import INSTRUCTION_DICT
from sieval.tasks._ifbench_fixed_checkers import (
    _fixed_checker_classes,
    fixed_ifbench_registry,
)

# The four subclasses are built inside a function, not at module scope, so that
# importing the repair module does not import the vendored fork -- `sieval/tasks`
# is walked wholesale by `import_all_tasks`, private modules included. Naming
# them here is what triggers the build, which is exactly the contract.
FIXED_CHECKERS = _fixed_checker_classes()
IndentStairsCheckerFixed = FIXED_CHECKERS["format:line_indent"]
SentTypeRatioCheckerFixed = FIXED_CHECKERS["ratio:sentence_type"]
WordsPositionCheckerFixed = FIXED_CHECKERS["words:words_position"]
SingleVowelParagraphCheckerFixed = FIXED_CHECKERS["words:vowel"]


def _check(cls, value, **kwargs):
    instruction = cls("id")
    instruction.build_description(**kwargs)
    return instruction.check_following(value)


def _both(fixed_cls, value, **kwargs):
    """(upstream verdict, repaired verdict) for the same input."""
    return (
        _check(fixed_cls.__mro__[1], value, **kwargs),
        _check(fixed_cls, value, **kwargs),
    )


# --------------------------------------------------------------------------
# format:line_indent -- blank lines removed while iterating
# --------------------------------------------------------------------------

_STAIRS = "alpha\n beta\n  gamma"


def test_line_indent_reduces_to_upstream_on_inputs_without_blank_lines():
    for value in (_STAIRS, "alpha\nbeta", " alpha\nbeta", "only one line"):
        up, fixed = _both(IndentStairsCheckerFixed, value)
        assert up == fixed, value


def test_line_indent_verdict_no_longer_depends_on_how_many_blank_lines():
    # Upstream's `for line in lines: lines.remove(line)` skips an element per
    # removal, so ONE blank line is removed and TWO leaves one behind at indent
    # 0, which breaks the chain. Same stair shape, opposite verdict.
    one_blank = "alpha\n\n beta\n  gamma"
    two_blank = "alpha\n\n\n beta\n  gamma"
    assert _check(upstream.IndentStairsChecker, one_blank) is True
    assert _check(upstream.IndentStairsChecker, two_blank) is False
    assert _check(IndentStairsCheckerFixed, one_blank) is True
    assert _check(IndentStairsCheckerFixed, two_blank) is True


def test_line_indent_still_rejects_a_response_that_does_not_indent():
    # The repair removes a formatting artefact, not the constraint. A preamble
    # is two consecutive lines at indent 0, which is what the instruction forbids.
    assert _check(IndentStairsCheckerFixed, f"Sure!\n{_STAIRS}") is False
    assert _check(IndentStairsCheckerFixed, "alpha\n beta\n gamma") is False


# --------------------------------------------------------------------------
# ratio:sentence_type -- endswith('.') and a vacuous 0 == 0
# --------------------------------------------------------------------------

_TWO_TO_ONE = "The sky is blue. Rain falls often. Is that so?"


def test_sentence_type_reduces_to_upstream_when_nothing_is_quoted():
    for value in (
        _TWO_TO_ONE,
        "Rain falls often. Why though?",
        "The sky is blue. Rain falls often. Birds fly south. Is that so?",
    ):
        up, fixed = _both(SentTypeRatioCheckerFixed, value)
        assert up == fixed, value


def test_sentence_type_counts_a_quoted_declarative():
    # A declarative that closes with a quote mark is still a declarative; only
    # `endswith('.')` says otherwise.
    value = 'He said "the sky is blue." Rain falls often. Is that so?'
    assert _check(upstream.SentTypeRatioChecker, value) is False
    assert _check(SentTypeRatioCheckerFixed, value) is True


def test_sentence_type_counts_a_quoted_interrogative():
    # The same repair, on the other count. Asserted separately because it is a
    # separate reachable divergence from upstream and the notes owe every one:
    # upstream tests `endswith('?')` just as blindly, and repairing only the
    # declarative side would compare a quote-aware count against a quote-blind
    # one.
    value = 'She said "Really?" He nodded. It is fine.'
    assert _check(upstream.SentTypeRatioChecker, value) is False
    assert _check(SentTypeRatioCheckerFixed, value) is True


def test_sentence_type_no_longer_passes_a_response_with_no_ratio_at_all():
    # `0 == 2 * 0` is True: upstream passes a response that engaged with the
    # instruction not at all. A false PASS is the direction that inflates a
    # score and the direction nobody goes looking for.
    value = "Wow! Amazing! Incredible!"
    assert _check(upstream.SentTypeRatioChecker, value) is True
    assert _check(SentTypeRatioCheckerFixed, value) is False


def test_sentence_type_keeps_the_exact_ratio_reading():
    # Deliberately NOT repaired: the instruction says "a 2:1 ratio", so 3:1 is
    # not it. A tolerance would invent a claim the item never made -- that is a
    # prompt-side change, which a verifier repair is not licensed to make.
    three_to_one = "The sky is blue. Rain falls often. Birds fly south. Is that so?"
    assert _check(SentTypeRatioCheckerFixed, three_to_one) is False


# --------------------------------------------------------------------------
# words:words_position -- positions counted over tokens, not words
# --------------------------------------------------------------------------

_KW = {"keyword": "whisper"}


def test_words_position_reduces_to_upstream_on_single_trailing_punctuation():
    for value in (
        "The whisper carried a whisper home.",
        "The whisper carried a shout home.",
        "Every whisper fades before the whisper ends.",
    ):
        up, fixed = _both(WordsPositionCheckerFixed, value, **_KW)
        assert up == fixed, value


def test_words_position_survives_two_trailing_punctuation_marks():
    # Upstream's `words[-3]` is right for exactly ONE trailing punctuation token
    # (the token list is the word list plus one, so token[-3] IS word[-2]) and
    # off by one for two. Same sentence, verdict decided by its final mark.
    value = "The whisper is a whisper here!?"
    assert _check(upstream.WordsPositionChecker, value, **_KW) is False
    assert _check(WordsPositionCheckerFixed, value, **_KW) is True


def test_words_position_no_longer_passes_a_response_at_the_wrong_position():
    # The same off-by-one in the other direction, and the worse one: upstream
    # reads word[-1] as if it were word[-2] and PASSES a response whose second
    # to last word is "a". A repair that only ever turned FAILs into PASSes
    # would be relaxing the constraint rather than correcting it.
    value = "The whisper is a whisper!?"
    assert _check(upstream.WordsPositionChecker, value, **_KW) is True
    assert _check(WordsPositionCheckerFixed, value, **_KW) is False


def test_words_position_survives_punctuation_near_the_front():
    # The same defect from the other end: `words[1]` is the comma.
    value = "Well, whisper drifts and then whisper stops."
    assert _check(upstream.WordsPositionChecker, value, **_KW) is False
    assert _check(WordsPositionCheckerFixed, value, **_KW) is True


def test_words_position_still_rejects_the_wrong_word():
    assert _check(WordsPositionCheckerFixed, "The shout is a shout!?", **_KW) is False
    assert _check(WordsPositionCheckerFixed, "whisper", **_KW) is False


# --------------------------------------------------------------------------
# words:vowel -- paragraphs counted by newline
# --------------------------------------------------------------------------

_ONE_PARA = "the beet needs sun"


def test_vowel_reduces_to_upstream_on_a_genuinely_single_line():
    for value in (_ONE_PARA, "the quick brown fox jumped over a lazy dog"):
        up, fixed = _both(SingleVowelParagraphCheckerFixed, value)
        assert up == fixed, value


def test_vowel_accepts_a_soft_wrapped_paragraph():
    # `split('\n')` calls every newline a paragraph break, so a wrapped
    # paragraph is rejected on line count before its vowels are ever examined.
    value = "the beet needs\nsun"
    assert _check(upstream.SingleVowelParagraphChecker, value) is False
    assert _check(SingleVowelParagraphCheckerFixed, value) is True


@pytest.mark.parametrize("blank", ["\n\n", "\r\n\r\n", "\n \n", "\n\t\n", "\n\xa0\n"])
def test_vowel_still_rejects_two_paragraphs(blank):
    # The single-paragraph requirement is declared by the item, so it stays;
    # only the definition of a break changes. Parametrized over how a blank line
    # can be spelled because this repair is the one that decides what a break
    # *is*: a pattern matching only `\n[ \t]*\n` leaves a CRLF response's
    # paragraphs joined, and two paragraphs then pass. Asserting the claim for
    # one separator is what let that through.
    value = f"the beet{blank}needs sun"
    assert _check(SingleVowelParagraphCheckerFixed, value) is False, repr(value)


def test_vowel_still_rejects_a_fourth_vowel():
    assert (
        _check(SingleVowelParagraphCheckerFixed, "the beet needs\nsunlight to grow")
        is False
    )


# --------------------------------------------------------------------------
# the registry overlay
# --------------------------------------------------------------------------


def test_registry_overlays_exactly_the_four_repaired_ids():
    registry = fixed_ifbench_registry()
    changed = {k for k, v in registry.items() if INSTRUCTION_DICT.get(k) is not v}
    assert changed == set(FIXED_CHECKERS)
    assert registry.keys() == INSTRUCTION_DICT.keys()


def test_registry_is_a_fresh_dict_and_never_the_vendored_global():
    # Mutating the shared registry would change how the unqualified task grades
    # in the same session -- including for samples already in flight.
    first = fixed_ifbench_registry()
    assert first is not INSTRUCTION_DICT
    assert first is not fixed_ifbench_registry()
    assert INSTRUCTION_DICT["format:line_indent"] is upstream.IndentStairsChecker


def test_every_repair_subclasses_the_checker_it_replaces():
    # Subclassing is what keeps `build_description` and the description pattern
    # upstream's by construction: a repair cannot change what an item ASKS.
    for instruction_id, fixed_cls in FIXED_CHECKERS.items():
        assert issubclass(fixed_cls, INSTRUCTION_DICT[instruction_id])
        own = set(vars(fixed_cls))
        assert not own & {"build_description", "get_instruction_args"}


def test_registry_refuses_to_mount_a_repair_whose_id_moved_upstream(monkeypatch):
    # After a re-vendor that renamed an id, a silent overlay would land on a key
    # nothing looks up and `_fixed` would grade identically while still claiming
    # a delta. Failing loudly here is the only place that is visible.
    # Patched at the vendored registry itself rather than at a name this module
    # re-exports: that is where a re-vendor would actually move the id, and the
    # overlay reads it fresh on every call.
    monkeypatch.setattr(
        "sieval.community.ifbench.instructions_registry.INSTRUCTION_DICT",
        {k: v for k, v in INSTRUCTION_DICT.items() if k != "words:vowel"},
    )
    with pytest.raises(KeyError, match="words:vowel"):
        fixed_ifbench_registry()


def test_registry_refuses_a_repair_whose_base_class_moved_upstream(monkeypatch):
    # The id survives a re-vendor but is re-bound to a different class: the
    # subclass would reintroduce the behaviour of a checker upstream dropped.
    monkeypatch.setattr(
        "sieval.community.ifbench.instructions_registry.INSTRUCTION_DICT",
        {**INSTRUCTION_DICT, "words:vowel": upstream.IndentStairsChecker},
    )
    with pytest.raises(TypeError, match="words:vowel"):
        fixed_ifbench_registry()
