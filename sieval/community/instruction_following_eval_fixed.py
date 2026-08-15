"""Repaired checkers for the IFEval instruction family.

Three of the 25 IFEval checkers grade something other than what their own
instruction says. This module holds the repairs, as mixins that override one
method each, plus the registries that mount them. Nothing here changes what an
item *asks*: every repair is to the code that decides whether an answer complied.

**Why this is original code in ``community/``, which the directory's CLAUDE.md
asks to be argued for.** The same three checkers are vendored twice — once under
``instruction_following_eval/`` (google-research) and once under ``multi_if/``
(Meta's copy of the same file) — and their bodies are logic-identical, differing
only in one import path (``instructions_util.generate_keywords`` against a flat
``generate_keywords``) and one logging call, neither of which a repair touches.
So this is the ``_sympy_guards.py`` situation exactly: holding the fix *outside*
both vendored packages serves upstream alignment rather than working against it,
because the alternative is two copies of every repair inside files whose value is
being diffable against upstream — "a fix duplicated into two files is a fix that
will eventually exist in only one of them".

**Nothing here is reachable from the unqualified tasks.** ``ifeval_0shot_gen``
and ``multi_if_0shot_gen`` keep grading through the vendored registry, byte for
byte; the repairs are visible only to the ``_fixed`` siblings, which say so in
their names. A faithful port stays available for reproducing published numbers,
and a repaired one for measuring a model.

The one change inside the vendored files is a keyword-only ``instruction_dict``
parameter on their four grader functions, defaulting to the vendored registry.
It was preferred over re-implementing the grading loops here: those loops are
~25 lines each of upstream logic, and a copy would drift silently the next time
the vendored files are re-synced, whereas a parameter cannot.

Every repair below is the narrowest one that fixes the defect, and each reduces
to upstream's own expression on the inputs upstream already handled — the tests
assert that reduction rather than asserting the repair in isolation.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import re
from collections.abc import Mapping

import langdetect

from sieval.community.instruction_following_eval import (
    instructions as _ifeval_upstream,
)
from sieval.community.instruction_following_eval import (
    instructions_registry as _ifeval_registry,
)
from sieval.community.multi_if import ifeval as _multi_if_upstream

#: The three instruction ids whose checker this module replaces. Declared rather
#: than derived so a rename upstream fails loudly at registry-build time instead
#: of silently un-applying a fix.
NTH_PARAGRAPH_FIRST_WORD = "length_constraints:nth_paragraph_first_word"
LETTER_FREQUENCY = "keywords:letter_frequency"
ENGLISH_CAPITAL = "change_case:english_capital"

FIXED_INSTRUCTION_IDS = frozenset(
    {NTH_PARAGRAPH_FIRST_WORD, LETTER_FREQUENCY, ENGLISH_CAPITAL}
)

#: Upstream's punctuation set for first-word extraction, lifted so the helper
#: below can run the same normalisation per token.
_PUNCTUATION = {".", ",", "?", "!", "'", '"'}


def _leading_word(token: str) -> str:
    """Upstream's first-word normalisation, lifted verbatim to run per token.

    Character for character the loop upstream runs on ``paragraph.split()[0]``:
    strip, drop leading quotes, then take letters up to the first punctuation
    mark, lowercased.
    """
    word = token.strip().lstrip("'").lstrip('"')
    out = ""
    for letter in word:
        if letter in _PUNCTUATION:
            break
        out += letter.lower()
    return out


def _is_ascii_letter(character: str) -> bool:
    """Upstream's accepted range for a frequency letter, as the predicate it is."""
    return 97 <= ord(character.lower()) <= 122


class _NthParagraphFirstWordFix:
    """Index the paragraph list that was counted; compare a multi-token value.

    **Defect A — the list counted is not the list indexed.** Upstream splits on
    ``\\n\\n``, then decrements a *count* for each blank chunk while indexing the
    *unfiltered* list::

        paragraphs = re.split(r"\\n\\n", value)
        num_paragraphs = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1
        ...
        paragraph = paragraphs[self._nth_paragraph - 1]

    So a blank chunk at or before index ``nth - 1`` changes which paragraph is
    checked — or hands the check a blank one — while the total it is compared
    against is computed on the other reading. The common case is a response whose
    *first* chunk is blank, which whole runs do uniformly depending on the chat
    template. (Two newlines have to meet before ``re.split`` yields an empty
    chunk, so a single leading ``\\n`` does not do it; a blank chunk *after* the
    target index is harmless, both readings then agreeing.) The two are
    reconciled the only way that keeps upstream's own ``num_paragraphs``: filter
    once, and index what was counted.

    **Defect B — a ``first_word`` that is not one word.** Upstream compares
    against ``paragraph.split()[0]``, a single whitespace-delimited token. Some
    slots store a multi-token phrase there, and the prompts that carry them ask
    for that same phrase, so the kwarg is faithful to the item and the comparison
    is what is wrong: a single token can never equal a multi-token string, so
    such a slot returns FAIL for every possible response.

    The repair reads how many tokens the *constraint's own value* spans and
    compares that many tokens of the paragraph, each normalised by upstream's own
    routine. For a single-token value — every well-formed item — ``n_tokens`` is
    1 and the expression reduces to upstream's, token for token. It does not
    relax the comparison: the phrase must still open the paragraph, in order.
    """

    # Set by the upstream `build_description` this mixin inherits alongside.
    _num_paragraphs: int
    _nth_paragraph: int
    _first_word: str

    def check_following(self, value: str) -> bool:
        # One definition of "paragraph": a `\n\n`-separated chunk with content.
        # Upstream computes the same number for `num_paragraphs`; it just kept
        # the unfiltered list around to index into.
        paragraphs = [p for p in re.split(r"\n\n", value) if p.strip()]
        num_paragraphs = len(paragraphs)

        if not 1 <= self._nth_paragraph <= num_paragraphs:
            return False
        paragraph = paragraphs[self._nth_paragraph - 1].strip()
        if not paragraph:  # unreachable after the filter; kept as upstream's guard
            return False

        # 1 for every well-formed item, which is upstream's path exactly. The
        # `or 1` covers an empty constraint value, where `split()` yields nothing
        # and slicing to 0 tokens would compare "" against "" and pass anything.
        n_tokens = len(self._first_word.split()) or 1
        opening = " ".join(_leading_word(t) for t in paragraph.split()[:n_tokens])

        return num_paragraphs == self._num_paragraphs and opening == self._first_word


class _LetterFrequencyFix:
    """Keep the character the item names.

    Upstream's ``build_description`` accepts a letter only when it is a single
    character in ``[a-z]``; anything else is silently replaced by
    ``random.choice(string.ascii_letters)``, drawn *freshly on every call* — so
    an item naming a non-alphabetic character is graded against a different
    letter each time it is scored.

    Everything else is deferred to upstream — the frequency default, the relation
    validation, the description pattern — so the only input whose handling
    changes is a single character outside ``[a-z]``. Upstream's fallback survives
    for values that are genuinely unusable (``None``, or more than one character
    after stripping): there the item states nothing gradable, and inventing a
    rule would be worse than upstream's.
    """

    _letter: str
    _frequency: int
    _comparison_relation: str
    _description_pattern: str

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        stripped = (letter or "").strip()
        substituting = len(stripped) == 1 and not _is_ascii_letter(stripped)
        # A placeholder upstream accepts, so its own validation still runs and no
        # random draw is consumed -- consuming one would perturb the global RNG
        # stream for every later checker, which is a side effect a grading fix
        # has no business having.
        description = super().build_description(  # type: ignore[misc]
            letter="a" if substituting else letter,
            let_frequency=let_frequency,
            let_relation=let_relation,
        )
        if substituting:
            self._letter = stripped.lower()
            description = self._description_pattern.format(
                letter=self._letter,
                let_frequency=self._frequency,
                let_relation=self._comparison_relation,
            )
        return description


class _EnglishCapitalFix:
    """Detect the language of a case-folded copy.

    Upstream is ``value.isupper() and langdetect.detect(value) == "en"``.
    ``langdetect`` profiles are built from lowercase text, so ALL-CAPS input is
    off-distribution for every profile and the verdict becomes unstable between
    the Latin-script languages — the same English paragraph can detect as ``de``.
    Because the checker's own precondition is that the response *is* all capitals,
    every response it ever sees is in exactly the state the detector handles worst.

    Only the argument to ``detect`` changes in what is *graded*. ``isupper()`` is
    still upstream's, still first, and still short-circuits, so the set of
    responses reaching the detector is exactly the set upstream sends there, and
    "all capital letters" is decided by code this mixin does not touch. The one
    other difference is that overriding the whole method drops upstream's
    log-on-exception (``logging.error`` in google-research's copy, ``logger.info``
    in Meta's -- the single line on which the two copies differ); it logged the
    entire response, and the verdict it accompanied is unchanged.

    The sibling ``change_case:english_lowercase`` is deliberately *not* repaired:
    it calls the detector on text that is already lowercase, which is the
    condition the profiles were built for, so it has no defect to fix.
    """

    def check_following(self, value: str) -> bool:
        assert isinstance(value, str)
        try:
            # `.lower()` only here: the response itself is never modified, and
            # the capitals requirement is evaluated above on the original.
            return value.isupper() and langdetect.detect(value.lower()) == "en"
        except langdetect.LangDetectException:
            # Upstream counts an undetectable text as following the instruction.
            # Kept for parity, though `isupper()` above makes it near-unreachable
            # -- an empty or uncased response fails before it.
            return True


class IFEvalParagraphFirstWordCheckFixed(
    _NthParagraphFirstWordFix, _ifeval_upstream.ParagraphFirstWordCheck
):
    """The IFEval registry's copy, repaired."""


class IFEvalLetterFrequencyCheckerFixed(
    _LetterFrequencyFix, _ifeval_upstream.LetterFrequencyChecker
):
    """The IFEval registry's copy, repaired."""


class IFEvalCapitalLettersEnglishCheckerFixed(
    _EnglishCapitalFix, _ifeval_upstream.CapitalLettersEnglishChecker
):
    """The IFEval registry's copy, repaired."""


class MultiIFParagraphFirstWordCheckFixed(
    _NthParagraphFirstWordFix, _multi_if_upstream.ParagraphFirstWordCheck
):
    """The Multi-IF registry's copy of the same checker, repaired identically."""


class MultiIFLetterFrequencyCheckerFixed(
    _LetterFrequencyFix, _multi_if_upstream.LetterFrequencyChecker
):
    """The Multi-IF registry's copy of the same checker, repaired identically."""


class MultiIFCapitalLettersEnglishCheckerFixed(
    _EnglishCapitalFix, _multi_if_upstream.CapitalLettersEnglishChecker
):
    """The Multi-IF registry's copy of the same checker, repaired identically."""


def _build(base: Mapping[str, type], fixes: Mapping[str, type]) -> dict[str, type]:
    """Overlay ``fixes`` on a copy of ``base``, refusing to add a new id.

    The refusal is the point: if an upstream re-sync renames one of the three
    ids, the fix would otherwise be mounted under a key nothing looks up and the
    ``_fixed`` task would quietly grade like the unqualified one.
    """
    missing = sorted(set(fixes) - set(base))
    if missing:
        raise KeyError(
            f"instruction id(s) not in the upstream registry: {missing}. "
            "The vendored checkers were renamed; update the constants in "
            "sieval.community.instruction_following_eval_fixed."
        )
    return {**base, **fixes}


def fixed_ifeval_registry() -> dict[str, type]:
    """The IFEval registry with the three repaired checkers substituted in.

    A fresh dict each call, and the vendored ``INSTRUCTION_DICT`` is never
    mutated: samples are graded concurrently, so a task that swapped the global
    registry would change how *other* tasks grade mid-run.
    """
    return _build(
        _ifeval_registry.INSTRUCTION_DICT,
        {
            NTH_PARAGRAPH_FIRST_WORD: IFEvalParagraphFirstWordCheckFixed,
            LETTER_FREQUENCY: IFEvalLetterFrequencyCheckerFixed,
            ENGLISH_CAPITAL: IFEvalCapitalLettersEnglishCheckerFixed,
        },
    )


def fixed_multi_if_registry() -> dict[str, type]:
    """The Multi-IF registry with the three repaired checkers substituted in."""
    return _build(
        _multi_if_upstream.INSTRUCTION_DICT,
        {
            NTH_PARAGRAPH_FIRST_WORD: MultiIFParagraphFirstWordCheckFixed,
            LETTER_FREQUENCY: MultiIFLetterFrequencyCheckerFixed,
            ENGLISH_CAPITAL: MultiIFCapitalLettersEnglishCheckerFixed,
        },
    )
