"""Repaired IFBench constraint checkers — the verifier, never the prompt.

Four of the checkers vendored in :mod:`sieval.community.ifbench.instructions`
grade something other than what their own instruction text asks for. This module
holds a repaired version of each and the registry overlay that mounts them;
:mod:`sieval.tasks.ifbench_0shot_gen_fixed` is the only caller.

Nothing here changes what an item *asks*. Every repair is to the code that
decides whether an answer complied.

**Why a separate module for one caller.** Not for sharing: one importer is why
it sits under ``tasks/`` rather than in ``community/``. It is a separate *file*
because the task module's whole claim is that it is the unqualified task plus
one overridden method, and two hundred lines of surgery on vendored code is not
something that claim survives being read beside.

**Registration must not pay for any of this.** ``import_all_tasks`` — which
``scripts/sync_meta_index.py`` and CI run — imports *every* module under
``sieval/tasks``, private ones included: the underscore keeps this module out of
the task *index*, not out of the *import scan*, so the discipline has to be kept
by the code rather than by the filename.
:mod:`sieval.community.ifbench.instructions` is a 2.3k-line fork pulling in NLTK
and a network fetch of its corpora, so it is imported inside
:func:`_fixed_checker_classes` and never at module scope — importing this module
is free, and *naming a repaired checker* is what costs.
:mod:`sieval.tasks._math_verify` keeps the same rule for the same reason.

**Subclasses rather than replacements.** ``build_description``'s defaults and
validation, ``get_instruction_args`` and the description pattern all stay
upstream's *by construction* rather than by review, so a repair cannot silently
change what an item says while claiming to change only how it is graded. Each
overrides exactly ``check_following``. A subclass needs its base loaded when it
is *defined*, which is why the four class statements live inside
:func:`_fixed_checker_classes` while the verdict logic they delegate to stays
out here, read, typed and tested without the fork present.

The four:

**``format:line_indent``** — upstream removes blank lines with

.. code-block:: python

    for line in lines:
        if not line.strip():
            lines.remove(line)

which mutates the list it is iterating and therefore skips an element on every
removal. A blank line that survives has indent 0 and breaks the "each line
indents further than the last" chain, so the verdict depends on *how many* blank
lines the response happened to contain — something the instruction neither
states nor could state. ``a\\n\\n b\\n  c`` (one blank line, removed) was already
True; ``a\\n\\n\\n b\\n  c`` (two, one survives) was False. Repaired by filtering
with a comprehension. A leading ``Sure!`` still fails, correctly — that is two
consecutive lines at indent 0, which is what the instruction forbids.

**``ratio:sentence_type``** — two independent defects in
``declarative_count == 2 * interrogative_count``:

1. Both counts test the raw final character — ``sentence.endswith('.')`` and
   ``sentence.endswith('?')`` — so a sentence closing on a quote or bracket
   (``... the sky is blue."``) is not counted at all. Repaired by looking past a
   trailing run of closing quotes/brackets before testing the terminal
   punctuation. Deliberately *not* "strip all punctuation": removing ``!`` or
   ``?`` would move a sentence into a different bucket.

   The repair is applied to **both** counts, so it recovers a quoted
   interrogative (``She said "Really?" He nodded. It is fine.`` — terminals
   ``?``, ``.``, ``.``, so 2 == 2*1, False upstream and True here) exactly as it
   recovers a quoted declarative. Repairing one side alone would compare a
   quote-aware count against a quote-blind one — a third reading of the
   instruction, and not one anybody asked for.
2. ``0 == 0`` is True, so a response with no interrogative at all — every
   sentence exclamatory, or nothing the splitter recognises as a sentence —
   **passes**. That is a false pass, the direction that inflates a score and the
   direction nobody notices. Repaired by requiring at least one interrogative.

Not repaired, deliberately: the exact-integer reading. 9 declaratives to 4
interrogatives is 2.25:1 and returns False; the instruction says "Maintain a 2:1
ratio", so False is the right answer. A tolerance would invent a claim the
instruction never made, which is a prompt-side change.

**``words:words_position``** — the instruction is "the second word ... and the
second to last word ... should be the word {keyword}", and the checker indexes
*tokens*. Its ``if words[-1] in string.punctuation: compare words[-3]`` is exactly
right for one trailing punctuation token — the token list is then the word list
plus one, so ``token[-3]`` *is* ``word[-2]`` — and wrong for anything else. A
response ending in two marks (``!?``, ``."``) shifts the index onto the last word,
which flips verdicts in **both** directions: it fails ``... a whisper here!?``,
which complies, and passes ``... is a whisper!?``, which does not. The same defect
bites from the front, where ``Well, whisper ...`` has ``words[1] == ','``.
Repaired by dropping every token that is entirely punctuation and comparing
``words[1]`` and ``words[-2]``. All three are the one defect of counting
punctuation as a word, when the instruction says *word*.

**``words:vowel``** — ``value.strip().split('\\n')`` then ``len != 1`` rejects on
**line** count. The single-paragraph requirement itself is declared (the prompts
say "Write a paragraph ...") so it stays; what is wrong is calling every newline
a paragraph break. Repaired by splitting on blank lines: a soft-wrapped paragraph
passes, two blank-line-separated paragraphs still fail.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import functools
import re
import string
import unicodedata
from typing import override

#: Trailing characters that close a sentence without ending it — a quoted
#: declarative is still a declarative. Deliberately not "all punctuation":
#: stripping `!` or `?` would change which bucket the sentence lands in.
_CLOSERS = "\"'”’»)]}"

#: A paragraph break is a blank line. A single newline inside a paragraph is a
#: soft wrap, which is what upstream's `split('\n')` mistakes for a break.
#:
#: Every spelling of a blank line counts, because a line ending cannot be what
#: decides whether a response is one paragraph or two: `\r?\n` reads CRLF as one
#: break, and `[^\S\r\n]` is "whitespace that is not a line break", so an
#: NBSP-padded blank line separates paragraphs like a space-padded one. Narrower
#: spellings fail open -- the halves stay joined, and two paragraphs pass a
#: checker whose only job is to reject them.
_PARA_SPLIT = re.compile(r"\r?\n[^\S\r\n]*\r?\n")


def _terminal(sentence: str) -> str:
    """The sentence's terminal punctuation, looking past closing quotes/brackets."""
    stripped = sentence.rstrip().rstrip(_CLOSERS).rstrip()
    return stripped[-1] if stripped else ""


def _is_punct_token(token: str) -> bool:
    """True if the token is punctuation only — i.e. not a word.

    Unicode-aware on purpose: ``word_tokenize`` emits ```` `` ````/``''`` for
    quotes and responses carry ``“``, ``’``, ``—``. ``string.punctuation`` alone
    misses every one of those and would leave in place the same off-by-one this
    repair exists to remove.
    """
    if not token:
        return True
    return all(
        char in string.punctuation or unicodedata.category(char).startswith("P")
        for char in token
    )


def _check_indent_stairs(value: str) -> bool:
    """``format:line_indent``, without the mutate-while-iterating blank removal."""
    lines = [line for line in value.split("\n") if line.strip()]
    for current, following in zip(lines, lines[1:], strict=False):
        current_indent = len(current) - len(current.lstrip(" "))
        next_indent = len(following) - len(following.lstrip(" "))
        if next_indent <= current_indent:
            return False
    return True


def _check_sent_type_ratio(value: str) -> bool:
    """``ratio:sentence_type`` counting quoted sentences, and not vacuously true.

    The 2:1 test itself is untouched: exact is what the instruction says.
    """
    from sieval.community.ifbench import instructions_util

    sentences = instructions_util.split_into_sentences(value)
    terminals = [_terminal(sentence) for sentence in sentences]
    declarative_count = sum(1 for t in terminals if t == ".")
    interrogative_count = sum(1 for t in terminals if t == "?")
    # `interrogative_count >= 1` is the false-pass repair: without it an
    # all-exclamatory response satisfies a ratio it never engaged with.
    return interrogative_count >= 1 and declarative_count == 2 * interrogative_count


def _check_words_position(value: str, keyword: str) -> bool:
    """``words:words_position`` positioned on words rather than on tokens."""
    from sieval.community.ifbench import instructions_util

    tokens = instructions_util.nltk.word_tokenize(value)
    words = [token for token in tokens if not _is_punct_token(token)]
    if len(words) < 2:
        return False
    return words[1].lower() == words[-2].lower() == keyword.lower()


def _check_single_vowel_paragraph(value: str) -> bool:
    """``words:vowel`` counting paragraphs by blank line, not by newline."""
    paragraphs = [p for p in _PARA_SPLIT.split(value.strip()) if p.strip()]
    if len(paragraphs) != 1:
        return False
    vowels = set("aeiou")
    used = {char for char in paragraphs[0].lower() if char in vowels}
    return len(used) <= 3


@functools.cache
def _fixed_checker_classes() -> dict[str, type]:
    """instruction id -> repaired checker. These keys are the whole story of what
    the ``_fixed`` task changes.

    The vendored fork is imported *here*, not at module scope: a subclass needs
    its base loaded at class-definition time, and ``import_all_tasks`` imports
    this module whether or not anything will grade through it. Cached, so the
    four classes are built once per process — samples grade concurrently and the
    registry is compared by identity.
    """
    from sieval.community.ifbench import instructions as upstream

    class IndentStairsCheckerFixed(upstream.IndentStairsChecker):
        """``format:line_indent``, without the mutate-while-iterating removal."""

        @override
        def check_following(self, value):
            return _check_indent_stairs(value)

    class SentTypeRatioCheckerFixed(upstream.SentTypeRatioChecker):
        """``ratio:sentence_type`` counting quoted sentences, and not vacuous."""

        @override
        def check_following(self, value):
            return _check_sent_type_ratio(value)

    class WordsPositionCheckerFixed(upstream.WordsPositionChecker):
        """``words:words_position`` positioned on words rather than on tokens."""

        @override
        def check_following(self, value):
            return _check_words_position(value, self._keyword)

    class SingleVowelParagraphCheckerFixed(upstream.SingleVowelParagraphChecker):
        """``words:vowel`` counting paragraphs by blank line, not by newline."""

        @override
        def check_following(self, value):
            return _check_single_vowel_paragraph(value)

    return {
        "format:line_indent": IndentStairsCheckerFixed,
        "ratio:sentence_type": SentTypeRatioCheckerFixed,
        "words:words_position": WordsPositionCheckerFixed,
        "words:vowel": SingleVowelParagraphCheckerFixed,
    }


def fixed_ifbench_registry() -> dict[str, type]:
    """Upstream's registry with the four repaired checkers overlaid.

    A **fresh dict per call**, never the vendored global: samples grade
    concurrently, and mutating the shared registry would change how the
    unqualified task grades in the same session.

    Refuses to mount a repair whose id or base class has moved upstream. A
    re-vendor that renames an id would land the overlay on a key nothing looks
    up; one that re-binds an id to a different class would reintroduce a checker
    upstream no longer uses. Either way ``_fixed`` would grade identically to the
    unqualified task while still claiming a delta, and here is the only place
    that is visible.
    """
    from sieval.community.ifbench.instructions_registry import INSTRUCTION_DICT

    registry = dict(INSTRUCTION_DICT)
    for instruction_id, fixed_cls in _fixed_checker_classes().items():
        base = registry.get(instruction_id)
        if base is None:
            raise KeyError(
                f"{instruction_id!r} has a repaired checker here but is not in "
                "the vendored IFBench registry; upstream renamed or removed it, "
                "and the repair would silently not apply."
            )
        if not issubclass(fixed_cls, base):
            raise TypeError(
                f"{instruction_id!r} is bound to {base.__name__} upstream, but "
                f"{fixed_cls.__name__} derives from "
                f"{fixed_cls.__mro__[1].__name__}; the repair would reintroduce "
                "a checker upstream no longer uses."
            )
        registry[instruction_id] = fixed_cls
    return registry
