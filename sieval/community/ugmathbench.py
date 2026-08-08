# UGMathBench (YangLabHKUST/UGMathBench) is distributed under GPL-3.0, so none
# of its harness code is vendored here. What IS reproduced from the pinned
# commit is the benchmark's *protocol* — the query template and answer-type
# descriptions a model must be shown for the run to be UGMathBench at all:
# https://github.com/YangLabHKUST/UGMathBench/blob/df47bfa639bfb89bdb0220036a7b2f216e72b0b3/utils.py
"""
UGMathBench prompting, answer extraction, and answer-type-aware grading.

UGMathBench problems carry a *sequence* of answers (a table to fill, several
sub-questions), each with its own answer type drawn from a fixed 10-type
vocabulary. Grading is therefore two-level: split the boxed response into as
many answers as the reference has, then compare each one under the rule its
type implies. A sample is correct only when every slot is.

Three pieces:

* :func:`build_prompt` — the upstream query template (single- vs multi-answer
  wording, plus the per-slot type description), reproduced from the pinned
  ``make_prompt``. This is what makes a run comparable to the paper.
* :func:`extract_predictions` — take the last ``\\boxed{...}``, normalize it,
  split on commas that sit outside brackets.
* :func:`judge_answers` — per-slot dispatch over the 10 answer types.

**The grader is an independent implementation, not a port.** Upstream's
``judge_rule.py`` is GPL-3.0 and cannot be carried into an Apache-2.0
distribution, so the comparison rules here were written against the answer-type
semantics the prompt itself states, on top of ``math-verify`` (already a sieval
dependency). Known behavioural deltas versus the reference judge:

* Symbolic equivalence starts from ``math-verify``'s parse/verify rather than
  upstream's bespoke ``parse_latex`` + ``simplify`` chain, and is more permissive
  on LaTeX shapes upstream's normalizer never learned. Numeric slots keep
  upstream's *relative* tolerance as a second chance, since ``math-verify``
  compares floats at fixed decimal rounding. Where it used to be *stricter* than
  upstream — no numeric sampling of free symbols — it no longer is:
  :func:`_same_function` closes that gap with a fixed ladder of substitution
  probes, arrived at independently rather than ported. See the parser note
  below for why that pass is load-bearing here.
* **The dataset's gold is sympy source, not LaTeX, and only one parser reads it.**
  ``math_verify.parse`` runs a LaTeX reader over both sides, and in LaTeX an
  unescaped ``sin`` is the product s*i*n while ``pi`` is p*i — so the stored
  ``7*sin(pi*x/5)+1`` becomes ``7*s*i*n*(i*p*x)/5 + 1`` and cannot match the
  model's ``7\\sin(\\frac{\\pi}{5}x)+1`` by any route except exact string
  equality. :func:`_parse_sympy_source` supplies the second reading.
  The first live run measured what this had been costing: **716 of 15,183
  samples** were graded wrong purely for it (EAcc 34.46 -> 38.49, AAcc 40.87 ->
  45.59, CAcc 48.07 -> 53.53, with **zero** verdicts moving right-to-wrong).
  Note the shape of the
  bug — it was invisible to the reference-replay measurement below, because
  replaying a gold as its own answer short-circuits on
  ``_squash(pred) == _squash(gold)`` and never reaches the symbolic path at
  all. **A self-replay canary exercises the fast path and is silent about the
  comparison logic it appears to certify.**
* No answer-type *inference*: upstream's ``is_equal`` retries every judgement
  method until one accepts, which lets a slot be graded by a rule its declared
  type did not ask for. Here the declared type decides, except inside OL/UOL
  elements (whose own types the dataset does not record).
* Extraction is strict, as ``eval_rule.py`` grades with
  ``Judger(strict_extract=True)``: no "guess the last LaTeX formula / the last
  number in the response" fallback, and the ``answer is`` / ``answer:`` hand-off
  is kept because upstream keeps it too (its no-box branch runs the same
  ``elif`` chain). Two smaller differences remain inside that branch and are
  *not* repairs — just shapes this reads differently:

  - upstream splits on ``herefore`` and keeps only the tail **before** looking
    for a box, so a response whose last box sits before its last "Therefore"
    loses the box entirely there, where :func:`_last_boxed_content` always
    takes the last box in the whole response;
  - upstream returns the **entire response** when the box is empty
    (``if not content: return text``), where :func:`extract_answer` treats an
    empty box as "no box" and falls through to the marker search.

  Both sit inside the 95.51% live agreement measured against upstream's judge,
  so neither is worth a divergence of its own — they are listed because a
  divergence list that quietly rounds off is not one.
* **Commas are split at a different bracket depth**, in three ways. All three
  are deliberate, and all three run in the same direction: a row upstream
  miscounts — and therefore grades wrong in every slot, whatever the model
  answered — this one counts correctly.

  - Upstream's ``split_by_comma`` counts ``<`` and ``>`` as brackets.
    :data:`_OPENERS` does not. In this dataset they are overwhelmingly the
    *relational operators*, a slot whose entire answer is ``<``, so an opening
    ``<`` swallows the comma after it; the grouping form ``\\langle`` /
    ``\\rangle`` is folded to parentheses before the scan and keeps working.
    Worth **6 rows** across ``Financial_mathematics_0300`` and
    ``Calculus_-_single_variable_0824`` (all three versions each).
  - Upstream lets its bracket depth go *negative*, so an unmatched closer parks
    everything after it below zero, where no comma splits again. This clamps at
    zero, which is worth **3 more rows** — ``Arithmetic_0071``, whose second
    slot is ``>``: upstream splits the first comma, drops to -1 on the ``>``,
    and never splits again.
  - This also tracks ``{}``, which keeps a comma inside a LaTeX group
    (``\\frac{a,b}{c}``) from splitting a slot in two. Worth **0 rows** on the
    references — they are plain sympy source — so it earns its place only on
    the prediction side, where the model writes LaTeX.

  Nine rows total, all of them ours-accepts / upstream-rejects, and they are
  counted inside the 552 below. The only ``<...>`` *pairs* in the 42,064 gold
  slots are ``<br />`` markup, so the grouping reading buys back nothing.
* Both sides are normalized *by the same pass*, references included. Upstream
  normalizes its references too — ``judge()`` runs ``norm_ans_str`` over the
  gold — but its extraction-time pass, ``Judger.normalize_answer``, runs on the
  prediction alone and rewrites shapes the reference never sees, notably
  ``sqrt(x)`` into ``sqrt{(}x)``. A plain-sympy reference such as
  ``sqrt(1.83985)`` therefore cannot match itself upstream. This asymmetry, not
  any single answer type, is the largest divergence in this module; it is
  measured below.
* A ``TF`` slot whose reference is not a boolean at all (9 of 1665, e.g. ``-22``,
  ``not real``) is compared as a value rather than graded wrong outright, which
  is what upstream's assert-then-swallow does. Those slots are otherwise
  unwinnable regardless of what the model answers.
* A numeric slot whose reference is exactly zero is compared absolutely
  (``|pred| <= tolerance``). Upstream divides by the reference, so a zero
  reference raises into a bare ``except`` and the slot falls through to the
  symbolic path instead.
* Per-slot verdicts are returned rather than only the sample-level ``all()``,
  so a wrong answer can be located without re-running the grader.
* **Some answers are refused rather than graded**, because the text being parsed
  is model output and ``parse_expr`` evaluates what it parses. Three shapes:

  - A boxed ``__import__('os').system(...)`` would otherwise run.
    :func:`~sieval.community._sympy_guards.sympy_globals` removes the builtins from the parse namespace.
  - A boxed ``eval("...")`` — or ``sympify``/``S``/``N``, or any name at all,
    since ``auto_symbol`` makes every unknown one callable — hands a *string*
    back to sympy, which re-sympifies it with its own default namespace and so
    gets the builtins back. :func:`~sieval.community._sympy_guards.quotes_free` refuses the quote instead of
    the callee, which is the only end of it that can be enumerated.
  - A boxed ``9^9^9^9`` asks for a 370-million-digit integer that never
    returns. :func:`~sieval.community._sympy_guards.evaluable` screens it out with an unevaluated pre-parse.
    Grading runs in a worker process, so this occupies one worker rather than
    the shared event loop, but a worker held forever is still a worker lost.

  A refused answer grades wrong. Upstream has none of these guards; they are
  not a divergence in what the benchmark *measures*, and no reachable
  comparison changes — the largest exponent in the pinned references is three
  digits, and not one of the 42,064 gold slots contains a quote.

23 of the 15,183 pinned rows (0.15%) cannot be graded correctly here even when
the reference is replayed verbatim as the answer. 19 are unwinnable upstream
too: the "comma-separated answers in one box" protocol cannot express them — a
comma inside an open-ended phrase, or an unbalanced bracket left by upstream's
own answer splitting. The other 4 are a shape this module loses and upstream
wins — a ``UOL`` reference whose top-level commas sit outside any bracket, which
:func:`split_answers` reads as separate slots
(``Calculus_-_single_variable_0384`` v1-v3, ``Complex_analysis_0035`` v3).

The prompt builder, by contrast, IS exact: it reproduces upstream's ``raw``
template byte-for-byte on all 15,183 rows.

Scores are nonetheless *not* the paper's, which is why the only task built on
this module is the ``_fixed`` variant (``ugmathbench_0shot_gen_fixed``) and the
unqualified name is left vacant.

How far from upstream, measured rather than argued. Upstream's judge was run as
a local *instrument* — GPL-3.0 restricts distribution, not use, and no upstream
code is vendored or redistributed here — over all 15,183 pinned rows, with each
row's own reference replayed back as a boxed answer:

* upstream accepts its own reference on 14,616 rows (96.27%);
* this module accepts 15,160 (99.85%);
* the 552 rows that disagree (3.64%) span 192 of 5,061 problems, an EAcc
  **ceiling** difference of **3.79 pp** — roughly five times the 0.70 pp
  binomial standard error at this sample size, so the divergence is material
  rather than noise;
* per answer slot, running upstream's own dispatch loop without its
  short-circuit: 788 of 42,064 slots disagree (1.87%), concentrated in ``EX``
  (350) and ``NV`` (322), with ``TF`` contributing exactly the 9 non-boolean
  references described above.

Direction matters more than magnitude here: **548 of those 552 are rows where
upstream rejects its own reference**, i.e. slots no model could win there. That
is repair, not drift, and the prediction-only normalization above is its
dominant cause; the non-boolean ``TF`` references contribute 9 and the
comma-splitting rules below 9. Only 4 rows go the other way (the ``UOL`` note
above). A *ceiling*, not an expectation: the
difference is realized only on a problem a model would otherwise answer
correctly in all three versions.

The measurement is a replay of stored references, so it bounds the grader's
divergence, not a model's score. What it cannot show is how the grader behaves
on real model prose — see the promotion criteria on the task class.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math
import re

from ._sympy_guards import evaluable, quotes_free, sympy_globals

#: The 16 subject configs of the HF dataset, in the order the benchmark lists
#: them. Doubles as the default load order, so a sliced run is reproducible.
SUBJECTS: tuple[str, ...] = (
    "Abstract_algebra",
    "Algebra",
    "Arithmetic",
    "Calculus_-_multivariable",
    "Calculus_-_single_variable",
    "Combinatorics",
    "Complex_analysis",
    "Differential_equations",
    "Financial_mathematics",
    "Geometry",
    "Linear_algebra",
    "Number_theory",
    "Probability",
    "Set_theory_and_logic",
    "Statistics",
    "Trigonometry",
)

#: Randomized versions per problem. EAcc is defined over all of them, so this
#: is a property of the benchmark, not a knob.
VERSIONS: int = 3

#: Answer-type code -> the description the prompt shows the model. Reproduced
#: from upstream ``make_prompt``'s ``type2descriptions``; ``{options}`` is
#: filled with the slot's option list for the two multiple-choice types.
TYPE_DESCRIPTIONS: dict[str, str] = {
    "UOL": (
        "an unordered list of answers surrounded by parentheses with any answer "
        'types, for example, (1, x^2, True), where "unordered list" means '
        "changing the order of elements results in the same answer"
    ),
    "OL": (
        "an ordered list of answers surrounded by parentheses with any answer "
        'types, for example, (1, x^2, True), where "ordered list" means changing '
        "the order of elements results in different answers"
    ),
    "INT": "a range inteval",
    "TF": "either True or False",
    "EX": "an expression",
    "EQ": "an equation",
    "MCS": "one option of a multiple choice question with options {options}",
    "MCM": (
        "more than one option concatenated without space or commas of a multiple "
        "choice question with options {options}, for example: BD"
    ),
    "NV": "a numerical value without units",
    "OE": (
        "a word, phrase, term or string that satisfies the requirements of the problem"
    ),
}

_PROMPT_HEAD = (
    "The following is an undergraduate-level mathematical problem in {subject}. "
    "You need to solve the problem by completing all placeholders [ANS].\n\n"
)
_PROMPT_SINGLE_TYPES = (
    "This problem involves only one placeholders [ANS] to be completed. "
    "The answer type is {descriptions}.\n\n"
)
_PROMPT_MULTI_TYPES = (
    "This problem involves {count} placeholders [ANS] to be completed. "
    "Their answer types are, in order, {descriptions}.\n\n"
)
_PROMPT_TAIL_SINGLE = (
    "Problem:\n{problem}\n\n"
    "All mathematical formulas and symbols you output should be represented with "
    'LaTeX. Please end your response with: "The final answer is \\boxed{ANSWER}", '
    "where ANSWER should be your final answer."
)
_PROMPT_TAIL_MULTI = (
    "Problem:\n{problem}\n\n"
    "All mathematical formulas and symbols you output should be represented with "
    'LaTeX. Please end your response with: "The final answers are \\boxed{ANSWER}"'
    ", where ANSWER should be the sequence of your final answers, separated by "
    "commas."
)


def describe_answer_type(answer_type: str, options: list[str] | None = None) -> str:
    """Render one slot's type description, filling MC options where the type asks.

    Upstream interpolates the option list with Python's ``list`` repr, so the
    model sees ``['A', 'B', 'C', 'D', 'E']``; kept as-is for prompt fidelity.

    The unknown-type guard is a sieval addition (upstream indexes the mapping
    directly). It raises ``LookupError`` rather than ``KeyError`` so the
    sentence reaches the CLI unquoted — ``KeyError.__str__`` is
    ``repr(args[0])``.
    """
    description = TYPE_DESCRIPTIONS.get(answer_type)
    if description is None:
        raise LookupError(
            f"unknown UGMathBench answer type {answer_type!r}; "
            f"expected one of {sorted(TYPE_DESCRIPTIONS)}"
        )
    if "{options}" in description:
        return description.format(options=list(options or []))
    return description


def _pad_options(
    answer_types: list[str], options: list[list[str]] | None
) -> list[list[str]]:
    """One option list per declared type, padded when the dataset is short.

    A handful of pinned rows declare fewer ``options`` entries than
    ``answer_type`` entries. Options only carry meaning for the two
    multiple-choice types, so a missing entry is a data gap, not a different
    question — pad rather than refuse to build the prompt.
    """
    padded = [list(entry) for entry in (options or [])][: len(answer_types)]
    padded.extend([] for _ in range(len(answer_types) - len(padded)))
    return padded


def build_prompt(
    subject: str,
    problem: str,
    n_answers: int,
    answer_types: list[str],
    options: list[list[str]] | None = None,
) -> str:
    """Build the UGMathBench query for one problem version.

    The single- and multi-answer wordings differ upstream (down to "The final
    answer is" vs "The final answers are"), and the count drives which one is
    used, so both are reproduced rather than unified.

    *n_answers* is the length of the reference answer sequence and is passed
    separately from *answer_types* on purpose: a few pinned rows declare fewer
    types than answers, and upstream's template takes the count from the answers
    while describing only the declared types. Reproducing that keeps the prompt
    byte-identical to the one the paper's numbers came from.

    The two branches read the type list differently, and that is upstream's
    shape rather than a simplification: its single-answer branch describes
    ``answer_type[0]`` alone, while its multi-answer branch joins every declared
    type. The two agree on every pinned row, since a row with one answer
    declares one type — but building the joined string once and using it in both
    branches would make this port's fidelity depend on that coincidence holding
    after a re-cut of the data, so each branch derives its own.
    """
    per_slot_options = _pad_options(answer_types, options)

    head = _PROMPT_HEAD.format(subject=subject)
    if n_answers == 1:
        first_type = answer_types[:1]
        descriptions = ", ".join(
            describe_answer_type(answer_type, slot_options)
            for answer_type, slot_options in zip(
                first_type, per_slot_options[:1], strict=True
            )
        )
        types = _PROMPT_SINGLE_TYPES.format(descriptions=descriptions)
        tail = _PROMPT_TAIL_SINGLE
    else:
        descriptions = ", ".join(
            describe_answer_type(answer_type, slot_options)
            for answer_type, slot_options in zip(
                answer_types, per_slot_options, strict=True
            )
        )
        types = _PROMPT_MULTI_TYPES.format(count=n_answers, descriptions=descriptions)
        tail = _PROMPT_TAIL_MULTI
    # `ANSWER` is a literal in the template, not a field to fill.
    return head + types + tail.replace("{problem}", problem)


# --- answer extraction -----------------------------------------------------

_BOXED_MARKERS = ("\\boxed", "\\fbox")
# Ordered by specificity: a response saying "the final answer is X" should not
# be cut at the bare "answer is".
_ANSWER_MARKERS = ("final answers are", "final answer is", "answer is", "answer:")
_LATEX_WRAPPERS = re.compile(r"\\(?:text|mathrm|mathbf|textbf|mbox)\s*\{([^{}]*)\}")
_SIMPLE_REMOVALS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\quad", "\\qquad")


def _last_boxed_content(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` / ``\\fbox{...}``.

    Brace-balanced rather than regex-greedy, so a nested ``\\frac{a}{b}`` inside
    the box survives intact.
    """
    start = max(text.rfind(marker) for marker in _BOXED_MARKERS)
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def normalize_answer(text: str) -> str:
    """Strip LaTeX decoration that never carries meaning for a comparison."""
    normalized = text.replace("∶", ":").replace("，", ",")
    normalized = normalized.replace("\\approx", "=").replace("\\simeq", "=")
    normalized = _LATEX_WRAPPERS.sub(r"\1", normalized)
    for token in _SIMPLE_REMOVALS:
        normalized = normalized.replace(token, "")
    # Percent signs and degree marks: the dataset stores bare values ("a
    # numerical value without units"), so a trailing unit is decoration too.
    normalized = normalized.replace("^{\\circ}", "").replace("^\\circ", "")
    normalized = normalized.replace("\\%", "").replace("%", "")
    return normalized.replace("$", "").strip()


def extract_answer(response: str) -> str | None:
    """Pull the answer segment out of a full model response.

    Strict, as upstream's ``eval_rule.py`` grades: the last box, else an
    explicit "the answer is" hand-off, else ``None``. There is deliberately no
    "take the last number in the response" fallback — the prompt mandates a box,
    and guessing inflates scores for models that ignored the format.
    """
    boxed = _last_boxed_content(response)
    if boxed is not None:
        normalized = normalize_answer(boxed)
        return normalized or None

    haystack = response.lower()
    for marker in _ANSWER_MARKERS:
        position = haystack.rfind(marker)
        if position >= 0:
            tail = response[position + len(marker) :].strip()
            # One line only: whatever follows a blank line is new prose.
            tail = tail.split("\n\n")[0].strip().rstrip(".")
            normalized = normalize_answer(tail)
            if normalized:
                return normalized
    return None


#: ``<`` and ``>`` are deliberately absent, unlike upstream's ``split_by_comma``
#: which counts them as brackets. In this dataset they are overwhelmingly the
#: *relational operators* — a slot whose whole answer is ``<`` — so treating an
#: opening ``<`` as a bracket swallows the comma after it and the slot count
#: comes out short, which grades every slot in the row wrong however good the
#: answer. Angle brackets as grouping earn nothing back: of the 42,064 gold
#: slots on the pinned revision the only ``<...>`` pairs are ``<br />`` markup,
#: and ``\langle`` / ``\rangle`` are folded to parentheses below.
_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = set(_OPENERS.values())
_SET_DELIMITERS = {
    "\\{": "(",
    "\\}": ")",
    "\\langle": "(",
    "\\rangle": ")",
    "\\lbrace": "(",
    "\\rbrace": ")",
}


def split_answers(text: str) -> list[str]:
    """Split a boxed answer into one string per ``[ANS]`` slot.

    Splits on commas at bracket depth zero, so a slot that is itself a list
    (``(1, 2, 3)``) or an interval (``(-\\infty, 5)``) stays whole. LaTeX set
    delimiters are folded to plain parentheses first so they nest like brackets.

    Three deliberate differences from upstream's ``split_by_comma``, all
    enumerated with their measured cost in the module docstring: ``{}`` counts
    as a bracket here and does not upstream; ``<`` and ``>`` do not count here
    and do upstream; and the depth clamps at zero instead of going negative.
    """
    folded = text
    for latex_delimiter, plain in _SET_DELIMITERS.items():
        folded = folded.replace(latex_delimiter, plain)

    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(folded):
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(folded[start:index])
            start = index + 1
    parts.append(folded[start:])
    return [part.strip().strip("$").strip() for part in parts]


# --- answer comparison -----------------------------------------------------

#: Upstream's ``norm_str2bool`` tests the single letters *before* lowercasing
#: (``if s in ['T', 'Y']``), so only the capitals are booleans; the word forms
#: are matched after a ``.lower()`` and so are case-insensitive. The asymmetry
#: looks like an oversight but it is load-bearing: ``t`` and ``y`` are ordinary
#: parameter names, and reading them as booleans marks a wrong answer right.
_TRUE_LETTERS = {"T", "Y"}
_FALSE_LETTERS = {"F", "N"}
_TRUE_WORDS = {"true", "yes"}
_FALSE_WORDS = {"false", "no"}
_WHITESPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    """Whitespace- and decoration-insensitive form, for the cheap equality path."""
    return _WHITESPACE.sub("", text.replace("$", "").replace("\\", "")).lower()


def _to_bool(text: str) -> bool | None:
    """Read a ``TF`` answer as a boolean, or ``None`` if it is not one.

    Only reachable from the ``TF`` branch of :func:`judge_answer`. Upstream
    gates the same conversion on the declared answer type — ``norm_ans_str``
    calls ``norm_str2bool`` only ``if ans_type == "TF"`` — and leaves the
    elements of a list alone, under a standing ``TODO: deal with OL with
    boolean``. Applying it to list elements instead reads the parameter names
    the dataset actually uses as truth values.
    """
    stripped = text.strip().strip(".")
    if stripped in _TRUE_LETTERS:
        return True
    if stripped in _FALSE_LETTERS:
        return False
    word = stripped.lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return None


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _parse_math(text: str) -> list:
    from math_verify import parse

    # `$`-wrapping steers the LaTeX extractor at a bare answer string. It does
    # NOT reliably read the plain-sympy shapes the dataset stores: `parse` runs
    # a LaTeX reader, where an unescaped `sin` is the product s*i*n and `pi` is
    # p*i, so the stored gold `7*sin(pi*x/5)+1` comes back as
    # `7*s*i*n*(i*p*x)/5 + 1`. Bare algebra ("x^3+2*x^2+6") survives; anything
    # naming a function does not. `_parse_sympy_source` is the other half.
    return parse(f"${text}$")


#: Deterministic probe points for :func:`_same_function`. A *fixed* ladder, not
#: a seeded RNG: a grader has to return the same verdict for the same pair on
#: every run and in every process, and a module-level RNG would make the answer
#: depend on how many comparisons preceded it. Off the integers and away from
#: 0 and 1, so the usual poles and fixed points are avoided, yet deliberately
#: kept small: an answer like ``e^{\cosh(4x)}`` overflows to infinity above
#: x ~ 2, and every overflowed probe is discarded, so a ladder reaching into
#: the twenties would leave too few usable points and mark a correct
#: exponential answer wrong.
_PROBES: tuple[float, ...] = (0.41, 0.73, 1.19, 1.57, 2.11)
_MIN_CLEAN_PROBES = 3
_MAX_FREE_SYMBOLS = 4

def _parse_sympy_source(text: str) -> list:
    """Parse UGMathBench's plain-sympy answer syntax as sympy source.

    The dataset stores gold answers as sympy-ish source rather than LaTeX
    (``x^3+2*x^2+6``, ``pi/6*(4^3-2^3)``, ``9*[sin(x)]^8*cos(x)``), while a
    model answers in LaTeX. Reading the gold with the LaTeX parser mangles it
    (see :func:`_parse_math`), so it is read here with sympy's own parser too
    and both readings are offered to the comparison.

    Square brackets are grouping in this dialect, ``^`` is exponentiation, and
    ``e`` / ``pi`` / ``ln`` / ``infinity`` are the constants and functions they
    look like — none of which sympy assumes by default.

    The text reaching this function is model output, and ``parse_expr``
    evaluates what it parses, so all three halves of that are guarded:
    :func:`~sieval.community._sympy_guards.sympy_globals` takes the interpreter out of the parse namespace,
    :func:`~sieval.community._sympy_guards.quotes_free` keeps a nested parse from handing it back, and
    :func:`~sieval.community._sympy_guards.evaluable` refuses arithmetic that would not finish.
    """
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    cleaned = (
        text.replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
        .replace("^", "**")
        .replace("infinity", "oo")
        .replace("$", "")
        .strip()
    )
    if not cleaned or not quotes_free(cleaned):
        return []
    local = {
        "e": sympy.E,
        "E": sympy.E,
        "pi": sympy.pi,
        "ln": sympy.log,
        "log": sympy.log,
        "oo": sympy.oo,
        "I": sympy.I,
        # The dataset spells inverse trig `arcsin`; sympy calls it `asin`.
        # Without the alias `arcsin(3/10)` parses as a *symbol* times a number,
        # which silently changes the free-symbol set and loses the comparison.
        "arcsin": sympy.asin,
        "arccos": sympy.acos,
        "arctan": sympy.atan,
        "arcsec": sympy.asec,
        "arccsc": sympy.acsc,
        "arccot": sympy.acot,
        "arcsinh": sympy.asinh,
        "arccosh": sympy.acosh,
        "arctanh": sympy.atanh,
    }
    out: list = []
    globals_ = sympy_globals()
    for transformations in _source_transformations():
        if not evaluable(cleaned, local, transformations):
            continue
        try:
            out.append(
                parse_expr(
                    cleaned,
                    local_dict=local,
                    global_dict=globals_,
                    transformations=transformations,
                )
            )
        except Exception:
            # Not sympy source under this reading (LaTeX, prose, an unbalanced
            # bracket) — the other readings are expected to handle those.
            continue
    return out


def _source_transformations():
    """Strict sympy source first, then the same with implicit multiplication.

    Both readings are kept rather than just the permissive one. Implicit
    multiplication is what a *prediction* sometimes needs (``5 - 5c`` is
    ``5 - 5*c``, and neither the LaTeX parser nor strict sympy reads it that
    way), but it also happily reinterprets a single symbol as a product, so it
    is offered as an extra candidate and never as a replacement.
    """
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        standard_transformations,
    )

    return (
        standard_transformations,
        standard_transformations + (implicit_multiplication_application,),
    )


def _sympy_candidates(text: str) -> list:
    """Every plausible sympy reading of *text*, from both parsers.

    Both are tried on both sides on purpose: the gold is usually sympy source
    and the prediction usually LaTeX, but neither is guaranteed, and committing
    a parser to a side would manufacture disagreements of its own.
    """
    import sympy

    out: list = []
    try:
        candidates = _parse_math(text) + _parse_sympy_source(text)
    except Exception:
        return out
    for item in candidates:
        if isinstance(item, sympy.Basic) and not any(item == seen for seen in out):
            out.append(item)
    return out


def _same_function(pred_expr, gold_expr, precision: float) -> bool:
    """Do two expressions denote the same function, by numeric substitution?

    Feeds identical values to identically *named* free symbols and compares the
    results. This is what catches equivalence that survives no string
    normalization — ``3\\cos(2\\sqrt{35}t)`` against ``3*cos(sqrt(980/7)*t)``,
    or ``x^0*e^(-8*x)`` against ``e^{-8x}``.

    Substitution is keyed on symbol *name*, never on the symbol object: the two
    parsers build ``Symbol('x')`` with different assumptions, so the objects
    compare unequal and a set union would yield two distinct ``x`` that then
    receive two different values — making every equivalent pair look unequal.

    Conservative by construction, because the only judgement this can change is
    wrong-to-right:

    * the two sides must involve the *same* set of symbol names, so an answer
      in ``x`` never matches one in ``t``;
    * every probe that evaluates cleanly must agree, and at least
      ``_MIN_CLEAN_PROBES`` of them must evaluate cleanly, so a pair that only
      survives at a single lucky point is rejected;
    * non-finite and complex results are discarded rather than compared.
    """
    import sympy

    pred_names = {symbol.name for symbol in pred_expr.free_symbols}
    gold_names = {symbol.name for symbol in gold_expr.free_symbols}
    if pred_names != gold_names or len(gold_names) > _MAX_FREE_SYMBOLS:
        return False

    if not gold_names:
        try:
            pred_value, gold_value = (
                complex(pred_expr.evalf()),
                complex(gold_expr.evalf()),
            )
        except Exception:
            return False
        if abs(pred_value.imag) > 1e-9 or abs(gold_value.imag) > 1e-9:
            return False
        return _close(pred_value.real, gold_value.real, precision)

    ordered = sorted(gold_names)
    clean = 0
    for probe in _PROBES:
        values = {
            name: sympy.Float(probe + 0.17 * index)
            for index, name in enumerate(ordered)
        }
        try:
            pred_value = complex(
                pred_expr.subs(
                    {s: values[s.name] for s in pred_expr.free_symbols}
                ).evalf()
            )
            gold_value = complex(
                gold_expr.subs(
                    {s: values[s.name] for s in gold_expr.free_symbols}
                ).evalf()
            )
        except Exception:
            continue
        parts = (pred_value.real, pred_value.imag, gold_value.real, gold_value.imag)
        # Reject only NaN and infinity. A magnitude ceiling would be wrong here:
        # the comparison below is *relative*, and an exponential answer is
        # legitimately enormous at these probes (e^cosh(4x) is ~1e40 at the
        # first one), so capping magnitude discards every probe and silently
        # marks a correct answer wrong.
        if any(not math.isfinite(part) for part in parts):
            continue  # undefined at this probe — it says nothing either way
        clean += 1
        if abs(pred_value.imag) > 1e-9 or abs(gold_value.imag) > 1e-9:
            if abs(pred_value - gold_value) > abs(gold_value) * precision * 1.01:
                return False
            continue
        if not _close(pred_value.real, gold_value.real, precision):
            return False
    return clean >= _MIN_CLEAN_PROBES


def _equivalent_by_substitution(pred: str, gold: str, precision: float) -> bool:
    """Last-chance equivalence check, over every parse of both sides."""
    try:
        pred_exprs = _sympy_candidates(pred)
        gold_exprs = _sympy_candidates(gold)
        for pred_expr in pred_exprs:
            for gold_expr in gold_exprs:
                if _same_function(pred_expr, gold_expr, precision):
                    return True
    except Exception:
        # Same contract as the rest of the module: an ungradeable answer is a
        # wrong answer, not a crashed run.
        return False
    return False


def _numeric_value(parsed: list) -> float | None:
    """Best-effort real value of a parsed expression, for relative tolerance."""
    for item in parsed:
        is_number = getattr(item, "is_number", False)
        if not is_number:
            continue
        try:
            value = complex(item.evalf())
        except (TypeError, ValueError, AttributeError):
            continue
        if abs(value.imag) < 1e-12:
            return value.real
    return None


def math_equal(pred: str, gold: str, precision: float = 1e-3) -> bool:
    """Compare two mathematical answers.

    Four chances, cheapest first: squashed string equality, plain-float
    comparison at *precision* (relative, as upstream), ``math-verify`` symbolic
    equivalence with a relative-tolerance retry on the parsed numeric values
    (``math-verify`` compares floats at fixed decimal rounding, which rejects
    pairs upstream's relative tolerance accepts), and finally equivalence by
    numeric substitution.

    The substitution pass exists because the first three all compare a LaTeX
    prediction against a gold that ``math-verify`` has read as LaTeX too — and
    the dataset does not store LaTeX. A gold naming any function comes back
    mangled (``sin`` as s*i*n, ``pi`` as p*i), so ``7\\sin(\\frac{\\pi}{5}x)+1``
    could not match the stored ``7*sin(pi*x/5)+1`` by any route but exact string
    equality. The first live run of this task measured the cost: of 1,634 wrong
    slots where extraction and reference agreed on slot count, 570 (34.9%) were
    this function's error rather than the model's, all of them in the free-form
    types and none in the structured ones.

    Because the new pass runs only after the others have said "not equal", the
    only verdict it can change is wrong-to-right; it cannot break a comparison
    that already succeeded.
    """
    if _squash(pred) == _squash(gold):
        return True

    pred_float, gold_float = _to_float(pred), _to_float(gold)
    if pred_float is not None and gold_float is not None:
        return _close(pred_float, gold_float, precision)

    try:
        from math_verify import verify

        parsed_gold, parsed_pred = _parse_math(gold), _parse_math(pred)
        if parsed_gold and parsed_pred and verify(parsed_gold, parsed_pred):
            return True
        gold_value, pred_value = (
            _numeric_value(parsed_gold),
            _numeric_value(parsed_pred),
        )
        # Accept on success, but do NOT reject on failure: both values come from
        # the LaTeX reading, which is the one known to mangle this dataset's
        # gold. Returning its verdict here made the substitution pass below
        # unreachable for every pair the mangling happens to turn into a
        # *number* — `2**100` reads as `2`, so a correct `2^100` was compared
        # against 2 and graded wrong without the pass ever running.
        if (
            gold_value is not None
            and pred_value is not None
            and _close(pred_value, gold_value, precision)
        ):
            return True
    except Exception:
        # math-verify raises on pathological input (unbalanced LaTeX, runaway
        # simplify). Fall through rather than return: a crash in one comparison
        # strategy is not evidence about the answer, and the substitution pass
        # below carries its own exception contract.
        pass
    return _equivalent_by_substitution(pred, gold, precision)


def _close(pred: float, gold: float, precision: float) -> bool:
    tolerance = precision * 1.01
    if gold == 0.0:
        return abs(pred) <= tolerance
    return abs((pred - gold) / gold) <= tolerance


def _option_letters(text: str, options: list[str]) -> list[str]:
    allowed = {option.lower() for option in options} or {
        chr(code) for code in range(ord("a"), ord("z") + 1)
    }
    return [char for char in text.lower() if char in allowed]


def _judge_multiple_choice_single(pred: str, gold: str, options: list[str]) -> bool:
    # Options are usually bare letters, but some problems label choices with
    # expressions ("Q(X)"), so compare the whole answer before touching brackets.
    if _squash(pred) == _squash(gold):
        return True
    target = gold.strip().lower()
    candidate = pred.strip().strip("[]().").strip()
    if candidate.lower() == target:
        return True
    # "D: 1/2" / "D. 1/2" / "D) 1/2" — the letter is the answer, the rest is echo.
    match = re.match(r"^([A-Za-z])\s*[:.)]", candidate)
    if match:
        return match.group(1).lower() == target
    letters = _option_letters(candidate, options)
    return len(letters) == 1 and letters[0] == target


def _judge_multiple_choice_multiple(pred: str, gold: str, options: list[str]) -> bool:
    # Same first move as the single-choice rule: options are usually bare
    # letters, but a problem may label its choices with words, and
    # `_option_letters` only ever matches single characters. Without this the
    # slot would be unwinnable whenever the options are not letters.
    if _squash(pred) == _squash(gold):
        return True
    gold_letters = sorted(_option_letters(gold, options))
    pred_letters = sorted(_option_letters(pred, options))
    return bool(gold_letters) and gold_letters == pred_letters


def _list_elements(text: str) -> list[str]:
    stripped = text.strip()
    while len(stripped) >= 2 and stripped[0] in "([<" and stripped[-1] in ")]>":
        stripped = stripped[1:-1].strip()
    return [element for element in split_answers(stripped) if element]


def _element_equal(pred: str, gold: str, precision: float) -> bool:
    """Compare one OL/UOL element, whose own answer type the dataset omits.

    Deliberately *not* boolean-aware. Upstream converts booleans only for a slot
    the dataset typed ``TF`` and never for the elements inside a list, and the
    elements here are overwhelmingly parameter names: reading ``t``, ``y``,
    ``f`` and ``n`` as truth values makes ``(x, t)`` match a gold ``(x, y)``.
    Thirteen OL/UOL references on the pinned data carry a bare ``t`` or ``y``.
    """
    return math_equal(pred, gold, precision)


def _judge_ordered_list(pred: str, gold: str, precision: float) -> bool:
    pred_items, gold_items = _list_elements(pred), _list_elements(gold)
    if len(pred_items) != len(gold_items):
        return False
    return all(
        _element_equal(p, g, precision)
        for p, g in zip(pred_items, gold_items, strict=True)
    )


def _judge_unordered_list(pred: str, gold: str, precision: float) -> bool:
    pred_items, gold_items = _list_elements(pred), _list_elements(gold)
    if len(pred_items) != len(gold_items):
        return False
    remaining = list(pred_items)
    for gold_item in gold_items:
        for index, pred_item in enumerate(remaining):
            if _element_equal(pred_item, gold_item, precision):
                remaining.pop(index)
                break
        else:
            return False
    return True


def judge_answer(
    pred: str,
    gold: str,
    answer_type: str,
    options: list[str] | None = None,
    precision: float = 1e-3,
) -> bool:
    """Grade one ``[ANS]`` slot under the rule its declared *answer_type* implies.

    Both sides are normalized first. The reference needs it as much as the
    prediction does: some stored answers carry the very decoration the answer
    type says they should not (a percent sign on a "numerical value without
    units"), and comparing a normalized prediction against a raw reference marks
    a right answer wrong.
    """
    pred, gold = normalize_answer(pred), normalize_answer(gold)
    slot_options = list(options or [])
    match answer_type:
        case "TF":
            gold_bool = _to_bool(gold)
            if gold_bool is None:
                # A handful of slots are typed TF but store a number or a word.
                # Upstream's TF judge asserts the reference is True/False and
                # grades the slot wrong when it is not, which makes it
                # unwinnable; compare the stored value instead.
                return math_equal(pred, gold, precision)
            return _to_bool(pred) is gold_bool
        case "MCS":
            return _judge_multiple_choice_single(pred, gold, slot_options)
        case "MCM":
            return _judge_multiple_choice_multiple(pred, gold, slot_options)
        case "OE":
            # A word or phrase: compare as text, never as mathematics.
            return _squash(pred) == _squash(gold)
        case "OL":
            return _judge_ordered_list(pred, gold, precision)
        case "UOL":
            return _judge_unordered_list(pred, gold, precision)
        case _:
            # NV / EX / EQ / INT — all mathematical values.
            return math_equal(pred, gold, precision)


def extract_predictions(response: str) -> list[str] | None:
    """Extract one predicted answer per ``[ANS]`` slot from a full response.

    ``None`` when no answer could be recovered at all — the caller records that
    as "not extracted" rather than as an empty answer.
    """
    extracted = extract_answer(response)
    if extracted is None:
        return None
    return split_answers(extracted)


def judge_answers(
    predictions: list[str] | None,
    golds: list[str],
    answer_types: list[str],
    options: list[list[str]] | None = None,
    precision: float = 1e-3,
) -> list[bool]:
    """Grade extracted answers against a problem's reference sequence.

    Returns one verdict per reference answer. A missing extraction, or a slot
    count that disagrees with the reference, grades every slot wrong: upstream
    rejects a miscounted answer outright rather than aligning a prefix.

    Every reference answer is graded. Where a row declares fewer types than
    answers, the undeclared slots fall through to the mathematical rule; that
    diverges from upstream, which silently truncates both sides to the declared
    types and so never looks at the trailing answers at all.
    """
    if predictions is None or len(predictions) != len(golds):
        return [False] * len(golds)

    effective_types = list(answer_types[: len(golds)])
    effective_types.extend("NV" for _ in range(len(golds) - len(effective_types)))
    per_slot_options = _pad_options(effective_types, options)

    return [
        judge_answer(pred, gold, answer_type, slot_options, precision)
        for pred, gold, answer_type, slot_options in zip(
            predictions, golds, effective_types, per_slot_options, strict=True
        )
    ]
