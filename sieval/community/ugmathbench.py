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

* Symbolic equivalence comes from ``math-verify``'s parse/verify rather than
  upstream's bespoke ``parse_latex`` + ``simplify`` + random-value-substitution
  chain. It is stricter on some near-equal expressions (no numeric sampling of
  free symbols) and more permissive on LaTeX shapes upstream's normalizer never
  learned. Numeric slots keep upstream's *relative* tolerance as a second
  chance, since ``math-verify`` compares floats at fixed decimal rounding.
* No answer-type *inference*: upstream's ``is_equal`` retries every judgement
  method until one accepts, which lets a slot be graded by a rule its declared
  type did not ask for. Here the declared type decides, except inside OL/UOL
  elements (whose own types the dataset does not record).
* Extraction is strict, matching upstream's ``Judger(strict_extract=True)`` as
  used by ``eval_rule.py``: no "guess the last number in the response" fallback.
* Both sides are normalized before comparison, references included. A few
  stored answers carry decoration their own type forbids (a percent sign on "a
  numerical value without units"), and normalizing only the prediction marks
  those right answers wrong.
* A ``TF`` slot whose reference is not a boolean at all (9 of 1665, e.g. ``-22``,
  ``not real``) is compared as a value rather than graded wrong outright, which
  is what upstream's assert-then-swallow does. Those slots are otherwise
  unwinnable regardless of what the model answers.
* Per-slot verdicts are returned rather than only the sample-level ``all()``,
  so a wrong answer can be located without re-running the grader.

Not a divergence, but worth knowing: 26 of the 15,183 pinned rows (0.17%) store
references the "comma-separated answers in one box" protocol cannot express —
a comma inside an open-ended phrase, or an unbalanced bracket left by upstream's
own answer splitting. They cannot be answered correctly here or upstream.

The prompt builder, by contrast, IS exact: it reproduces upstream's ``raw``
template byte-for-byte on all 15,183 rows. Scores are nonetheless *not*
guaranteed to reproduce the paper's rule-based numbers, which is why the task
ships ``status="experimental"``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re

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
    """
    description = TYPE_DESCRIPTIONS.get(answer_type)
    if description is None:
        raise KeyError(
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
    """
    per_slot_options = _pad_options(answer_types, options)
    descriptions = ", ".join(
        describe_answer_type(answer_type, slot_options)
        for answer_type, slot_options in zip(
            answer_types, per_slot_options, strict=True
        )
    )

    head = _PROMPT_HEAD.format(subject=subject)
    if n_answers == 1:
        types = _PROMPT_SINGLE_TYPES.format(descriptions=descriptions)
        tail = _PROMPT_TAIL_SINGLE
    else:
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


_OPENERS = {"(": ")", "[": "]", "{": "}", "<": ">"}
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

    Unlike upstream this also tracks ``{}``, which keeps a comma inside a LaTeX
    group (``\\frac{a,b}{c}``) from splitting a slot in two.
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

_TRUE_WORDS = {"t", "y", "true", "yes"}
_FALSE_WORDS = {"f", "n", "false", "no"}
_WHITESPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    """Whitespace- and decoration-insensitive form, for the cheap equality path."""
    return _WHITESPACE.sub("", text.replace("$", "").replace("\\", "")).lower()


def _to_bool(text: str) -> bool | None:
    word = text.strip().strip(".").lower()
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

    # `$`-wrapping steers the LaTeX extractor at a bare answer string; the Expr
    # config behind it still catches the plain-sympy shapes the dataset stores
    # ("x^3+2*x^2+6", "(-infinity,5)").
    return parse(f"${text}$")


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

    Three chances, cheapest first: squashed string equality, plain-float
    comparison at *precision* (relative, as upstream), then ``math-verify``
    symbolic equivalence with a final relative-tolerance retry on the parsed
    numeric values (``math-verify`` compares floats at fixed decimal rounding,
    which rejects pairs upstream's relative tolerance accepts).
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
        if gold_value is not None and pred_value is not None:
            return _close(pred_value, gold_value, precision)
    except Exception:
        # math-verify raises on pathological input (unbalanced LaTeX, runaway
        # simplify). An ungradeable answer is a wrong answer, not a crashed run.
        return False
    return False


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
    gold_letters = sorted(_option_letters(gold, options))
    pred_letters = sorted(_option_letters(pred, options))
    return bool(gold_letters) and gold_letters == pred_letters


def _list_elements(text: str) -> list[str]:
    stripped = text.strip()
    while len(stripped) >= 2 and stripped[0] in "([<" and stripped[-1] in ")]>":
        stripped = stripped[1:-1].strip()
    return [element for element in split_answers(stripped) if element]


def _element_equal(pred: str, gold: str, precision: float) -> bool:
    """Compare one OL/UOL element, whose own answer type the dataset omits."""
    pred_bool, gold_bool = _to_bool(pred), _to_bool(gold)
    if gold_bool is not None:
        return pred_bool is gold_bool
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
