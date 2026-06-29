"""MathArena-aligned prompting and answer extraction.

Mirrors the eth-sri/matharena (MIT) reference implementation so sieval's
AIME/HMMT 2026 tasks elicit and parse answers the same way matharena.ai does:

* the per-competition ``instruction`` strings from
  ``configs/competitions/{aime,hmmt}/*.yaml`` (final answer in ``\\boxed{}``;
  AIME additionally states the 0-999 integer range);
* the answer extractor is vendored from ``src/matharena/parser.py`` (see the
  ``# adapted from`` header on the extraction section below): take the LAST
  ``\\boxed{}``/``\\fbox{}`` (brace-balanced via recursive regex, inner boxes
  stripped, with the upstream approximation/decimal walk-back) and, when
  ``strict_parsing`` is false, fall back to the last bare integer.

Equivalence judging stays on the ``math-verify`` library in the task layer;
this module only covers prompt construction and answer extraction, so the
upstream sympy/``parse_answer`` grading machinery is intentionally NOT vendored
and the functions return the raw answer ``str`` instead of upstream's
``(value, WarningType)`` tuples.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re

# Instruction strings copied verbatim from the matharena competition configs at
# https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/{aime/aime_2026,hmmt/hmmt_feb_2026}.yaml
# (YAML escapes ``\\boxed{{}}`` which resolves to ``\boxed{}``.)
AIME_INSTRUCTION = (
    "Put your final answer within \\boxed{}.\n"
    "The answer is an integer between 0 and 999 inclusive."
)
HMMT_INSTRUCTION = "Put your final answer within \\boxed{}."


def build_prompt(instruction: str, problem: str) -> str:
    """matharena prompt: the competition instruction followed by the problem."""
    return f"{instruction}\n\n{problem}"


# ---------------------------------------------------------------------------
# adapted from https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/src/matharena/parser.py
#
# Faithful port of the boxed-answer extraction slice. Deliberate adaptations:
#   * functions return ``str | None`` rather than upstream's
#     ``(value, WarningType)`` — sieval does not consume warning levels;
#   * the sympy ``parse_answer`` / ``AnswerList`` grading path is omitted —
#     equivalence is judged by ``math-verify`` in the task layer;
#   * the recursive ``regex`` patterns are compiled lazily (the ``regex`` third-
#     party module is an optional ``[math]`` dependency) so importing a task
#     module does not require the extra, matching sieval's import discipline.
#     Upstream imports ``regex`` at module top.
# ---------------------------------------------------------------------------

APPROX_RE = re.compile(r"\\approx|≈|approximately|approx\.?|about", re.I)
EXACT_BOX_TOKENS = (r"\frac", r"\sqrt", r"\binom", "/", "^")
_LAST_INT = re.compile(r"\b\d+\b")

# Recursive brace-balanced box patterns use the ``regex`` module's ``(?2)``
# subroutine call, which stdlib ``re`` cannot express. Compiled on first use.
_BOX_PATTERN = None
_INNER_BOX_PATTERN = None


def _box_pattern():
    global _BOX_PATTERN
    if _BOX_PATTERN is None:
        import regex

        _BOX_PATTERN = regex.compile(r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}")
    return _BOX_PATTERN


def _inner_box_pattern():
    global _INNER_BOX_PATTERN
    if _INNER_BOX_PATTERN is None:
        import regex

        _INNER_BOX_PATTERN = regex.compile(r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}")
    return _INNER_BOX_PATTERN


def remove_inner_boxed(match: str) -> str:
    """Unwrap any nested ``\\boxed{X}``/``\\fbox{X}`` to ``X`` (matharena parity)."""
    matches = list(_inner_box_pattern().finditer(match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def contains_approximation(s: str) -> bool:
    """Whether a boxed answer looks like a trailing approximation."""
    return APPROX_RE.search(s) is not None


def should_prefer_previous_boxed_over_approximation(s: str) -> bool:
    if not contains_approximation(s):
        return False

    prefix = APPROX_RE.split(s, maxsplit=1)[0].strip()
    if not prefix:
        return True
    if "=" in prefix or any(token in prefix for token in EXACT_BOX_TOKENS):
        return False
    return re.fullmatch(r"[\s$(){}\[\].,0-9+\-]+", prefix) is not None


def is_decimal_approximation_box(s: str) -> bool:
    """Whether a boxed value is only a decimal approximation."""
    s = s.replace(r"\displaystyle", "")
    s = s.replace("$", "").replace(",", "").strip()
    s = re.sub(r"\\[,;:! ]", "", s)
    s = re.sub(r"\s+", "", s)
    return re.fullmatch(r"[-+]?\d+\.\d+(?:\.\.\.)?", s) is not None


def looks_like_exact_boxed_expression(s: str) -> bool:
    """Whether a boxed expression is a plausible exact answer."""
    if contains_approximation(s) or is_decimal_approximation_box(s):
        return False
    return any(token in s for token in EXACT_BOX_TOKENS)


def find_last_boxed_content(text: str, list_answer: bool = False) -> str | None:
    """Content of the last ``\\boxed{}``/``\\fbox{}``, with the upstream walk-back.

    When several boxes are present, prefer an earlier exact box over a trailing
    approximation (``\\approx``/``≈``) or a bare decimal approximation.
    """
    matches = list(_box_pattern().finditer(text))
    if not matches:
        return None

    if len(matches) > 1 and list_answer:
        # find all boxed content on the same line (no \n in between) as the last box
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(_box_pattern().finditer(split_text[i]))
            if len(matches_line) > 0:
                # If a full list answer is boxed repeatedly on the final line,
                # joining all boxes duplicates it. Multiple scalar boxes are
                # still joined, e.g. \boxed{2}, \boxed{3}.
                if any("," in match.group(2) for match in matches_line):
                    returned_boxed = matches_line[-1].group(2)
                else:
                    returned_boxed = ",".join(
                        [match.group(2) for match in matches_line]
                    )
                return remove_inner_boxed(returned_boxed)

    selected_match = matches[-1]
    if len(matches) > 1 and should_prefer_previous_boxed_over_approximation(
        selected_match.group(2)
    ):
        for match in reversed(matches[:-1]):
            if not contains_approximation(match.group(2)):
                selected_match = match
                break
    if len(matches) > 1 and is_decimal_approximation_box(selected_match.group(2)):
        for match in reversed(matches[:-1]):
            if looks_like_exact_boxed_expression(match.group(2)):
                selected_match = match
                break

    return remove_inner_boxed(selected_match.group(2))


def extract_boxed_answer(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{}``/``\\fbox{}``, or None."""
    answer = find_last_boxed_content(text)
    return answer.strip() if answer is not None else None


def extract_last_integer(text: str) -> str | None:
    """matharena's non-strict fallback: the last bare integer in the text."""
    matches = _LAST_INT.findall(text)
    return matches[-1] if matches else None


def extract_answer(text: str, strict_parsing: bool = False) -> str | None:
    """matharena-aligned extraction.

    Take the last ``\\boxed{}``; if absent and ``strict_parsing`` is false,
    fall back to the last integer (both AIME 2026 and HMMT Feb 2026 set
    ``strict_parsing: false``).
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    if not strict_parsing:
        return extract_last_integer(text)
    return None
