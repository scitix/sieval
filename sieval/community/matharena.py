"""MathArena-aligned prompting and answer extraction.

Mirrors the eth-sri/matharena (MIT) reference implementation so sieval's
AIME/HMMT 2026 tasks elicit and parse answers the same way matharena.ai does:

* the per-competition ``instruction`` strings from
  ``configs/competitions/{aime,hmmt}/*.yaml`` (final answer in ``\\boxed{}``;
  AIME additionally states the 0-999 integer range);
* ``extract_answer`` reproduces ``src/matharena/parser.py``: take the LAST
  ``\\boxed{}``/``\\fbox{}`` (brace-balanced, inner boxes stripped) and, when
  ``strict_parsing`` is false, fall back to the last integer in the text.

Equivalence judging stays on the ``math-verify`` library in the task layer;
this module only covers prompt construction and answer extraction.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re

# Instruction strings copied verbatim from the matharena competition configs.
# (YAML escapes ``\\boxed{{}}`` which resolves to ``\boxed{}``.)
AIME_INSTRUCTION = (
    "Put your final answer within \\boxed{}.\n"
    "The answer is an integer between 0 and 999 inclusive."
)
HMMT_INSTRUCTION = "Put your final answer within \\boxed{}."

_BOX_OPEN = re.compile(r"\\(?:boxed|fbox)\s*\{")
_LAST_INT = re.compile(r"-?\d+")


def build_prompt(instruction: str, problem: str) -> str:
    """matharena prompt: the competition instruction followed by the problem."""
    return f"{instruction}\n\n{problem}"


def _iter_boxed(text: str):
    """Yield the brace-balanced content of each ``\\boxed{...}``/``\\fbox{...}``."""
    for m in _BOX_OPEN.finditer(text):
        depth = 1
        i = m.end()
        start = i
        while i < len(text) and depth:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            yield text[start : i - 1]


def _strip_inner_boxed(s: str) -> str:
    """Unwrap any nested ``\\boxed{X}``/``\\fbox{X}`` to ``X`` (matharena parity)."""
    while True:
        m = _BOX_OPEN.search(s)
        if not m:
            return s
        depth = 1
        i = m.end()
        start = i
        while i < len(s) and depth:
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth:
            return s  # unbalanced — leave as-is
        s = s[: m.start()] + s[start : i - 1] + s[i:]


def extract_boxed_answer(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{}``/``\\fbox{}``, or None."""
    boxes = list(_iter_boxed(text))
    if not boxes:
        return None
    return _strip_inner_boxed(boxes[-1]).strip()


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
