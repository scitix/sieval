"""Unit tests for the DeepSeek-Math eval adaptation.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest

from sieval.community.deepseek_math import (
    STOP_WORDS,
    eval_math,
    extract_math_answer,
    extract_math_few_shot_cot_answer,
    format_prompt,
    is_correct,
    math_equal,
    symbolic_equal,
)

_FA = "\nFinal Answer: The final answer is ${}$. I hope it is correct."


def test_stop_words_have_leading_newline():
    assert STOP_WORDS == ["\nProblem:"]


def test_format_prompt_minerva_layout():
    p = format_prompt("Find $x$.", "")
    assert p.count("Problem:\n") == 5  # 4 baked exemplars + query
    assert p.endswith("Problem:\nFind $x$.\n\nSolution:")  # rstrip drops trailing nl


def test_extract_prefers_final_answer_line():
    assert extract_math_few_shot_cot_answer("q", f"work{_FA.format('24')}", "cot") == [
        "24"
    ]


def test_extract_truncates_hallucinated_next_problem():
    text = f"work{_FA.format('24')}\n\nProblem:\nnext\n\nSolution: $99$"
    assert extract_math_few_shot_cot_answer("q", text, "cot") == ["24"]


def test_extract_falls_back_to_boxed():
    # no "Final Answer" line -> boxed extraction
    assert extract_math_few_shot_cot_answer("q", "so $\\boxed{7}$", "cot") == ["7"]


def test_reference_answer_is_list_from_boxed_solution():
    assert extract_math_answer("q", "thus $\\boxed{[2,5)}$.", "cot") == ["[2,5)"]


def test_eval_math_set_matches_lists():
    # order-insensitive multiset matching of pred vs ref lists
    assert bool(eval_math({"prediction": ["3", "1", "2"], "answer": ["1", "2", "3"]}))
    assert not bool(eval_math({"prediction": ["1", "2"], "answer": ["1", "2", "3"]}))


def test_eval_math_percentage_numeric_layer():
    # math_equal numeric layer: 50\% == 0.5 (no parse_latex needed)
    assert bool(eval_math({"prediction": ["50\\%"], "answer": ["0.5"]}))


# --- execution safety: symbolic_equal must not run the answer it grades ------


def _payload(target):
    return f"__import__('os').system('touch {target}')"


@pytest.mark.parametrize("entry", ["symbolic_equal", "math_equal", "is_correct"])
def test_grading_a_payload_executes_nothing(entry, tmp_path):
    """Upstream runs this. Reachable from a model's extracted answer."""
    target = tmp_path / entry
    call = {
        "symbolic_equal": lambda p: symbolic_equal(p, "1"),
        "math_equal": lambda p: math_equal(p, "1"),
        "is_correct": lambda p: is_correct({"prediction": p, "answer": "1"}),
    }[entry]
    assert call(_payload(target)) is False
    assert not target.exists()


def test_quoteless_payload_via_the_raw_string_fallback(tmp_path):
    """The second path: unparseable text used to reach `simplify`/`N` verbatim.

    Guarding only `parse_expr` would leave this open — `sympify` uses sympy's
    default namespace, where `__import__` resolves with no quote required.
    """
    target = tmp_path / "fallthrough"

    # chr() builds every string without a quote, so the quote screen alone
    # would not catch it; the refusal of the raw-string fallback does.
    def chrs(text):
        return "+".join(f"chr({ord(c)})" for c in text)

    payload = f"__import__({chrs('os')}).system({chrs(f'touch {target}')})"
    assert symbolic_equal(payload, "1") is False
    assert not target.exists()


def test_unparseable_answer_is_refused_not_sympified():
    """The behavioural shape of the divergence: no verdict is invented."""
    assert symbolic_equal("not math at all", "1") is False


@pytest.mark.parametrize(
    "prediction, reference",
    [
        ("\\frac{1}{2}", "0.5"),
        ("x^{1/2}", "\\sqrt{x}"),
        ("7(x-3)(x+3)", "7(x+3)(x-3)"),
        ("\\frac{3\\sqrt{20}}{5}", "\\frac{6\\sqrt{5}}{5}"),
    ],
)
def test_symbolic_equality_still_works(prediction, reference):
    """The guards must not cost the symbolic path they protect.

    All four are real disagreements from the stored MATH run that only the
    symbolic path resolves — they are exactly what is lost when `parse_latex`
    is unavailable, so they also pin that the `[math]` extra's antlr4 pin is
    doing its job.

    Spelled in LaTeX on purpose: `parse_latex` is tried first and *succeeds*
    on `x**2 - 1` by silently truncating at the `**` (returning `x`), so a
    Python-spelled case never reaches `parse_expr` and tests nothing here.
    That is upstream behaviour, unchanged by the guards.
    """
    assert symbolic_equal(prediction, reference) is True
    assert bool(eval_math({"prediction": [prediction], "answer": [reference]}))
