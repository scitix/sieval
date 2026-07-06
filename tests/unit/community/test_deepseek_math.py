"""Unit tests for the DeepSeek-Math eval adaptation.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.deepseek_math import (
    STOP_WORDS,
    eval_math,
    extract_math_answer,
    extract_math_few_shot_cot_answer,
    format_prompt,
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
