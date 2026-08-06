"""Unit tests for the vendored AGIEval prompt / parse / score layer.

Pins the behaviours the port's fidelity claims rest on: the exact upstream prompt
strings, the family routing, and the three comparison rules (including the
gaokao-mathqa multi-letter-gold quirk that is kept on purpose).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.community.agieval.dataset_loader import (
    CHINESE_CLOZE_SUBSETS,
    CHINESE_QA_SUBSETS,
    ENGLISH_CLOZE_SUBSETS,
    ENGLISH_QA_SUBSETS,
    MATH_SUBSETS,
    SUBSETS,
    second_stage_prompt,
    zero_shot_prompt,
)
from sieval.community.agieval.evaluation import (
    LEADERBOARD_EN_MCQ_SUBSETS,
    LEADERBOARD_ZH_MCQ_SUBSETS,
    evaluate_single_sample,
)
from sieval.community.agieval.math_equivalence import is_equiv
from sieval.community.agieval.post_process import post_process


def _row(**overrides) -> dict:
    row = {
        "passage": None,
        "question": "Q?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4"],
        "label": "A",
        "answer": None,
    }
    row.update(overrides)
    return row


def test_subsets_cover_the_21_v1_1_files_without_overlap():
    families = (
        ENGLISH_QA_SUBSETS,
        CHINESE_QA_SUBSETS,
        ENGLISH_CLOZE_SUBSETS,
        CHINESE_CLOZE_SUBSETS,
    )
    assert len(SUBSETS) == 21
    assert len(set(SUBSETS)) == 21
    assert sum(len(family) for family in families) == 21


def test_leaderboard_groups_are_not_the_prompt_families():
    # gaokao-english is prompted in English but averaged as Chinese upstream;
    # a port that reused the prompt families for reporting would get both wrong.
    assert "gaokao-english" in ENGLISH_QA_SUBSETS
    assert "gaokao-english" in LEADERBOARD_ZH_MCQ_SUBSETS
    assert "gaokao-english" not in LEADERBOARD_EN_MCQ_SUBSETS
    assert len(LEADERBOARD_EN_MCQ_SUBSETS) == 8
    assert len(LEADERBOARD_ZH_MCQ_SUBSETS) == 11
    # The math group spans both languages and both formats.
    assert set(MATH_SUBSETS) == {
        "sat-math",
        "aqua-rat",
        "gaokao-mathqa",
        "math",
        "gaokao-mathcloze",
    }


def test_zero_shot_prompt_english_qa_matches_upstream_string():
    prompt = zero_shot_prompt("sat-math", _row(passage="P. "))
    assert prompt == (
        "P. Q: Q? Answer Choices: (A)1 (B)2 (C)3 (D)4\n"
        "A: Among A through D, the answer is"
    )


def test_zero_shot_prompt_letter_tracks_option_count():
    five = ["(A)1", "(B)2", "(C)3", "(D)4", "(E)5"]
    assert zero_shot_prompt("aqua-rat", _row(options=five)).endswith(
        "Among A through E, the answer is"
    )


def test_zero_shot_prompt_chinese_qa_matches_upstream_string():
    prompt = zero_shot_prompt("gaokao-mathqa", _row())
    assert prompt == "问题：Q? 选项：(A)1 (B)2 (C)3 (D)4\n答案：从A到D, 我们应选择"


def test_zero_shot_prompt_cloze_subsets_omit_options():
    assert zero_shot_prompt("math", _row(options=[], label=None, answer="7")) == (
        "Q: Q?\nA: The answer is"
    )
    assert zero_shot_prompt(
        "gaokao-mathcloze", _row(options=[], label=None, answer="7")
    ) == ("问题：Q?\n答案：")


def test_zero_shot_prompt_rejects_unknown_subset():
    with pytest.raises(ValueError, match="Unknown AGIEval subset"):
        zero_shot_prompt("mmlu", _row())


def test_second_stage_prompt_appends_the_family_cue():
    assert second_stage_prompt("sat-math", "CTX", "reasoning") == (
        "CTX\nreasoning\nTherefore, among A through E, the answer is"
    )
    assert second_stage_prompt("gaokao-mathqa", "CTX", "推理") == (
        "CTX\n推理\n因此，从A到D, 我们应选择"
    )
    assert second_stage_prompt("math", "CTX", "r") == (
        "CTX\nr\nTherefore, the answer is"
    )
    assert second_stage_prompt("gaokao-mathcloze", "CTX", "r") == "CTX\nr\n因此，答案是"


def test_post_process_single_answer_takes_first_capital_letter():
    assert post_process("logiqa-en", " C.") == "C"
    # Upstream's parser is positional, not semantic: the first A-F character
    # wins, even mid-word ("Among" -> A), which is why it only ever runs on the
    # terse second-stage reply and not on a chain of thought.
    assert post_process("logiqa-en", "Among the options, D is right") == "A"
    assert post_process("logiqa-en", "no letters here") is None


def test_post_process_multi_answer_subsets_return_every_letter():
    assert post_process("jec-qa-kd", "(A) and (C)") == ["A", "C"]
    assert post_process("gaokao-physics", " B") == ["B"]
    assert post_process("jec-qa-ca", "no letters") is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (" $\\boxed{\\frac{1}{2}}$", "\\frac{1}{2}"),
        (" $x = 42$", "42"),
        (" 42", "42"),
        (" the value is 3.5", "3.5"),
        ("The answer is therefore 8", "8"),
        (" no answer at all", None),
    ],
)
def test_post_process_cloze_extracts_the_math_answer(reply, expected):
    assert post_process("math", reply) == expected


def test_post_process_rejects_unknown_subset():
    with pytest.raises(ValueError, match="Unknown AGIEval subset"):
        post_process("mmlu", "A")


def test_evaluate_single_sample_compares_multi_answer_as_a_set():
    assert evaluate_single_sample("jec-qa-kd", ["A"], "A") is True
    # Order-insensitive, and a superset is wrong rather than partially right.
    assert evaluate_single_sample("jec-qa-ca", ["B", "A"], "A") is False
    assert evaluate_single_sample("gaokao-physics", None, "A") is False


def test_evaluate_single_sample_uses_math_equivalence_for_cloze():
    assert evaluate_single_sample("math", "\\frac{1}{2}", "0.5") is True
    assert evaluate_single_sample("gaokao-mathcloze", "2", "3") is False
    assert evaluate_single_sample("math", None, "2") is False


def test_evaluate_single_sample_is_exact_elsewhere_including_the_quirk():
    assert evaluate_single_sample("sat-math", "D", "D") is True
    # 7 gaokao-mathqa rows ship multi-letter gold while the subset is scored by
    # exact compare, so no single-letter prediction can win them. Kept verbatim
    # from upstream — see the task's reference_impl.notes.
    assert evaluate_single_sample("gaokao-mathqa", "A", "AD") is False


def test_is_equiv_applies_the_normalizations_sieval_community_math_drops():
    # These four are exactly the steps sieval.community.math.strip_string omits,
    # which is why AGIEval keeps its own copy.
    assert is_equiv("2 cm", "2cm") is True  # space removal
    assert is_equiv("50\\%", "50") is True  # percent removal
    assert is_equiv("90^\\circ", "90") is True  # degree removal
    assert is_equiv("k = 5", "5") is True  # leading "k = " strip
    assert is_equiv(None, "5") is False
