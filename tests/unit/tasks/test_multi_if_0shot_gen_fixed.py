"""Registration and seam contract for the corrected Multi-IF task.

The repairs themselves are tested in
``tests/unit/community/test_instruction_following_eval_fixed.py``. What this
module pins is that the variant is the unqualified task plus *one* overridden
method — conversation walking, per-turn grading, the cumulative constraint lists
and the per-language pooling all inherited — plus the one defect that is
Multi-IF-specific: a ``first_word`` spanning more than one token, which upstream
cannot pass with any response at all.

Every fixture is synthetic. Nothing here quotes Multi-IF's prompts or kwargs.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio

import pytest

from sieval.core.tasks.meta import get_task_meta
from sieval.tasks.multi_if_0shot_gen import MultiIFZeroShotGenTask
from sieval.tasks.multi_if_0shot_gen_fixed import MultiIFZeroShotGenFixedTask

_NTH = "length_constraints:nth_paragraph_first_word"
_NO_COMMA = "punctuation:no_comma"


def test_meta():
    meta = get_task_meta(MultiIFZeroShotGenFixedTask)
    base = get_task_meta(MultiIFZeroShotGenTask)
    assert meta.name == "multi_if_0shot_gen_fixed"
    # Resolved through the MRO, since the subclass re-declares no generic base.
    assert meta.dataset == base.dataset
    assert meta.eval_mode == base.eval_mode
    assert meta.deps_group == base.deps_group == "multi-if"
    assert meta.status == "stable"


def test_reference_impl_quantifies_the_divergence():
    reference_impl = get_task_meta(MultiIFZeroShotGenFixedTask).reference_impl
    assert reference_impl is not None
    notes = reference_impl.notes
    for instruction_id in (
        "length_constraints:nth_paragraph_first_word",
        "keywords:letter_frequency",
        "change_case:english_capital",
    ):
        assert instruction_id in notes
    assert "SCORE IMPACT" in notes
    assert "69.07→70.52" in notes


def _task(cls):
    # `__new__`: the constructor requires a model with a bound dialect_id, and
    # nothing below reaches the model.
    return cls.__new__(cls)


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


def test_the_base_task_still_grades_through_the_vendored_registry():
    assert _task(MultiIFZeroShotGenTask)._instruction_dict() is None


def test_the_fixed_task_returns_the_repaired_registry():
    from sieval.community.instruction_following_eval_fixed import (
        fixed_multi_if_registry,
    )

    registry = _task(MultiIFZeroShotGenFixedTask)._instruction_dict()
    assert registry == fixed_multi_if_registry()


def test_the_fixed_task_uses_multi_ifs_own_registry_not_ifevals():
    # The two vendored copies are logic-identical for the three repaired
    # checkers, but not for the other 22: routing Multi-IF through IFEval's
    # registry would quietly change the multilingual checkers.
    from sieval.community.instruction_following_eval_fixed import (
        fixed_ifeval_registry,
    )
    from sieval.community.multi_if import ifeval as multi_if_upstream

    registry = _task(MultiIFZeroShotGenFixedTask)._instruction_dict()
    assert registry != fixed_ifeval_registry()
    unrepaired = set(registry) - {
        _NTH,
        "keywords:letter_frequency",
        "change_case:english_capital",
    }
    for instruction_id in unrepaired:
        assert (
            registry[instruction_id]
            is multi_if_upstream.INSTRUCTION_DICT[instruction_id]
        )


def test_the_fixed_task_hands_out_a_fresh_registry_each_call():
    task = _task(MultiIFZeroShotGenFixedTask)
    assert task._instruction_dict() is not task._instruction_dict()


def test_exactly_one_method_differs_from_the_unqualified_task():
    overrides = {
        name
        for name, value in vars(MultiIFZeroShotGenFixedTask).items()
        if callable(value) and hasattr(MultiIFZeroShotGenTask, name)
    }
    assert overrides == {"_instruction_dict"}
    for name in ("preprocess", "infer", "postprocess", "feedback", "report"):
        assert getattr(MultiIFZeroShotGenFixedTask, name) is getattr(
            MultiIFZeroShotGenTask, name
        )


def _sample(instruction_ids, kwargs) -> dict:
    # One turn is enough: what differs between the two tasks is per-response
    # grading, and the conversation walk is inherited and tested elsewhere.
    return {
        "key": "k:1:en",
        "language": "English",
        "turns": [
            {
                "prompt": "write something",
                "instruction_id_list": list(instruction_ids),
                # JSON-encoded in the dataset and decoded by `feedback`.
                "kwargs": list(kwargs),
            }
        ],
    }


def _judge(cls, raw, response):
    post = {
        "rollouts": [
            {
                "index": 0,
                "extracted": True,
                "prediction": [{"turn": 1, "response": response}],
            }
        ]
    }
    _final, judgement = asyncio.run(_task(cls).feedback(post, _Ctx(raw)))
    return judgement


def test_the_two_tasks_agree_on_a_turn_with_no_repaired_constraint():
    raw = _sample([_NO_COMMA], ["{}"])
    base = _judge(MultiIFZeroShotGenTask, raw, "no commas here")
    fixed = _judge(MultiIFZeroShotGenFixedTask, raw, "no commas here")
    assert base == fixed


def test_the_fixed_task_reads_the_paragraph_the_prompt_asked_for():
    raw = _sample(
        [_NTH],
        ['{"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}'],
    )
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    base = _judge(MultiIFZeroShotGenTask, raw, response)
    fixed = _judge(MultiIFZeroShotGenFixedTask, raw, response)
    assert base["metrics"]["turn_1_strict_follow_all"] is False
    assert fixed["metrics"]["turn_1_strict_follow_all"] is True


def test_upstream_cannot_pass_a_multi_token_first_word_at_all():
    # The Multi-IF-only defect, and the reason it is worth a repair rather than a
    # note: upstream compares against one whitespace-delimited token, so a slot
    # whose value spans two returns FAIL for *every* response. A check that
    # cannot pass measures nothing -- it is not a hard constraint, it is a
    # constant.
    raw = _sample(
        [_NTH],
        ['{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "once upon"}'],
    )
    for response in (
        "alpha one\n\nonce upon a time",
        "alpha one\n\nonce a time",
        "alpha one\n\nupon once a time",
    ):
        assert (
            _judge(MultiIFZeroShotGenTask, raw, response)["metrics"][
                "turn_1_strict_follow_all"
            ]
            is False
        )

    # The repair grades it: the phrase must open the paragraph, in order.
    passed = _judge(MultiIFZeroShotGenFixedTask, raw, "alpha one\n\nonce upon a time")
    assert passed["metrics"]["turn_1_strict_follow_all"] is True
    for response in ("alpha one\n\nonce a time", "alpha one\n\nupon once a time"):
        assert (
            _judge(MultiIFZeroShotGenFixedTask, raw, response)["metrics"][
                "turn_1_strict_follow_all"
            ]
            is False
        )


def test_a_blank_first_word_stays_ungradeable_in_both_tasks():
    # Three slots in the pinned set ship an empty value. A constraint that states
    # nothing to check is not this repair's to invent a rule for, so it keeps
    # failing exactly as upstream does -- the repair is not a backfill.
    raw = _sample(
        [_NTH],
        ['{"num_paragraphs": 1, "nth_paragraph": 1, "first_word": ""}'],
    )
    for response in ("alpha one", "\n\nalpha one"):
        base = _judge(MultiIFZeroShotGenTask, raw, response)
        fixed = _judge(MultiIFZeroShotGenFixedTask, raw, response)
        assert base["metrics"]["turn_1_strict_follow_all"] is False
        assert fixed["metrics"]["turn_1_strict_follow_all"] is False


def test_a_mixed_turn_moves_only_at_the_repaired_constraint():
    raw = _sample(
        [_NO_COMMA, _NTH],
        ["{}", '{"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}'],
    )
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    base = _judge(MultiIFZeroShotGenTask, raw, response)
    fixed = _judge(MultiIFZeroShotGenFixedTask, raw, response)
    assert base["extra"]["turn_1"]["strict"]["follow_instruction_list"] == [True, False]
    assert fixed["extra"]["turn_1"]["strict"]["follow_instruction_list"] == [True, True]
    assert base["metrics"]["turn_1_strict_instruction_level"] == pytest.approx(0.5)
    assert fixed["metrics"]["turn_1_strict_instruction_level"] == pytest.approx(1.0)


def test_report_is_inherited_verbatim():
    assert MultiIFZeroShotGenFixedTask.report is MultiIFZeroShotGenTask.report
