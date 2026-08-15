"""Registration and seam contract for the corrected IFEval task.

The repairs themselves are tested in
``tests/unit/community/test_instruction_following_eval_fixed.py``. What this
module pins is the claim the variant is built on: that it is the unqualified
task plus *one* overridden method, so any measured delta is the repaired
checkers and cannot be a second change riding along.

Fixtures avoid ``change_case:*`` constraints on purpose — those route through
``langdetect``, which upstream leaves unseeded, so a verdict built on them can
flip between runs.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio
import subprocess
import sys

import pytest

from sieval.core.tasks.meta import get_task_meta
from sieval.tasks.ifeval_0shot_gen import IFEvalZeroShotGenTask
from sieval.tasks.ifeval_0shot_gen_fixed import IFEvalZeroShotGenFixedTask

_NTH = "length_constraints:nth_paragraph_first_word"
_NO_COMMA = "punctuation:no_comma"


def test_import_does_not_pull_evaluation_lib():
    # Same contract as the unqualified task's, and one more: the repair module
    # reaches the vendored checkers and langdetect, so importing it at module
    # scope would make *registration* -- which imports every task module --
    # pay for them.
    code = (
        "import sys\n"
        "import sieval.tasks.ifeval_0shot_gen_fixed\n"
        "assert 'sieval.community.instruction_following_eval.evaluation_lib' "
        "not in sys.modules, 'evaluation_lib must be lazy-imported'\n"
        "assert 'sieval.community.instruction_following_eval_fixed' "
        "not in sys.modules, 'the repair module must be lazy-imported'\n"
        "assert 'langdetect' not in sys.modules, 'langdetect must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules
    # don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_meta():
    meta = get_task_meta(IFEvalZeroShotGenFixedTask)
    base = get_task_meta(IFEvalZeroShotGenTask)
    assert meta.name == "ifeval_0shot_gen_fixed"
    # The dataset FK resolves through the MRO: the subclass never re-declares a
    # generic base, so this is what proves it inherited one rather than silently
    # registering against a different sample type.
    assert meta.dataset == base.dataset
    assert meta.eval_mode == base.eval_mode
    assert meta.n_shot == base.n_shot == 0
    assert meta.deps_group == base.deps_group == "ifeval"
    # Not gated on reproducing a published number: the divergence is carried by
    # the name, and what `_fixed` owes is the quantified delta in `notes`.
    assert meta.status == "stable"


def test_reference_impl_quantifies_the_divergence():
    # `_fixed` is licensed by a defect, not a preference, and owes two things:
    # every divergence enumerated, and a measured score impact. An unmeasured
    # fork is not a fix.
    reference_impl = get_task_meta(IFEvalZeroShotGenFixedTask).reference_impl
    assert reference_impl is not None
    notes = reference_impl.notes
    for instruction_id in (
        "length_constraints:nth_paragraph_first_word",
        "keywords:letter_frequency",
        "change_case:english_capital",
    ):
        assert instruction_id in notes
    assert "SCORE IMPACT" in notes
    assert "92.79→95.38" in notes


def _task(cls):
    # `__new__`: the constructor requires a model with a bound dialect_id, and
    # nothing below reaches the model.
    return cls.__new__(cls)


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


def test_the_base_task_still_grades_through_the_vendored_registry():
    # The unqualified name tracks upstream, bugs included. `None` is what makes
    # the graders fall back to the vendored `INSTRUCTION_DICT`, so this is the
    # assertion that the repairs are unreachable from it.
    assert _task(IFEvalZeroShotGenTask)._instruction_dict() is None


def test_the_fixed_task_returns_the_repaired_registry():
    from sieval.community.instruction_following_eval_fixed import (
        fixed_ifeval_registry,
    )

    registry = _task(IFEvalZeroShotGenFixedTask)._instruction_dict()
    assert registry == fixed_ifeval_registry()


def test_the_fixed_task_hands_out_a_fresh_registry_each_call():
    # Samples grade concurrently; a shared dict would let one sample's grading
    # change another's.
    task = _task(IFEvalZeroShotGenFixedTask)
    assert task._instruction_dict() is not task._instruction_dict()


def test_exactly_one_method_differs_from_the_unqualified_task():
    # The whole evidential weight of the measured delta rests on this: prompt,
    # graders, records and report are inherited, so the two tasks cannot diverge
    # anywhere except at the registry.
    overrides = {
        name
        for name, value in vars(IFEvalZeroShotGenFixedTask).items()
        if callable(value) and hasattr(IFEvalZeroShotGenTask, name)
    }
    assert overrides == {"_instruction_dict"}
    for name in ("preprocess", "infer", "postprocess", "feedback", "report"):
        assert getattr(IFEvalZeroShotGenFixedTask, name) is getattr(
            IFEvalZeroShotGenTask, name
        )


def _sample(instruction_ids, kwargs) -> dict:
    return {
        "key": 1,
        "prompt": "write something",
        "instruction_id_list": list(instruction_ids),
        "kwargs": list(kwargs),
    }


def _judge(cls, raw, response):
    post = {"rollouts": [{"index": 0, "extracted": True, "prediction": response}]}
    _final, judgement = asyncio.run(_task(cls).feedback(post, _Ctx(raw)))
    return judgement


def test_the_two_tasks_agree_on_a_sample_with_no_repaired_constraint():
    # Not "the scores match" but "the records match": 22 of the 25 checkers are
    # the same objects in both registries, so a sample built only from those must
    # produce a byte-identical judgement.
    raw = _sample([_NO_COMMA], [{}])
    base = _judge(IFEvalZeroShotGenTask, raw, "no commas here")
    fixed = _judge(IFEvalZeroShotGenFixedTask, raw, "no commas here")
    assert base == fixed


def test_the_fixed_task_reads_the_paragraph_the_prompt_asked_for():
    # The response opens with a blank line, which is what whole runs do; upstream
    # then reads paragraph 1 in slot 2 and fails a compliant answer.
    raw = _sample(
        [_NTH],
        [{"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}],
    )
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    base = _judge(IFEvalZeroShotGenTask, raw, response)
    fixed = _judge(IFEvalZeroShotGenFixedTask, raw, response)
    assert base["metrics"]["strict_follow_all"] is False
    assert fixed["metrics"]["strict_follow_all"] is True
    # The headline is derived from the same metric, so it moves with it.
    assert base["rollouts"][0]["correct"] is False
    assert fixed["rollouts"][0]["correct"] is True


def test_the_loose_reading_was_already_masking_the_paragraph_defect():
    # Why loose moves less than strict, as a test rather than a claim: loose
    # re-tries the response with its first line stripped, which deletes the very
    # blank chunk that shifts the index. It was accidentally hiding the defect,
    # not immune to it.
    raw = _sample(
        [_NTH],
        [{"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}],
    )
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    base = _judge(IFEvalZeroShotGenTask, raw, response)
    assert base["metrics"]["strict_follow_all"] is False
    assert base["metrics"]["loose_follow_all"] is True


def test_the_fixed_task_does_not_pass_a_response_that_missed_the_constraint():
    raw = _sample(
        [_NTH],
        [{"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}],
    )
    fixed = _judge(
        IFEvalZeroShotGenFixedTask,
        raw,
        "\n\nalpha one\n\nWRONG two\n\ngamma three",
    )
    assert fixed["metrics"]["strict_follow_all"] is False


def test_a_mixed_sample_moves_only_at_the_repaired_constraint():
    # Two constraints, one repaired and one not. The unrepaired verdict must be
    # identical in both arms -- that is what makes an instruction-level delta
    # attributable rather than merely correlated.
    raw = _sample(
        [_NO_COMMA, _NTH],
        [{}, {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "beta"}],
    )
    response = "\n\nalpha one\n\nbeta two\n\ngamma three"
    base = _judge(IFEvalZeroShotGenTask, raw, response)
    fixed = _judge(IFEvalZeroShotGenFixedTask, raw, response)
    base_followed = base["extra"]["strict"]["follow_instruction_list"]
    fixed_followed = fixed["extra"]["strict"]["follow_instruction_list"]
    assert base_followed == [True, False]
    assert fixed_followed == [True, True]
    assert base["metrics"]["strict_instruction_level"] == pytest.approx(0.5)
    assert fixed["metrics"]["strict_instruction_level"] == pytest.approx(1.0)


def test_report_is_inherited_verbatim():
    # Deliberately not overridden: `check_report_declarations` treats a
    # `super().report()` call as a new definition rather than a delegate, and the
    # pooled report has nothing variant-specific in it anyway.
    assert IFEvalZeroShotGenFixedTask.report is IFEvalZeroShotGenTask.report
