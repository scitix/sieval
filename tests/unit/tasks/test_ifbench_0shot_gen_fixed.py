"""Registration and seam contract for the corrected IFBench task.

The repairs themselves are tested in
``tests/unit/tasks/test__ifbench_fixed_checkers.py``. What this module pins is
the claim the variant is built on: that it is the unqualified task plus *one*
overridden method, so the measured delta is the repaired checkers and cannot be
a second change riding along.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from sieval.core.tasks.meta import get_task_meta
from sieval.tasks.ifbench_0shot_gen import IFBenchZeroShotGenTask
from sieval.tasks.ifbench_0shot_gen_fixed import IFBenchZeroShotGenFixedTask

_INDENT = "format:line_indent"
_TITLE = "format:title_case"


def test_import_does_not_pull_evaluation_lib():
    # Same contract as the unqualified task's, and one more: the repair module
    # subclasses the vendored checkers and reaches NLTK through them, so
    # defining it at module scope would make *registration* -- which imports
    # every task module -- pay for both.
    code = (
        "import sys\n"
        "import sieval.tasks.ifbench_0shot_gen_fixed\n"
        "assert 'sieval.community.ifbench.evaluation_lib' not in sys.modules, (\n"
        "    'evaluation_lib must be lazy-imported')\n"
        "assert 'sieval.tasks._ifbench_fixed_checkers' not in sys.modules, (\n"
        "    'the repair module must be lazy-imported')\n"
        "assert 'nltk' not in sys.modules, 'nltk must be lazy-imported'\n"
    )
    # A fresh interpreter, so pytest's already-loaded modules cannot mask it.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_the_registration_scan_does_not_need_the_ifbench_extras():
    # `import_all_tasks()` -- what `scripts/sync_meta_index.py` runs, and what
    # CI's preflight runs it through -- imports EVERY module under
    # `sieval/tasks`, private ones included. The underscore keeps the repair
    # module out of the task *index*; it does not keep it out of the *import
    # scan*, so the module owes the same discipline every other task module
    # keeps (`_math_verify` lazy-imports `math_verify` inside its function for
    # the same reason): no optional dependency at module scope. Defining the
    # four subclasses at module scope broke exactly this, and a green local run
    # could not see it because the `ifbench` extras were installed.
    #
    # `emoji` is blocked rather than assumed absent: it stands in for the whole
    # `ifbench` extra, so the test keeps its teeth in an environment that has it.
    code = (
        "import sys\n"
        "class _NoEmoji:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'emoji' or name.startswith('emoji.'):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _NoEmoji())\n"
        "from sieval.core.tasks.meta import import_all_tasks\n"
        "import_all_tasks()\n"
        "assert 'sieval.community.ifbench.instructions' not in sys.modules, (\n"
        "    'registration must not load the vendored checker fork')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_the_repair_module_registers_no_task():
    # A non-task module may sit in `sieval/tasks` only if it registers nothing.
    # Asserted through the synced index rather than the scanner's private
    # helper, so this keeps holding if the scan is reimplemented: what matters
    # is that the repair module contributes no registry entry. It is imported by
    # the scan either way -- see the test above.
    repair = Path(__file__).parents[3] / "sieval/tasks/_ifbench_fixed_checkers.py"
    assert repair.exists()
    assert repair.name.startswith("_")

    index = json.loads(
        (Path(__file__).parents[3] / "sieval/meta/index.json").read_text()
    )
    names = {task["name"] for task in index["tasks"]}
    assert "ifbench_0shot_gen_fixed" in names
    assert "_ifbench_fixed_checkers" not in names
    assert not any(name.startswith("_") for name in names)


def test_meta():
    meta = get_task_meta(IFBenchZeroShotGenFixedTask)
    base = get_task_meta(IFBenchZeroShotGenTask)
    assert meta.name == "ifbench_0shot_gen_fixed"
    # The dataset FK resolves through the MRO: the subclass never re-declares a
    # generic base, so this proves it inherited one rather than silently
    # registering against a different sample type.
    assert meta.dataset == base.dataset
    assert meta.eval_mode == base.eval_mode
    assert meta.n_shot == base.n_shot == 0
    assert meta.deps_group == base.deps_group == "ifbench"
    # Deliberately NOT inherited. The base is `experimental` about a
    # published-number reproduction that this task does not attempt, so that
    # reason does not carry over; what a `_fixed` owes for `stable` is a
    # quantified score impact, which the notes carry. The divergence is
    # announced by the name -- saying it again in the status would read as "not
    # ready" rather than "deliberately different".
    assert meta.status == "stable"


def test_reference_impl_quantifies_the_divergence():
    # `_fixed` is licensed by a defect, not a preference, and owes two things:
    # every divergence enumerated, and a measured score impact. An unmeasured
    # fork is not a fix.
    reference_impl = get_task_meta(IFBenchZeroShotGenFixedTask).reference_impl
    assert reference_impl is not None
    notes = reference_impl.notes
    for instruction_id in (
        "format:line_indent",
        "ratio:sentence_type",
        "words:words_position",
        "words:vowel",
    ):
        assert instruction_id in notes
    assert "SCORE IMPACT" in notes
    # The headline metric's delta, on the full official 300-prompt set.
    assert "39.2917→39.3333" in notes


def test_the_notes_do_not_imply_all_four_repairs_moved_the_number():
    # Two of the four flip nothing on the measured responses. A reader given
    # four repairs and one delta will assume the delta came from four.
    reference_impl = get_task_meta(IFBenchZeroShotGenFixedTask).reference_impl
    assert reference_impl is not None
    assert "flip nothing here" in reference_impl.notes


def _task(cls):
    # `__new__`: the constructor requires a model with a bound dialect_id, and
    # nothing below reaches the model.
    return cls.__new__(cls)


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


def test_the_base_task_still_grades_through_the_vendored_registry():
    # The unqualified name tracks upstream, bugs included. `None` is what makes
    # the graders fall back to the vendored INSTRUCTION_DICT, so this is the
    # assertion that the repairs are unreachable from it.
    assert _task(IFBenchZeroShotGenTask)._instruction_dict() is None


def test_the_fixed_task_returns_the_repaired_registry():
    from sieval.tasks._ifbench_fixed_checkers import fixed_ifbench_registry

    registry = _task(IFBenchZeroShotGenFixedTask)._instruction_dict()
    assert registry == fixed_ifbench_registry()


def test_the_fixed_task_hands_out_a_fresh_registry_each_call():
    # Samples grade concurrently; a shared dict would let one sample's grading
    # change another's.
    task = _task(IFBenchZeroShotGenFixedTask)
    assert task._instruction_dict() is not task._instruction_dict()


def test_exactly_one_method_differs_from_the_unqualified_task():
    # The whole evidential weight of the measured delta rests on this: prompt,
    # graders, records and report are inherited, so the two tasks cannot diverge
    # anywhere except at the registry lookup.
    overrides = {
        name
        for name, value in vars(IFBenchZeroShotGenFixedTask).items()
        if callable(value) and hasattr(IFBenchZeroShotGenTask, name)
    }
    assert overrides == {"_instruction_dict"}
    for name in ("preprocess", "infer", "postprocess", "feedback", "report"):
        assert getattr(IFBenchZeroShotGenFixedTask, name) is getattr(
            IFBenchZeroShotGenTask, name
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
    # Not "the scores match" but "the records match": every checker outside the
    # four is the same object in both registries, so a sample built only from
    # those must produce an identical judgement, field for field.
    raw = _sample([_TITLE], [{}])
    response = "This Is A Title Cased Response"
    assert _judge(IFBenchZeroShotGenTask, raw, response) == _judge(
        IFBenchZeroShotGenFixedTask, raw, response
    )


def test_the_fixed_task_grades_the_stairs_the_prompt_asked_for():
    # Two blank lines: upstream's mutate-while-iterating removal leaves one
    # behind at indent 0 and breaks the chain. This is the flip that produced
    # essentially the whole measured delta.
    raw = _sample([_INDENT], [{}])
    response = "alpha\n\n\n beta\n  gamma"
    assert (
        _judge(IFBenchZeroShotGenTask, raw, response)["metrics"]["strict_follow_all"]
        is False
    )
    assert (
        _judge(IFBenchZeroShotGenFixedTask, raw, response)["metrics"][
            "strict_follow_all"
        ]
        is True
    )


def test_the_fixed_task_does_not_pass_a_response_that_missed_the_constraint():
    # The repair removes a formatting artefact from the verdict, not the
    # constraint from the item.
    raw = _sample([_INDENT], [{}])
    judgement = _judge(IFBenchZeroShotGenFixedTask, raw, "alpha\n beta\n gamma")
    assert judgement["metrics"]["strict_follow_all"] is False


def test_the_loose_reading_masks_the_indent_constraint_in_both_tasks():
    # Loose retries the response with its first line dropped, and dropping the
    # first line of a flat block leaves a shorter block that happens to climb.
    # That is upstream's loose semantics, unchanged here -- worth pinning so a
    # future reader does not mistake it for the repair being too permissive, and
    # worth knowing when reading the headline, which is the LOOSE number.
    raw = _sample([_INDENT], [{}])
    for cls in (IFBenchZeroShotGenTask, IFBenchZeroShotGenFixedTask):
        judgement = _judge(cls, raw, "alpha\n beta\n gamma")
        assert judgement["metrics"]["strict_follow_all"] is False
        assert judgement["metrics"]["loose_follow_all"] is True


def test_a_mixed_sample_moves_only_at_the_repaired_constraint():
    # The per-constraint lists are what report() pools, so "the score moved" is
    # only trustworthy if the slot that moved is the repaired one.
    raw = _sample([_TITLE, _INDENT], [{}, {}])
    response = "Alpha\n\n\n Beta\n  Gamma"
    base = _judge(IFBenchZeroShotGenTask, raw, response)
    fixed = _judge(IFBenchZeroShotGenFixedTask, raw, response)
    # The title_case slot is graded by a checker neither registry touches, so it
    # must hold its verdict while the indent slot moves.
    assert base["extra"]["strict"]["follow_instruction_list"] == [True, False]
    assert fixed["extra"]["strict"]["follow_instruction_list"] == [True, True]


def test_report_is_inherited_verbatim():
    # Including the headline pointer: IFBench's is LOOSE prompt-level, the
    # opposite of IFEval's, and a variant that quietly reported the strict
    # reading would look like a repair that moved the score a lot more.
    raw = _sample([_TITLE], [{}])
    judgement = _judge(IFBenchZeroShotGenFixedTask, raw, "This Is A Title")

    class _Final:
        feedback_result = judgement

    report = asyncio.run(_task(IFBenchZeroShotGenFixedTask).report([_Final()], []))
    assert report["score_key"] == "loose_prompt_level_accuracy"
    assert report["score"] == report["loose_prompt_level_accuracy"]
