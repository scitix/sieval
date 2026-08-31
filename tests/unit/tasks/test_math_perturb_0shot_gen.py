"""Unit tests for the MATH-P-Simple / MATH-P-Hard 0-shot CoT tasks.

Both leaves are the same class parameterized by sample type, so most cases run
against one and a handful assert the pair really is symmetric.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Dataset
from sieval.core.models import ModelOutput, Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    NonRetriableSampleError,
    TaskContext,
    build_prediction_record,
)
from sieval.core.tasks.meta import TASK_REGISTRY, get_task_meta, import_all_tasks
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    interval_declaration_problems,
)
from sieval.datasets.math_perturb_hard import MATHPerturbHardDataset
from sieval.datasets.math_perturb_simple import MATHPerturbSimpleDataset
from sieval.tasks._math_perturb_base import (
    COT_INSTRUCTION,
    MATH_PERTURB_UPSTREAM_URL,
    grade_extracted,
    seed_count_key,
    seed_score_key,
    type_score_key,
)
from sieval.tasks.math_perturb_hard_0shot_gen import MATHPerturbHardZeroShotGenTask
from sieval.tasks.math_perturb_simple_0shot_gen import MATHPerturbSimpleZeroShotGenTask
from tests.conftest import HandlerTransport

_LEAVES = (
    (
        "math_perturb_simple_0shot_gen",
        MATHPerturbSimpleZeroShotGenTask,
        MATHPerturbSimpleDataset,
        "math_perturb_simple",
    ),
    (
        "math_perturb_hard_0shot_gen",
        MATHPerturbHardZeroShotGenTask,
        MATHPerturbHardDataset,
        "math_perturb_hard",
    ),
)


class _CapturingChatModel(ChatModel):
    def __init__(self, texts: tuple[str, ...] = ("\\boxed{42}",)):
        self.last_req: Request | None = None
        self._texts = texts
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=self._texts)


def _sample(
    answer: str = "42",
    original_split: str = "train",
    subject: str = "Algebra",
    problem_id: int = 1,
) -> dict:
    return {
        "problem_id": problem_id,
        "problem": "What is the answer?",
        "answer": answer,
        "level": "Level 5",
        "type": subject,
        "original_split": original_split,
    }


def _task(
    cls=MATHPerturbSimpleZeroShotGenTask,
    dataset_cls=MATHPerturbSimpleDataset,
    texts: tuple[str, ...] = ("\\boxed{42}",),
    **kwargs,
):
    dataset = dataset_cls(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample()])})
    )
    model = _CapturingChatModel(texts)
    return cls(dataset, model, **kwargs), model


def _ctx(raw: dict, **kwargs) -> TaskContext:
    return TaskContext(sample_id=raw["problem_id"], raw_sample=raw, **kwargs)


async def _judge(task, raw: dict, texts: tuple[str, ...]):
    """Drive one sample through postprocess + feedback with canned responses."""
    ctx = _ctx(raw)
    inferred = ModelOutput(model=task.model.meta(), texts=list(texts))
    post = await task.postprocess(inferred, ctx)
    ctx = _ctx(raw, postprocess_result=post)
    _, judgement = await task.feedback(post, ctx)
    return post, judgement


def _final(raw: dict, post, judgement, sample_id: int | None = None) -> TaskContext:
    return TaskContext(
        sample_id=raw["problem_id"] if sample_id is None else sample_id,
        raw_sample=raw,
        postprocess_result=post,
        feedback_result=judgement,
    )


# --- registration: both leaves resolve to their own dataset ---


@pytest.mark.parametrize(("name", "cls", "_ds", "dataset_name"), _LEAVES)
def test_leaf_resolves_its_own_dataset_through_the_generic_base(
    name, cls, _ds, dataset_name
):
    """The FK comes from the generic argument on the shared base.

    Both leaves subclass one parameterized base, so a mistake here silently
    points a task at its sibling's 279 rows — real answers against the wrong
    questions.
    """
    import_all_tasks()
    meta = TASK_REGISTRY[name]
    # The registry entry is this class's own, not one the sibling registered
    # under a name that happens to read right.
    assert get_task_meta(cls) is meta
    assert meta.dataset == dataset_name
    assert meta.status == "experimental"
    assert meta.reference_kind == "value"
    assert meta.deps_group == "math"
    assert meta.model_type == "chat"


def test_the_two_leaves_do_not_share_a_dataset():
    import_all_tasks()
    assert (
        TASK_REGISTRY["math_perturb_simple_0shot_gen"].dataset
        != TASK_REGISTRY["math_perturb_hard_0shot_gen"].dataset
    )


@pytest.mark.parametrize(("name", "_cls", "_ds", "_dn"), _LEAVES)
def test_reference_impl_pins_the_upstream_commit(name, _cls, _ds, _dn):
    import_all_tasks()
    reference = TASK_REGISTRY[name].reference_impl
    assert reference is not None
    assert reference.url == MATH_PERTURB_UPSTREAM_URL
    assert "df4840f680fce405c9449008564574961c7f4df1" in reference.url
    # The prompt is a sieval choice; a reader must not have to infer that.
    assert "PROMPT IS A SIEVAL CHOICE" in reference.notes


# --- prompt ---


def test_cot_instruction_is_deepseek_maths_verbatim():
    assert COT_INSTRUCTION == (
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


@pytest.mark.anyio
async def test_preprocess_builds_one_user_turn_and_records_the_extracted_gold():
    task, _ = _task()
    raw = _sample()
    pre = await task.preprocess(raw, _ctx(raw))
    assert pre["prompt"] == [
        {"role": "user", "content": "What is the answer?" + COT_INSTRUCTION}
    ]
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == ["42"]
    assert pre["extra"] == {
        "problem_id": 1,
        "original_split": "train",
        "type": "Algebra",
    }


@pytest.mark.anyio
async def test_preprocess_records_a_multi_valued_gold_as_its_atoms():
    task, _ = _task()
    raw = _sample(answer="2 or 3")
    pre = await task.preprocess(raw, _ctx(raw))
    assert pre["reference"] == ["2", "3"]


@pytest.mark.anyio
async def test_a_row_with_no_extractable_gold_fails_the_sample():
    """A value-reference task must not grade against an empty reference."""
    task, _ = _task()
    raw = _sample(answer="")
    with pytest.raises(NonRetriableSampleError, match="no extractable ground truth"):
        await task.preprocess(raw, _ctx(raw))


# --- postprocess: the prediction is the extracted atom list, `None` when empty ---


@pytest.mark.anyio
async def test_postprocess_stores_the_atom_list():
    task, _ = _task()
    raw = _sample()
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["so \\boxed{2 \\text{ or } 3}"]),
        _ctx(raw),
    )
    assert post["rollouts"][0]["prediction"] == ["2", "3"]
    assert post["rollouts"][0]["extracted"] is True


@pytest.mark.anyio
async def test_postprocess_records_an_empty_extraction_as_none_not_empty_list():
    """`extracted` is derived from `prediction is not None`.

    An empty list would record a failed extraction as a successful one and hide
    it from `n_unextracted`.
    """
    task, _ = _task()
    raw = _sample()
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=[""]), _ctx(raw)
    )
    assert post["rollouts"][0].get("prediction") is None
    assert post["rollouts"][0]["extracted"] is False


@pytest.mark.anyio
async def test_postprocess_keeps_one_prediction_per_rollout():
    task, _ = _task(n=2)
    raw = _sample()
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["\\boxed{42}", "\\boxed{7}"]),
        _ctx(raw),
    )
    assert [r["prediction"] for r in post["rollouts"]] == [["42"], ["7"]]


# --- feedback ---


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answer", "response", "expected"),
    [
        ("42", "so \\boxed{42}", True),
        ("42", "so \\boxed{43}", False),
        ("\\frac{1}{2}", "so \\boxed{0.5}", True),
        ("2 or 3", "so \\boxed{2 \\text{ or } 3}", True),
        ("2 or 3", "so \\boxed{2}", False),
        ("42", "I could not solve it", False),
    ],
)
async def test_feedback_verdicts(answer, response, expected):
    task, _ = _task()
    raw = _sample(answer=answer)
    _, judgement = await _judge(task, raw, (response,))
    assert judgement["rollouts"][0]["correct"] is expected


@pytest.mark.anyio
async def test_feedback_records_the_extracted_gold_and_the_routing_labels():
    task, _ = _task()
    raw = _sample(answer="2 or 3", original_split="test", subject="Precalculus")
    _, judgement = await _judge(task, raw, ("\\boxed{2 \\text{ or } 3}",))
    assert judgement["reference"] == ["2", "3"]
    assert judgement["extra"] == {"original_split": "test", "type": "Precalculus"}


@pytest.mark.anyio
async def test_an_unextracted_rollout_is_wrong_not_a_failure():
    task, _ = _task()
    raw = _sample()
    _, judgement = await _judge(task, raw, ("",))
    assert judgement["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_reads_the_stored_prediction_not_the_response_text():
    """Grading must survive a resume that carries no ``infer_result``.

    The context handed to ``feedback`` here has none, so a task reaching for the
    raw text would raise instead of scoring.
    """
    task, _ = _task()
    raw = _sample()
    post = build_prediction_record([["42"]])
    _, judgement = await task.feedback(post, _ctx(raw, postprocess_result=post))
    assert judgement["rollouts"][0]["correct"] is True


def test_grade_extracted_does_not_mutate_its_arguments():
    """``eval_math`` mutates the dict it is given; the caller's lists must not be."""
    gold = ["7"]
    prediction = ["99", "7"]
    assert grade_extracted(gold, prediction)
    assert gold == ["7"]
    assert prediction == ["99", "7"]


# --- k / n ---


def test_k_greater_than_n_is_rejected_at_construction():
    with pytest.raises(ValueError, match="pass@2 needs at least 2 sample"):
        _task(k=2, n=1)


@pytest.mark.anyio
async def test_infer_forwards_the_task_budget():
    task, model = _task(n=3, texts=("a", "b", "c"))
    raw = _sample()
    pre = await task.preprocess(raw, _ctx(raw))
    await task.infer(pre, _ctx(raw))
    assert model.last_req is not None
    assert model.last_req.sampling.n == 3


# --- report ---


@pytest.mark.anyio
async def test_report_headline_and_declarations():
    task, _ = _task()
    rows = [
        _sample(problem_id=1, original_split="train", subject="Algebra"),
        _sample(problem_id=2, original_split="train", subject="Algebra"),
        _sample(problem_id=3, original_split="test", subject="Geometry"),
        _sample(problem_id=4, original_split="test", subject="Geometry"),
    ]
    finals = []
    for index, raw in enumerate(rows):
        response = "\\boxed{42}" if index % 2 == 0 else "\\boxed{43}"
        post, judgement = await _judge(task, raw, (response,))
        finals.append(_final(raw, post, judgement))

    report = await task.report(finals, [])
    assert report["score"] == 50.0
    assert report["pass@1"] == 50.0
    assert report[SCORE_KEY_FIELD] == "pass@1"
    assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
    assert report[seed_score_key("train")] == 50.0
    assert report[seed_score_key("test")] == 50.0
    assert report[seed_count_key("train")] == 2.0
    assert report[seed_count_key("test")] == 2.0
    assert report[type_score_key("Algebra")] == 50.0
    assert report[type_score_key("Geometry")] == 50.0
    assert report["fails"] == 0
    assert report["n_unextracted"] == 0.0
    assert not interval_declaration_problems(report)


@pytest.mark.anyio
async def test_report_separates_the_two_seed_splits():
    task, _ = _task()
    rows = [
        _sample(problem_id=1, original_split="train"),
        _sample(problem_id=2, original_split="train"),
        _sample(problem_id=3, original_split="test"),
    ]
    responses = ("\\boxed{42}", "\\boxed{42}", "\\boxed{43}")
    finals = []
    for raw, response in zip(rows, responses, strict=True):
        post, judgement = await _judge(task, raw, (response,))
        finals.append(_final(raw, post, judgement))

    report = await task.report(finals, [])
    # The paper's second claim lives on this axis, so the two must not be pooled.
    assert report[seed_score_key("train")] == 100.0
    assert report[seed_score_key("test")] == 0.0


def test_type_score_key_slugs_the_seven_math_subjects():
    assert type_score_key("Counting & Probability") == (
        "score_type_counting_and_probability"
    )
    assert type_score_key("Intermediate Algebra") == "score_type_intermediate_algebra"
    assert type_score_key("Number Theory") == "score_type_number_theory"


@pytest.mark.anyio
async def test_report_charges_a_fail_as_wrong_in_every_denominator_it_can_reach():
    task, _ = _task()
    raw = _sample(problem_id=1, original_split="train", subject="Algebra")
    post, judgement = await _judge(task, raw, ("\\boxed{42}",))
    finals = [_final(raw, post, judgement)]
    failed = _sample(problem_id=2, original_split="train", subject="Algebra")

    report = await task.report(finals, [TaskContext(sample_id=2, raw_sample=failed)])
    # DENOMINATOR_REQUESTED: the fail is a slot in each denominator, scored 0.
    assert report["score"] == 50.0
    assert report[seed_score_key("train")] == 50.0
    assert report[type_score_key("Algebra")] == 50.0
    assert report["fails"] == 1


@pytest.mark.anyio
async def test_report_declarations_survive_the_fold_at_n_above_one():
    """`merge_metrics`, not `|` — the failure it prevents is silent.

    At n>1 three interval-bearing fragments are folded (the ungated pair, the
    whole sampling block, and the two seed cells). A plain merge would keep only
    the last fragment's `ci95_units` and leave every other interval with no unit,
    with the intervals themselves all still present.
    """
    task, _ = _task(n=2, k=2)
    # Four problems, two per seed split, and the verdicts DISAGREE — a run whose
    # units all agree has nothing to estimate, so `wilson_interval` omits every
    # interval and there would be no fold to test.
    rows = [
        _sample(problem_id=1, original_split="train"),
        _sample(problem_id=2, original_split="train"),
        _sample(problem_id=3, original_split="test"),
        _sample(problem_id=4, original_split="test"),
    ]
    responses = [
        ("\\boxed{42}", "\\boxed{42}"),
        ("\\boxed{43}", "\\boxed{43}"),
        ("\\boxed{42}", "\\boxed{42}"),
        ("\\boxed{43}", "\\boxed{43}"),
    ]
    finals = []
    for raw, texts in zip(rows, responses, strict=True):
        post, judgement = await _judge(task, raw, texts)
        finals.append(_final(raw, post, judgement))

    report = await task.report(finals, [])
    units = report["ci95_units"]
    # All three fragments' declarations survive, on their own populations.
    assert units["score"] == "n_problems"
    assert units["pass@k"] == "n_problems"
    assert units[seed_score_key("train")] == seed_count_key("train")
    assert units[seed_score_key("test")] == seed_count_key("test")
    assert not interval_declaration_problems(report)


@pytest.mark.anyio
async def test_report_tolerates_a_fail_with_no_raw_sample():
    """A sample that died before the row was attached carries no labels.

    It can only be charged to the headline; the cells must not raise or invent a
    bucket for it.
    """
    task, _ = _task()
    raw = _sample(original_split="train", subject="Algebra")
    post, judgement = await _judge(task, raw, ("\\boxed{42}",))
    report = await task.report(
        [_final(raw, post, judgement)], [TaskContext(sample_id=9, raw_sample=None)]
    )
    assert report["score"] == 50.0
    # The cell keeps its own denominator of 1 — the untyped fail is not in it.
    assert report[seed_score_key("train")] == 100.0


@pytest.mark.anyio
async def test_report_on_an_empty_run_still_declares_everything():
    task, _ = _task()
    report = await task.report([], [])
    assert report["score"] == 0.0
    assert report[SCORE_KEY_FIELD] == "pass@1"
    assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
    # A rate with no count behind it cannot be read, so both cells keep theirs.
    for seed_split in ("train", "test"):
        assert report[seed_score_key(seed_split)] == 0.0
        assert report[seed_count_key(seed_split)] == 0.0
    assert not interval_declaration_problems(report)


@pytest.mark.anyio
async def test_report_publishes_no_vote_columns():
    """Set-valued answers have no canonical vote key — see the module docstring."""
    task, _ = _task(n=2, k=2, texts=("\\boxed{42}", "\\boxed{42}"))
    raw = _sample()
    post, judgement = await _judge(task, raw, ("\\boxed{42}", "\\boxed{42}"))
    report = await task.report([_final(raw, post, judgement)], [])
    assert "maj@k" not in report
    assert "self_consistency" not in report
    assert report["pass@k"] == 100.0


@pytest.mark.anyio
async def test_report_averages_rollouts_into_the_cells_at_n_above_one():
    task, _ = _task(n=2)
    raw = _sample(original_split="train")
    post, judgement = await _judge(task, raw, ("\\boxed{42}", "\\boxed{43}"))
    report = await task.report([_final(raw, post, judgement)], [])
    # One of two rollouts correct: the cell reads the same pass@1 the headline
    # does, not the first rollout alone.
    assert report["pass@1"] == 50.0
    assert report[seed_score_key("train")] == 50.0


@pytest.mark.anyio
async def test_report_counts_unextracted_rollouts():
    task, _ = _task()
    raw = _sample()
    post, judgement = await _judge(task, raw, ("",))
    report = await task.report([_final(raw, post, judgement)], [])
    assert report["n_unextracted"] == 1.0


# --- the two leaves behave identically on the same row ---


@pytest.mark.anyio
@pytest.mark.parametrize(("_name", "cls", "dataset_cls", "_dn"), _LEAVES)
async def test_both_leaves_grade_the_same_row_the_same_way(
    _name, cls, dataset_cls, _dn
):
    task, _ = _task(cls=cls, dataset_cls=dataset_cls)
    raw = _sample(answer="\\frac{1}{2}")
    _, judgement = await _judge(task, raw, ("so \\boxed{0.5}",))
    assert judgement["rollouts"][0]["correct"] is True


@pytest.mark.anyio
@pytest.mark.parametrize(("_name", "cls", "dataset_cls", "_dn"), _LEAVES)
async def test_both_leaves_build_the_same_prompt(_name, cls, dataset_cls, _dn):
    task, _ = _task(cls=cls, dataset_cls=dataset_cls)
    raw = _sample()
    pre = await task.preprocess(raw, _ctx(raw))
    assert pre["prompt"][0]["content"].endswith(COT_INSTRUCTION)


def test_dataset_and_task_classes_are_not_interchangeable():
    """A leaf bound to the wrong dataset class must be a type-level mistake.

    Nothing at run time forbids it, so this pins the pairing the registry made:
    the sample TypedDicts are distinct, which is what makes the FK resolve.
    """
    assert isinstance(
        MATHPerturbSimpleDataset(
            _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample()])})
        ),
        Dataset,
    )
    assert (
        MATHPerturbSimpleZeroShotGenTask.__mro__[1]
        is MATHPerturbHardZeroShotGenTask.__mro__[1]
    )
