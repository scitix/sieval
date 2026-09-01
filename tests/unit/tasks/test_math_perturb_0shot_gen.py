"""Unit tests for the MATH-P-Simple / MATH-P-Hard 0-shot CoT tasks.

Both leaves are the same class parameterized by sample type, so most cases run
against one and a handful assert the pair really is symmetric.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import REPEAT_GROUP_COLUMN, Dataset
from sieval.core.models import ModelOutput, Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    NonRetriableSampleError,
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
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
from sieval.tasks import _math_perturb_base
from sieval.tasks._math_perturb_base import (
    COT_INSTRUCTION,
    MATH_PERTURB_UPSTREAM_URL,
    MATH_SUBJECTS,
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
    # `stable` since the alignment run: deepseek-math-7b-rl -- the model whose
    # own zero-shot CoT turn this prompt is -- puts the published pair inside
    # three same-config runs. See the base module's ALIGNMENT notes.
    assert meta.status == "stable"
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
@pytest.mark.parametrize(
    "response",
    [
        "\\boxed{}",
        "\\boxed{ }",
        "So the answer is \\boxed{\\text{}}",
        "\\boxed{\\,}",
        "\\boxed{.}",
        # The shape a reply truncated at `max_tokens` takes -- which is the case
        # `n_unextracted` is read to detect, so it is the one that must not hide.
        "Let me work through this. The answer is ",
    ],
)
async def test_an_empty_ATOM_is_unextracted_not_a_successful_extraction(
    response: str,
):
    """`[""]` is a truthy list, so `if atoms` would call it extracted.

    Upstream's extractor returns a one-element list holding the empty string for
    every response here, not the empty list the sibling test above covers. Both
    have to become `None`, or `n_unextracted` reads 0 for a run that extracted
    nothing — and the gold side already guards this exact shape (`gold_atoms`).
    """
    task, _ = _task()
    raw = _sample()
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=[response]), _ctx(raw)
    )
    assert post["rollouts"][0].get("prediction") is None
    assert post["rollouts"][0]["extracted"] is False
    _, judgement = await task.feedback(post, _ctx(raw, postprocess_result=post))
    # Verdict-neutral: an empty prediction was already scored wrong, via
    # `math_equal`'s empty-prediction refusal. Only the health count moves.
    assert judgement["rollouts"][0]["correct"] is False
    assert (await task.report([_final(raw, post, judgement)], []))[
        "n_unextracted"
    ] == 1.0


@pytest.mark.anyio
async def test_a_partly_empty_atom_list_stays_a_prediction():
    """`any`, not `all` — a wrong answer is not a missing one.

    A reply that boxes twice and leaves one blank extracts to `["42", ""]`, which
    upstream's all-must-match rule then scores wrong. That is a prediction, and
    recording it as unextracted would blame the parser for the model's answer.
    """
    task, _ = _task()
    raw = _sample()
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["I get \\boxed{42} and \\boxed{}"]),
        _ctx(raw),
    )
    assert post["rollouts"][0].get("prediction") == ["42", ""]
    assert post["rollouts"][0]["extracted"] is True
    _, judgement = await task.feedback(post, _ctx(raw, postprocess_result=post))
    # Upstream requires every atom matched, so the blank one loses the sample.
    assert judgement["rollouts"][0]["correct"] is False
    assert (await task.report([_final(raw, post, judgement)], []))[
        "n_unextracted"
    ] == 0.0


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


class _Raiser:
    """An async ``run_cpu_bound`` stand-in that always raises *exc*.

    Counts its calls, so a test cannot pass because the patch was never reached —
    which is what a rename of the grading call site would otherwise look like.
    """

    def __init__(self, exc: type[BaseException]):
        self._exc = exc
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise self._exc("grader stub")


@pytest.mark.anyio
async def test_a_grader_timeout_is_a_wrong_answer(monkeypatch):
    """The half that stays swallowed: a grade that could not be computed IN TIME.

    The prediction is a shape `simplify` cannot bound, which is the model's
    problem, and `report` counts fails in the denominator either way.
    """
    task, _ = _task()
    stub = _Raiser(TimeoutError)
    monkeypatch.setattr(_math_perturb_base, "run_cpu_bound", stub)
    raw = _sample()
    post = build_prediction_record([["42"]])
    _, judgement = await task.feedback(post, _ctx(raw, postprocess_result=post))
    assert judgement["rollouts"][0]["correct"] is False
    assert stub.calls > 0, "the grading call site moved; this intercepted nothing"


@pytest.mark.anyio
@pytest.mark.parametrize("exc", [ValueError, AttributeError, ImportError, OSError])
async def test_a_broken_grader_propagates_instead_of_scoring_zero(exc, monkeypatch):
    """A grader that is BROKEN rather than slow must not read as a wrong answer.

    This benchmark is unusually exposed to it: a missing `lark` makes every
    symbolic comparison fall through to string equality and understates the score
    by 5-6 points without raising at all, and the shapes that DO raise (a dead
    worker, an optional dependency absent from the environment) used to be scored
    0 with `fails` left at 0. Propagated, the runner records
    `exception::<class>` on the sample and `fails` becomes the signal.
    """
    task, _ = _task()
    stub = _Raiser(exc)
    monkeypatch.setattr(_math_perturb_base, "run_cpu_bound", stub)
    raw = _sample()
    post = build_prediction_record([["42"]])
    with pytest.raises(exc):
        await task.feedback(post, _ctx(raw, postprocess_result=post))
    assert stub.calls > 0, "the grading call site moved; this intercepted nothing"


@pytest.mark.anyio
async def test_moving_a_sample_into_fails_moves_no_published_rate():
    """Why propagating costs nothing here, asserted over the RICH report.

    `math_500`'s version of this covers a headline and one interval; this report
    also carries two seed cells with their own populations and seven subject
    cells, and a fail reaches those denominators only through `raw_sample`. So
    the neutrality has to be checked on every rate this task publishes, not just
    on `score`.
    """
    task, _ = _task()

    def _judged(problem_id: int, *, correct: bool, split: str, subject: str):
        raw = _sample(problem_id=problem_id, original_split=split, subject=subject)
        return raw, TaskContext(
            sample_id=problem_id,
            raw_sample=raw,
            feedback_result=build_judgement_record(
                ["42"],
                [build_rollout_judgement(0, correct)],
                extra={"original_split": split, "type": subject},
            ),
            postprocess_result=build_prediction_record([["42" if correct else "0"]]),
        )

    survivors = [
        _judged(0, correct=True, split="train", subject="Algebra")[1],
        _judged(1, correct=True, split="train", subject="Algebra")[1],
        _judged(2, correct=False, split="test", subject="Geometry")[1],
    ]
    bad_raw, bad_as_final = _judged(3, correct=False, split="test", subject="Geometry")
    bad_as_fail = TaskContext(sample_id=3, raw_sample=bad_raw)

    before = await task.report([*survivors, bad_as_final], [])
    after = await task.report(survivors, [bad_as_fail])

    # Every RATE and every POPULATION is identical -- the seed cells and all seven
    # subject cells included, which is what proves the fail still reaches their
    # denominators through `raw_sample`.
    rates = [
        key
        for key in before
        if key.startswith(("score", "pass@", "n_problems", "n_unextracted"))
        and not key.endswith("_ci95")
        and key != SCORE_KEY_FIELD
    ]
    assert len(rates) >= 12, rates  # headline + 2 seed + 2 counts + 7 subjects
    for key in rates:
        assert before[key] == after[key], key
    assert before["fails"] == 0
    assert after["fails"] == 1

    # What moves is intervals only, and one of them VANISHES: the `test` seed cell
    # keeps a single judged problem, and `wilson_interval` needs 0 < p < 1. Its
    # `ci95_units` entry goes with it, so the pair stays a pair and the report is
    # still declaration-clean -- the count is published unconditionally either way.
    # Pre-existing `_clustered_interval` semantics, reached more often now; on a
    # real 279-row run a 115-problem cell does not go uniform.
    assert "score_seed_test_ci95" in before
    assert "score_seed_test_ci95" not in after
    assert "score_seed_test" not in after["ci95_units"]
    assert after["n_problems_seed_test"] == before["n_problems_seed_test"]
    assert interval_declaration_problems(before) == []
    assert interval_declaration_problems(after) == []


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


@pytest.mark.anyio
async def test_an_unexpected_seed_split_is_published_rather_than_dropped():
    """A third `original_split` value gets its own cell, like an eighth subject.

    The subject loop takes the union with what was observed; the seed loop used
    a fixed two-element list, so a row carrying anything else was counted in the
    headline and in no seed column at all — the two seed rates silently stopped
    summing to the total. The pinned data is train/test only, so this is drift
    insurance, not a live case.
    """
    task, _ = _task()

    def _judged(problem_id: int, split: str):
        raw = _sample(problem_id=problem_id, original_split=split)
        return TaskContext(
            sample_id=problem_id,
            raw_sample=raw,
            feedback_result=build_judgement_record(
                ["42"],
                [build_rollout_judgement(0, True)],
                extra={"original_split": split, "type": "Algebra"},
            ),
            postprocess_result=build_prediction_record([["42"]]),
        )

    finals = [_judged(0, "train"), _judged(1, "test"), _judged(2, "validation")]
    report = await task.report(finals, [])

    # The two declared cells still publish, unconditionally, as before.
    assert "score_seed_train" in report
    assert "score_seed_test" in report
    # And the unexpected one is reported instead of vanishing.
    assert report["score_seed_validation"] == 100.0
    assert report["n_problems_seed_validation"] == 1.0
    # The cells account for every problem -- the property that broke.
    assert sum(
        report[key]
        for key in report
        if key.startswith("n_problems_seed_") and not key.endswith("_ci95")
    ) == len(finals)


def test_k_greater_than_n_is_rejected_at_construction():
    with pytest.raises(ValueError, match="pass@2 needs at least 2 sample"):
        _task(k=2, n=1)


def test_a_missing_lark_backend_is_refused_at_construction(monkeypatch):
    """The one grader failure no call-site handler can reach.

    `parse_latex(backend="lark")` raises ImportError when the package is gone,
    and `symbolic_equal`'s bare `except` — upstream's control flow, kept —
    turns it into a False verdict. So `feedback` never sees an exception, and
    the run finishes with fails=0 and a score 5-6 points low. Construction is
    the only place it can be caught.
    """
    import importlib.util as importlib_util

    real = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **kw: None if name == "lark" else real(name, *a, **kw),
    )
    with pytest.raises(ImportError, match="needs the `lark` LaTeX backend"):
        _task()


def test_the_lark_check_names_the_group_and_its_stale_install(monkeypatch):
    """The message has to be actionable for the case that actually happens.

    `deps_group="math"` is satisfied by an environment that installed the group
    before `lark` was added to it, so "install the math group" alone would read
    as already-done. Asserted so the explanation cannot be trimmed away.
    """
    import importlib.util as importlib_util

    real = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **kw: None if name == "lark" else real(name, *a, **kw),
    )
    with pytest.raises(ImportError) as excinfo:
        _task()
    message = str(excinfo.value)
    assert "pdm install -G math" in message
    assert "before `lark` was added" in message


def test_the_lark_check_passes_in_a_normal_environment():
    """Discriminating companion: the guard must not reject a working install."""
    task, _ = _task()
    assert task is not None


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
async def test_all_seven_subject_cells_are_published_whatever_the_draw_held():
    """A column that appears only when the draw contains it cannot be keyed on.

    Same contract as the two seed cells: one `limit` or `filter` is enough to
    leave a subject unobserved, and a breakdown that grows and loses columns
    between runs is one a consumer has to guess the shape of.
    """
    task, _ = _task()
    raw = _sample(problem_id=1, subject="Algebra")
    post, judgement = await _judge(task, raw, ("\\boxed{42}",))

    report = await task.report([_final(raw, post, judgement)], [])
    assert len(MATH_SUBJECTS) == 7
    for subject in MATH_SUBJECTS:
        assert type_score_key(subject) in report
    assert report[type_score_key("Algebra")] == 100.0
    # Unobserved: 0 over 0, published rather than absent.
    assert report[type_score_key("Geometry")] == 0.0


@pytest.mark.anyio
async def test_a_subject_outside_the_seven_is_reported_not_dropped():
    """The union, not `MATH_SUBJECTS` alone — the risk a fixed list introduces.

    If the pinned source ever grows an eighth subject, publishing only the seven
    would silently omit its rows from the breakdown while still counting them in
    the headline.
    """
    task, _ = _task()
    raw = _sample(problem_id=1, subject="Topology")
    post, judgement = await _judge(task, raw, ("\\boxed{42}",))

    report = await task.report([_final(raw, post, judgement)], [])
    assert report[type_score_key("Topology")] == 100.0
    for subject in MATH_SUBJECTS:
        assert type_score_key(subject) in report


@pytest.mark.anyio
async def test_a_seed_cell_over_a_repeated_split_collapses_the_copies():
    """`_restrict`: without it a cell counts each copy as its own problem.

    `problem_groups` collapses the whole split, but a seed cell covers a SUBSET
    of it, so the grouping has to be narrowed positionally or `metric_interval`
    is handed one key per value it does not have — and a population inflated by
    the repeat factor narrows the interval by its square root.
    """
    rows = [
        _sample(problem_id=1, original_split="train"),
        _sample(problem_id=1, original_split="train"),
        _sample(problem_id=2, original_split="test"),
        _sample(problem_id=2, original_split="test"),
    ]
    for row, group in zip(rows, (1, 1, 2, 2), strict=True):
        row[REPEAT_GROUP_COLUMN] = group
    dataset = MATHPerturbSimpleDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(r) for r in rows])})
    )
    task = MATHPerturbSimpleZeroShotGenTask(dataset, _CapturingChatModel())

    finals = []
    for index, raw in enumerate(rows):
        post, judgement = await _judge(task, raw, ("\\boxed{42}",))
        finals.append(_final(raw, post, judgement, sample_id=index))

    grouping = task.problem_groups(finals)
    assert grouping is not None and grouping.n_problems == 2

    report = await task.report(finals, [])
    # Two copies of one problem per split, so each cell is ONE problem -- not the
    # two samples its denominator counts.
    for seed_split in ("train", "test"):
        assert report[seed_count_key(seed_split)] == 1.0
    assert not interval_declaration_problems(report)


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
