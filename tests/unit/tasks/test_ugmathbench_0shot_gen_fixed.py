"""Unit tests for the corrected UGMathBench task: stage plumbing and metrics.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import REPEAT_GROUP_COLUMN
from sieval.core.models import ModelMeta, ModelOutput, Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskAction,
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    PROBLEM_COUNT_FIELD,
    SCORE_CI_FIELD,
    interval_declaration_problems,
    wilson_interval,
)
from sieval.datasets.ugmathbench import UGMathBenchDataset
from sieval.tasks.ugmathbench_0shot_gen_fixed import (
    VERSION_COUNT_FIELD,
    UGMathBenchZeroShotGenFixedTask,
)
from tests.conftest import HandlerTransport


def _sample(
    problem_id: str = "Algebra_0001",
    version: int = 1,
    answer: list[str] | None = None,
    answer_type: list[str] | None = None,
) -> dict:
    answers = answer if answer is not None else ["4"]
    return {
        "id": problem_id,
        "subject": "Algebra",
        "topic": "Linear equations",
        "subtopic": "Solving",
        "level": "2",
        "keywords": ["algebra"],
        "version": version,
        "problem": "Solve $x+1=5$. [ANS]",
        "answer": answers,
        "answer_type": answer_type or ["NV"] * len(answers),
        "options": [[] for _ in answers],
    }


def _inferred(*texts: str) -> ModelOutput:
    meta: ModelMeta = {"model": "mock-chat", "api_base": None, "default_params": {}}
    return ModelOutput(model=meta, texts=list(texts))


def _task(precision: float = 1e-3) -> UGMathBenchZeroShotGenFixedTask:
    sample = _sample()
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([sample])})
    )
    model = ChatModel(model="mock-chat", api_key="fake")
    return UGMathBenchZeroShotGenFixedTask(dataset, model, precision=precision)


def _judged(
    problem_id: str, version: int, correct: bool, subject: str = "Algebra"
) -> TaskContext:
    return TaskContext(
        sample_id=f"{problem_id}-v{version}",
        feedback_result=build_judgement_record(
            ["1"],
            [build_rollout_judgement(0, correct)],
            extra={"problem_id": problem_id, "version": version, "subject": subject},
        ),
    ).to_final()


def _all_versions(
    problem_id: str, verdicts: list[bool], subject: str = "Algebra"
) -> list[TaskContext]:
    return [
        _judged(problem_id, version, correct, subject)
        for version, correct in enumerate(verdicts, start=1)
    ]


def _failed(problem_id: str, version: int) -> TaskContext:
    """A sample that died before feedback but still knows which problem it is."""
    return TaskContext(
        sample_id=f"{problem_id}-v{version}",
        raw_sample=_sample(problem_id, version),
    ).to_failed(None, "error", "boom")


def test_precision_must_be_positive():
    with pytest.raises(ValueError, match="precision must be > 0"):
        _task(precision=0)


@pytest.mark.anyio
async def test_preprocess_builds_the_benchmark_prompt_and_carries_grouping_keys():
    raw = _sample(version=2)
    record = await _task().preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert record["prompt"][0]["role"] == "user"
    content = record["prompt"][0]["content"]
    assert "undergraduate-level mathematical problem in Algebra" in content
    assert "The final answer is \\boxed{ANSWER}" in content
    assert record["reference"] == ["4"]
    assert record["extra"]["problem_id"] == "Algebra_0001"
    assert record["extra"]["version"] == 2


@pytest.mark.anyio
async def test_postprocess_records_one_prediction_per_answer_slot():
    record = await _task().postprocess(
        _inferred("The final answers are \\boxed{1, 2}"), TaskContext(sample_id=0)
    )
    assert record["rollouts"][0]["prediction"] == ["1", "2"]
    assert record["rollouts"][0]["extracted"] is True


@pytest.mark.anyio
async def test_postprocess_marks_an_unboxed_response_as_not_extracted():
    record = await _task().postprocess(
        _inferred("I could not solve it."), TaskContext(sample_id=0)
    )
    assert record["rollouts"][0]["extracted"] is False
    assert record["rollouts"][0].get("prediction") is None


@pytest.mark.anyio
async def test_feedback_grades_every_slot_and_records_grouping_keys():
    raw = _sample(answer=["1", "2"], answer_type=["NV", "NV"])
    post = build_prediction_record([["1", "3"]])
    final, judgement = await _task().feedback(
        post, TaskContext(sample_id=0, raw_sample=raw)
    )

    assert final is True
    assert judgement["n_correct"] == 0  # one slot wrong -> the sample is wrong
    rollout = judgement["rollouts"][0]
    assert rollout["extra"]["per_answer"] == [True, False]
    assert rollout["metrics"]["answer_accuracy"] == 0.5
    assert judgement["extra"]["problem_id"] == "Algebra_0001"


@pytest.mark.anyio
async def test_feedback_without_a_raw_sample_fails_the_sample():
    """No gold to compare against means no verdict, not a wrong-by-default one.

    Failing costs neither metric: AAcc counts versions as
    `len(finals) + len(fails)`, and EAcc keeps the problem's place through
    report()'s failed-sample loop.
    """
    with pytest.raises(ValueError, match="no raw sample to grade against"):
        await _task().feedback(
            build_prediction_record([["1"]]), TaskContext(sample_id=0, raw_sample=None)
        )


@pytest.mark.anyio
async def test_effective_accuracy_needs_every_version():
    finals = [
        *_all_versions("p1", [True, True, True]),
        *_all_versions("p2", [True, True, False]),
    ]
    report = await _task().report(finals, [])

    assert report["n_problems"] == 2
    assert report["eacc"] == 50.0  # only p1 is correct in all three versions
    assert report["aacc"] == pytest.approx(500 / 6)  # 5 of 6 versions
    assert report["cacc"] == 100.0  # both are right at least once
    assert report["delta"] == pytest.approx(report["aacc"] - report["eacc"])
    assert report["relative_delta"] == pytest.approx(
        (report["aacc"] - report["eacc"]) * 100 / report["eacc"]
    )
    assert report["score"] == report["eacc"]


@pytest.mark.anyio
async def test_a_problem_missing_a_version_cannot_be_an_effective_hit():
    finals = _all_versions("p1", [True, True])  # third version never judged
    report = await _task().report(finals, [])

    assert report["incomplete_problems"] == 1
    assert report["eacc"] == 0.0
    assert report["cacc"] == 100.0


@pytest.mark.anyio
async def test_failed_samples_count_against_the_average():
    finals = _all_versions("p1", [True, True])
    report = await _task().report(finals, [_failed("p1", 3)])

    assert report["fails"] == 1.0
    assert report["n_versions_judged"] == 2.0
    assert report["aacc"] == pytest.approx(200 / 3)  # 2 correct out of 3 versions


@pytest.mark.anyio
async def test_a_wholly_failed_problem_stays_in_the_effective_accuracy_denominator():
    # Every version of p2 failed, so p2 contributes no judgement at all. It must
    # still occupy a slot: dropping it would compute EAcc over the survivors and
    # report 100.0 for a run that answered half the problems.
    finals = _all_versions("p1", [True, True, True])
    report = await _task().report(finals, [_failed("p2", v) for v in (1, 2, 3)])

    assert report["n_problems"] == 2.0
    assert report["eacc"] == 50.0
    assert report["cacc"] == 50.0
    assert report["incomplete_problems"] == 1.0
    assert report["unattributed_fails"] == 0.0


@pytest.mark.anyio
async def test_a_failed_sample_is_identified_from_its_prompt_record():
    # Persisted contexts are not required to carry raw_sample; the prompt record
    # carries the same grouping keys, so identity survives either way.
    raw = _sample("p2", 1)
    ctx = await _task().preprocess(raw, TaskContext(sample_id="p2-v1", raw_sample=raw))
    orphan = TaskContext(sample_id="p2-v1", preprocess_result=ctx).to_failed(
        None, "error", "boom"
    )
    report = await _task().report(_all_versions("p1", [True, True, True]), [orphan])

    assert report["n_problems"] == 2.0
    assert report["unattributed_fails"] == 0.0


@pytest.mark.anyio
async def test_a_fail_with_no_identity_is_counted_not_silently_dropped():
    orphan = TaskContext(sample_id="?").to_failed(None, "error", "boom")
    report = await _task().report(_all_versions("p1", [True, True, True]), [orphan])

    assert report["unattributed_fails"] == 1.0
    assert report["n_problems"] == 1.0  # nothing to attribute it to


@pytest.mark.anyio
async def test_extra_rollouts_do_not_become_pass_at_n():
    # A model configured with n > 1 must not turn a version into "any rollout
    # was right" -- that would inflate every accuracy built on top of it.
    ctx = TaskContext(
        sample_id=0,
        feedback_result=build_judgement_record(
            ["1"],
            [build_rollout_judgement(0, False), build_rollout_judgement(1, True)],
            extra={"problem_id": "p1", "version": 1, "subject": "Algebra"},
        ),
    ).to_final()
    report = await _task().report([ctx], [])
    assert report["aacc"] == 0.0


@pytest.mark.anyio
async def test_per_subject_effective_accuracy_is_reported():
    finals = [
        *_all_versions("a1", [True, True, True], subject="Algebra"),
        *_all_versions("g1", [False, True, True], subject="Geometry"),
    ]
    report = await _task().report(finals, [])

    assert report["eacc_algebra"] == 100.0
    assert report["eacc_geometry"] == 0.0


@pytest.mark.anyio
async def test_empty_run_reports_the_same_keys():
    report = await _task().report([], [])
    for key in ("score", "eacc", "aacc", "cacc", "delta", "relative_delta", "fails"):
        assert key in report
    assert report["score"] == 0.0


def _judged_without_extra(problem_id: str, version: int, correct: bool) -> TaskContext:
    """A judged version whose judgement lost its grouping keys.

    What ``feedback()`` emits when ``raw_sample`` is gone: a wrong-by-default
    verdict. The prompt record still names the problem.
    """
    return TaskContext(
        sample_id=f"{problem_id}-v{version}",
        preprocess_result=build_prompt_record(
            [{"role": "user", "content": "q"}],
            reference=["1"],
            extra={
                "problem_id": problem_id,
                "version": version,
                "subject": "Algebra",
            },
        ),
        feedback_result=build_judgement_record(
            ["1"], [build_rollout_judgement(0, correct)]
        ),
    ).to_final()


@pytest.mark.anyio
async def test_a_judged_version_without_grouping_keys_is_recovered_from_the_prompt():
    # Otherwise the problem leaves EAcc's denominator while its three wrong
    # verdicts stay in AAcc's, so EAcc is computed over the survivors — biased
    # *upward*, in the direction that flatters the run, and silently.
    good = _all_versions("p1", [True, True, True])
    lost = [_judged_without_extra("p2", version, False) for version in (1, 2, 3)]

    report = await _task().report(good + lost, [])

    assert report["n_problems"] == 2
    assert report["eacc"] == 50.0  # not 100.0
    assert report["aacc"] == 50.0
    assert report["unattributed_finals"] == 0.0


@pytest.mark.anyio
async def test_a_failed_version_still_names_its_problem_from_the_prompt_record():
    """Where the lost-`raw_sample` guarantee lives now that feedback raises.

    The branch that used to hand-build a wrong-by-default judgement did this by
    calling `_identify` itself. report()'s failed-sample loop calls the same
    `_identify`, so the problem keeps its place in the EAcc denominator — the
    invariant is unchanged, one code path lighter. Without it, all three versions
    of `p9` would leave `by_problem` while staying in AAcc's denominator, biasing
    EAcc upward.
    """
    good = _all_versions("p1", [True, True, True])
    lost = [
        TaskContext(
            sample_id=f"p9-v{version}",
            raw_sample=None,
            preprocess_result=build_prompt_record(
                [{"role": "user", "content": "q"}],
                reference=["1"],
                extra={"problem_id": "p9", "version": version, "subject": "Algebra"},
            ),
        ).to_failed(TaskAction.FEEDBACK, "exception::ValueError", "no raw sample")
        for version in (1, 2, 3)
    ]

    report = await _task().report(good, lost)

    # p9 held its place: two problems, and only p1 is correct in every version.
    assert report["n_problems"] == 2
    assert report["eacc"] == 50.0
    assert report["unattributed_fails"] == 0.0


@pytest.mark.anyio
async def test_an_unrecoverable_version_is_counted_rather_than_dropped():
    # Nothing left to recover from, so EAcc really is an upper bound here. The
    # point is that the run says so instead of reporting a clean number.
    good = _all_versions("p1", [True, True, True])
    lost = [
        TaskContext(
            sample_id=f"p2-v{version}",
            feedback_result=build_judgement_record(
                ["1"], [build_rollout_judgement(0, False)]
            ),
        ).to_final()
        for version in (1, 2, 3)
    ]

    report = await _task().report(good + lost, [])

    assert report["unattributed_finals"] == 3.0
    assert report["eacc"] > report["aacc"]  # the invariant this makes visible


# --- the headline interval --------------------------------------------------


@pytest.mark.anyio
async def test_the_interval_is_clustered_on_problems_not_versions():
    """The headline is EAcc, a per-PROBLEM rate, so the population is problems.

    One sample is one *(problem, version)* pair, so reading the versions as
    independent problems would narrow the interval by the same `sqrt(times)` an
    uncollapsed repeat does, wearing a different name.
    """
    finals = [
        *_all_versions("p1", [True, True, True]),
        *_all_versions("p2", [True, True, True]),
        *_all_versions("p3", [True, True, False]),
        *_all_versions("p4", [False, False, False]),
    ]
    report = await _task().report(finals, [])

    assert report[PROBLEM_COUNT_FIELD] == 4.0  # problems, not the 12 versions
    interval = report[SCORE_CI_FIELD]
    assert isinstance(interval, list)
    lo, hi = interval
    assert lo < report["eacc"] < hi
    # `eacc` is `score` under its own name, so it repeats the headline bounds;
    # `cacc` is the other per-problem rate and gets its own, over "correct in ANY
    # version" rather than "in every one". They read 50.0 and 75.0 here, so a
    # `cacc` bound copied from the headline fails to contain its own number.
    assert report["eacc_ci95"] == [lo, hi]
    cacc_interval = report["cacc_ci95"]
    assert isinstance(cacc_interval, list)
    assert cacc_interval != interval
    assert cacc_interval[0] < report["cacc"] < cacc_interval[1]
    # Two units in one report, so `aacc` is declared on its own population and
    # the three per-problem rates on theirs.
    assert report["ci95_units"] == {
        "score": PROBLEM_COUNT_FIELD,
        "eacc": PROBLEM_COUNT_FIELD,
        "cacc": PROBLEM_COUNT_FIELD,
        "aacc": VERSION_COUNT_FIELD,
    }
    # `delta` and `relative_delta` combine aggregates from BOTH units, so no
    # per-unit value has either as its mean and neither gets an interval.
    assert "delta_ci95" not in report
    assert "relative_delta_ci95" not in report
    # The task tests call report() directly, so the runner's finalizer never sees
    # this dict -- run the validator here or a missing declaration ships.
    assert interval_declaration_problems(report) == []

    # It is exactly the interval over the four per-problem EAcc indicators --
    # two of four correct in every version.
    expected = wilson_interval([1.0, 1.0, 0.0, 0.0], 4)
    assert expected is not None
    assert (lo, hi) == expected

    # And the collapsing is what widened it: the same verdicts read per VERSION
    # -- 8 of 12 correct, which is AAcc's axis -- give a strictly narrower one.
    # That narrower interval is AAcc's own, and it is published as such rather
    # than not at all: 12 versions, declared as 12.
    per_version = wilson_interval([1.0] * 8 + [0.0] * 4, 12)
    assert per_version is not None
    assert hi - lo > per_version[1] - per_version[0]
    assert report[VERSION_COUNT_FIELD] == 12.0
    assert report["aacc_ci95"] == list(per_version)
    aacc_interval = report["aacc_ci95"]
    assert isinstance(aacc_interval, list)
    assert aacc_interval[0] < report["aacc"] < aacc_interval[1]


@pytest.mark.anyio
async def test_the_interval_is_on_eaccs_axis_not_aaccs():
    """A mean over versions would land on AAcc, which is a different number.

    20 problems, each correct in one version of three: EAcc is 0 and AAcc is
    33.3. Collapsing the versions by their MEAN -- what a grouped
    `interval_metrics` call would do -- brackets 33.3; EAcc's all-versions
    reduction brackets 0. The two do not overlap, so this fails loudly if the
    interval is ever computed on the version axis.
    """
    finals = [
        final
        for problem in range(20)
        for final in _all_versions(f"p{problem}", [True, False, False])
    ]
    report = await _task().report(finals, [])

    assert report[PROBLEM_COUNT_FIELD] == 20.0
    assert report["eacc"] == 0.0
    assert report["aacc"] == pytest.approx(100 / 3)
    interval = report[SCORE_CI_FIELD]
    assert isinstance(interval, list)
    lo, hi = interval
    # p == 0 exactly: the one-sided Clopper-Pearson limit over 20 problems.
    assert lo == 0.0
    assert hi < report["aacc"]


@pytest.mark.anyio
async def test_a_repeated_version_does_not_inflate_the_population():
    """Copies of a version share their `problem_id`, so `by_problem` absorbs them.

    A repeat-wrapped run is degenerate for EAcc for an unrelated reason -- a
    problem carrying six verdicts is not judged on exactly `VERSIONS` of them, so
    it counts as incomplete and can never be a hit -- but the POPULATION must
    still be the problem count. Reading the copies as extra problems is the
    narrowing this task's grouping exists to prevent.
    """
    finals = [
        final
        for problem in ("p1", "p2")
        for _ in range(2)
        for final in _all_versions(problem, [True, True, True])
    ]
    report = await _task().report(finals, [])

    assert report[PROBLEM_COUNT_FIELD] == 2.0  # not 4, and not the 12 samples
    assert report["incomplete_problems"] == 2.0


def test_problem_groups_is_off_because_report_already_collapsed():
    """Even on a repeated split, where the base implementation WOULD group.

    `report`'s `by_problem` reduction is the collapse, and it is nonlinear:
    letting core collapse the same samples again by their mean would put the
    interval on AAcc's axis.
    """
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample(), _sample()])})
    ).repeat(2)
    task = UGMathBenchZeroShotGenFixedTask(
        dataset, ChatModel(model="mock-chat", api_key="fake")
    )
    test_set = task.dataset.test_set
    assert test_set is not None
    # The premise: the split really is repeat-stamped, so the base implementation
    # would return a grouping here rather than None for want of a column.
    assert REPEAT_GROUP_COLUMN in test_set.column_names

    finals = [task.make_context(i) for i in range(len(test_set))]
    assert task.problem_groups(finals) is None


# --- n / k sampling wiring -------------------------------------------------


class _CapturingChatModel(ChatModel):
    """Records the merged request kwargs, and honours `n` the way a backend does."""

    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_kwargs = {**self._kwargs, "n": req.sampling.n}
        return Response(texts=(r"\boxed{4}",) * req.sampling.n)


@pytest.mark.anyio
async def test_infer_forwards_n_to_the_model():
    """Without this, `n=4` enables the sampling metrics over a one-rollout draw."""
    sample = _sample()
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([sample])})
    )
    model = _CapturingChatModel()
    task = UGMathBenchZeroShotGenFixedTask(dataset, model, k=4, n=4)

    pre = await task.preprocess(sample, TaskContext(sample_id=0))
    out = await task.infer(pre, TaskContext(sample_id=0))

    assert model.last_kwargs.get("n") == 4
    # and the whole draw survives postprocess, one prediction per rollout
    post = await task.postprocess(out, TaskContext(sample_id=0))
    assert len(post["rollouts"]) == 4


@pytest.mark.anyio
async def test_sampling_metrics_use_the_aacc_denominator():
    """A failed version counts as wrong, as it does for AAcc.

    Averaging over the judged versions alone would bias these upward over
    survivors -- the defect the EAcc warnings in this module describe.
    """
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample()])})
    )
    task = UGMathBenchZeroShotGenFixedTask(
        dataset, ChatModel(model="mock-chat", api_key="fake"), k=2, n=2
    )

    def judged(problem_id: str, version: int, verdicts: list[bool]) -> TaskContext:
        return TaskContext(
            sample_id=f"{problem_id}-v{version}",
            postprocess_result=build_prediction_record([["4"]] * len(verdicts)),
            feedback_result=build_judgement_record(
                ["4"],
                [build_rollout_judgement(i, c) for i, c in enumerate(verdicts)],
                extra={
                    "problem_id": problem_id,
                    "version": version,
                    "subject": "Algebra",
                },
            ),
        ).to_final()

    finals = [judged("p1", v, [True, True]) for v in (1, 2, 3)]
    fails = [TaskContext(sample_id="p2-v1", raw_sample=_sample("p2", 1))]

    report = await task.report(finals, fails)

    assert report["n"] == 2.0
    assert report["k"] == 2.0
    assert report["n_short"] == 0.0
    # 3 judged and solved over a denominator of 4 -> 75, not the 100 a
    # survivors-only denominator would report.
    assert report["pass@1"] == pytest.approx(75.0)
    assert report["avg@n"] == pytest.approx(75.0)
    # Every judged version here reads 1.0, so there is no spread between the
    # units to estimate from and the keys ship without a bound -- omitted, never
    # a zero-width interval claiming a certainty this run does not have.
    units = report["ci95_units"]
    assert isinstance(units, dict)
    assert "pass@1" not in units
    assert "pass@1_ci95" not in report
    assert wilson_interval([1.0] * 3, 4) is None
    assert interval_declaration_problems(report) == []


@pytest.mark.anyio
async def test_the_sampling_block_is_clustered_on_versions_not_problems():
    """UGMathBench is the one task whose sampling block is not per problem.

    Each entry `aggregate` folds is one *(problem, version)* pair, over AAcc's
    denominator, so every key of that block declares `n_versions`. Copying the
    headline's declaration would quote a per-version width over a population of
    problems -- the same narrowing an uncollapsed repeat produces.
    """
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample()])})
    )
    task = UGMathBenchZeroShotGenFixedTask(
        dataset, ChatModel(model="mock-chat", api_key="fake"), k=2, n=2
    )

    def judged(problem_id: str, version: int, verdicts: list[bool]) -> TaskContext:
        return TaskContext(
            sample_id=f"{problem_id}-v{version}",
            postprocess_result=build_prediction_record([["4"]] * len(verdicts)),
            feedback_result=build_judgement_record(
                ["4"],
                [build_rollout_judgement(i, c) for i, c in enumerate(verdicts)],
                extra={
                    "problem_id": problem_id,
                    "version": version,
                    "subject": "Algebra",
                },
            ),
        ).to_final()

    # Four versions with three different per-version pass@1 values, so the block
    # has something to estimate and the six keys do not all coincide.
    finals = [
        judged("p1", 1, [True, True]),
        judged("p1", 2, [True, False]),
        judged("p1", 3, [False, False]),
        judged("p2", 1, [True, True]),
    ]
    report = await task.report(finals, [])

    assert report[VERSION_COUNT_FIELD] == 4.0
    assert report[PROBLEM_COUNT_FIELD] == 2.0
    units = report["ci95_units"]
    assert isinstance(units, dict)
    # The whole block on the version axis, the headline and its per-problem
    # siblings on theirs.
    assert units["pass@1"] == VERSION_COUNT_FIELD
    assert units["avg@n"] == VERSION_COUNT_FIELD
    assert units["pass@k"] == VERSION_COUNT_FIELD
    assert units["aacc"] == VERSION_COUNT_FIELD
    assert units["score"] == PROBLEM_COUNT_FIELD
    assert units["cacc"] == PROBLEM_COUNT_FIELD
    # The interval set is the metric set: every key the block folded is
    # declared, and nothing the budget gated out is.
    for key in ("pass@1", "avg@n", "pass@k", "pass^k", "maj@k", "self_consistency"):
        assert (key in report) == (key in units), key
    # Each key is estimated on its OWN per-version values, not on a shared one:
    # pass@1 averages 1.0 / 0.5 / 0.0 / 1.0 while pass@2 averages the
    # solved-at-least-once indicator, so the two bounds differ.
    expected = wilson_interval([1.0, 0.5, 0.0, 1.0], 4)
    assert expected is not None
    assert report["pass@1_ci95"] == list(expected)
    assert report["pass@k_ci95"] != report["pass@1_ci95"]
    assert interval_declaration_problems(report) == []
