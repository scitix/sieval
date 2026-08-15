"""Unit tests for the corrected UGMathBench task: stage plumbing and metrics.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

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
from sieval.datasets.ugmathbench import UGMathBenchDataset
from sieval.tasks.ugmathbench_0shot_gen_fixed import UGMathBenchZeroShotGenFixedTask
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
