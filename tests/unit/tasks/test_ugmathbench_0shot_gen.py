"""Unit tests for the UGMathBench task: stage plumbing and the version metrics.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import subprocess
import sys

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelMeta, ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.ugmathbench import UGMathBenchDataset
from sieval.tasks.ugmathbench_0shot_gen import UGMathBenchZeroShotGenTask


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


def _task(precision: float = 1e-3) -> UGMathBenchZeroShotGenTask:
    sample = _sample()
    dataset = UGMathBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([sample])})
    )
    model = ChatModel(model="mock-chat", api_key="fake")
    return UGMathBenchZeroShotGenTask(dataset, model, precision=precision)


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
async def test_feedback_without_a_raw_sample_is_wrong_not_a_crash():
    final, judgement = await _task().feedback(
        build_prediction_record([["1"]]), TaskContext(sample_id=0, raw_sample=None)
    )
    assert final is True
    assert judgement["n_correct"] == 0


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
    failed = TaskContext(sample_id="p1-v3").to_failed(None, "error", "boom")
    report = await _task().report(finals, [failed])

    assert report["fails"] == 1.0
    assert report["n_versions_judged"] == 2.0
    assert report["aacc"] == pytest.approx(200 / 3)  # 2 correct out of 3 versions


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


def test_import_does_not_pull_math_verify():
    code = (
        "import sys\n"
        "import sieval.tasks.ugmathbench_0shot_gen\n"
        "assert 'math_verify' not in sys.modules, "
        "'math_verify must be lazy-imported'\n"
    )
    # Fresh interpreter so pytest's already-loaded modules don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
