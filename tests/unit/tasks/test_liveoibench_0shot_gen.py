"""Unit tests for the LiveOIBench 0-shot chat task.

Three things are load-bearing here and none of them live in the evaluator:

* what goes on the wire — the problem's own limits, the grader sources, and a
  test *directory* rather than ~140 MB of inlined cases;
* mapping the returned verdict vector onto subtasks by the names the evaluator
  reports, rather than by an ordering the task re-derives;
* best-of-n, which is how upstream reports a model — not mean, not pass@k.

The execution guard itself is the vendored evaluator's, a separate deployable
with its own tests.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import json
import os

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    NonRetriableSampleError,
    TaskContext,
    build_prediction_record,
)
from sieval.datasets.liveoibench import LiveOIBenchDataset
from sieval.tasks.liveoibench_0shot_gen import LiveOIBenchZeroShotGenTask
from tests.conftest import HandlerTransport

PROBLEM_ID = "IOI-2025-contest-beechtree"
# Two subtasks: the first pays 30 for both its tests, the second 70 for its one.
SUBTASKS = {
    "1": {"task": "Subtask 1", "score": 30, "testcases": ["a1", "a2"]},
    "2": {"task": "Subtask 2", "score": 70, "testcases": ["b1"]},
}
CASE_NAMES = ["a1", "a2", "b1"]


class _StubChatModel(ChatModel):
    def __init__(
        self, replies: tuple[str, ...] = ("```beechtree.cpp\nint main(){}\n```",)
    ):
        self._replies = replies
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=self._replies[: req.sampling.n])


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Evaluator:
    """Stands in for the code-eval service, one scripted reply per rollout."""

    def __init__(self, verdict_sets: list[list[bool]]):
        self._verdict_sets = verdict_sets
        self.bodies: list[dict] = []
        self.deadlines: list[float] = []

    async def post(self, url, *, json, timeout):
        _ = url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        verdicts = self._verdict_sets[len(self.bodies) - 1]
        return _Response(
            {
                "status": all(verdicts),
                "msg": "graded",
                "data": {
                    "n_cases": len(verdicts),
                    "n_passed": sum(verdicts),
                    "case_verdicts": verdicts,
                    "case_names": CASE_NAMES,
                    "peak_memory_mb": 12.5,
                },
            }
        )

    async def aclose(self) -> None:
        return None


def _raw(
    tests_dir: str = "/staged/IOI/2025/contest/beechtree/tests", **overrides
) -> dict:
    return {
        "problem_id": PROBLEM_ID,
        "competition": "IOI",
        "contest": "contest",
        "task_name": "beechtree",
        "problem_statement": "# Beech Tree\n",
        "time_limit": 1.5,
        "memory_limit": 2048.0,
        "task_type": "batch",
        "difficulty": 12,
        "algorithms": '["tree"]',
        "grader_code": "",
        "starter_code": "",
        "subtasks": json.dumps(SUBTASKS),
        "contest_id": "IOI-2025-contest",
        "tests_dir": tests_dir,
    } | overrides


def _task(raw: dict, evaluator: _Evaluator | None = None, **kwargs):
    dataset = LiveOIBenchDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([raw])})
    )
    task = LiveOIBenchZeroShotGenTask(dataset, _StubChatModel(), **kwargs)
    if evaluator is not None:
        task._http_client = evaluator  # the real client is never used
    return task


async def _grade(raw: dict, predictions: list[str], verdict_sets, **kwargs):
    evaluator = _Evaluator(verdict_sets)
    task = _task(raw, evaluator, **kwargs)
    try:
        _, judgement = await task.feedback(
            build_prediction_record(predictions),
            TaskContext(sample_id=0, raw_sample=raw),
        )
    finally:
        await task.shutdown()
    return judgement, evaluator


# --------------------------------------------------------------------------- #
# Prompt and extraction
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_the_prompt_is_a_single_user_turn():
    # Upstream's default run sends no system message.
    task = _task(_raw())
    record = await task.preprocess(_raw(), TaskContext(sample_id=0))
    await task.shutdown()
    assert [m["role"] for m in record["prompt"]] == ["user"]
    assert "Time limit: 1.5 seconds" in record["prompt"][0]["content"]
    assert "'```beechtree.cpp'" in record["prompt"][0]["content"]


@pytest.mark.anyio
async def test_extraction_prefers_the_tasks_own_fence():
    task = _task(_raw())
    reply = (
        "Here is a sketch:\n```cpp\nint wrong(){}\n```\n"
        "And the answer:\n```beechtree.cpp\nint right(){}\n```\n"
    )
    record = await task.postprocess(
        Response(texts=(reply,)), TaskContext(sample_id=0, raw_sample=_raw())
    )
    await task.shutdown()
    assert record["rollouts"][0]["prediction"] == "int right(){}"


@pytest.mark.anyio
async def test_a_reply_with_no_code_records_a_miss():
    task = _task(_raw())
    record = await task.postprocess(
        Response(texts=("I cannot solve this.",)),
        TaskContext(sample_id=0, raw_sample=_raw()),
    )
    await task.shutdown()
    assert record["rollouts"][0].get("prediction") is None
    assert record["rollouts"][0]["extracted"] is False


# --------------------------------------------------------------------------- #
# What goes on the wire
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_the_request_carries_the_problems_own_limits_and_a_test_directory():
    _, evaluator = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    (body,) = evaluator.bodies
    assert body["source"] == "liveoibench" and body["lang"] == "cpp"
    assert body["timeout_per_case"] == 1.5
    assert body["memory_limit"] == 2048
    assert body["entry_filename"] == "beechtree.cpp"
    # A directory, not ~140 MB of inlined cases.
    assert body["test_dir"].endswith("beechtree/tests")
    assert "test" not in body


@pytest.mark.anyio
async def test_grader_sources_are_sent_for_the_problems_that_ship_them():
    raw = _raw(
        grader_code=json.dumps(
            {"grader.cpp": "int main(){}", "beechtree.h": "void f();"}
        )
    )
    _, evaluator = await _grade(raw, ["void f(){}"], [[True, True, True]])
    assert evaluator.bodies[0]["files"] == {
        "grader.cpp": "int main(){}",
        "beechtree.h": "void f();",
    }


@pytest.mark.anyio
async def test_inline_mode_ships_the_cases_read_from_disk(tmp_path):
    for name, payload in (("a1", "1\n"), ("a2", "2\n"), ("b1", "3\n")):
        (tmp_path / f"{name}.in").write_text(payload)
        (tmp_path / f"{name}.out").write_text(payload)
    raw = _raw(tests_dir=str(tmp_path))
    _, evaluator = await _grade(
        raw, ["int main(){}"], [[True, True, True]], inline_tests=True
    )
    body = evaluator.bodies[0]
    assert "test_dir" not in body
    assert body["test"]["names"] == CASE_NAMES
    assert body["test"]["inputs"] == ["1\n", "2\n", "3\n"]


@pytest.mark.anyio
async def test_inline_mode_refuses_a_test_input_with_no_expected_output(tmp_path):
    (tmp_path / "a1.in").write_text("1\n")
    raw = _raw(tests_dir=str(tmp_path))
    with pytest.raises(NonRetriableSampleError, match="no .out"):
        await _grade(raw, ["int main(){}"], [[True]], inline_tests=True)


@pytest.mark.anyio
async def test_the_http_deadline_outlasts_every_budget_the_evaluator_can_spend(
    tmp_path,
):
    for name in CASE_NAMES:
        (tmp_path / f"{name}.in").write_text("1\n")
        (tmp_path / f"{name}.out").write_text("1\n")
    _, evaluator = await _grade(
        _raw(tests_dir=str(tmp_path)), ["int main(){}"], [[True, True, True]]
    )
    # 60s compile bound + 3 cases * (1.5 * 1.2 + 5) + 30s slack.
    assert evaluator.deadlines[0] == pytest.approx(60.0 + 3 * 6.8 + 30.0)


# --------------------------------------------------------------------------- #
# Verdicts to subtask scores
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_verdicts_are_mapped_onto_subtasks_by_the_names_the_evaluator_reports():
    # b1 alone pays 70; a1/a2 are incomplete so subtask 1 pays nothing.
    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, False, True]])
    assert judgement["score"] == 70.0
    assert judgement["metrics"]["relative_score"] == 70.0
    assert judgement["metrics"]["tests_passed_pct"] == pytest.approx(200 / 3)
    assert judgement["rollouts"][0]["extra"]["subtasks"]["1"]["score"] == 0
    assert judgement["rollouts"][0]["extra"]["subtasks"]["2"]["score"] == 70


@pytest.mark.anyio
async def test_a_full_solve_is_correct_and_aces():
    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    assert judgement["rollouts"][0]["correct"] is True
    assert judgement["metrics"]["ace"] is True
    assert judgement["score"] == 100.0


@pytest.mark.anyio
async def test_the_judgement_records_no_reference_and_names_the_rubric():
    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    assert judgement["reference"] is None
    assert judgement["extra"]["total_points"] == 100
    assert judgement["extra"]["n_subtasks"] == 2
    assert judgement["extra"]["problem_id"] == PROBLEM_ID


@pytest.mark.anyio
async def test_the_best_rollout_is_reported_not_the_first_or_the_mean():
    # Upstream samples n and reports the highest-scoring candidate.
    judgement, _ = await _grade(
        _raw(),
        ["int a(){}", "int b(){}", "int c(){}"],
        [[True, True, False], [True, False, True], [False, False, False]],
        n=3,
    )
    assert judgement["score"] == 70.0, "the 70-point rollout must win"
    assert judgement["extra"]["best_rollout"] == 1
    assert judgement["n_rollouts"] == 3


@pytest.mark.anyio
async def test_ties_on_score_are_broken_by_tests_passed():
    # Both score 0; the second passed more tests, so it is the better candidate.
    judgement, _ = await _grade(
        _raw(),
        ["int a(){}", "int b(){}"],
        [[False, False, False], [True, False, False]],
        n=2,
    )
    assert judgement["score"] == 0.0
    assert judgement["extra"]["best_rollout"] == 1


@pytest.mark.anyio
async def test_an_evaluator_without_the_cpp_path_is_an_error_not_a_zero():
    # `status` alone cannot attribute subtasks, so scoring from it would report
    # every partial solution as zero.
    class _OldEvaluator(_Evaluator):
        async def post(self, url, *, json, timeout):
            self.bodies.append(json)
            return _Response(
                {"status": False, "msg": "not supported data source", "data": None}
            )

    task = _task(_raw(), _OldEvaluator([]))
    try:
        with pytest.raises(RuntimeError, match="no per-case verdicts"):
            await task.feedback(
                build_prediction_record(["int main(){}"]),
                TaskContext(sample_id=0, raw_sample=_raw()),
            )
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_a_problem_with_no_rubric_fails_the_sample_without_retrying():
    task = _task(_raw(subtasks="{}"))
    try:
        with pytest.raises(NonRetriableSampleError, match="no subtasks"):
            await task.feedback(
                build_prediction_record(["int main(){}"]),
                TaskContext(sample_id=0, raw_sample=_raw(subtasks="{}")),
            )
    finally:
        await task.shutdown()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
class _Final:
    """A finalized sample as `report` reads it: the judgement plus the
    prediction record `health_metrics` counts extraction misses from."""

    def __init__(self, judgement: dict, predictions: list[str | None] | None = None):
        self.feedback_result = judgement
        self.postprocess_result = build_prediction_record(
            predictions if predictions is not None else ["int main(){}"]
        )


@pytest.mark.anyio
async def test_the_report_averages_over_every_requested_sample():
    task = _task(_raw())
    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    try:
        # One perfect problem plus one pipeline failure -> 50, not 100.
        report = await task.report([_Final(judgement)], [object()])
    finally:
        await task.shutdown()
    assert report["relative_score"] == 50.0
    assert report["score"] == report["relative_score"]
    assert report["score_key"] == "relative_score"
    assert report["denominator_policy"] == "requested"
    assert report["ace_rate"] == 50.0
    assert report["fails"] == 1.0


@pytest.mark.anyio
async def test_an_empty_run_still_declares_its_headline():
    task = _task(_raw())
    try:
        report = await task.report([], [object()])
    finally:
        await task.shutdown()
    assert report["score"] == 0.0
    assert report["score_key"] == "relative_score"
    assert report["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_human_metrics_come_from_the_staged_contestant_table(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    contestants = tmp_path / "contest_results.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contest_id": "IOI-2025-contest",
                    "gold_cutoff": 90.0,
                    "silver_cutoff": 60.0,
                    "bronze_cutoff": 30.0,
                    "contestants_ranking": json.dumps(
                        [{"Rank": 1, "beechtree": 100}, {"Rank": 2, "beechtree": 40}]
                    ),
                }
            ]
        ),
        contestants,
    )

    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    task = _task(_raw())
    task.dataset._contestants_path = str(contestants)
    try:
        report = await task.report([_Final(judgement)], [])
    finally:
        await task.shutdown()

    # 100 beats the 40 and ties the 100 -> 50th percentile, and clears gold.
    assert report["human_percentile"] == 50.0
    assert report["gold_rate"] == 100.0
    assert report["n_contests"] == 1.0


@pytest.mark.anyio
async def test_a_missing_contestant_table_leaves_the_scores_reportable(tmp_path):
    judgement, _ = await _grade(_raw(), ["int main(){}"], [[True, True, True]])
    task = _task(_raw())
    task.dataset._contestants_path = str(tmp_path / "absent.parquet")
    try:
        report = await task.report([_Final(judgement)], [])
    finally:
        await task.shutdown()
    assert report["relative_score"] == 100.0
    assert "human_percentile" not in report


@pytest.mark.anyio
async def test_the_task_is_registered_under_its_file_name():
    from sieval.core.tasks.meta import get_task_class

    assert get_task_class("liveoibench_0shot_gen") is LiveOIBenchZeroShotGenTask
    assert os.path.basename(__file__) == "test_liveoibench_0shot_gen.py"
