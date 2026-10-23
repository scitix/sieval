"""Tests for ``TaskRunner.plan_run``: what a run would do, decided read-only.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import hashlib
from pathlib import Path

import pytest

from sieval.core.runners import MultiTaskRunner, StageResumeError
from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.consts import TaskAction
from sieval.core.tasks.loader import TaskLoader
from tests.conftest import (
    MockChatModel,
    MockDataset,
    MockSelectiveFailModel,
    make_config,
)
from tests.integration.resume.conftest import (
    RESUME_SAMPLES,
    CountingMockChatModel,
    ResumeTask,
)

ANSWERS = {"Q1": "A1", "Q2": "A2", "Q3": "A3"}


class AgainTask(ResumeTask):
    """With *again*, ``feedback`` asks once more after the first answer."""

    def __init__(self, dataset, model, name=None, again=False):
        super().__init__(dataset=dataset, model=model, name=name)
        self._again = again

    async def feedback(self, post, ctx):
        done, result = await super().feedback(post, ctx)
        return done and not (self._again and ctx.iteration == 0), result


def _tree(root: Path) -> dict[str, str]:
    """Every file under *root* with its content digest."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


async def _record(tmp_path: Path, samples=RESUME_SAMPLES, model=None, **config) -> Path:
    result_dir = tmp_path / "recorded"
    task = ResumeTask(
        dataset=MockDataset(samples),
        model=model or MockChatModel(answers=ANSWERS),
        name="t",
    )
    config = make_config(tmp_path, result_dir=str(result_dir), **config)
    await TaskRunner(task, config).arun()
    return result_dir


def _runner(tmp_path: Path, result_dir: Path, samples=RESUME_SAMPLES, **config):
    model = CountingMockChatModel(answers=ANSWERS)
    task = ResumeTask(dataset=MockDataset(samples), model=model, name="t")
    config.setdefault("resume", True)
    return TaskRunner(
        task, make_config(tmp_path, result_dir=str(result_dir), **config)
    ), model


class TestPlanResume:
    @pytest.mark.anyio
    async def test_a_fresh_run_infers_every_sample(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path, tmp_path / "none", resume=False)
        plan = await runner.plan_run()
        assert plan == {
            "start": "fresh",
            "recorded": None,
            "samples": 3,
            "outcomes": None,
            "new": 3,
            "to_infer": 3,
            "to_grade": 3,
        }
        assert not (tmp_path / "none").exists()

    @pytest.mark.anyio
    async def test_a_finished_run_is_reused_whole(self, tmp_path: Path) -> None:
        result_dir = await _record(tmp_path)
        before = _tree(result_dir)
        runner, model = _runner(tmp_path, result_dir)

        plan = await runner.plan_run()

        assert plan["start"] == "reuse"
        assert plan["outcomes"] is not None
        assert (plan["outcomes"]["kept"], plan["outcomes"]["rereported"]) == (3, 0)
        assert (plan["to_infer"], plan["to_grade"]) == (0, 0)
        assert model.call_count == 0
        assert _tree(result_dir) == before

    @pytest.mark.anyio
    async def test_new_samples_are_counted_to_infer(self, tmp_path: Path) -> None:
        result_dir = await _record(tmp_path, samples=RESUME_SAMPLES[:2])
        before = _tree(result_dir)
        runner, model = _runner(tmp_path, result_dir)

        plan = await runner.plan_run()

        assert plan["start"] == "resume"
        assert plan["outcomes"] == {
            "kept": 2,
            "rereported": 0,
            "regraded": 0,
            "repostprocessed": 0,
            "restarted": 0,
            "dropped": 0,
        }
        assert (plan["new"], plan["to_infer"], plan["to_grade"]) == (1, 1, 1)
        assert model.call_count == 0
        assert _tree(result_dir) == before

    @pytest.mark.anyio
    async def test_a_failed_inference_is_counted_to_infer(self, tmp_path: Path) -> None:
        failing = MockSelectiveFailModel(fail_samples={"Q3"}, answers=ANSWERS)
        result_dir = await _record(tmp_path, model=failing)
        before = _tree(result_dir)
        runner = TaskRunner(
            ResumeTask(dataset=MockDataset(RESUME_SAMPLES), model=failing, name="t"),
            make_config(tmp_path, result_dir=str(result_dir), resume=True),
        )

        plan = await runner.plan_run()

        assert (plan["new"], plan["to_infer"], plan["to_grade"]) == (0, 1, 1)
        assert _tree(result_dir) == before

    @pytest.mark.anyio
    async def test_a_failure_past_max_retries_is_not_counted(
        self, tmp_path: Path
    ) -> None:
        failing = MockSelectiveFailModel(fail_samples={"Q3"}, answers=ANSWERS)
        result_dir = await _record(tmp_path, model=failing, max_retries=0)
        runner = TaskRunner(
            ResumeTask(dataset=MockDataset(RESUME_SAMPLES), model=failing, name="t"),
            make_config(
                tmp_path, result_dir=str(result_dir), resume=True, max_retries=0
            ),
        )
        plan = await runner.plan_run()
        assert (plan["to_infer"], plan["to_grade"]) == (0, 0)

    @pytest.mark.anyio
    async def test_resume_from_another_dir_reads_it_and_copies_nothing(
        self, tmp_path: Path
    ) -> None:
        recorded = await _record(tmp_path, samples=RESUME_SAMPLES[:2])
        before = _tree(recorded)
        target = tmp_path / "new"
        runner, _ = _runner(tmp_path, target, resume=recorded)

        plan = await runner.plan_run()

        assert plan["start"] == "resume"
        assert plan["recorded"] == str(recorded)
        assert plan["outcomes"]["kept"] == 2
        assert plan["to_infer"] == 1
        assert _tree(recorded) == before
        assert not target.exists()

    @pytest.mark.anyio
    async def test_a_resume_that_would_be_refused_raises(self, tmp_path: Path) -> None:
        result_dir = await _record(tmp_path, samples=RESUME_SAMPLES[:2])
        task = ResumeTask(
            dataset=MockDataset(RESUME_SAMPLES),
            # Other request arguments: another inference identity.
            model=MockChatModel(answers=ANSWERS).with_args(temperature=0.9),
            name="t",
        )
        runner = TaskRunner(
            task, make_config(tmp_path, result_dir=str(result_dir), resume=True)
        )
        with pytest.raises(StageResumeError, match="inference"):
            await runner.plan_run()

    @pytest.mark.anyio
    async def test_the_multi_runner_maps_a_refusal_to_its_task(
        self, tmp_path: Path
    ) -> None:
        result_dir = await _record(tmp_path, samples=RESUME_SAMPLES[:2])
        multi = MultiTaskRunner()
        multi.add_task(
            ResumeTask(
                dataset=MockDataset(RESUME_SAMPLES),
                model=MockChatModel(answers=ANSWERS).with_args(temperature=0.9),
                name="t",
            ),
            make_config(tmp_path, result_dir=str(result_dir), resume=True),
        )
        plans = await multi.plan_run()
        assert isinstance(plans["t"], StageResumeError)


class TestUnservedCandidate:
    @pytest.mark.anyio
    async def test_a_resume_with_nothing_to_infer_runs(self, tmp_path: Path) -> None:
        result_dir = await _record(tmp_path)
        (result_dir / "report.json").unlink()  # finalize through the full path
        model = CountingMockChatModel(answers=ANSWERS)
        runner = TaskRunner(
            ResumeTask(dataset=MockDataset(RESUME_SAMPLES), model=model, name="t"),
            make_config(tmp_path, result_dir=str(result_dir), resume=True),
            unserved_candidate=True,
        )
        report = await runner.arun()
        assert report["completed"] == 3
        assert model.call_count == 0

    @pytest.mark.anyio
    async def test_a_resume_with_samples_to_infer_is_refused_unrecorded(
        self, tmp_path: Path
    ) -> None:
        result_dir = await _record(tmp_path, samples=RESUME_SAMPLES[:2])
        before = _tree(result_dir)
        model = CountingMockChatModel(answers=ANSWERS)
        runner = TaskRunner(
            ResumeTask(dataset=MockDataset(RESUME_SAMPLES), model=model, name="t"),
            make_config(tmp_path, result_dir=str(result_dir), resume=True),
            unserved_candidate=True,
        )
        with pytest.raises(RuntimeError, match="1 sample"):
            await runner.arun()
        assert model.call_count == 0
        assert _tree(result_dir) == before

    @pytest.mark.anyio
    async def test_a_regrade_that_sends_a_sample_back_fails_it_at_infer(
        self, tmp_path: Path
    ) -> None:
        # The plan saw nothing to infer; the new grader's feedback iterates.
        result_dir = tmp_path / "recorded"
        answers = AgainTask(
            dataset=MockDataset(RESUME_SAMPLES), model=MockChatModel(ANSWERS), name="t"
        )
        await TaskRunner(
            answers, make_config(tmp_path, result_dir=str(result_dir))
        ).arun()

        def resume(model, **runner):
            task = AgainTask(
                dataset=MockDataset(RESUME_SAMPLES), model=model, name="t", again=True
            )
            config = make_config(tmp_path, result_dir=str(result_dir), resume=True)
            return TaskRunner(task, config, **runner)

        model = CountingMockChatModel(answers=ANSWERS)
        unserved = resume(model, unserved_candidate=True)
        assert (await unserved.plan_run())["to_infer"] == 0
        report = await resume(model, unserved_candidate=True).arun()

        # Failed by the guard at infer, before any request: not a dead endpoint.
        assert model.call_count == 0
        assert (report["completed"], report["failed"]) == (0, 3)
        recorded = await TaskLoader(answers, result_dir).load_initial_state()
        assert {
            (c.iteration, c.error_action, c.error_reason) for c in recorded.values()
        } == {(1, TaskAction.INFER, "exception::RuntimeError")}

        # A resume with the engine up retries them.
        report = await resume(model).arun()
        assert model.call_count == 3
        assert (report["completed"], report["failed"]) == (3, 0)
