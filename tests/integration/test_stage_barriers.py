"""Pattern: stage barriers and ``stop_after`` through the real runners.

Every sample of every task finishes a barrier stage before any starts the
next; a run that stops after a stage writes no report, and the resume that
continues it ends with the same stage identities and ``requests.digest`` as a
run without barriers. A stub model delays one sample to show the barrier holds.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path

import anyio
import orjson
import pytest

from sieval.core.runners import MultiTaskRunner
from sieval.core.runners.runner import BARRIER_ITERATION_REASON
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config, prompt_of

SAMPLES = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(4)]


class _Clock:
    """The order stage starts and ends happen in, across every task."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    def mark(self, task: str, what: str, question: str) -> None:
        self.events.append((task, what, question))

    def first(self, what: str) -> int:
        return min(i for i, e in enumerate(self.events) if e[1] == what)

    def last(self, what: str) -> int:
        return max(i for i, e in enumerate(self.events) if e[1] == what)


class SlowModel(MockChatModel):
    """Answers each question with its answer; *slow* takes a while."""

    def __init__(self, slow: str | None = None, **kwargs) -> None:
        super().__init__(
            answers={s["question"]: s["answer"] for s in SAMPLES}, **kwargs
        )
        self._slow = slow

    async def _stub_arun(self, req):
        if prompt_of(req) == self._slow:
            await anyio.sleep(0.2)
        return await super()._stub_arun(req)


class ClockedTask(Task):
    model_type = "chat"

    #: Where stage starts are recorded; set on the instance, so it is not a
    #: constructor argument and not in the grader identity.
    clock: "_Clock | None" = None

    def __init__(self, dataset, model, name=None, iterate=False):
        super().__init__(dataset=dataset, model=model, name=name)
        self._iterate = iterate

    @property
    def _clock(self) -> "_Clock | None":
        return type(self).clock if self._own_clock is None else self._own_clock

    _own_clock: "_Clock | None" = None

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        out = await self.model.agenerate(pre)
        if self._clock is not None:
            self._clock.mark(self.name, "inferred", pre)
        return out

    async def postprocess(self, inf, ctx):
        if self._clock is not None:
            self._clock.mark(self.name, "postprocess", ctx.raw_sample["question"])
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        done = not self._iterate or ctx.iteration > 0
        return done, {"correct": post == ctx.raw_sample["answer"]}

    async def report(self, finals, fails):
        correct = sum(1 for f in finals if f.feedback_result["correct"])
        return {"correct": correct, "total": len(finals) + len(fails)}


def _task(name: str, clock=None, slow=None, iterate=False) -> ClockedTask:
    task = ClockedTask(
        MockDataset(SAMPLES), SlowModel(slow=slow), name=name, iterate=iterate
    )
    task._own_clock = clock
    return task


class FailingModel(SlowModel):
    """Raises on Q1, so that sample fails every attempt."""

    async def _stub_arun(self, req):
        if prompt_of(req) == "Q1":
            raise RuntimeError("Q1 always fails")
        return await super()._stub_arun(req)


def _failing_task() -> ClockedTask:
    return ClockedTask(MockDataset(SAMPLES), FailingModel(), name="a")


def _meta(root: Path, task: str) -> dict:
    return orjson.loads((root / task / "meta.json").read_bytes())


def _runner(root: Path, **kwargs) -> MultiTaskRunner:
    return MultiTaskRunner(result_dir=str(root), **kwargs)


def _config(tmp_path: Path, **overrides):
    return make_config(tmp_path, result_dir=None, **overrides)


class TestBarrier:
    @pytest.mark.anyio
    async def test_no_postprocess_starts_until_every_sample_of_every_task_inferred(
        self, tmp_path: Path
    ) -> None:
        clock = _Clock()
        hooks: list[str] = []

        async def hook(stage: str) -> None:
            hooks.append(stage)

        runner = _runner(tmp_path / "r", barriers=["inference"])
        runner.on_barrier(hook)
        runner.add_task(_task("a", clock, slow="Q2"), config=_config(tmp_path))
        runner.add_task(_task("b", clock), config=_config(tmp_path))
        reports = await runner.arun()

        assert reports["a"]["correct"] == reports["b"]["correct"] == 4
        # Q2 of task a answers last; nothing of either task graded before it.
        assert clock.last("inferred") < clock.first("postprocess")
        assert clock.events[clock.last("inferred")] == ("a", "inferred", "Q2")
        assert hooks == ["inference"]

    @pytest.mark.anyio
    async def test_without_a_barrier_a_sample_is_graded_before_the_last_answers(
        self, tmp_path: Path
    ) -> None:
        # The control: streaming is unchanged when no barrier is asked for.
        clock = _Clock()
        runner = _runner(tmp_path / "r")
        runner.add_task(_task("a", clock, slow="Q2"), config=_config(tmp_path))
        await runner.arun()
        assert clock.first("postprocess") < clock.last("inferred")

    @pytest.mark.anyio
    async def test_added_barriers_and_stop_after_are_the_effective_barriers(
        self, tmp_path: Path
    ) -> None:
        runner = _runner(tmp_path / "r", stop_after="feedback")
        runner.add_barrier("preprocess").add_barrier("inference")
        assert runner.effective_barriers() == ("preprocess", "inference", "feedback")


class TestStopAfter:
    async def _baseline(self, tmp_path: Path) -> dict:
        runner = _runner(tmp_path / "plain")
        runner.add_task(_task("a"), config=_config(tmp_path))
        await runner.arun()
        return _meta(tmp_path / "plain", "a")

    @pytest.mark.anyio
    async def test_stops_after_inference_and_a_resume_continues_identically(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "staged"
        clock = _Clock()
        runner = _runner(root, stop_after="inference")
        runner.add_task(_task("a", clock), config=_config(tmp_path))
        reports = await runner.arun()

        assert reports == {"a": {"stopped_after": "inference"}}
        assert not (root / "a" / "report.json").exists()
        assert not [e for e in clock.events if e[1] == "postprocess"]
        meta = _meta(root, "a")
        (run,) = meta["runs"]
        assert run["stopped_after"] == "inference"
        assert run["reran"] == ["data", "preprocess", "inference"]
        assert "reported" not in run

        clock.events.clear()
        resumed = _runner(root)
        resumed.add_task(_task("a", clock), config=_config(tmp_path, resume=True))
        report = (await resumed.arun())["a"]

        assert report["correct"] == 4
        assert not [e for e in clock.events if e[1] == "inferred"]
        meta = _meta(root, "a")
        assert [r.get("stopped_after") for r in meta["runs"]] == ["inference", None]
        assert meta["runs"][-1]["reran"] == ["postprocess", "feedback", "report"]
        baseline = await self._baseline(tmp_path)
        assert meta["identity"] == baseline["identity"]
        assert meta["identity"]["inference"]["requests"]["digest"] is not None

    @pytest.mark.anyio
    async def test_several_stops_are_several_runs(self, tmp_path: Path) -> None:
        root = tmp_path / "staged"
        for stop, resume in (("preprocess", False), ("postprocess", True)):
            runner = _runner(root, stop_after=stop)
            runner.add_task(_task("a"), config=_config(tmp_path, resume=resume))
            await runner.arun()
        resumed = _runner(root)
        resumed.add_task(_task("a"), config=_config(tmp_path, resume=True))
        await resumed.arun()

        runs = _meta(root, "a")["runs"]
        assert [r.get("stopped_after") for r in runs] == [
            "preprocess",
            "postprocess",
            None,
        ]
        assert [r["reran"] for r in runs] == [
            ["data", "preprocess"],
            ["inference", "postprocess"],
            ["feedback", "report"],
        ]
        assert (
            _meta(root, "a")["identity"] == (await self._baseline(tmp_path))["identity"]
        )

    @pytest.mark.anyio
    async def test_stop_after_report_reports_and_records_the_stop(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "r"
        runner = _runner(root, stop_after="report")
        runner.add_task(_task("a"), config=_config(tmp_path))
        report = (await runner.arun())["a"]
        assert report["correct"] == 4
        assert _meta(root, "a")["runs"][0]["stopped_after"] == "report"

    @pytest.mark.anyio
    async def test_a_task_config_stop_after_must_match_the_session(
        self, tmp_path: Path
    ) -> None:
        runner = _runner(tmp_path / "r", stop_after="inference")
        with pytest.raises(ValueError, match="session-wide"):
            runner.add_task(
                _task("a"), config=_config(tmp_path, stop_after="postprocess")
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "barrier", [{"stop_after": "inference"}, {"barriers": ["postprocess"]}]
    )
    async def test_a_sample_out_of_retries_does_not_hold_a_resume_barrier(
        self, tmp_path: Path, barrier: dict
    ) -> None:
        # The retry limit fails Q1 before any stage runs; a barrier still
        # counting it as unfinished would never open.
        root = tmp_path / "r"
        first = _runner(root)
        first.add_task(_failing_task(), config=_config(tmp_path, max_retries=0))
        await first.arun()

        resumed = _runner(root, **barrier)
        resumed.add_task(
            _failing_task(), config=_config(tmp_path, resume=True, max_retries=0)
        )
        with anyio.fail_after(10):
            await resumed.arun()
        assert _meta(root, "a")["runs"][-1]["ended"]

    def test_stop_after_needs_each_stage_recorded(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="record_each_stage"):
            _config(tmp_path, stop_after="inference", record_each_stage=False)


class TestIteratingTasks:
    @pytest.mark.anyio
    @pytest.mark.parametrize("stage", ["inference", "postprocess"])
    async def test_a_barrier_inside_an_iteration_is_refused_before_running(
        self, tmp_path: Path, stage: str
    ) -> None:
        clock = _Clock()
        runner = _runner(tmp_path / "r", barriers=[stage])
        runner.add_task(
            _task("loop", clock), config=_config(tmp_path, max_iterations=3)
        )
        with pytest.raises(ValueError, match="Task 'loop' can iterate"):
            await runner.arun()
        assert clock.events == []

    @pytest.mark.anyio
    async def test_a_barrier_on_feedback_lets_a_task_iterate(
        self, tmp_path: Path
    ) -> None:
        runner = _runner(tmp_path / "r", barriers=["feedback"])
        runner.add_task(
            _task("loop", iterate=True), config=_config(tmp_path, max_iterations=3)
        )
        assert (await runner.arun())["loop"]["total"] == 4

    @pytest.mark.anyio
    async def test_a_sample_asking_to_iterate_past_a_barrier_fails_retriably(
        self, tmp_path: Path
    ) -> None:
        # No max_iterations: the barrier is admitted, the iteration is not.
        root = tmp_path / "r"
        runner = _runner(root, barriers=["inference"])
        runner.add_task(_task("loop", iterate=True), config=_config(tmp_path))
        report = (await runner.arun())["loop"]
        assert (report["correct"], report["total"]) == (0, 4)
        shards = list((root / "loop").rglob("*.jsonl"))
        assert any(BARRIER_ITERATION_REASON in s.read_text() for s in shards)


class TestPlan:
    @pytest.mark.anyio
    async def test_planning_refuses_an_iterating_task_under_the_barrier(
        self, tmp_path: Path
    ) -> None:
        runner = _runner(tmp_path / "r", stop_after="inference")
        runner.add_task(_task("loop"), config=_config(tmp_path, max_iterations=2))
        (plan,) = (await runner.plan_run()).values()
        assert isinstance(plan, ValueError)
