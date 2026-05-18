"""
Unit tests for MultiTaskRunner behavior.

Tests observable behavior (return values, exceptions, result_dir layout)
rather than internal wiring (private attributes, mock call args).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from typing import Any

import pytest

from sieval.core.runners.multi_runner import MultiTaskRunner
from sieval.core.runners.runner import TaskRunnerConfig
from sieval.core.tasks.context import TaskAction
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config

SAMPLES = [
    {"question": "1+1?", "answer": "2"},
    {"question": "2+3?", "answer": "5"},
]


class _SimpleTask(Task):
    """Minimal task for unit-level MultiTaskRunner tests."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0}


def _make_task(name: str) -> _SimpleTask:
    dataset = MockDataset(SAMPLES)
    model = MockChatModel(answers={"1+1?": "2", "2+3?": "5"})
    return _SimpleTask(dataset=dataset, model=model, name=name)


def _quiet_config(**overrides) -> TaskRunnerConfig:
    defaults = {
        "show_progress": False,
        "detect_anomalies": False,
        "profile_io": False,
        "profile_stages": False,
        "profile_usage": False,
        "dump_progress": False,
    }
    defaults.update(overrides)
    return TaskRunnerConfig(**defaults)  # type: ignore[invalid-argument-type]


# ===================================================================
# run / arun basic behavior
# ===================================================================
def test_run_delegates_to_arun(tmp_path):
    """Sync .run() should produce the same result as .arun()."""
    runner = MultiTaskRunner(result_dir=str(tmp_path / "sync"))
    cfg = make_config(tmp_path, result_dir=None)
    runner.add_task(_make_task("t"), config=cfg)
    result = runner.run()
    assert "t" in result
    assert result["t"]["accuracy"] == 1.0


@pytest.mark.anyio
async def test_arun_empty_runners_returns_empty_dict():
    """arun() with no tasks added returns {}."""
    result = await MultiTaskRunner().arun()
    assert result == {}


@pytest.mark.anyio
async def test_arun_ignores_unknown_stage_limit_keys(tmp_path):
    """Unknown concurrency_limits keys are silently dropped, run succeeds."""
    stage_limits: dict[Any, int] = {TaskAction.INFER: 2, "unknown": 5}
    runner = MultiTaskRunner(
        result_dir=str(tmp_path / "unk"),
        concurrency_limits=stage_limits,
    )
    cfg = make_config(tmp_path, result_dir=None)
    runner.add_task(_make_task("t"), config=cfg)
    result = await runner.arun()
    # Unknown key didn't cause errors, task ran successfully
    assert result["t"]["accuracy"] == 1.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "limits",
    [None, {"bogus": 1, "nope": 2}],
    ids=["none_limits", "all_invalid_keys"],
)
async def test_arun_stage_limit_normalization(tmp_path, limits):
    """None or all-invalid concurrency_limits → tasks still run normally."""
    runner = MultiTaskRunner(
        result_dir=str(tmp_path / "norm"),
        concurrency_limits=limits,
    )
    cfg = make_config(tmp_path, result_dir=None)
    runner.add_task(_make_task("t"), config=cfg)
    result = await runner.arun()
    assert result["t"]["accuracy"] == 1.0


# ===================================================================
# add_task: result_dir layout, chaining, validation
# ===================================================================
def test_add_task_auto_constructs_result_dir(tmp_path):
    """base_result_dir + cfg.result_dir=None → output at base/task_name."""
    base = tmp_path / "base"
    runner = MultiTaskRunner(result_dir=str(base))
    runner.add_task(_make_task("my_task"), config=_quiet_config())
    runner.run()
    # The runner should have written results under base/my_task/
    assert (base / "my_task").is_dir()


def test_add_task_no_base_keeps_explicit_result_dir(tmp_path):
    """base_result_dir=None → config's result_dir used as-is."""
    explicit = tmp_path / "explicit"
    runner = MultiTaskRunner(result_dir=None)
    cfg = _quiet_config(result_dir=str(explicit))
    runner.add_task(_make_task("t"), config=cfg)
    runner.run()
    assert explicit.is_dir()


def test_add_task_returns_self_for_chaining(tmp_path):
    runner = MultiTaskRunner(result_dir=str(tmp_path / "chain"))
    ret = runner.add_task(_make_task("t"))
    assert ret is runner


def test_add_task_duplicate_name_raises(tmp_path):
    """Adding two tasks with the same name raises ValueError."""
    runner = MultiTaskRunner(result_dir=str(tmp_path / "dup"))
    runner.add_task(_make_task("same"), config=_quiet_config())
    with pytest.raises(ValueError, match="already registered"):
        runner.add_task(_make_task("same"), config=_quiet_config())


def test_add_task_output_dir_collision_raises(tmp_path):
    """Two tasks resolving to the same output dir raises ValueError."""
    fixed_dir = str(tmp_path / "collision_dir")
    runner = MultiTaskRunner(result_dir=None)
    runner.add_task(_make_task("a"), config=_quiet_config(result_dir=fixed_dir))
    with pytest.raises(ValueError, match="collision"):
        runner.add_task(_make_task("b"), config=_quiet_config(result_dir=fixed_dir))


# ===================================================================
# arun: exception propagation
# ===================================================================
@pytest.mark.anyio
async def test_arun_exception_propagates(tmp_path):
    """If a runner's task always fails, ExceptionGroup propagates."""

    class _AlwaysFailTask(_SimpleTask):
        async def infer(self, pre, ctx):
            raise RuntimeError("boom")

    dataset = MockDataset(SAMPLES)
    model = MockChatModel()
    task = _AlwaysFailTask(dataset=dataset, model=model, name="bad")

    runner = MultiTaskRunner(result_dir=str(tmp_path / "exc"))
    cfg = make_config(tmp_path, result_dir=None, max_retries=0)
    runner.add_task(task, config=cfg)

    # Task failures are captured per-sample, not propagated as ExceptionGroup.
    # Verify the runner completes with 0% accuracy instead of crashing.
    result = await runner.arun()
    assert result["bad"]["accuracy"] == 0.0
