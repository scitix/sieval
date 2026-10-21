"""
Resume speed benchmarks.

Measures time to resume from various completion percentages and
the fast-path cached report return.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path

import pytest

from sieval.core.runners.runner import TaskRunner

# Integration mock for zero-latency completion of remaining samples
from tests.conftest import (
    BenchmarkTask,
    MockCountingChatModel,
    PerfTimer,
    make_large_dataset,
    make_perf_config,
    require_available_memory_gb,
    write_completed_samples,
)


# ===================================================================
# Resume benchmarks
# ===================================================================
class TestResumeByCompletion:
    """Measure resume time at various completion percentages."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("completion_pct", [50, 99])
    async def test_resume_by_completion_pct(
        self, tmp_path: Path, completion_pct: int
    ) -> None:
        n_total = 1000
        n_completed = int(n_total * completion_pct / 100)

        result_dir = tmp_path / f"resume_{completion_pct}"

        # Phase 1: Write completed samples
        await write_completed_samples(
            result_dir, n_completed, name=f"resume_{completion_pct}"
        )

        # Phase 2: Resume and complete remaining
        dataset = make_large_dataset(n_total, payload_size=50)
        model = MockCountingChatModel(default_answer="A0")  # Zero latency for speed
        task = BenchmarkTask(
            dataset=dataset, model=model, name=f"resume_{completion_pct}"
        )
        config = make_perf_config(
            tmp_path / f"cfg_{completion_pct}",
            result_dir=str(result_dir),
            resume=True,
            concurrency_limit=64,
        )

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        n_remaining = n_total - n_completed
        print(
            f"PERF: resume {completion_pct}% "
            f"(remaining={n_remaining}) => {timer.elapsed:.3f}s"
        )
        assert report is not None
        # The resume keeps the seeded samples: only the rest reach the model
        assert model.call_count == n_remaining
        # Resume should complete in reasonable time; higher completion % = faster
        max_time = max(1.0, n_remaining / 100)  # ~10ms per remaining sample
        assert timer.elapsed < max_time, (
            f"Resume {completion_pct}% too slow: "
            f"{timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )


class TestCachedReportFastResume:
    """Measure fast-path: cached report.json return."""

    @pytest.mark.anyio
    async def test_cached_report_fast_resume(self, tmp_path: Path) -> None:
        """When all samples are done and report.json exists, return instantly."""
        n_total = 2000
        result_dir = tmp_path / "fast_resume"

        # A finished run: its report.json is what the fast path returns
        await write_completed_samples(result_dir, n_total, name="fast_resume")

        # Resume — should hit the fast path
        dataset = make_large_dataset(n_total, payload_size=50)
        model = MockCountingChatModel(default_answer="A0")
        task = BenchmarkTask(dataset=dataset, model=model, name="fast_resume")
        config = make_perf_config(
            tmp_path / "cfg_fast",
            result_dir=str(result_dir),
            resume=True,
        )

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        print(f"PERF: cached_report resume n={n_total} => {timer.elapsed:.4f}s")
        assert report is not None
        assert model.call_count == 0
        assert timer.elapsed < 3.0, f"Fast resume too slow: {timer.elapsed:.3f}s"


# ===================================================================
# Stress tests
# ===================================================================
@pytest.mark.stress
class TestResumeStress:
    @pytest.mark.anyio
    async def test_resume_large_checkpoint(self, tmp_path: Path) -> None:
        """Stress: resume with 50,000 completed samples on disk."""
        require_available_memory_gb(8.0)
        n_completed = 50000
        result_dir = tmp_path / "stress_resume"

        await write_completed_samples(result_dir, n_completed, name="stress_resume")

        dataset = make_large_dataset(n_completed, payload_size=50)
        model = MockCountingChatModel(default_answer="A0")
        task = BenchmarkTask(dataset=dataset, model=model, name="stress_resume")
        config = make_perf_config(
            tmp_path / "cfg_stress",
            result_dir=str(result_dir),
            resume=True,
        )

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        print(f"STRESS: resume_50k => {timer.elapsed:.2f}s")
        assert report is not None
