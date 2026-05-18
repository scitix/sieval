"""
Pipeline throughput benchmarks (diagnostic / unit-level).

Complements the acceptance benchmark test (test_performance_acceptance.py) by
isolating
specific subsystem behaviors:
- record_each_stage toggle overhead
- I/O profile impact (long-input workloads)
- Iteration overhead linearity
- Time-to-first-sample latency

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import time
from pathlib import Path
from typing import Any

import pytest

from sieval.core.runners.runner import TaskRunner
from tests.conftest import (
    BenchmarkTask,
    IOProfile,
    LatencyMockChatModel,
    MultiIterBenchmarkTask,
    PerfTimer,
    make_large_dataset,
    make_perf_config,
    require_available_memory_gb,
    samples_per_second,
)


class TestRecordEachStageOverhead:
    """Compare record_each_stage=True vs False overhead."""

    @pytest.mark.anyio
    async def test_record_each_stage_overhead(self, tmp_path: Path) -> None:
        n_samples = 500

        results: dict[str, float] = {}
        for record_each in [True, False]:
            label = "each_stage" if record_each else "final_only"
            dataset = make_large_dataset(n_samples, payload_size=50)
            model = LatencyMockChatModel(
                latency_s=0.01,
                latency_jitter=0.0,
                output_size=50,
            )
            task = BenchmarkTask(dataset=dataset, model=model, name=f"record_{label}")
            config = make_perf_config(
                tmp_path / label,
                concurrency_limit=64,
                record_each_stage=record_each,
            )

            runner = TaskRunner(task, config)
            timer = PerfTimer()
            with timer:
                await runner.arun()
            results[label] = timer.elapsed

        overhead_pct = (
            (results["each_stage"] - results["final_only"])
            / results["final_only"]
            * 100
            if results["final_only"] > 0
            else 0
        )
        print(
            f"PERF: record_each_stage overhead={overhead_pct:.1f}% "
            f"(each={results['each_stage']:.3f}s, "
            f"final={results['final_only']:.3f}s)"
        )
        assert overhead_pct < 200, (
            f"record_each_stage overhead too high: {overhead_pct:.1f}%"
        )


class TestThroughputByIOProfile:
    """Measure throughput for long-input workloads not covered by acceptance test."""

    # Only keep profiles NOT covered by acceptance test scenarios.
    # "long_in_short_out" tests large-input / small-output which is
    # a different stress pattern from the acceptance test's long-output scenario.
    _DIAGNOSTIC_PROFILES = [
        IOProfile(
            "long_in_short_out",
            input_size=4000,
            output_size=50,
            latency_s=0.01,
            latency_jitter=0.0,
        ),
    ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("profile", _DIAGNOSTIC_PROFILES, ids=lambda p: p.name)
    async def test_throughput_by_io_profile(
        self, tmp_path: Path, profile: IOProfile
    ) -> None:
        n_samples = 200
        task, _ = BenchmarkTask.from_profile(
            profile, n_samples=n_samples, name=f"ioprofile_{profile.name}"
        )
        config = make_perf_config(tmp_path / profile.name, concurrency_limit=64)

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        assert report is not None
        sps = samples_per_second(n_samples, timer.elapsed)
        print(
            f"PERF: io_profile={profile.name} n={n_samples} => "
            f"{sps:.1f} sps, {timer.elapsed:.3f}s"
        )
        assert sps > 50, f"IO profile {profile.name} throughput too low: {sps:.1f} sps"


class TestIterationOverheadRatio:
    """Verify iteration loop scales linearly (unique diagnostic check)."""

    @pytest.mark.anyio
    async def test_iteration_overhead_ratio(self, tmp_path: Path) -> None:
        """Compare 1-iteration vs 3-iteration: overhead should scale ~linearly."""
        n_samples = 300
        results: dict[int, float] = {}

        for n_iter in [1, 3]:
            dataset = make_large_dataset(n_samples, payload_size=50)
            model = LatencyMockChatModel(
                latency_s=0.005,
                latency_jitter=0.0,
                output_size=50,
            )
            task = MultiIterBenchmarkTask(
                dataset=dataset,
                model=model,
                name=f"iter_ratio_{n_iter}",
                finalize_after=n_iter,
            )
            config = make_perf_config(
                tmp_path / f"ratio_{n_iter}",
                concurrency_limit=64,
                max_iterations=n_iter + 1,
            )

            runner = TaskRunner(task, config)
            timer = PerfTimer()
            with timer:
                await runner.arun()
            results[n_iter] = timer.elapsed

        ratio = results[3] / results[1] if results[1] > 0 else 0
        print(
            f"PERF: iteration_overhead_ratio "
            f"1_iter={results[1]:.3f}s, 3_iter={results[3]:.3f}s, "
            f"ratio={ratio:.1f}x (ideal ~3.0x)"
        )
        # 3 iterations should take roughly 3x the time of 1 iteration,
        # but framework overhead means it should be less than 5x
        assert ratio < 5.0, (
            f"Iteration overhead non-linear: {ratio:.1f}x (expected <5.0x)"
        )


class TestTimeToFirstSample:
    """Measure time from arun() start to first preprocess() call.

    With eager init, the runner must enumerate the entire dataset before
    any sample begins execution.  With lazy init, the first sample starts
    almost immediately after hydrate + manifest sync.
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize("n_samples", [1000, 5000, 20000])
    async def test_time_to_first_preprocess(
        self, tmp_path: Path, n_samples: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Time from runner.arun() entry to first preprocess() invocation."""
        dataset = make_large_dataset(n_samples, payload_size=50)
        model = LatencyMockChatModel(latency_s=0.0, latency_jitter=0.0, output_size=50)
        task = BenchmarkTask(dataset=dataset, model=model, name=f"ttfs_{n_samples}")
        config = make_perf_config(tmp_path / f"ttfs_{n_samples}", concurrency_limit=64)
        runner = TaskRunner(task, config)

        first_preprocess_ts: float | None = None
        original_preprocess = task.preprocess

        async def _tracking_preprocess(raw: Any, ctx: Any) -> str:
            nonlocal first_preprocess_ts
            if first_preprocess_ts is None:
                first_preprocess_ts = time.perf_counter()
            return await original_preprocess(raw, ctx)

        monkeypatch.setattr(task, "preprocess", _tracking_preprocess)

        start_ts = time.perf_counter()
        await runner.arun()

        assert first_preprocess_ts is not None
        ttfs = first_preprocess_ts - start_ts
        total = time.perf_counter() - start_ts
        print(
            f"PERF: time_to_first_sample n={n_samples} => "
            f"ttfs={ttfs * 1000:.1f}ms, total={total:.3f}s"
        )
        # First sample should start within 500ms regardless of dataset size
        assert ttfs < 0.5, (
            f"Time to first sample too slow: {ttfs * 1000:.1f}ms "
            f"(limit 500ms) for n={n_samples}"
        )


# Stress tests
# ===================================================================
@pytest.mark.stress
class TestThroughputStress:
    @pytest.mark.anyio
    async def test_throughput_10k(self, tmp_path: Path) -> None:
        """Stress: 10,000 samples through the full pipeline."""
        require_available_memory_gb(4.0)
        n_samples = 10000
        dataset = make_large_dataset(n_samples, payload_size=50)
        model = LatencyMockChatModel(
            latency_s=0.005,
            latency_jitter=0.0,
            output_size=50,
        )
        task = BenchmarkTask(dataset=dataset, model=model, name="stress_10k")
        config = make_perf_config(tmp_path, concurrency_limit=128)

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        assert report is not None
        sps = samples_per_second(n_samples, timer.elapsed)
        print(f"STRESS: 10k throughput => {sps:.1f} sps, {timer.elapsed:.1f}s")
