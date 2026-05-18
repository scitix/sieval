"""
Concurrency scaling benchmarks.

Measures throughput vs concurrency limit, CompositeLimiter overhead,
and MultiTaskRunner coordination cost.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path

import anyio
import pytest

from sieval.core.runners.multi_runner import MultiTaskRunner
from sieval.core.runners.runner import TaskRunner
from sieval.core.utils.concurrency import CompositeLimiter
from tests.conftest import (
    BenchmarkTask,
    LatencyMockChatModel,
    PerfTimer,
    make_large_dataset,
    make_perf_config,
    samples_per_second,
)


class TestThroughputVsConcurrency:
    """Measure how throughput scales with concurrency limits."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("concurrency", [1, 4, 16])
    async def test_throughput_vs_concurrency(
        self, tmp_path: Path, concurrency: int
    ) -> None:
        n_samples = 200
        dataset = make_large_dataset(n_samples, payload_size=50)
        model = LatencyMockChatModel(latency_s=0.01, latency_jitter=0.0, output_size=50)
        task = BenchmarkTask(
            dataset=dataset,
            model=model,
            name=f"conc_{concurrency}",
        )
        config = make_perf_config(
            tmp_path / f"conc_{concurrency}",
            concurrency_limit=concurrency,
        )

        runner = TaskRunner(task, config)
        timer = PerfTimer()
        with timer:
            report = await runner.arun()

        assert report is not None
        sps = samples_per_second(n_samples, timer.elapsed)
        # Theoretical max: concurrency / latency (pipelined across 4 stages)
        theoretical_max = concurrency / 0.01
        efficiency = sps / theoretical_max * 100 if theoretical_max > 0 else 0
        print(
            f"PERF: concurrency={concurrency} => "
            f"{sps:.1f} sps, efficiency={efficiency:.1f}%, "
            f"{timer.elapsed:.3f}s"
        )
        # At concurrency=1, theoretical_max = 100 sps; expect at least 30% efficiency.
        # At higher concurrency, expect at least 20% (scheduling overhead grows).
        min_efficiency = 30.0 if concurrency == 1 else 20.0
        assert efficiency > min_efficiency, (
            f"Efficiency too low at concurrency={concurrency}: "
            f"{efficiency:.1f}% (expected >{min_efficiency:.0f}%)"
        )


class TestCompositeLimiterOverhead:
    """Measure per-acquire overhead of CompositeLimiter."""

    @pytest.mark.anyio
    async def test_composite_limiter_overhead(self) -> None:
        """CompositeLimiter with 2 limiters vs 0 limiters."""
        n_iterations = 10000

        # Baseline: no limiters
        limiter_none = CompositeLimiter()
        timer_none = PerfTimer()
        with timer_none:
            for _ in range(n_iterations):
                async with limiter_none:
                    pass

        # With 2 limiters (global + stage)
        global_lim = anyio.CapacityLimiter(1000)
        stage_lim = anyio.CapacityLimiter(500)
        limiter_two = CompositeLimiter(global_lim, stage_lim)
        timer_two = PerfTimer()
        with timer_two:
            for _ in range(n_iterations):
                async with limiter_two:
                    pass

        overhead_per_acquire_ms = (
            (timer_two.elapsed - timer_none.elapsed) / n_iterations * 1000
        )
        print(
            f"PERF: CompositeLimiter overhead={overhead_per_acquire_ms:.4f} ms/acquire "
            f"(none={timer_none.elapsed:.4f}s, two={timer_two.elapsed:.4f}s)"
        )
        assert overhead_per_acquire_ms < 1.0, (
            f"Limiter overhead too high: {overhead_per_acquire_ms:.4f} ms"
        )


class TestMultiTaskRunnerOverhead:
    """Compare MultiTaskRunner vs sequential individual TaskRunners."""

    @pytest.mark.anyio
    async def test_multi_task_runner_overhead(self, tmp_path: Path) -> None:
        n_samples = 200
        n_tasks = 4
        # Use higher latency so actual work dominates over fixed coordination
        # overhead (saver init, task scheduling, etc.)
        latency = 0.05

        # Run individually (sequentially)
        total_individual = 0.0
        for t in range(n_tasks):
            dataset = make_large_dataset(n_samples, payload_size=50)
            model = LatencyMockChatModel(
                latency_s=latency,
                latency_jitter=0.0,
                output_size=50,
            )
            task = BenchmarkTask(dataset=dataset, model=model, name=f"ind_{t}")
            config = make_perf_config(tmp_path / f"ind_{t}", concurrency_limit=32)
            runner = TaskRunner(task, config)
            timer = PerfTimer()
            with timer:
                await runner.arun()
            total_individual += timer.elapsed

        # Run via MultiTaskRunner (parallel)
        multi = MultiTaskRunner(
            result_dir=str(tmp_path / "multi"),
            concurrency_limit=256,
        )
        for t in range(n_tasks):
            dataset = make_large_dataset(n_samples, payload_size=50)
            model = LatencyMockChatModel(
                latency_s=latency,
                latency_jitter=0.0,
                output_size=50,
            )
            task = BenchmarkTask(dataset=dataset, model=model, name=f"multi_{t}")
            cfg = make_perf_config(tmp_path / f"multi_{t}", concurrency_limit=32)
            multi.add_task(task, config=cfg)

        timer_multi = PerfTimer()
        with timer_multi:
            await multi.arun()

        speedup = (
            total_individual / timer_multi.elapsed if timer_multi.elapsed > 0 else 0
        )
        print(
            f"PERF: multi_task total={timer_multi.elapsed:.3f}s "
            f"vs individual_sum={total_individual:.3f}s "
            f"(speedup={speedup:.1f}x)"
        )
        # MultiTaskRunner should not be slower than sequential
        # (allow 20% tolerance for scheduling/coordination overhead)
        assert timer_multi.elapsed < total_individual * 1.2, (
            f"MultiTaskRunner significantly slower than sequential: "
            f"{timer_multi.elapsed:.3f}s vs {total_individual:.3f}s"
        )
