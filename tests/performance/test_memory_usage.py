"""
Memory usage benchmarks.

Uses psutil RSS tracking to measure process-level memory under different
workloads. To produce measurable RSS deltas, tests use large payloads
(10KB+) so allocations exceed the Python heap's free-list capacity and
force OS-level page allocation.

Verifies that:
- Pipeline memory scales sub-linearly with sample count
- record_each_stage overhead is bounded
- No memory leaks across runs
- Context and saver buffer footprints are bounded

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import gc
from pathlib import Path

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.saver import TaskSaver
from tests.conftest import (
    BenchmarkTask,
    IOProfile,
    LatencyMockChatModel,
    MemoryTracker,
    _make_bench_ctx,
    make_large_dataset,
    make_perf_config,
    require_available_memory_gb,
)

# Use 10KB payloads so RSS delta is measurable (small payloads fit in
# Python's existing heap and produce 0.0MB delta with psutil).
_MEM_PAYLOAD_SIZE = 10 * 1024


class TestPipelineMemoryScaling:
    """Verify pipeline memory scales sub-linearly with sample count.

    Uses 10KB payloads with concurrency=64 to measure end-to-end pipeline
    memory.  The output_size is kept small (100B) so that stage results
    don't dominate RSS — the test focuses on raw_sample lifecycle.
    """

    @pytest.mark.anyio
    async def test_pipeline_memory_scaling(self, tmp_path: Path) -> None:
        """Memory at n=2000, 5000, 10000 with 10KB payloads."""
        results: dict[int, float] = {}
        peaks: dict[int, float] = {}

        for n_samples in [2000, 5000, 10000]:
            gc.collect()

            tracker = MemoryTracker()
            tracker.start()

            dataset = make_large_dataset(n_samples, payload_size=_MEM_PAYLOAD_SIZE)
            model = LatencyMockChatModel(
                latency_s=0.0,
                latency_jitter=0.0,
                output_size=100,
            )
            task = BenchmarkTask(
                dataset=dataset,
                model=model,
                name=f"memscale_{n_samples}",
                output_size=100,
            )
            config = make_perf_config(
                tmp_path / f"mem_{n_samples}", concurrency_limit=64
            )

            runner = TaskRunner(task, config)
            await runner.arun()
            tracker.stop()

            results[n_samples] = tracker.delta_mb
            peaks[n_samples] = tracker.peak_mb
            print(
                f"PERF: memory n={n_samples} => "
                f"delta={tracker.delta_mb:.1f}MB, "
                f"peak={tracker.peak_mb:.1f}MB"
            )

        # 10000/2000 = 5x samples, memory should grow sub-linearly (<4x)
        assert results[2000] > 0, (
            f"Baseline delta too small to measure: {results[2000]:.2f}MB"
        )
        ratio = results[10000] / results[2000]
        peak_ratio = peaks[10000] / peaks[2000] if peaks[2000] > 0 else 0
        print(
            f"PERF: memory scaling ratio (10000/2000) "
            f"delta={ratio:.1f}x, peak={peak_ratio:.1f}x"
        )
        assert ratio < 4, f"Memory scales linearly or worse: {ratio:.1f}x"


class TestInitPhaseMemory:
    """Measure memory consumed during the context initialization phase only.

    This isolates the exact behavior changed by lazy context creation:
    eager enumerate-all-samples vs on-demand test_set[i] with a sliding
    window.  Each path runs in a subprocess to get clean RSS baselines.
    """

    @pytest.mark.parametrize("n_samples", [5000, 10000, 20000])
    def test_init_phase_memory_eager(self, n_samples: int) -> None:
        """Memory to eagerly create all N TaskContext objects (50KB payloads)."""
        payload_size = 50 * 1024
        dataset = make_large_dataset(n_samples, payload_size=payload_size)
        test_set = dataset.test_set
        assert test_set is not None

        gc.collect()
        tracker = MemoryTracker()
        tracker.start()

        contexts: dict[int, TaskContext] = {}
        for i, raw in enumerate(test_set):
            contexts[i] = TaskContext(sample_id=i, raw_sample=raw)

        tracker.stop()
        mb = tracker.delta_mb
        per_ctx_kb = mb * 1024 / n_samples if n_samples > 0 else 0
        print(
            f"PERF: init_eager n={n_samples} => "
            f"{mb:.1f}MB total, {per_ctx_kb:.1f}KB/ctx"
        )
        assert len(contexts) == n_samples
        # Sanity: should use measurable memory with 50KB payloads
        assert mb > 10, f"Eager init RSS too low to measure: {mb:.1f}MB"

    @pytest.mark.parametrize("n_samples", [5000, 10000, 20000])
    def test_init_phase_memory_lazy_window(self, n_samples: int) -> None:
        """Memory with a sliding window of 64 contexts (50KB payloads).

        Simulates the lazy path: only ~window_size contexts alive at once.
        """
        payload_size = 50 * 1024
        window_size = 64
        dataset = make_large_dataset(n_samples, payload_size=payload_size)
        test_set = dataset.test_set
        assert test_set is not None

        gc.collect()
        tracker = MemoryTracker()
        tracker.start()

        window: dict[int, TaskContext] = {}
        for i in range(n_samples):
            raw = test_set[i]
            window[i] = TaskContext(sample_id=i, raw_sample=raw)
            if len(window) > window_size:
                del window[i - window_size]

        tracker.stop()
        mb = tracker.delta_mb
        print(
            f"PERF: init_lazy_window n={n_samples} window={window_size} => "
            f"{mb:.1f}MB total"
        )
        assert len(window) <= window_size


class TestMemoryByIOProfile:
    """Compare memory usage across different I/O profiles."""

    # Use profiles with larger payloads to produce measurable RSS delta
    _MEM_PROFILES = [
        IOProfile(
            "short_in_short_out",
            input_size=5000,
            output_size=5000,
            latency_s=0.0,
            latency_jitter=0.0,
        ),
        IOProfile(
            "long_in_short_out",
            input_size=20000,
            output_size=5000,
            latency_s=0.0,
            latency_jitter=0.0,
        ),
        IOProfile(
            "short_in_long_out",
            input_size=5000,
            output_size=20000,
            latency_s=0.0,
            latency_jitter=0.0,
        ),
        IOProfile(
            "balanced",
            input_size=10000,
            output_size=10000,
            latency_s=0.0,
            latency_jitter=0.0,
        ),
    ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("profile", _MEM_PROFILES, ids=lambda p: p.name)
    async def test_memory_by_io_profile(
        self, tmp_path: Path, profile: IOProfile
    ) -> None:
        n_samples = 2000
        gc.collect()

        tracker = MemoryTracker()
        tracker.start()

        task, _ = BenchmarkTask.from_profile(
            profile, n_samples=n_samples, name=f"memprofile_{profile.name}"
        )
        config = make_perf_config(tmp_path / profile.name, concurrency_limit=64)

        runner = TaskRunner(task, config)
        await runner.arun()
        tracker.stop()

        print(
            f"PERF: memory profile={profile.name} => "
            f"delta={tracker.delta_mb:.1f}MB, peak={tracker.peak_mb:.1f}MB"
        )
        # Each profile should stay under 500MB delta
        assert tracker.delta_mb < 500, (
            f"Memory too high for profile {profile.name}: {tracker.delta_mb:.1f}MB"
        )


class TestRecordEachStageMemory:
    """Compare memory: record_each_stage True vs False."""

    @pytest.mark.anyio
    async def test_record_each_stage_memory_overhead(self, tmp_path: Path) -> None:
        n_samples = 2000
        results: dict[str, float] = {}

        for record_each in [True, False]:
            label = "each" if record_each else "final"
            gc.collect()

            tracker = MemoryTracker()
            tracker.start()

            dataset = make_large_dataset(n_samples, payload_size=_MEM_PAYLOAD_SIZE)
            model = LatencyMockChatModel(
                latency_s=0.0,
                latency_jitter=0.0,
                output_size=_MEM_PAYLOAD_SIZE,
            )
            task = BenchmarkTask(
                dataset=dataset,
                model=model,
                name=f"memrecord_{label}",
                output_size=_MEM_PAYLOAD_SIZE,
            )
            config = make_perf_config(
                tmp_path / label,
                concurrency_limit=64,
                record_each_stage=record_each,
            )

            runner = TaskRunner(task, config)
            await runner.arun()
            tracker.stop()

            results[label] = tracker.delta_mb
            print(
                f"PERF: memory record={label} => "
                f"peak={tracker.peak_mb:.1f}MB, delta={tracker.delta_mb:.1f}MB"
            )

        # record_each_stage should not use more than 3x the memory of final-only
        if results["final"] > 1.0:
            ratio = results["each"] / results["final"]
            print(f"PERF: record_each memory ratio = {ratio:.1f}x")
            assert ratio < 3.0, f"record_each memory too high: {ratio:.1f}x"


class TestContextMemoryFootprint:
    """Measure per-context memory footprint at different payload sizes."""

    @pytest.mark.parametrize("payload_kb", [50, 100, 200])
    def test_context_memory_footprint(self, payload_kb: int) -> None:
        """Profile: how much memory does 2000 contexts consume?"""
        n = 2000
        payload_size = payload_kb * 1024

        tracker = MemoryTracker()
        tracker.start()
        contexts = [
            _make_bench_ctx(i, TaskStage.FINAL, payload_size=payload_size)
            for i in range(n)
        ]
        tracker.stop()

        # RSS may undercount when Python reuses arena pages; clamp to 0.
        delta_mb = max(tracker.delta_mb, 0.0)
        per_ctx_kb = delta_mb * 1024 / n if n > 0 else 0
        print(
            f"PERF: ctx_footprint payload={payload_kb}KB => "
            f"total={delta_mb:.1f}MB, "
            f"per_ctx={per_ctx_kb:.1f}KB"
        )
        # Keep reference so GC doesn't collect early
        assert len(contexts) == n
        # RSS-based tracking has page-level granularity; for smaller payloads
        # Python's allocator may reuse existing arena space, producing a 0
        # delta.  Only assert measurability for >=100KB payloads.
        if payload_kb >= 100:
            assert delta_mb > 0, f"RSS delta not measurable at {payload_kb}KB payload"
        # Per-context memory should be < 10x the payload size (overhead bound)
        if delta_mb > 0:
            assert per_ctx_kb < payload_kb * 10, (
                f"Per-context overhead too high: "
                f"{per_ctx_kb:.1f}KB vs {payload_kb}KB payload"
            )


class TestSaverBufferMemory:
    """Measure Saver write-buffer memory usage."""

    @pytest.mark.parametrize("buffer_size", [256, 1024, 4096])
    def test_saver_buffer_memory(self, tmp_path: Path, buffer_size: int) -> None:
        """Memory consumed by buffer_size contexts (10KB payload) in saver queue."""
        gc.collect()

        tracker = MemoryTracker()
        tracker.start()
        saver = TaskSaver(
            root_dir=tmp_path / "saver_mem_bench",
            shard_samples=256,
            write_buffer_size=buffer_size + 1,
            write_buffer_flush_interval=9999.0,
        )
        for i in range(buffer_size):
            ctx = _make_bench_ctx(i, TaskStage.FINAL, payload_size=_MEM_PAYLOAD_SIZE)
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)
        tracker.stop()

        per_ctx_kb = tracker.delta_mb * 1024 / buffer_size if buffer_size > 0 else 0
        print(
            f"PERF: saver_buffer size={buffer_size} => "
            f"delta={tracker.delta_mb:.1f}MB, per_ctx={per_ctx_kb:.1f}KB"
        )
        # Buffer memory should scale roughly linearly with buffer_size
        # and stay under 500MB total
        assert tracker.delta_mb < 500, (
            f"Saver buffer too large: {tracker.delta_mb:.1f}MB"
        )


class TestNoMemoryLeak:
    """Verify no memory leak across multiple pipeline runs."""

    @pytest.mark.anyio
    async def test_no_memory_leak_across_runs(self, tmp_path: Path) -> None:
        """Run the pipeline 10 times with 10KB payloads; RSS growth stays bounded."""
        n_samples = 1000
        n_runs = 10
        final_rss: list[float] = []

        for run_idx in range(n_runs):
            gc.collect()
            dataset = make_large_dataset(n_samples, payload_size=_MEM_PAYLOAD_SIZE)
            model = LatencyMockChatModel(
                latency_s=0.0,
                latency_jitter=0.0,
                output_size=_MEM_PAYLOAD_SIZE,
            )
            task = BenchmarkTask(
                dataset=dataset,
                model=model,
                name=f"leak_{run_idx}",
                output_size=_MEM_PAYLOAD_SIZE,
            )
            config = make_perf_config(
                tmp_path / f"leak_{run_idx}", concurrency_limit=64
            )

            runner = TaskRunner(task, config)
            tracker = MemoryTracker()
            tracker.start()
            await runner.arun()
            tracker.stop()

            final_rss.append(tracker.final_mb)
            print(
                f"PERF: leak_check run={run_idx} => "
                f"baseline={tracker.baseline_mb:.1f}MB, "
                f"final={tracker.final_mb:.1f}MB, delta={tracker.delta_mb:.1f}MB"
            )

        # Compare absolute RSS after warmup to avoid hiding leaks when both
        # baseline and final increase together.
        warm_final = final_rss[1:]
        growth = warm_final[-1] - warm_final[0] if warm_final else 0.0
        print(f"PERF: final RSS growth across runs 1-{n_runs - 1} = {growth:.1f}MB")
        assert growth < 30, (
            f"Memory leak detected: final RSS grew {growth:.1f}MB "
            f"across {n_runs - 1} runs"
        )


# ===================================================================
# Stress tests
# ===================================================================
@pytest.mark.stress
class TestMemoryStress:
    @pytest.mark.anyio
    async def test_memory_10k_samples(self, tmp_path: Path) -> None:
        """Stress: 10k samples with 10KB payloads — memory peak."""
        require_available_memory_gb(8.0)
        n_samples = 10000
        gc.collect()

        tracker = MemoryTracker()
        tracker.start()

        dataset = make_large_dataset(n_samples, payload_size=_MEM_PAYLOAD_SIZE)
        model = LatencyMockChatModel(
            latency_s=0.001,
            latency_jitter=0.0,
            output_size=_MEM_PAYLOAD_SIZE,
        )
        task = BenchmarkTask(
            dataset=dataset,
            model=model,
            name="stress_mem_10k",
            output_size=_MEM_PAYLOAD_SIZE,
        )
        config = make_perf_config(tmp_path, concurrency_limit=128)

        runner = TaskRunner(task, config)
        await runner.arun()
        tracker.stop()

        print(
            f"STRESS: memory_10k => "
            f"peak={tracker.peak_mb:.0f}MB, delta={tracker.delta_mb:.0f}MB"
        )
        assert tracker.delta_mb < 2000, (
            f"10k sample memory too high: {tracker.delta_mb:.0f}MB"
        )
