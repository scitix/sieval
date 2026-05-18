"""
I/O overhead benchmarks for TaskSaver and TaskLoader.

Measures flush performance, manifest write scaling, hydration speed,
compensation scan, and cross-stage dependency loading overhead.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path

import pytest

from sieval.core.tasks.context import TaskContext, TaskManifest, TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.saver import TaskSaver
from tests.conftest import (
    BenchmarkTask,
    LatencyMockChatModel,
    PerfTimer,
    _make_bench_ctx,
    make_large_dataset,
    require_available_memory_gb,
    samples_per_second,
)


# ===================================================================
# Helpers
# ===================================================================
async def _write_contexts_batch(
    root: Path,
    contexts: list[TaskContext],
    shard_samples: int = 256,
) -> None:
    """Write a batch of contexts to disk via TaskSaver."""
    saver = TaskSaver(
        root_dir=root,
        shard_samples=shard_samples,
        shard_write_concurrency=8,
        write_buffer_size=len(contexts) + 1,
        write_buffer_flush_interval=9999.0,
    )
    for ctx in contexts:
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)
    await saver.flush()


def _make_mock_task(n_samples: int = 10) -> BenchmarkTask:
    """Create a minimal BenchmarkTask for loader tests."""
    dataset = make_large_dataset(n_samples, payload_size=50)
    model = LatencyMockChatModel(latency_s=0, latency_jitter=0.0, output_size=50)
    return BenchmarkTask(dataset=dataset, model=model, name="io_bench")


# ===================================================================
# Saver flush benchmarks
# ===================================================================
class TestSaverFlushPerformance:
    """Measure TaskSaver.flush() at scale."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("n_contexts", [100, 1000, 5000])
    async def test_flush_batch_performance(
        self, tmp_path: Path, n_contexts: int
    ) -> None:
        """Time to flush N contexts to disk in a single batch."""
        saver = TaskSaver(
            root_dir=tmp_path / "flush_bench",
            shard_samples=256,
            shard_write_concurrency=8,
            write_buffer_size=n_contexts + 1,
            write_buffer_flush_interval=9999.0,
        )

        for i in range(n_contexts):
            ctx = _make_bench_ctx(i, TaskStage.FINAL)
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)

        timer = PerfTimer()
        with timer:
            await saver.flush()

        cps = samples_per_second(n_contexts, timer.elapsed)
        print(f"PERF: flush n={n_contexts} => {timer.elapsed:.4f}s ({cps:.0f} ctx/s)")
        # Allow up to ~0.2s per 1k contexts with a small fixed floor.
        max_time = max(0.3, n_contexts / 1000 * 0.2)
        assert timer.elapsed < max_time, (
            f"Flush too slow: {timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )

    @pytest.mark.anyio
    async def test_flush_large_payloads(self, tmp_path: Path) -> None:
        """Flush 500 contexts with ~10KB payload each."""
        n = 500
        payload_size = 10 * 1024

        saver = TaskSaver(
            root_dir=tmp_path / "large_payload",
            shard_samples=256,
            shard_write_concurrency=8,
            write_buffer_size=n + 1,
            write_buffer_flush_interval=9999.0,
        )

        for i in range(n):
            ctx = _make_bench_ctx(i, TaskStage.FINAL, payload_size=payload_size)
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)

        timer = PerfTimer()
        with timer:
            await saver.flush()

        print(f"PERF: large_payload flush n={n} => {timer.elapsed:.4f}s")
        assert timer.elapsed < 10.0, (
            f"Large payload flush too slow: {timer.elapsed:.3f}s"
        )


class TestManifestWriteScaling:
    """Measure manifest write time vs manifest size."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("n_entries", [1000, 5000, 10000])
    async def test_manifest_write_scaling(self, tmp_path: Path, n_entries: int) -> None:
        """Manifest rewrite time with N existing entries."""
        saver = TaskSaver(
            root_dir=tmp_path / f"manifest_{n_entries}",
            shard_samples=1024,
        )

        # Pre-populate manifest
        manifest: dict[int, TaskManifest] = {}
        for i in range(n_entries):
            manifest[i] = TaskManifest(
                sample_id=i,
                stage=TaskStage.FINAL.value,
                iteration=0,
                final=True,
                failed=False,
            )

        saver.sync_manifest(manifest)  # type: ignore[arg-type]

        # Add one more item to trigger write
        ctx = _make_bench_ctx(n_entries, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)

        timer = PerfTimer()
        with timer:
            await saver.flush()

        print(f"PERF: manifest_write n={n_entries} => {timer.elapsed:.4f}s")
        # Allow up to ~0.1s per 1k manifest entries with a small fixed floor.
        max_time = max(0.3, n_entries / 1000 * 0.1)
        assert timer.elapsed < max_time, (
            f"Manifest write too slow: {timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )


# ===================================================================
# Loader hydration benchmarks
# ===================================================================
class TestLoaderHydrationPerformance:
    """Measure TaskLoader hydration speed."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("n_samples", [2000])
    async def test_hydration_speed(self, tmp_path: Path, n_samples: int) -> None:
        """Time to hydrate N samples from disk."""
        root = tmp_path / "hydrate_bench"
        contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n_samples)]
        await _write_contexts_batch(root, contexts, shard_samples=256)

        task = _make_mock_task(n_samples)
        loader = TaskLoader(
            task=task,
            root_dir=root,
            shard_read_concurrency=8,
        )

        timer = PerfTimer()
        with timer:
            loaded = await loader.load_initial_state()

        sps = samples_per_second(n_samples, timer.elapsed)
        print(
            f"PERF: hydrate n={n_samples} => {timer.elapsed:.4f}s ({sps:.0f} samples/s)"
        )
        assert len(loaded) == n_samples
        # Scale threshold: ~0.1s per 1000 samples
        max_time = max(0.5, n_samples / 1000 * 0.1)
        assert timer.elapsed < max_time, (
            f"Hydration too slow: {timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )

    @pytest.mark.anyio
    async def test_compensation_scan_speed(self, tmp_path: Path) -> None:
        """Time for compensation scan (rebuild manifest from idx files)."""
        root = tmp_path / "compensate_bench"
        n_samples = 2000
        contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n_samples)]
        await _write_contexts_batch(root, contexts, shard_samples=256)

        # Delete manifest to force compensation scan
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

        task = _make_mock_task(n_samples)
        loader = TaskLoader(
            task=task,
            root_dir=root,
            shard_read_concurrency=8,
        )

        timer = PerfTimer()
        with timer:
            loaded = await loader.load_initial_state()

        print(f"PERF: compensation_scan n={n_samples} => {timer.elapsed:.4f}s")
        assert len(loaded) == n_samples
        assert timer.elapsed < 5.0, f"Compensation scan too slow: {timer.elapsed:.3f}s"

    @pytest.mark.anyio
    async def test_dependency_loading_overhead(self, tmp_path: Path) -> None:
        """Measure overhead of cross-stage dependency loading.

        Writes snapshots at each stage, then hydrates with all dependencies.
        Compares against single-stage-only hydration.
        """
        n_samples = 500
        stages = [
            TaskStage.PREPROCESSED,
            TaskStage.INFERRED,
            TaskStage.POSTPROCESSED,
            TaskStage.FINAL,
        ]

        # Write snapshots for each stage (simulating record_each_stage=True)
        root_dep = tmp_path / "dep_bench"
        for stage in stages:
            contexts = [_make_bench_ctx(i, stage) for i in range(n_samples)]
            # Write each stage to its own subdirectory
            saver = TaskSaver(
                root_dir=root_dep,
                shard_samples=256,
                shard_write_concurrency=8,
                write_buffer_size=n_samples + 1,
                write_buffer_flush_interval=9999.0,
            )
            for ctx in contexts:
                # Make snapshot for non-terminal stages
                if stage != TaskStage.FINAL:
                    ctx = ctx.make_snapshot()
                saver._update_manifest_entry(ctx)
                saver._stage_queue.append(ctx)
            await saver.flush()

        # Hydrate with dependencies
        task_dep = _make_mock_task(n_samples)
        loader_dep = TaskLoader(
            task=task_dep,
            root_dir=root_dep,
            shard_read_concurrency=8,
        )

        timer_dep = PerfTimer()
        with timer_dep:
            _loaded_dep = await loader_dep.load_initial_state()

        # Compare: hydrate without dependency stages (final only)
        root_nodep = tmp_path / "nodep_bench"
        final_contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n_samples)]
        await _write_contexts_batch(root_nodep, final_contexts, shard_samples=256)

        task_nodep = _make_mock_task(n_samples)
        loader_nodep = TaskLoader(
            task=task_nodep,
            root_dir=root_nodep,
            shard_read_concurrency=8,
        )

        timer_nodep = PerfTimer()
        with timer_nodep:
            _loaded_nodep = await loader_nodep.load_initial_state()

        overhead_pct = (
            (timer_dep.elapsed - timer_nodep.elapsed) / timer_nodep.elapsed * 100
            if timer_nodep.elapsed > 0
            else 0
        )
        print(
            f"PERF: dep_loading overhead={overhead_pct:.1f}% "
            f"(dep={timer_dep.elapsed:.4f}s, nodep={timer_nodep.elapsed:.4f}s)"
        )
        # Dependency loading adds cross-stage reads; overhead should be bounded
        # (4 stages means ~4x more reads, expect <300% overhead)
        assert overhead_pct < 300, (
            f"Dependency loading overhead too high: {overhead_pct:.1f}%"
        )


# ===================================================================
# Stress tests
# ===================================================================
@pytest.mark.stress
class TestIOStress:
    @pytest.mark.anyio
    async def test_flush_20k_contexts(self, tmp_path: Path) -> None:
        """Stress: flush 20,000 contexts across many shards."""
        require_available_memory_gb(4.0)
        n = 20000

        saver = TaskSaver(
            root_dir=tmp_path / "stress_flush",
            shard_samples=256,
            shard_write_concurrency=8,
            write_buffer_size=n + 1,
            write_buffer_flush_interval=9999.0,
        )

        for i in range(n):
            ctx = _make_bench_ctx(i, TaskStage.FINAL)
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)

        timer = PerfTimer()
        with timer:
            await saver.flush()

        cps = samples_per_second(n, timer.elapsed)
        print(f"STRESS: flush_20k => {cps:.0f} ctx/s, {timer.elapsed:.2f}s")
