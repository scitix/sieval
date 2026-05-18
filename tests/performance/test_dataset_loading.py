"""
Dataset loading and initialization benchmarks.

Measures HFDataset construction, iteration overhead, and context creation speed.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.tasks.context import TaskContext
from tests.conftest import (
    PerfMockDataset,
    PerfTimer,
    make_large_dataset,
    require_available_memory_gb,
    samples_per_second,
)


class TestDatasetConstruction:
    """Benchmark HFDataset.from_list() construction speed."""

    @pytest.mark.parametrize("n_samples", [1000, 5000, 20000])
    def test_dataset_construction_speed(self, n_samples: int) -> None:
        """Time to build an HFDataset from a list of dicts."""
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n_samples)]

        timer = PerfTimer()
        with timer:
            HFDataset.from_list(samples)

        sps = samples_per_second(n_samples, timer.elapsed)
        print(
            f"PERF: dataset_construction n={n_samples} => "
            f"{sps:.0f} samples/s, {timer.elapsed:.4f}s"
        )
        # Scale threshold linearly: ~0.25s per 1000 samples
        max_time = max(1.0, n_samples / 1000 * 0.25)
        assert timer.elapsed < max_time, (
            f"Dataset construction too slow: "
            f"{timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )


class TestContextInitialization:
    """Benchmark creating TaskContext from dataset iteration."""

    @pytest.mark.parametrize("n_samples", [1000, 5000, 20000])
    def test_context_initialization_speed(self, n_samples: int) -> None:
        """Time to iterate dataset and create TaskContext objects."""
        dataset = make_large_dataset(n_samples, payload_size=50)
        test_set = dataset.test_set
        assert test_set is not None

        timer = PerfTimer()
        with timer:
            contexts = {}
            for i, raw in enumerate(test_set):
                contexts[i] = TaskContext(sample_id=i, raw_sample=raw)

        sps = samples_per_second(n_samples, timer.elapsed)
        print(
            f"PERF: context_init_enumerate n={n_samples} "
            f"=> {sps:.0f} ctx/s, {timer.elapsed:.4f}s"
        )
        assert len(contexts) == n_samples
        # Scale threshold: ~0.25s per 1000 samples (includes HF iteration)
        max_time = max(1.0, n_samples / 1000 * 0.25)
        assert timer.elapsed < max_time, (
            f"Context init too slow: {timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )

    @pytest.mark.parametrize("n_samples", [1000, 5000, 20000])
    def test_context_initialization_random_access(self, n_samples: int) -> None:
        """Time to create contexts via test_set[i] random access (lazy path).

        This benchmarks the actual code path used by the runner's deferred
        context creation: individual Arrow row access via __getitem__.
        """
        dataset = make_large_dataset(n_samples, payload_size=50)
        test_set = dataset.test_set
        assert test_set is not None

        timer = PerfTimer()
        with timer:
            contexts = {}
            for i in range(n_samples):
                raw = test_set[i]
                contexts[i] = TaskContext(sample_id=i, raw_sample=raw)

        sps = samples_per_second(n_samples, timer.elapsed)
        print(
            f"PERF: context_init_random_access n={n_samples} "
            f"=> {sps:.0f} ctx/s, {timer.elapsed:.4f}s"
        )
        assert len(contexts) == n_samples
        # Random access may be slower than sequential enumerate; allow 2x headroom
        max_time = max(1.0, n_samples / 1000 * 0.50)
        assert timer.elapsed < max_time, (
            f"Context random access too slow: "
            f"{timer.elapsed:.3f}s (limit {max_time:.1f}s)"
        )


class TestDatasetIterationOverhead:
    """Compare HF Dataset iteration vs plain list iteration."""

    def test_dataset_iteration_overhead(self) -> None:
        """HF Dataset iteration overhead stays within an expected bound."""
        n = 10000
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n)]
        hf_ds = HFDataset.from_list(samples)

        # Raw list iteration
        timer_list = PerfTimer()
        with timer_list:
            for s in samples:
                _ = s["question"]

        # HF Dataset iteration
        timer_hf = PerfTimer()
        with timer_hf:
            for s in hf_ds:
                _ = s["question"]

        ratio = timer_hf.elapsed / timer_list.elapsed if timer_list.elapsed > 0 else 0
        print(
            f"PERF: iteration overhead list={timer_list.elapsed:.4f}s, "
            f"hf={timer_hf.elapsed:.4f}s, ratio={ratio:.1f}x"
        )
        # HF Dataset uses Arrow columnar storage, so per-row Python iteration is
        # much slower than raw list iteration. Guard against severe regression.
        assert ratio < 600, (
            f"HF iteration overhead regression: {ratio:.1f}x (expect <600x)"
        )


class TestLargePayloadSamples:
    """Benchmark loading samples with large payloads (e.g. long text)."""

    def test_large_sample_payload(self) -> None:
        """1000 samples with ~50KB payload each."""
        n = 1000
        payload_size = 50 * 1024  # 50KB

        timer = PerfTimer()
        with timer:
            dataset = make_large_dataset(n, payload_size=payload_size)

        assert dataset.test_set is not None

        timer_iter = PerfTimer()
        with timer_iter:
            for s in dataset.test_set:
                _ = s["question"]

        print(
            f"PERF: large_payload construction={timer.elapsed:.4f}s, "
            f"iteration={timer_iter.elapsed:.4f}s"
        )
        total = timer.elapsed + timer_iter.elapsed
        assert total < 10.0, f"Large payload loading too slow: {total:.3f}s"


class TestDatasetOperations:
    """Benchmark select/shuffle operations."""

    def test_dataset_select_and_shuffle(self) -> None:
        """select() and shuffle() on 50k-sample dataset."""
        n = 50000
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n)]
        hf_ds = HFDataset.from_list(samples)
        ds_dict = HFDatasetDict({"test": hf_ds})
        dataset = PerfMockDataset.__new__(PerfMockDataset)
        dataset._dataset_dict = ds_dict

        timer_select = PerfTimer()
        with timer_select:
            dataset.select(1000)

        timer_shuffle = PerfTimer()
        with timer_shuffle:
            dataset.shuffle(seed=42)

        print(
            f"PERF: select={timer_select.elapsed:.4f}s, "
            f"shuffle={timer_shuffle.elapsed:.4f}s"
        )
        total = timer_select.elapsed + timer_shuffle.elapsed
        assert total < 3.0, f"Dataset operations too slow: {total:.3f}s"


# ===================================================================
# Stress tests
# ===================================================================
@pytest.mark.stress
class TestDatasetStress:
    def test_dataset_100k(self) -> None:
        """Stress: build and iterate 100k-sample dataset."""
        require_available_memory_gb(4.0)
        n = 100000

        timer = PerfTimer()
        with timer:
            dataset = make_large_dataset(n, payload_size=50)
            assert dataset.test_set is not None
            for s in dataset.test_set:
                _ = s["question"]

        sps = samples_per_second(n, timer.elapsed)
        print(f"STRESS: dataset_100k => {sps:.0f} samples/s, {timer.elapsed:.2f}s")
