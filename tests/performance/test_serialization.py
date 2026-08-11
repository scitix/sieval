"""
Serialization / deserialization benchmarks.

Measures TaskContext.serialize() and dict_to_obj() throughput at scale
and with varying payload sizes.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import orjson
import pytest

from sieval.core.tasks.context import TaskStage
from sieval.core.utils.serialization import dict_to_obj, global_type_registry
from tests.conftest import (
    COMMON_PROFILES,
    PerfTimer,
    _make_bench_ctx,
    samples_per_second,
)


@contextmanager
def _gc_paused() -> Iterator[None]:
    """Take Python's cyclic collector out of a timed *ratio*.

    A gen2 collection costs in proportion to the whole live heap -- ~0.3 ms at 5k
    tracked objects, ~300 ms at 2.4M -- and by the time a suite run reaches here
    that heap is thousands of other tests' fixtures, which this loop never touches.
    What it does touch is the trigger: ``obj_to_dict`` builds a container per node
    and the caller retains every result, so its net allocations drive gen0 -> gen1
    -> gen2. ``orjson.dumps`` returns one untracked ``bytes`` and triggers almost
    nothing, so the tax lands on the numerator alone and the ratio amplifies it:
    the same code has measured 12x and 184x against a 50x bar, deciding only
    whether one gen2 fell inside the window.

    Only the ratio needs this. The absolute bounds elsewhere in this file keep
    enough headroom to absorb a collection, and they are the ones whose numbers
    should stay comparable with a real run's.

    Reference counting is untouched, so nothing accumulates but cycles made inside
    the block.
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


class TestSerializeThroughput:
    """Measure TaskContext.serialize() throughput."""

    @pytest.mark.parametrize("n_contexts", [1000, 5000])
    def test_serialize_throughput(self, n_contexts: int) -> None:
        """Contexts serialized per second (small payload)."""
        contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n_contexts)]

        timer = PerfTimer()
        with timer:
            for ctx in contexts:
                ctx.serialize(store_type_metadata=True, include_meta=True)

        cps = samples_per_second(n_contexts, timer.elapsed)
        print(
            f"PERF: serialize n={n_contexts} => {cps:.0f} ctx/s, {timer.elapsed:.4f}s"
        )
        assert cps > 1000, f"Serialize throughput too low: {cps:.0f} ctx/s"

    @pytest.mark.parametrize("payload_kb", [1, 10, 50])
    def test_serialize_large_payload(self, payload_kb: int) -> None:
        """Serialization time for contexts with large payloads."""
        n = 100
        payload_size = payload_kb * 1024
        contexts = [
            _make_bench_ctx(i, TaskStage.FINAL, payload_size=payload_size)
            for i in range(n)
        ]

        timer = PerfTimer()
        with timer:
            for ctx in contexts:
                ctx.serialize(store_type_metadata=True, include_meta=True)

        print(f"PERF: serialize payload={payload_kb}KB n={n} => {timer.elapsed:.4f}s")
        assert timer.elapsed < 10.0, (
            f"Large payload serialize too slow: {timer.elapsed:.3f}s"
        )

    def test_orjson_vs_serialize_breakdown(self) -> None:
        """Time breakdown: Python obj_to_dict vs orjson.dumps."""
        n = 1000
        contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n)]

        dicts: list[dict] = []
        timer_ser = PerfTimer()
        timer_json = PerfTimer()
        # Both halves under one pause, so the comparison stays symmetric: leaving
        # the collector on for the C half would let the deferred collections land
        # there instead, inflating the denominator and masking a real regression.
        with _gc_paused():
            # Measure serialize (Python side)
            with timer_ser:
                for ctx in contexts:
                    dicts.append(
                        ctx.serialize(store_type_metadata=True, include_meta=True)
                    )

            # Measure orjson.dumps (C side)
            with timer_json:
                for d in dicts:
                    orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY)

        ratio = timer_ser.elapsed / timer_json.elapsed if timer_json.elapsed > 0 else 0
        print(
            f"PERF: serialize={timer_ser.elapsed:.4f}s, "
            f"orjson.dumps={timer_json.elapsed:.4f}s, "
            f"ratio={ratio:.1f}x"
        )
        # Python serialize should not be more than 50x slower than orjson C code
        assert ratio < 50, f"serialize overhead vs orjson too high: {ratio:.1f}x"


class TestDeserializeThroughput:
    """Measure dict_to_obj() throughput."""

    def test_deserialization_throughput(self) -> None:
        """dict_to_obj throughput for typed reconstruction."""
        n = 1000
        contexts = [_make_bench_ctx(i, TaskStage.FINAL) for i in range(n)]
        serialized = [
            ctx.serialize(store_type_metadata=True, include_meta=True)
            for ctx in contexts
        ]
        json_blobs = [orjson.dumps(d) for d in serialized]

        timer = PerfTimer()
        with timer:
            for blob in json_blobs:
                obj = orjson.loads(blob)
                for field_name in (
                    "preprocess_result",
                    "infer_result",
                    "postprocess_result",
                    "feedback_result",
                ):
                    if field_name in obj:
                        dict_to_obj(obj[field_name], global_type_registry)

        dps = samples_per_second(n, timer.elapsed)
        print(f"PERF: deserialize n={n} => {dps:.0f} ctx/s, {timer.elapsed:.4f}s")
        assert dps > 1000, f"Deserialize throughput too low: {dps:.0f} ctx/s"


class TestSerializeByIOProfile:
    """Measure serialization cost across different I/O profiles."""

    @pytest.mark.parametrize("profile", COMMON_PROFILES, ids=lambda p: p.name)
    def test_serialize_by_io_profile(self, profile) -> None:
        """Serialize 500 contexts per I/O profile to compare overhead."""
        n = 500
        contexts = [
            _make_bench_ctx(i, TaskStage.FINAL, payload_size=profile.output_size)
            for i in range(n)
        ]

        timer = PerfTimer()
        with timer:
            for ctx in contexts:
                ctx.serialize(store_type_metadata=True, include_meta=True)

        cps = samples_per_second(n, timer.elapsed)
        print(
            f"PERF: serialize profile={profile.name} n={n} => "
            f"{cps:.0f} ctx/s, {timer.elapsed:.4f}s"
        )
        # All profiles should maintain at least 1000 ctx/s
        assert cps > 1000, (
            f"Serialize throughput too low for {profile.name}: {cps:.0f} ctx/s"
        )
