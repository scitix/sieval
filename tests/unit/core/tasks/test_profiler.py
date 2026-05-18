"""
Tests for sieval.core.tasks.profiler — TaskTokenStats, TaskProfiler.

Prefer behavior assertions (summary output and public getters) over
private container introspection.
"""

import io
from unittest.mock import patch

import orjson
import pytest
from loguru import logger

from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.profiler import (
    TaskProfiler,
    TaskProfilerContext,
    TaskTokenStats,
    _compute_timing_stats,
)


def _make_ctx_with_meta(stage_meta: dict) -> TaskContext:
    return TaskContext(sample_id=0, raw_sample={}, stage_meta=stage_meta)


def _capture_logs(fn) -> str:
    sink = io.StringIO()
    logger_id = logger.add(sink, format="{message}")
    try:
        fn()
    finally:
        logger.remove(logger_id)
    return sink.getvalue()


class TestTaskTokenStats:
    def test_update_and_bucket_stats(self):
        stats = TaskTokenStats(thresholds=[100, 1000], labels=["<100", "100-1k", ">1k"])
        for value in [50, 500, 2000]:
            stats.update(value)

        assert stats.count == 3
        assert stats.total == 2550
        assert stats.min == 50
        assert stats.max == 2000
        assert stats.avg == pytest.approx(850.0)
        assert stats.buckets["<100"] == 1
        assert stats.buckets["100-1k"] == 1
        assert stats.buckets[">1k"] == 1

        boundary = TaskTokenStats(thresholds=[100], labels=["<=100", ">100"])
        boundary.update(100)
        boundary.update(99)
        # bisect_right(100, [100]) = 1, so 100 falls into the second bucket.
        assert boundary.buckets[">100"] == 1
        assert boundary.buckets["<=100"] == 1

    def test_clear(self):
        stats = TaskTokenStats(thresholds=[100], labels=["<=100", ">100"])
        stats.update(50)
        stats.update(200)
        stats.clear()
        assert stats.count == 0
        assert stats.total == 0
        assert stats.min == float("inf")
        assert stats.max == float("-inf")
        assert stats.buckets == {}

    def test_avg_empty(self):
        stats = TaskTokenStats(thresholds=[100], labels=["<=100", ">100"])
        assert stats.avg == 0.0


class TestTaskProfilerRecordIO:
    def test_record_io_respects_profile_flag(self):
        enabled = TaskProfiler(profile_io=True)
        enabled.record_io("read_shard", 0.5)
        enabled.record_io("read_shard", 0.3)
        assert enabled.get_io_aggregates()["read_shard"] == [0.5, 0.3]

        disabled = TaskProfiler(profile_io=False)
        disabled.record_io("read_shard", 0.5)
        assert disabled.get_io_aggregates() == {}

    def test_should_profile_usage_reflects_flag(self):
        assert TaskProfiler(profile_usage=True).should_profile_usage() is True
        assert TaskProfiler(profile_usage=False).should_profile_usage() is False


class TestTaskProfilerRecordUsage:
    def test_record_model_usage_paths(self):
        usage = {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}

        enabled = TaskProfiler(profile_usage=True)
        enabled.record_model_usage(usage, stage_name="inferred")
        enabled_out = _capture_logs(enabled.log_summary)
        assert "Token Usage Summary" in enabled_out
        assert "Total Tokens Used: 1,200" in enabled_out
        assert "Input Tokens: 1,000" in enabled_out
        assert "Output Tokens: 200" in enabled_out
        assert "Stage: inferred" in enabled_out

        disabled = TaskProfiler(profile_usage=False)
        disabled.record_model_usage(usage, stage_name="inferred")
        assert _capture_logs(disabled.log_summary).strip() == ""

        no_stage = TaskProfiler(profile_usage=True)
        no_stage.record_model_usage(usage, stage_name=None)
        assert _capture_logs(no_stage.log_summary).strip() == ""

        none_usage = TaskProfiler(profile_usage=True)
        none_usage.record_model_usage(None, stage_name="inferred")
        assert _capture_logs(none_usage.log_summary).strip() == ""

        zero_usage = TaskProfiler(profile_usage=True)
        zero_usage.record_model_usage(
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            stage_name="inferred",
        )
        assert _capture_logs(zero_usage.log_summary).strip() == ""


class TestTaskProfilerAggregateStageTimings:
    def test_aggregate_stage_timings_enabled(self):
        profiler = TaskProfiler(profile_stages=True)
        ctx1 = _make_ctx_with_meta(
            {
                "preprocessed": [{"timing_s": 0.1}],
                "inferred": [{"timing_s": 1.5}, {"timing_s": 2.0}],
                "ignored": [{"model_calls": []}],
            }
        )
        ctx2 = _make_ctx_with_meta({"inferred": [{"timing_s": 1.0}]})

        profiler.aggregate_stage_timings({0: ctx1, 1: ctx2})
        agg = profiler.get_stage_aggregates()
        assert agg["preprocessed"] == [0.1]
        assert sorted(agg["inferred"]) == [1.0, 1.5, 2.0]
        assert agg.get("ignored", []) == []

    def test_aggregate_stage_timings_disabled(self):
        profiler = TaskProfiler(profile_stages=False)
        ctx = _make_ctx_with_meta({"inferred": [{"timing_s": 1.0}]})
        profiler.aggregate_stage_timings({0: ctx})
        assert profiler.get_stage_aggregates() == {}


class TestTaskProfilerAggregateTokenUsage:
    def test_aggregate_token_usage_enabled(self):
        profiler = TaskProfiler(profile_usage=True)
        ctx1 = _make_ctx_with_meta(
            {
                "inferred": [
                    {
                        "model_calls": [
                            {
                                "usage": {
                                    "input_tokens": 500,
                                    "output_tokens": 100,
                                    "total_tokens": 600,
                                }
                            },
                            {
                                "usage": {
                                    "input_tokens": 300,
                                    "output_tokens": 50,
                                    "total_tokens": 350,
                                }
                            },
                        ]
                    }
                ]
            }
        )
        ctx2 = _make_ctx_with_meta(
            {
                "inferred": [
                    {
                        "model_calls": [
                            {
                                "usage": {
                                    "input_tokens": 200,
                                    "output_tokens": 20,
                                    "total_tokens": 220,
                                }
                            },
                            {"usage": "invalid_usage_payload"},
                        ]
                    }
                ]
            }
        )

        profiler.aggregate_token_usage({0: ctx1, 1: ctx2})
        output = _capture_logs(profiler.log_summary)
        assert "Input Tokens: 1,000" in output
        assert "Output Tokens: 170" in output
        assert "Stage Total: 1,170" in output

    def test_aggregate_token_usage_disabled(self):
        profiler = TaskProfiler(profile_usage=False)
        ctx = _make_ctx_with_meta(
            {
                "inferred": [
                    {
                        "model_calls": [
                            {
                                "usage": {
                                    "input_tokens": 100,
                                    "output_tokens": 50,
                                    "total_tokens": 150,
                                }
                            }
                        ]
                    }
                ]
            }
        )
        profiler.aggregate_token_usage({0: ctx})
        assert _capture_logs(profiler.log_summary).strip() == ""

    def test_aggregate_token_usage_clears_previous_stage_stats(self):
        profiler = TaskProfiler(profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            stage_name="inferred",
        )

        # No usage in this pass -> existing stage stats should be cleared first.
        profiler.aggregate_token_usage({0: _make_ctx_with_meta({"inferred": [{}]})})

        assert _capture_logs(profiler.log_summary).strip() == ""


class TestTaskProfilerClear:
    def test_clear_all(self):
        profiler = TaskProfiler(
            profile_io=True, profile_stages=True, profile_usage=True
        )
        profiler.record_io("op", 1.0)
        profiler.record_model_usage(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            stage_name="inferred",
        )
        profiler.clear()
        assert profiler.get_io_aggregates() == {}
        assert profiler.get_stage_aggregates() == {}
        assert _capture_logs(profiler.log_summary).strip() == ""


class TestTaskProfilerLogSummary:
    def test_log_summary_with_data(self):
        profiler = TaskProfiler(
            task_name="AllTest",
            profile_usage=True,
            profile_io=True,
            profile_stages=True,
        )
        profiler.record_model_usage(
            {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
            stage_name="inferred",
        )
        profiler.record_model_usage(
            {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
            stage_name="inferred",
        )
        profiler.record_io("read_shard", 0.5)
        profiler.record_io("read_shard", 0.3)
        profiler.record_io("write_shard", 1.2)
        profiler.aggregate_stage_timings(
            {
                0: _make_ctx_with_meta(
                    {
                        "preprocessed": [{"timing_s": 0.1}],
                        "inferred": [{"timing_s": 1.5}, {"timing_s": 2.0}],
                    }
                ),
                1: _make_ctx_with_meta(
                    {
                        "preprocessed": [{"timing_s": 0.2}],
                        "inferred": [{"timing_s": 1.0}],
                    }
                ),
            }
        )

        output = _capture_logs(profiler.log_summary)
        assert "Token Usage Summary" in output
        assert "I/O Profile Summary" in output
        assert "Stage Profile Summary" in output
        assert "Input Tokens" in output
        assert "1,500" in output
        assert "300" in output
        assert "read_shard" in output
        assert "write_shard" in output
        assert "preprocessed" in output
        assert "inferred" in output

    def test_log_summary_reports_input_bucket_boundary_for_8192(self):
        profiler = TaskProfiler(task_name="Boundary", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 8192, "output_tokens": 0, "total_tokens": 8192},
            stage_name="inferred",
        )

        output = _capture_logs(profiler.log_summary)

        assert "Stage: inferred" in output
        assert "8k-16k: 1" in output

    def test_log_summary_reports_default_input_and_output_distributions(self):
        """
        Keep default bucket thresholds/labels behavior stable at boundaries.
        This validates user-visible summary output, not internal containers.
        """
        profiler = TaskProfiler(task_name="BoundaryAll", profile_usage=True)
        for input_tokens, output_tokens in [
            (8191, 1023),
            (8192, 1024),
            (16384, 4096),
            (32768, 8192),
            (131072, 16384),
        ]:
            profiler.record_model_usage(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                stage_name="inferred",
            )

        output = _capture_logs(profiler.log_summary)

        assert "Dist: <8k: 1, 8k-16k: 1, 16k-32k: 1, 32k-128k: 1, >128k: 1" in output
        assert "Dist: <1k: 1, 1k-4k: 1, 4k-8k: 1, 8k-16k: 1, >16k: 1" in output

    def test_log_summary_empty(self):
        profiler = TaskProfiler(
            task_name="EmptyTest",
            profile_usage=True,
            profile_io=True,
            profile_stages=True,
        )
        output = _capture_logs(profiler.log_summary)
        # No data recorded, so no output should be produced.
        assert output.strip() == ""

    def test_log_summary_uses_union_of_stage_names_and_adds_stage_totals(self):
        profiler = TaskProfiler(task_name="UnionStages", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120},
            stage_name="preprocessed",
        )
        profiler.record_model_usage(
            {"input_tokens": 0, "output_tokens": 30, "total_tokens": 30},
            stage_name="feedback",
        )
        profiler.record_model_usage(
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            stage_name="inferred",
        )

        output = _capture_logs(profiler.log_summary)
        assert "Stage: preprocessed" in output
        assert "Stage: feedback" in output
        assert "Stage: inferred" in output
        assert "Input" in output
        assert "Output" in output
        # inferred stage total must be input + output, not subtraction.
        assert "Stage Total: 30" in output


class TestComputeTimingStats:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _compute_timing_stats([])

    def test_single_element(self):
        s = _compute_timing_stats([0.5])
        assert s["count"] == 1
        assert s["total_s"] == pytest.approx(0.5)
        assert s["avg_s"] == pytest.approx(0.5)
        assert s["min_s"] == pytest.approx(0.5)
        assert s["max_s"] == pytest.approx(0.5)
        assert s["p50_s"] == pytest.approx(0.5)
        assert s["p90_s"] == pytest.approx(0.5)
        assert s["p95_s"] == pytest.approx(0.5)
        assert s["p99_s"] == pytest.approx(0.5)

    def test_two_elements(self):
        s = _compute_timing_stats([0.1, 0.9])
        assert s["count"] == 2
        assert s["min_s"] == pytest.approx(0.1)
        assert s["max_s"] == pytest.approx(0.9)
        # int(0.5 * 2) == 1, so p50 is sorted_durs[1] == 0.9
        assert s["p50_s"] == pytest.approx(0.9)

    def test_hundred_elements(self):
        durs = [float(i) for i in range(100)]
        s = _compute_timing_stats(durs)
        assert s["count"] == 100
        assert s["total_s"] == pytest.approx(sum(durs))
        assert s["avg_s"] == pytest.approx(sum(durs) / 100)
        assert s["min_s"] == pytest.approx(0.0)
        assert s["max_s"] == pytest.approx(99.0)
        # int(0.5 * 100) == 50, so p50 is sorted_durs[50] == 50.0
        assert s["p50_s"] == pytest.approx(50.0)
        # int(0.9 * 100) == 90, so p90 is sorted_durs[90] == 90.0
        assert s["p90_s"] == pytest.approx(90.0)
        # int(0.95 * 100) == 95, so p95 is sorted_durs[95] == 95.0
        assert s["p95_s"] == pytest.approx(95.0)
        # int(0.99 * 100) == 99, so p99 is sorted_durs[99] == 99.0
        assert s["p99_s"] == pytest.approx(99.0)

    def test_unsorted_input(self):
        s = _compute_timing_stats([3.0, 1.0, 2.0])
        assert s["min_s"] == pytest.approx(1.0)
        assert s["max_s"] == pytest.approx(3.0)
        # int(0.5 * 3) == 1, so p50 is sorted_durs[1] == 2.0
        assert s["p50_s"] == pytest.approx(2.0)


class TestTaskProfilerContext:
    @pytest.mark.anyio
    async def test_context_records_io_duration_when_enabled(self):
        profiler = TaskProfiler(profile_io=True)

        with patch(
            "sieval.core.tasks.profiler.time.perf_counter",
            side_effect=[10.0, 10.25],
        ):
            async with TaskProfilerContext(profiler, "read_shard", io_operation=True):
                pass

        assert profiler.get_io_aggregates()["read_shard"] == [0.25]


class TestTaskProfilerToDict:
    def test_all_dimensions_enabled(self):
        profiler = TaskProfiler(
            task_name="AllDims",
            profile_io=True,
            profile_stages=True,
            profile_usage=True,
        )
        profiler.record_io("read_shard", 0.5)
        profiler.record_io("read_shard", 0.3)
        profiler.record_model_usage(
            {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
            stage_name="inferred",
        )
        profiler.aggregate_stage_timings(
            {
                0: _make_ctx_with_meta(
                    {"inferred": [{"timing_s": 1.5}, {"timing_s": 2.0}]}
                ),
            }
        )

        result = profiler.to_dict()

        assert "meta" in result
        assert result["meta"]["task_name"] == "AllDims"
        assert result["meta"]["config"]["profile_io"] is True
        assert result["meta"]["config"]["profile_stages"] is True
        assert result["meta"]["config"]["profile_usage"] is True
        assert "generated_at" in result["meta"]

        assert "token_usage" in result
        assert "inferred" in result["token_usage"]
        assert result["token_usage"]["inferred"]["input"]["total"] == 1000
        assert result["token_usage"]["inferred"]["output"]["total"] == 200

        assert "io" in result
        assert result["io"]["read_shard"]["count"] == 2

        assert "stages" in result
        assert result["stages"]["inferred"]["count"] == 2

    def test_usage_only(self):
        """Default config: only profile_usage=True."""
        profiler = TaskProfiler(task_name="UsageOnly", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
            stage_name="inferred",
        )
        result = profiler.to_dict()

        assert "meta" in result
        assert "token_usage" in result
        assert "io" not in result
        assert "stages" not in result

    def test_no_data_produces_meta_only(self):
        """All dimensions enabled but no data recorded."""
        profiler = TaskProfiler(
            task_name="Empty",
            profile_io=True,
            profile_stages=True,
            profile_usage=True,
        )
        result = profiler.to_dict()

        assert "meta" in result
        assert "token_usage" not in result
        assert "io" not in result
        assert "stages" not in result

    def test_token_stats_serialization(self):
        """Verify bucket distribution is included in token stats."""
        profiler = TaskProfiler(task_name="Buckets", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            stage_name="inferred",
        )
        result = profiler.to_dict()

        input_stats = result["token_usage"]["inferred"]["input"]
        assert input_stats["count"] == 1
        assert input_stats["total"] == 100
        assert input_stats["min"] == 100
        assert input_stats["max"] == 100
        assert input_stats["avg"] == pytest.approx(100.0)
        assert "<8k" in input_stats["buckets"]


class TestTaskProfilerSave:
    @pytest.mark.anyio
    async def test_save_writes_valid_json(self, tmp_path):
        profiler = TaskProfiler(
            task_name="SaveTest",
            profile_usage=True,
        )
        profiler.record_model_usage(
            {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
            stage_name="inferred",
        )
        await profiler.save(tmp_path)

        profile_path = tmp_path / "profile.json"
        assert profile_path.exists()

        data = orjson.loads(profile_path.read_bytes())
        assert data["meta"]["task_name"] == "SaveTest"
        assert "token_usage" in data

    @pytest.mark.anyio
    async def test_save_atomic_no_tmp_left(self, tmp_path):
        profiler = TaskProfiler(task_name="Atomic", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            stage_name="inferred",
        )
        await profiler.save(tmp_path)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    @pytest.mark.anyio
    async def test_save_skips_when_no_data(self, tmp_path):
        profiler = TaskProfiler(
            task_name="NoData",
            profile_io=False,
            profile_stages=False,
            profile_usage=False,
        )
        await profiler.save(tmp_path)

        assert not (tmp_path / "profile.json").exists()

    @pytest.mark.anyio
    async def test_save_io_error_does_not_raise(self, tmp_path):
        profiler = TaskProfiler(task_name="IOError", profile_usage=True)
        profiler.record_model_usage(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            stage_name="inferred",
        )

        with patch(
            "sieval.core.tasks.profiler.anyio.open_file",
            side_effect=OSError("disk full"),
        ):
            await profiler.save(tmp_path)
        # Should not raise — error is logged.

        assert not (tmp_path / "profile.json").exists()
        assert list(tmp_path.glob("*.tmp")) == []
