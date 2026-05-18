"""Task profiling: I/O timings, stage execution, and token usage statistics."""

import bisect
import contextlib
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, TypedDict

import anyio
import orjson
from loguru import logger

from sieval.core.models import ModelUsage

from .context import TaskContext


class TaskTokenStats:
    """Bucket-based token count statistics.

    Tracks count, total, min, max, and per-bucket distribution for a stream
    of token counts.  Bucket boundaries are defined by *thresholds* and the
    corresponding human-readable *labels* (one more label than thresholds).
    """

    def __init__(self, thresholds: list[int], labels: list[str]):
        assert len(labels) == len(thresholds) + 1
        self.thresholds = thresholds
        self.labels = labels

        self.count = 0
        self.total = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.buckets: dict[str, int] = defaultdict(int)

    def update(self, count: int):
        self.count += 1
        self.total += count
        if count < self.min:
            self.min = count
        if count > self.max:
            self.max = count

        # Determine bucket
        idx = bisect.bisect_right(self.thresholds, count)
        label = self.labels[idx]
        self.buckets[label] += 1

    def clear(self):
        self.count = 0
        self.total = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.buckets.clear()

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


class ProfileConfigSnapshot(TypedDict):
    profile_io: bool
    profile_stages: bool
    profile_usage: bool


class ProfileMeta(TypedDict):
    generated_at: str
    task_name: str
    config: ProfileConfigSnapshot


class ProfileTimingStats(TypedDict):
    count: int
    total_s: float
    avg_s: float
    min_s: float
    max_s: float
    p50_s: float
    p90_s: float
    p95_s: float
    p99_s: float


class ProfileTokenStats(TypedDict):
    count: int
    total: int
    avg: float
    min: int
    max: int
    buckets: dict[str, int]


class ProfileStageTokenUsage(TypedDict, total=False):
    input: ProfileTokenStats
    output: ProfileTokenStats


class ProfileReport(TypedDict, total=False):
    meta: ProfileMeta
    token_usage: dict[str, ProfileStageTokenUsage]
    io: dict[str, ProfileTimingStats]
    stages: dict[str, ProfileTimingStats]


def _compute_timing_stats(durations: list[float]) -> ProfileTimingStats:
    """Compute timing statistics using nearest-rank percentiles (no interpolation).

    Percentile indices use ``int(p * N)`` clamped to ``N-1``.  This rounds
    toward the *upper* neighbour for small sample sizes (e.g. p50 of two
    elements returns the larger value).
    """
    if not durations:
        raise ValueError("durations must be non-empty")
    sorted_durs = sorted(durations)
    count = len(sorted_durs)
    total = sum(sorted_durs)
    return ProfileTimingStats(
        count=count,
        total_s=total,
        avg_s=total / count,
        min_s=sorted_durs[0],
        max_s=sorted_durs[-1],
        p50_s=sorted_durs[min(int(0.5 * count), count - 1)],
        p90_s=sorted_durs[min(int(0.9 * count), count - 1)],
        p95_s=sorted_durs[min(int(0.95 * count), count - 1)],
        p99_s=sorted_durs[min(int(0.99 * count), count - 1)],
    )


class TaskProfiler:
    """Collects I/O timings, stage execution timings, and token usage for a task run."""

    def __init__(
        self,
        task_name: str = "Task",
        profile_io: bool = False,
        profile_stages: bool = False,
        profile_usage: bool = True,
    ):
        self._task_name = task_name
        self._profile_io = profile_io
        self._profile_stages = profile_stages
        self._profile_usage = profile_usage

        # operation_name -> [durations]
        self._io_aggregates: dict[str, list[float]] = defaultdict(list)
        # stage_name -> [durations]
        self._stage_aggregates: dict[str, list[float]] = defaultdict(list)
        # stage_name -> TaskTokenStats
        self._stage_input_tokens: dict[str, TaskTokenStats] = {}
        self._stage_output_tokens: dict[str, TaskTokenStats] = {}

    def should_profile_io(self) -> bool:
        return self._profile_io

    def should_profile_stages(self) -> bool:
        return self._profile_stages

    def should_profile_usage(self) -> bool:
        return self._profile_usage

    def record_io(self, operation: str, duration: float) -> None:
        if not self._profile_io:
            return
        self._io_aggregates[operation].append(duration)

    def _get_or_create_stage_token_stats(
        self, stage_name: str
    ) -> tuple[TaskTokenStats, TaskTokenStats]:
        if stage_name not in self._stage_input_tokens:
            self._stage_input_tokens[stage_name] = TaskTokenStats(
                thresholds=[8192, 16384, 32768, 131072],
                labels=["<8k", "8k-16k", "16k-32k", "32k-128k", ">128k"],
            )
        if stage_name not in self._stage_output_tokens:
            self._stage_output_tokens[stage_name] = TaskTokenStats(
                thresholds=[1024, 4096, 8192, 16384],
                labels=["<1k", "1k-4k", "4k-8k", "8k-16k", ">16k"],
            )
        return self._stage_input_tokens[stage_name], self._stage_output_tokens[
            stage_name
        ]

    def record_model_usage(
        self,
        usage: dict[str, int] | ModelUsage | None,
        stage_name: str | None = None,
    ) -> None:
        if not usage or not self._profile_usage or not stage_name:
            return

        stage_input, stage_output = self._get_or_create_stage_token_stats(stage_name)
        pt = usage.get("input_tokens", 0)
        ct = usage.get("output_tokens", 0)
        if pt > 0:
            stage_input.update(pt)
        if ct > 0:
            stage_output.update(ct)

    def aggregate_stage_timings(self, contexts: dict[str | int, TaskContext]) -> None:
        """Rebuild per-stage timing distributions from persisted task contexts."""
        if not self._profile_stages:
            return
        self._stage_aggregates.clear()
        for ctx in contexts.values():
            if ctx.stage_meta:
                for stage_name, meta_list in ctx.stage_meta.items():
                    for meta in meta_list:
                        timing = meta.get("timing_s")
                        if isinstance(timing, int | float):
                            self._stage_aggregates[stage_name].append(float(timing))

    def aggregate_token_usage(self, contexts: dict[str | int, TaskContext]) -> None:
        """Rebuild per-stage token usage statistics from persisted task contexts."""
        if not self._profile_usage:
            return
        # Clear per-stage stats
        for stats in self._stage_input_tokens.values():
            stats.clear()
        for stats in self._stage_output_tokens.values():
            stats.clear()

        for ctx in contexts.values():
            if ctx.stage_meta:
                for stage_name, meta_list in ctx.stage_meta.items():
                    # Get or create per-stage stats
                    stage_input, stage_output = self._get_or_create_stage_token_stats(
                        stage_name
                    )
                    for meta in meta_list:
                        # Read usage from model_calls (new structure)
                        model_calls = meta.get("model_calls", [])
                        for call in model_calls:
                            usage = call.get("usage")
                            if isinstance(usage, dict):
                                # Record to per-stage stats
                                pt = usage.get("input_tokens", 0)
                                ct = usage.get("output_tokens", 0)
                                if pt > 0:
                                    stage_input.update(pt)
                                if ct > 0:
                                    stage_output.update(ct)

    def log_summary(self) -> None:
        """Emit a formatted profiling report (token usage, I/O, stages) via loguru."""
        header = f"[{self._task_name}]"

        if self._profile_usage and (
            self._stage_input_tokens or self._stage_output_tokens
        ):
            # Calculate global totals from per-stage stats
            total_input = sum(
                stats.total for stats in self._stage_input_tokens.values()
            )
            total_output = sum(
                stats.total for stats in self._stage_output_tokens.values()
            )
            total_tokens = total_input + total_output

            if total_tokens > 0:
                logger.info("=== {} Token Usage Summary ===", header)
                logger.info("   Total Tokens Used: {:,}", total_tokens)
                logger.info("   Input Tokens: {:,}", total_input)
                logger.info("   Output Tokens: {:,}", total_output)

                # Log per-stage token usage with distribution
                logger.info("=== {} Per-Stage Token Usage ===", header)
                # Get all stage names
                stage_names = sorted(
                    set(self._stage_input_tokens.keys())
                    | set(self._stage_output_tokens.keys())
                )
                for stage_name in stage_names:
                    input_stats = self._stage_input_tokens.get(stage_name)
                    output_stats = self._stage_output_tokens.get(stage_name)

                    if (input_stats and input_stats.count > 0) or (
                        output_stats and output_stats.count > 0
                    ):
                        logger.info("   Stage: {}", stage_name)
                        if input_stats and input_stats.count > 0:
                            self._log_token_stats("  Input", input_stats)
                        if output_stats and output_stats.count > 0:
                            self._log_token_stats("  Output", output_stats)
                        stage_total = (input_stats.total if input_stats else 0) + (
                            output_stats.total if output_stats else 0
                        )
                        logger.info("     Stage Total: {:,}", stage_total)

        if self._profile_io and self._io_aggregates:
            logger.info("=== {} I/O Profile Summary ===", header)
            self._log_aggregated_stats(self._io_aggregates, prefix=header)

        if self._profile_stages and self._stage_aggregates:
            logger.info("=== {} Stage Profile Summary ===", header)
            self._log_aggregated_stats(self._stage_aggregates, prefix=header)

    def _log_token_stats(self, name: str, stats: TaskTokenStats):
        if stats.count == 0:
            return

        # Base stats
        logger.opt(raw=True).info(
            f"   {name:<15} | "
            f"reqs: {stats.count:>5} | "
            f"avg: {stats.avg:>9.1f} | "
            f"min: {stats.min:>9} | "
            f"max: {stats.max:>9} | "
            f"sum: {stats.total:>9,}\n"
        )
        # Buckets
        sorted_buckets = sorted(
            stats.buckets.items(), key=lambda x: stats.labels.index(x[0])
        )
        bucket_strs = [f"{k}: {v}" for k, v in sorted_buckets]
        logger.opt(raw=True).info(f"     Dist: {', '.join(bucket_strs)}\n")

    def _log_aggregated_stats(
        self, aggregates: dict[str, list[float]], prefix: str = ""
    ) -> None:
        if not aggregates:
            return

        valid_items = {k: v for k, v in aggregates.items() if v}
        if not valid_items:
            return

        max_name_len = max(len(k) for k in valid_items)
        for op_name, durations in sorted(valid_items.items()):
            s = _compute_timing_stats(durations)

            msg = (
                f"{prefix}   {op_name:<{max_name_len}} | "
                f"cnt: {s['count']:>5} | "
                f"avg: {s['avg_s']:>9.4f}s | "
                f"min: {s['min_s']:>9.4f}s | "
                f"p50: {s['p50_s']:>9.4f}s | "
                f"p95: {s['p95_s']:>9.4f}s | "
                f"max: {s['max_s']:>9.4f}s | "
                f"all: {s['total_s']:>9.2f}s\n"
            )
            logger.opt(raw=True).info(msg)

    def _token_stats_to_dict(self, stats: TaskTokenStats) -> ProfileTokenStats:
        """Convert a TaskTokenStats instance to a serializable dict."""
        return ProfileTokenStats(
            count=stats.count,
            total=stats.total,
            avg=stats.avg,
            min=int(stats.min),
            max=int(stats.max),
            buckets=dict(stats.buckets),
        )

    def to_dict(self) -> ProfileReport:
        """Serialize collected profiling data into a structured dict."""
        report = ProfileReport(
            meta=ProfileMeta(
                generated_at=datetime.now(UTC).isoformat(),
                task_name=self._task_name,
                config=ProfileConfigSnapshot(
                    profile_io=self._profile_io,
                    profile_stages=self._profile_stages,
                    profile_usage=self._profile_usage,
                ),
            )
        )

        # Token usage
        if self._profile_usage:
            stage_names = sorted(
                set(self._stage_input_tokens.keys())
                | set(self._stage_output_tokens.keys())
            )
            token_usage: dict[str, ProfileStageTokenUsage] = {}
            for stage_name in stage_names:
                entry = ProfileStageTokenUsage()
                input_stats = self._stage_input_tokens.get(stage_name)
                output_stats = self._stage_output_tokens.get(stage_name)
                if input_stats and input_stats.count > 0:
                    entry["input"] = self._token_stats_to_dict(input_stats)
                if output_stats and output_stats.count > 0:
                    entry["output"] = self._token_stats_to_dict(output_stats)
                if entry:
                    token_usage[stage_name] = entry
            if token_usage:
                report["token_usage"] = token_usage

        # I/O timings
        if self._profile_io:
            io_stats: dict[str, ProfileTimingStats] = {}
            for op_name, durations in sorted(self._io_aggregates.items()):
                if durations:
                    io_stats[op_name] = _compute_timing_stats(durations)
            if io_stats:
                report["io"] = io_stats

        # Stage timings
        if self._profile_stages:
            stage_stats: dict[str, ProfileTimingStats] = {}
            for stage_name, durations in sorted(self._stage_aggregates.items()):
                if durations:
                    stage_stats[stage_name] = _compute_timing_stats(durations)
            if stage_stats:
                report["stages"] = stage_stats

        return report

    async def save(self, root_dir: Path) -> None:
        """Atomically write profile.json to root_dir.

        Skips writing if no profiling data was collected.
        """
        report = self.to_dict()
        # Skip if only meta — no actual data sections
        if not any(k in report for k in ("token_usage", "io", "stages")):
            return

        profile_path = root_dir / "profile.json"
        tmp_path = profile_path.with_suffix(".tmp")
        try:
            async with await anyio.open_file(tmp_path, "wb") as f:
                await f.write(orjson.dumps(report))
            await anyio.Path(tmp_path).replace(profile_path)
            logger.info("Saved profile to: {}", profile_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                await anyio.Path(tmp_path).unlink(missing_ok=True)
            logger.error("Failed to save profile.json: {}", e)

    def get_io_aggregates(self) -> dict[str, list[float]]:
        return {k: v.copy() for k, v in self._io_aggregates.items()}

    def get_stage_aggregates(self) -> dict[str, list[float]]:
        return {k: v.copy() for k, v in self._stage_aggregates.items()}

    def clear(self) -> None:
        self._io_aggregates.clear()
        self._stage_aggregates.clear()
        for stats in self._stage_input_tokens.values():
            stats.clear()
        for stats in self._stage_output_tokens.values():
            stats.clear()


class TaskProfilerContext:
    """Async context manager that times an operation for a TaskProfiler."""

    def __init__(
        self, profiler: TaskProfiler, operation: str, io_operation: bool = True
    ):
        self._profiler = profiler
        self._operation = operation
        self._is_io = io_operation
        self._start: float | None = None

    async def __aenter__(self) -> Self:
        if (self._is_io and self._profiler.should_profile_io()) or (
            not self._is_io and self._profiler.should_profile_stages()
        ):
            self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None and self._is_io:
            duration = time.perf_counter() - self._start
            self._profiler.record_io(self._operation, duration)
