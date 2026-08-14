"""Task profiling: I/O timings, stage execution, and token usage statistics."""

import bisect
import contextlib
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, NotRequired, Self, TypedDict

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


class UsageBreakdownStats:
    """One optional usage field, summed against the count it breaks down.

    ``parent_total`` accumulates only over the calls that actually reported the
    field, so a share is taken against comparable tokens.  Dividing by every
    call's output would understate reasoning on a fleet where only some servers
    report it, and the error grows silently with the non-reporting share.
    """

    def __init__(self, parent_key: str):
        self.parent_key = parent_key
        self.total = 0
        self.parent_total = 0
        self.calls = 0

    def update(self, value: int, parent: int) -> None:
        self.total += value
        self.parent_total += parent
        self.calls += 1

    @property
    def share(self) -> float | None:
        """The ratio, or ``None`` where there is no parent to take it against.

        ``0.0`` would be the same lie the optional counts exist to avoid: a
        server that reports 50 reasoning tokens against 0 completion tokens --
        exactly the one counting reasoning outside its completion count -- has
        not measured a 0% share, it has made the ratio undefined.  ``total``
        still carries the number that was reported.
        """
        return self.total / self.parent_total if self.parent_total > 0 else None


def _usage_count(usage: Mapping[str, object], key: str) -> int | None:
    """Read one token count, tolerating the untyped shapes read back from disk.

    ``None`` means the key was absent or unusable -- never zero, which is a
    reported measurement.
    """
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


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


class ProfileTokenBreakdown(TypedDict):
    """An optional usage field aggregated over the calls that reported it.

    ``share`` is ``total / parent_total``, and both denominators cover only
    ``calls_reporting``.  Compare that against ``calls_total`` before reading
    the share as representative of the run.  It is **absent** when
    ``parent_total`` is 0, since the ratio is undefined rather than zero.
    """

    total: int
    parent: str
    parent_total: int
    share: NotRequired[float]
    calls_reporting: int
    calls_total: int


class ProfileStageTokenUsage(TypedDict, total=False):
    input: ProfileTokenStats
    output: ProfileTokenStats
    breakdown: dict[str, ProfileTokenBreakdown]
    reported_total_mismatches: int


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
        self._stage_breakdown: dict[str, dict[str, UsageBreakdownStats]] = {}
        self._stage_usage_calls: dict[str, int] = {}
        self._stage_total_mismatches: dict[str, int] = {}

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

    _BREAKDOWN_PARENTS: ClassVar[Mapping[str, str]] = {
        "reasoning_tokens": "output_tokens",
        "accepted_prediction_tokens": "output_tokens",
        "rejected_prediction_tokens": "output_tokens",
        "cached_tokens": "input_tokens",
    }

    def _accumulate_usage(self, stage_name: str, usage: Mapping[str, object]) -> None:
        """Fold one call's usage into the per-stage statistics.

        Shared by the live path and the resume rebuild.  Those two carried the
        same input/output logic twice, so a field added to one of them would
        have been dropped by the other on the next resume.
        """
        stage_input, stage_output = self._get_or_create_stage_token_stats(stage_name)
        parents = {
            "input_tokens": _usage_count(usage, "input_tokens") or 0,
            "output_tokens": _usage_count(usage, "output_tokens") or 0,
        }
        if parents["input_tokens"] > 0:
            stage_input.update(parents["input_tokens"])
        if parents["output_tokens"] > 0:
            stage_output.update(parents["output_tokens"])

        self._stage_usage_calls[stage_name] = (
            self._stage_usage_calls.get(stage_name, 0) + 1
        )
        breakdown = self._stage_breakdown.setdefault(stage_name, {})
        for field, parent_key in self._BREAKDOWN_PARENTS.items():
            value = _usage_count(usage, field)
            if value is None:
                continue
            stats = breakdown.setdefault(field, UsageBreakdownStats(parent_key))
            stats.update(value, parents[parent_key])
        if _usage_count(usage, "reported_total_tokens") is not None:
            self._stage_total_mismatches[stage_name] = (
                self._stage_total_mismatches.get(stage_name, 0) + 1
            )

    def record_model_usage(
        self,
        usage: dict[str, int] | ModelUsage | None,
        stage_name: str | None = None,
    ) -> None:
        if not usage or not self._profile_usage or not stage_name:
            return
        self._accumulate_usage(stage_name, usage)

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
        # Clear per-stage stats.  The breakdown accumulators have to go too:
        # they are additive, so a rebuild that kept them would double-count
        # every call it then re-reads.
        for stats in self._stage_input_tokens.values():
            stats.clear()
        for stats in self._stage_output_tokens.values():
            stats.clear()
        self._stage_breakdown.clear()
        self._stage_usage_calls.clear()
        self._stage_total_mismatches.clear()

        for ctx in contexts.values():
            if ctx.stage_meta:
                for stage_name, meta_list in ctx.stage_meta.items():
                    for meta in meta_list:
                        # Read usage from model_calls (new structure)
                        model_calls = meta.get("model_calls", [])
                        for call in model_calls:
                            usage = call.get("usage")
                            # Same admission test as the live path, not a
                            # looser one: a usage the live path never counted
                            # must not appear in calls_total after a resume.
                            if isinstance(usage, dict) and usage and stage_name:
                                self._accumulate_usage(stage_name, usage)

    def _breakdown_to_dict(self, stage_name: str) -> dict[str, ProfileTokenBreakdown]:
        calls_total = self._stage_usage_calls.get(stage_name, 0)
        result: dict[str, ProfileTokenBreakdown] = {}
        for field, stats in sorted(self._stage_breakdown.get(stage_name, {}).items()):
            entry = ProfileTokenBreakdown(
                total=stats.total,
                parent=stats.parent_key,
                parent_total=stats.parent_total,
                calls_reporting=stats.calls,
                calls_total=calls_total,
            )
            # Omitted, not zeroed: an undefined ratio is not a measured 0%.
            if stats.share is not None:
                entry["share"] = stats.share
            result[field] = entry
        return result

    def _log_usage_breakdown(self, stage_name: str) -> None:
        """Log the optional counts as shares, with their reporting denominator.

        The denominator is printed on every line rather than only when partial:
        a share whose coverage is stated only sometimes reads as full coverage
        the rest of the time, which is the failure this whole field set exists
        to avoid.  Fields no server reported have no accumulator, so they emit
        no line at all rather than a misleading zero.
        """
        calls_total = self._stage_usage_calls.get(stage_name, 0)
        for field, stats in sorted(self._stage_breakdown.get(stage_name, {}).items()):
            parent = stats.parent_key.removesuffix("_tokens")
            share = stats.share
            # A 0-token parent makes the ratio undefined, and printing "0.0%"
            # there reads as a measurement of no usage rather than no basis.
            ratio = f"{share:.1%} of {parent}" if share is not None else f"no {parent}"
            logger.info(
                "     {}: {:,} ({}; {} of {} calls reporting)",
                field.removesuffix("_tokens"),
                stats.total,
                ratio,
                stats.calls,
                calls_total,
            )
        mismatches = self._stage_total_mismatches.get(stage_name, 0)
        if mismatches:
            logger.info(
                "     reported total differed from prompt+completion on {} of {} calls",
                mismatches,
                calls_total,
            )

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
                        # Breakdowns are shares of the two counts above, so they
                        # are logged after the stage total and never folded in.
                        stage_total = (input_stats.total if input_stats else 0) + (
                            output_stats.total if output_stats else 0
                        )
                        logger.info("     Stage Total: {:,}", stage_total)
                        self._log_usage_breakdown(stage_name)

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
                breakdown = self._breakdown_to_dict(stage_name)
                if breakdown:
                    entry["breakdown"] = breakdown
                mismatches = self._stage_total_mismatches.get(stage_name, 0)
                if mismatches:
                    entry["reported_total_mismatches"] = mismatches
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
