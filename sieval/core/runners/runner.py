"""Single-task execution engine with dual-stream compute/storage architecture."""

import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import anyio
import orjson
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger

from sieval import __version__
from sieval.core.models import ModelOutput
from sieval.core.tasks.anomaly import TaskAnomalyDetector
from sieval.core.tasks.concurrency import (
    compute_stream_buffer_capacity,
    get_limiter_for,
    min_limit,
    prepare_limiters,
)
from sieval.core.tasks.context import (
    TaskAction,
    TaskContext,
    TaskManifest,
    TaskStage,
    TaskStageMeta,
    TaskStageOutput,
)
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.profiler import TaskProfiler
from sieval.core.tasks.progress import TaskProgress
from sieval.core.tasks.saver import TaskSaver
from sieval.core.tasks.task import Task
from sieval.core.types import JSONValue
from sieval.core.utils.concurrency import CompositeLimiter
from sieval.core.utils.meta import build_stage_meta, report_versions

from .resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_reject_message,
    resume_version_verdict,
)

# Type Aliases
type TaskStageMetaHook = Callable[[Any, TaskStage, TaskContext], TaskStageMeta | None]
type TaskStageMetaHooks = dict[TaskStage, TaskStageMetaHook]


class ResultDirExistsError(FileExistsError):
    """Raised when ``result_dir`` exists with persisted data. ``.path`` is
    the offending directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Result directory '{path}' already exists and contains data. "
            "To continue, set `auto_resume=True`; "
            "to start fresh, specify a different `result_dir` "
            "or delete the directory manually."
        )


@dataclass
class TaskRunnerConfig:
    """Configuration for :class:`TaskRunner`.

    Controls execution behavior, persistence, concurrency, profiling, and
    progress reporting for a single task run.

    Attributes:
        result_dir: Output directory for shards, manifest, and report.
            If ``None``, auto-generated under ``outputs/<task_name>/<timestamp>``.
        auto_resume: When ``True``, detect and resume from existing result_dir.
    """

    result_dir: str | None = None
    auto_resume: bool = False

    profile_io: bool = False  # timings
    profile_stages: bool = False  # timings
    profile_usage: bool = True  # tokens

    concurrency_limit: int | None = None
    concurrency_limits: (
        dict[
            TaskAction | Literal["preprocess", "infer", "postprocess", "feedback"], int
        ]
        | None
    ) = None

    shard_samples: int = 1024
    shard_write_concurrency: int = 8
    shard_read_concurrency: int = 8
    write_buffer_size: int = 64
    write_buffer_flush_interval: float = 16.0
    record_each_stage: bool = True
    record_type_metadata: bool = True
    record_meta: bool = True
    stage_meta_hook: TaskStageMetaHook | None = None
    # Per-stage hooks take priority over stage_meta_hook for matching stages;
    # stages not in stage_meta_hooks fall back to stage_meta_hook (if set).
    stage_meta_hooks: TaskStageMetaHooks | None = None

    # None = unbounded iterations. Use only when task.feedback() is guaranteed
    # to eventually return finalize=True; otherwise execution will loop forever.
    max_iterations: int | None = None
    max_retries: int | None = None

    show_progress: bool = True
    progress_log_interval: float = 15.0  # 15s
    progress_log_pct_interval: float = 10.0  # 10%
    dump_progress: bool = True
    progress_dump_interval: float = 1.0  # 1s

    detect_anomalies: bool = True
    detect_anomalies_on_resume: bool = True

    # Metadata-only; engine-level determinism is enforced by backend translators.
    deterministic: bool = False


def gate_resume_version(root_dir: Path, current_version: str) -> None:
    """Refuse to resume across an incompatible sieval version.

    Reads the format-stable ``version`` from ``root_dir/meta.json`` and
    applies :func:`resume_version_verdict`. Missing, unreadable, or
    version-less ``meta.json`` is fail-closed (reject). Raises
    :class:`ResumeVersionError` on reject; logs a warning on a compatible
    non-exact resume; silent on an exact match.
    """
    meta_path = root_dir / "meta.json"
    v_run: object = None
    try:
        meta = orjson.loads(meta_path.read_bytes())
        v_run = meta.get("version") if isinstance(meta, dict) else None
    except (OSError, orjson.JSONDecodeError):
        v_run = None

    if not isinstance(v_run, str):
        raise ResumeVersionError(
            format_reject_message(
                "<missing>",
                current_version,
                "meta.json is missing, unreadable, or has no version",
            )
        )

    verdict = resume_version_verdict(v_run, current_version)
    if verdict.action is ResumeAction.REJECT:
        raise ResumeVersionError(
            format_reject_message(v_run, current_version, verdict.reason)
        )
    if verdict.action is ResumeAction.COMPATIBLE:
        logger.warning(
            "Resuming across compatible sieval versions (persisted={}, current={}); "
            "per-record provenance will record the blend.",
            v_run,
            current_version,
        )


class TaskRunner:
    """Core execution engine for a single evaluation task.

    Implements a **dual-stream architecture**: a *compute stream* runs task
    stages concurrently across samples, while a *storage stream* persists
    results to append-only sharded JSONL files asynchronously.

    Lifecycle::

        arun()
         ├─ Fast Resume check (cached report.json)
         ├─ Load / compensate disk state  (TaskLoader)
         ├─ Hydrate in-memory contexts from shards
         ├─ Seed compute stream (resumed + new samples)
         ├─ Dual-stream loop
         │   ├─ Compute dispatcher  (_compute_dispatcher)
         │   │   └─ _run_one_stage → _execute_stage_logic
         │   └─ Storage worker      (TaskSaver.consume_stream)
         ├─ Finalize: profiler summary, report, anomaly report
         └─ Cleanup: task.shutdown()

    Concurrency is hierarchically throttled::

        global (MultiTaskRunner) → task (concurrency_limit)
            → stage (concurrency_limits) → model (Model._limiter)

    Persistence uses ``record_each_stage`` mode:
        * ``True`` (default): snapshot after each stage for fine-grained resume.
        * ``False``: only FINAL/FAILED + iteration boundaries; lower I/O.
    """

    def __init__(
        self,
        task: Task,
        config: TaskRunnerConfig | None = None,
    ):
        self._task = task
        self._config = config or TaskRunnerConfig()

        # States
        self._contexts: dict[str | int, TaskContext] = {}
        self._hydrated_ids: set[str | int] = set()
        self._total_samples: int = 0
        # {sample_id: {iteration: {rule: [indices]}}}
        # populated in the dispatcher as each FINAL context passes through
        self._anomaly_results: dict[str | int, dict[int, dict[str, list[int]]]] = (
            defaultdict(dict)
        )
        self._auto_resume = self._config.auto_resume
        # Paths
        self._resumed_from_existing, self._root_dir = self._resolve_result_dir(
            self._config.result_dir, task, self._auto_resume
        )
        if self._resumed_from_existing:
            gate_resume_version(self._root_dir, __version__)
            logger.info("Auto resumed from: {}", self._root_dir)

        # Profiling
        self._profiler = TaskProfiler(
            task_name=task.name,
            profile_io=self._config.profile_io,
            profile_stages=self._config.profile_stages,
            profile_usage=self._config.profile_usage,
        )

        # Concurrency & Limits
        self._local_limit = self._config.concurrency_limit
        self._local_stage_limits_cfg = self._normalize_limits(
            self._config.concurrency_limits
        )
        self._stream_buffer_capacity = compute_stream_buffer_capacity(
            self._config.record_each_stage,
            self._local_limit,
            self._local_stage_limits_cfg,
        )
        self._local_limiter: anyio.CapacityLimiter | None = None
        self._local_stage_limiters: dict[str, anyio.CapacityLimiter] = {}

        # Execution Config
        self._max_iterations = self._config.max_iterations
        self._max_retries = self._config.max_retries

        # Persistence Components
        self._record_each_stage = self._config.record_each_stage
        self._loader = TaskLoader(
            task=task,
            root_dir=self._root_dir,
            shard_read_concurrency=self._config.shard_read_concurrency,
            profiler=self._profiler,
        )
        self._saver = TaskSaver(
            root_dir=self._root_dir,
            shard_samples=self._config.shard_samples,
            shard_write_concurrency=self._config.shard_write_concurrency,
            write_buffer_size=self._config.write_buffer_size,
            write_buffer_flush_interval=self._config.write_buffer_flush_interval,
            record_type_metadata=self._config.record_type_metadata,
            record_meta=self._config.record_meta,
            profiler=self._profiler,
            deterministic=self._config.deterministic,
        )

        # Progress Tracking (Initialized in arun)
        self._progress: TaskProgress | None = None
        self._active_counts: dict[str, int] = defaultdict(int)
        self._interrupted: bool = False

        # Anomaly Detection
        self._anomaly_detector = TaskAnomalyDetector(root_dir=self._root_dir)

        # Runtime Context (may be injected by MultiTaskRunner)
        self._shared_global_limiter: anyio.CapacityLimiter | None = None
        self._shared_stage_limiters: dict[str, anyio.CapacityLimiter] = {}
        self._progress_position: int = 0

    def set_runtime_context(
        self,
        shared_global_limiter: anyio.CapacityLimiter | None,
        shared_stage_limiters: dict[str, anyio.CapacityLimiter],
        progress_position: int,
        shared_global_limit: int | None,
        shared_stage_limits: dict[str, int],
    ) -> None:
        """Inject shared concurrency context from :class:`MultiTaskRunner`.

        Merges external (shared) limits with this runner's local limits
        by taking the minimum of each, then recalculates the stream buffer
        capacity for the tighter effective constraints.
        """
        self._shared_global_limiter = shared_global_limiter
        self._shared_stage_limiters = shared_stage_limiters
        self._progress_position = progress_position

        # 1. Effective Global Limit
        effective_global = min_limit(self._local_limit, shared_global_limit)

        # 2. Effective Stage Limits
        effective_stages = {}
        # Merge keys from both configs
        all_stages = set(self._local_stage_limits_cfg.keys()) | set(
            shared_stage_limits.keys()
        )
        for stage in all_stages:
            local_val = self._local_stage_limits_cfg.get(stage)
            shared_val = shared_stage_limits.get(stage)
            effective_stages[stage] = min_limit(local_val, shared_val)

        # 3. Re-calculate buffer capacity using the tighter constraints
        self._stream_buffer_capacity = compute_stream_buffer_capacity(
            self._record_each_stage,
            effective_global,
            effective_stages,
        )

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def run(self) -> Any:
        return anyio.run(self.arun)

    async def arun(self) -> Any:
        """Execute the full task pipeline asynchronously.

        Orchestrates the complete lifecycle: fast resume check → load state
        → hydrate → dual-stream execution → finalize report.  Handles
        ``KeyboardInterrupt`` gracefully by flushing pending writes.

        Returns:
            The report produced by ``task.report()``, or ``None`` if
            interrupted before completion.
        """
        # 0. Check for cached report (Fast Resume)
        # If report.json exists and is valid, we consider the task completed.
        if self._auto_resume:
            cached_report = await self._loader.load_cached_report()
            if cached_report is not None:
                # Check if we need to retry failures
                has_failures, all_final = await self._loader.get_manifest_status()
                if not has_failures and all_final:
                    # Generate anomaly report if needed on resume
                    if (
                        self._config.detect_anomalies
                        and self._config.detect_anomalies_on_resume
                    ):
                        # Load existing report to check if regeneration needed
                        await self._anomaly_detector.load()
                        if self._anomaly_detector.needs_regeneration():
                            logger.info(
                                "Anomaly detection rules changed, "
                                "regenerating anomaly report"
                            )
                            # Load contexts to generate anomaly report
                            self._contexts = await self._loader.load_initial_state()
                            # Hydrate only FINAL and FAILED samples
                            await self._loader.hydrate(
                                self._contexts,
                                self._hydrated_ids,
                                include_stages={TaskStage.FINAL, TaskStage.FAILED},
                                prepare_retries=False,
                                record_each_stage=self._record_each_stage,
                            )
                            # Only detects anomalies in the latest iteration
                            # (loader keeps the highest-iteration offset).
                            await self._anomaly_detector.generate_and_save(
                                self._contexts,
                                self._task.name,
                                task_tags=self._task.tags,
                                backup_if_changed=True,
                            )
                        else:
                            logger.info(
                                "Anomaly detection rules unchanged, "
                                "using cached anomaly report"
                            )
                    return cached_report

        try:
            # 0.5. Lifecycle Setup
            await self._task.setup()

            # Write the version handshake up front so interrupted-then-resumed
            # runs always have a meta.json for the resume gate to read.
            await self._saver.write_run_meta()

            # 1. Load Initial State
            self._contexts = await self._loader.load_initial_state()

            # 2. Determine sample universe (without materializing all raw_samples)
            dataset_size = (
                len(self._task.dataset.test_set) if self._task.dataset.test_set else 0
            )
            pending_ids: list[int] = [
                i for i in range(dataset_size) if i not in self._contexts
            ]
            self._total_samples = max(len(self._contexts), dataset_size)

            # 3. Hydrate Data
            await self._loader.hydrate(
                self._contexts,
                self._hydrated_ids,
                include_stages=(
                    {
                        TaskStage.FINAL,
                        TaskStage.FAILED,
                        TaskStage.INITIAL,
                        TaskStage.PREPROCESSED,
                        TaskStage.INFERRED,
                        TaskStage.POSTPROCESSED,
                        TaskStage.FEEDBACK,
                    }
                    if self._auto_resume
                    else {
                        TaskStage.INITIAL,
                        TaskStage.PREPROCESSED,
                        TaskStage.INFERRED,
                        TaskStage.POSTPROCESSED,
                        TaskStage.FEEDBACK,
                    }
                ),
                prepare_retries=self._auto_resume,
                record_each_stage=self._record_each_stage,
            )

            # 4. Prepare Runtime State
            initial_manifest: dict[str | int, TaskManifest] = {}
            # For exit check
            finals_count = 0
            fails_count = 0
            # For progress init
            completed_ids = []
            failed_ids = []
            anomaly_ids = []
            anomaly_details = defaultdict(int)

            for sid, ctx in self._contexts.items():
                # 4.1. Enforce Retry Limits
                # retry_count is pre-incremented by loader._prepare_failed_retries
                # before each resume attempt, so that runner can compare
                # retry_count > max_retries: retry_count=1 means "on retry #1",
                # so max_retries=1 allows exactly one retry.
                if (
                    self._auto_resume
                    and self._max_retries is not None
                    and not ctx.is_terminal()
                    and ctx.retry_count > self._max_retries
                ):
                    logger.warning(
                        "Sample {} exceeded max retries ({} > {}). Marking as FAILED.",
                        sid,
                        ctx.retry_count,
                        self._max_retries,
                    )
                    # Pipeline update: mutate local variable `ctx` and update storage
                    ctx = ctx.to_failed(
                        None, "retry_limit", f"Max retries {self._max_retries} reached"
                    )
                    self._contexts[sid] = ctx

                # 4.2. Sync Manifest
                entry: TaskManifest = {
                    "sample_id": ctx.sample_id,
                    "stage": ctx.stage.value,
                    "iteration": ctx.iteration,
                    "final": ctx.stage == TaskStage.FINAL,
                    "failed": ctx.stage == TaskStage.FAILED,
                }
                if ctx.error_action:
                    entry["error_action"] = ctx.error_action.value
                if ctx.error_reason:
                    entry["error_reason"] = ctx.error_reason
                if ctx.retry_count:
                    entry["retry_count"] = ctx.retry_count
                initial_manifest[sid] = entry

                # 4.3. Collect Stats for Exit Check & Progress
                if ctx.is_terminal():
                    completed_ids.append(sid)
                    if ctx.stage == TaskStage.FINAL:
                        finals_count += 1
                        # Check for anomalies in successful runs
                        anomalies = self._anomaly_detector.detect(
                            ctx, task_tags=self._task.tags
                        )
                        if anomalies:
                            anomaly_ids.append(sid)
                            for rule in anomalies:
                                anomaly_details[rule] += 1
                            # Rebuild _anomaly_results to include pre-resume samples
                            self._anomaly_results[ctx.sample_id][ctx.iteration] = {
                                rule: sorted(indices)
                                for rule, indices in anomalies.items()
                            }
                    elif ctx.stage == TaskStage.FAILED:
                        fails_count += 1
                        failed_ids.append(sid)

            # Sync Saver Manifest
            self._saver.sync_manifest(initial_manifest)

            # 5. Check for Early Exit
            total_count = self._total_samples
            if total_count == 0 or (finals_count + fails_count == total_count):
                # Hydrate results for report generation
                await self._loader.hydrate(
                    self._contexts,
                    self._hydrated_ids,
                    include_stages={TaskStage.FINAL, TaskStage.FAILED},
                    prepare_retries=False,
                    record_each_stage=self._record_each_stage,
                )
                # Re-fetch finals and fails fully hydrated
                finals, fails = self._final_and_failed()
                report = await self._task.report(finals, fails)
                # Always save report on completion
                await self._save_report_with_versions(report, finals, fails)
                return report

            # 6. Setup Progress Tracker
            self._progress = TaskProgress(
                total=self._total_samples,
                desc=f"Running {self._task.name}",
                position=self._progress_position,
                show_progress=self._config.show_progress,
                log_interval=self._config.progress_log_interval,
                log_pct_interval=self._config.progress_log_pct_interval,
                root_dir=self._root_dir,
                dump_progress=self._config.dump_progress,
                dump_interval=self._config.progress_dump_interval,
            )
            self._progress.init_state(
                completed_ids,
                failed_ids=failed_ids,
                anomaly_ids=anomaly_ids,
                anomaly_details=anomaly_details,
            )

            # 7. Execution Loop (Dual Streams)
            compute_send, compute_recv = anyio.create_memory_object_stream(
                self._stream_buffer_capacity
            )
            storage_send, storage_recv = anyio.create_memory_object_stream(
                self._stream_buffer_capacity
            )

            async with anyio.create_task_group() as tg:
                # Setup Concurrency: Always create local limiters
                # This ensures task-specific limits are always respected
                self._local_limiter, self._local_stage_limiters = prepare_limiters(
                    self._local_limit, self._local_stage_limits_cfg
                )

                # Start Storage Worker
                tg.start_soon(self._saver.consume_stream, storage_recv)

                # Start Compute Dispatcher
                tg.start_soon(
                    self._compute_dispatcher,
                    compute_recv,
                    compute_send,
                    storage_send,
                    tg,
                )

                # Seed resumed non-terminal contexts (backfill raw_sample if needed)
                for ctx in self._contexts.values():
                    if not ctx.is_terminal():
                        if ctx.raw_sample is None:
                            ctx = self._ensure_raw_sample(ctx)
                            self._contexts[ctx.sample_id] = ctx
                        await compute_send.send(ctx)

                # Seed new pending samples — create context on demand
                for sid in pending_ids:
                    ctx = self._task.make_context(sid)
                    self._contexts[sid] = ctx
                    await compute_send.send(ctx)

            # 8. Finalize
            self._profiler.aggregate_stage_timings(self._contexts)
            self._profiler.aggregate_token_usage(self._contexts)
            self._profiler.log_summary()
            await self._profiler.save(self._root_dir)

            finals, fails = self._final_and_failed()
            report = await self._task.report(finals, fails)
            # Always save report on completion
            await self._save_report_with_versions(report, finals, fails)
            # Backstop: create-if-absent, normally a no-op (written at run start).
            await self._saver.write_run_meta()
            # Generate anomaly report
            if self._config.detect_anomalies:
                await self._anomaly_detector.generate_and_save_from_results(
                    self._anomaly_results,
                    self._task.name,
                    total_samples=self._total_samples,
                    final_count=len(finals),
                    failed_count=len(fails),
                    backup_if_changed=False,
                )

            if self._progress:
                self._progress.close()
            return report

        except (KeyboardInterrupt, anyio.get_cancelled_exc_class()):
            if not self._interrupted:
                self._interrupted = True
                logger.warning("Interrupted. Saving progress...")
                try:
                    with anyio.CancelScope(shield=True):
                        await self._saver.flush()
                        await self._saver.write_run_meta()
                        # Generate anomaly report on interrupt
                        if self._config.detect_anomalies:
                            finals_i, fails_i = self._final_and_failed()
                            await self._anomaly_detector.generate_and_save_from_results(
                                self._anomaly_results,
                                self._task.name,
                                total_samples=self._total_samples,
                                final_count=len(finals_i),
                                failed_count=len(fails_i),
                                backup_if_changed=False,
                            )
                except Exception as e:
                    logger.error("Flush failed: {}", e)
            if self._progress:
                self._progress.close()
            return None

        finally:
            # X. Lifecycle Shutdown
            await self._task.shutdown()

    async def _compute_dispatcher(
        self,
        recv_stream: MemoryObjectReceiveStream[TaskContext],
        compute_send: MemoryObjectSendStream[TaskContext],
        storage_send: MemoryObjectSendStream[TaskContext],
        tg: TaskGroup,
    ) -> None:
        """Central dispatch loop for the compute stream.

        Reads contexts (initial seeds and completed stage results) from
        *recv_stream*, forwards terminal contexts to the storage stream,
        updates progress, detects anomalies, and schedules the next stage
        for non-terminal contexts via the task group.  Closes all streams
        once every sample reaches a terminal state.
        """
        # Count initially active tasks to know when to stop
        total_tasks = self._total_samples
        completed_tasks = sum(1 for c in self._contexts.values() if c.is_terminal())

        async with recv_stream, compute_send, storage_send:
            async for ctx in recv_stream:
                # Update local state
                self._contexts[ctx.sample_id] = ctx
                self._hydrated_ids.add(ctx.sample_id)

                terminal = ctx.is_terminal()
                if terminal:
                    await storage_send.send(ctx)
                    # Store anomaly results (not context) to avoid retaining
                    # full context objects across iterations.
                    if ctx.stage == TaskStage.FINAL:
                        anomalies = self._anomaly_detector.detect(
                            ctx, task_tags=self._task.tags
                        )
                        if anomalies:
                            self._anomaly_results[ctx.sample_id][ctx.iteration] = {
                                rule: sorted(indices)
                                for rule, indices in anomalies.items()
                            }
                else:
                    if self._record_each_stage and ctx.stage != TaskStage.INITIAL:
                        await storage_send.send(ctx.make_snapshot())
                    elif (
                        not self._record_each_stage
                        and ctx.stage == TaskStage.INITIAL
                        and ctx.iteration > 0
                    ):
                        # Iteration boundary: save full context so that failed
                        # samples can resume from the last completed iteration
                        # instead of restarting from scratch.
                        # Must be full context (not snapshot) because INITIAL
                        # stage has no result_field in STAGE_TO_RESULT_FIELD,
                        # so a snapshot would lose all result data.
                        await storage_send.send(ctx)

                # Update Progress
                if terminal and self._progress:
                    is_failed = ctx.stage == TaskStage.FAILED
                    # For progress tracking, only pass rule names (not indices)
                    anomaly_rules = None
                    if not is_failed and ctx.sample_id in self._anomaly_results:
                        iter_results = self._anomaly_results[ctx.sample_id].get(
                            ctx.iteration
                        )
                        if iter_results:
                            anomaly_rules = set(iter_results.keys())
                    self._progress.update(
                        sample_id=ctx.sample_id,
                        current_hydrated_count=len(self._hydrated_ids),
                        failed=is_failed,
                        anomalies=anomaly_rules,
                    )
                elif self._progress:
                    self._progress.tick(len(self._hydrated_ids))

                # Check global completion
                if terminal:
                    completed_tasks += 1
                    if completed_tasks >= total_tasks:
                        break
                    continue

                # Schedule next stage
                tg.start_soon(self._run_one_stage, ctx, compute_send)

    async def _run_one_stage(
        self, ctx: TaskContext, compute_send: MemoryObjectSendStream[TaskContext]
    ) -> None:
        """Acquire concurrency limiters and execute a single pipeline stage.

        Applies both the shared (global) and local (task-level) capacity
        limiters for the stage's action, tracks active counts for progress
        display, then delegates to :meth:`_execute_stage_logic`.
        """
        action = ctx.next_action()
        if not action:
            return

        shared_limiter = get_limiter_for(
            action, self._shared_global_limiter, self._shared_stage_limiters
        )
        local_limiter = get_limiter_for(
            action, self._local_limiter, self._local_stage_limiters
        )
        async with CompositeLimiter(shared_limiter, local_limiter):
            self._active_counts[action.value] += 1
            if self._progress:
                self._progress.set_status(self._active_counts, len(self._hydrated_ids))
            try:
                await self._execute_stage_logic(ctx, compute_send, action)
            finally:
                self._active_counts[action.value] -= 1
                if self._progress:
                    self._progress.set_status(
                        self._active_counts, len(self._hydrated_ids)
                    )

    async def _execute_stage_logic(
        self,
        ctx: TaskContext,
        compute_send: MemoryObjectSendStream[TaskContext],
        action: TaskAction,
    ) -> None:
        """Route to the appropriate task method and build the next context.

        Calls ``task.preprocess / infer / postprocess / feedback`` according
        to *action*, collects timing and metadata (auto + user-provided +
        hook), records token usage, and sends the resulting context back to
        the compute stream.  On exception, transitions the context to FAILED.
        """
        should_measure_time = (
            self._profiler.should_profile_stages() or self._config.record_meta
        )
        start_time = time.perf_counter() if should_measure_time else None

        def _finalize_timing() -> float | None:
            if start_time is None:
                return None
            return time.perf_counter() - start_time

        def _resolve(r: Any) -> tuple[Any, TaskStageMeta | None]:
            # Stage-to-stage value is transparent: return what user returns.
            # If a box is returned, meta is extracted, and the box is preserved.
            if isinstance(r, TaskStageOutput):
                return r, r.meta
            return r, None

        def _build_auto_meta(
            timing_s: float | None,
            inner: Any,
        ) -> TaskStageMeta:
            # Handle single `ModelOutput` and `list[ModelOutput]`
            if isinstance(inner, ModelOutput):
                return build_stage_meta(inner, timing_s=timing_s)
            elif (
                isinstance(inner, list)
                and inner
                and all(isinstance(item, ModelOutput) for item in inner)
            ):
                return build_stage_meta(*inner, timing_s=timing_s)
            else:
                return build_stage_meta(timing_s=timing_s)

        def _get_stage_meta_hook(stage: TaskStage) -> TaskStageMetaHook | None:
            hooks = self._config.stage_meta_hooks
            if hooks and stage in hooks:
                return hooks[stage]
            # Fall back to the global hook for stages not covered by stage_meta_hooks.
            return self._config.stage_meta_hook

        def _run_stage(
            stage: TaskStage,
            raw_result: Any,
            finalize: bool | None = None,
        ) -> TaskContext:
            # 1. Resolve value and optional user meta (box preserved)
            val, out_meta = _resolve(raw_result)
            timing_s = _finalize_timing()
            inner = val.value if isinstance(val, TaskStageOutput) else val

            # 2. Auto meta always recorded; user meta merges on top
            meta = _build_auto_meta(timing_s, inner)

            # 3. Merge user-provided meta then hook
            if out_meta:
                meta.update(out_meta)
            hook = _get_stage_meta_hook(stage)
            if hook:
                hook_meta = hook(val, stage, ctx)
                if hook_meta:
                    meta.update(hook_meta)

            # 4. Record usage from model_calls
            model_calls = meta.get("model_calls", [])
            for call in model_calls:
                if call.get("usage"):
                    self._profiler.record_model_usage(
                        call["usage"], stage_name=stage.value
                    )

            # 5. Build next context state for this stage
            if stage == TaskStage.PREPROCESSED:
                return ctx.to_preprocessed(val, meta=meta)
            if stage == TaskStage.INFERRED:
                return ctx.to_inferred(val, meta=meta)
            if stage == TaskStage.POSTPROCESSED:
                return ctx.to_postprocessed(val, meta=meta)
            # FEEDBACK handled separately
            if stage == TaskStage.FEEDBACK:
                next_ctx = ctx.to_feedback(val, meta=meta)
                if finalize:
                    return next_ctx.to_final()
                if (
                    self._max_iterations is not None
                    and ctx.iteration + 1 >= self._max_iterations
                ):
                    return next_ctx.to_failed(
                        None,
                        "iteration_limit",
                        f"Max iterations {self._max_iterations} reached",
                    )
                return next_ctx.iterate()
            return ctx

        try:
            match action:
                case TaskAction.PREPROCESS:
                    r = await self._task.preprocess(ctx.raw_sample, ctx)
                    new_ctx = _run_stage(TaskStage.PREPROCESSED, r)

                case TaskAction.INFER:
                    r = await self._task.infer(ctx.preprocess_result, ctx)
                    new_ctx = _run_stage(TaskStage.INFERRED, r)

                case TaskAction.POSTPROCESS:
                    r = await self._task.postprocess(ctx.infer_result, ctx)
                    new_ctx = _run_stage(TaskStage.POSTPROCESSED, r)

                case TaskAction.FEEDBACK:
                    finalize, fb = await self._task.feedback(
                        ctx.postprocess_result, ctx
                    )
                    new_ctx = _run_stage(TaskStage.FEEDBACK, fb, finalize=finalize)

            await compute_send.send(new_ctx)

        except Exception as e:
            logger.opt(exception=True).error(
                "Stage {} failed for {}", action, ctx.sample_id
            )
            new_ctx = ctx.to_failed(
                action, f"exception::{e.__class__.__name__}", str(e)
            )
            await compute_send.send(new_ctx)

    def _ensure_raw_sample(self, ctx: TaskContext) -> TaskContext:
        if ctx.raw_sample is not None:
            return ctx
        sid = ctx.sample_id
        test_set = self._task.dataset.test_set
        if test_set and isinstance(sid, int) and 0 <= sid < len(test_set):
            return replace(ctx, raw_sample=test_set[sid])
        return ctx

    def _final_and_failed(self) -> tuple[list[TaskContext], list[TaskContext]]:
        finals = [c for c in self._contexts.values() if c.stage == TaskStage.FINAL]
        fails = [c for c in self._contexts.values() if c.stage == TaskStage.FAILED]
        return finals, fails

    async def _save_report_with_versions(
        self,
        report: JSONValue,
        finals: list[TaskContext],
        fails: list[TaskContext],
    ) -> None:
        """Inject the distinct producing-version list into a dict report, then save.

        Aggregates over the in-memory terminal contexts' ``stage_meta`` at zero
        extra I/O; :func:`report_versions` owns the ``"unknown"`` sentinel rule.
        Non-dict reports are saved unchanged; ``None`` skips the save.
        """
        if report is None:
            return
        if isinstance(report, dict):
            cast(dict[str, JSONValue], report)["sieval_versions"] = report_versions(
                (c.stage_meta for c in finals),
                (c.stage_meta for c in fails),
            )
        await self._saver.save_report(report)

    def _resolve_result_dir(
        self, result_dir: str | None, task: Task, auto_resume: bool
    ) -> tuple[bool, Path]:
        if result_dir:
            p = Path(result_dir).expanduser()
            if p.exists() and p.is_file():
                raise ValueError("result_dir is a file")
            if p.exists() and (p / "manifest.json").exists():
                if not auto_resume:
                    raise ResultDirExistsError(p)
                return True, p
            return False, p
        root = Path("outputs") / task.name
        if auto_resume and root.exists():
            candidates = [
                d
                for d in root.iterdir()
                if d.is_dir() and re.fullmatch(r"\d{14}", d.name)
            ]
            if candidates:
                latest = max(candidates, key=lambda x: x.name)
                if (latest / "manifest.json").exists():
                    return True, latest
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return False, root / ts

    def _normalize_limits(self, raw_limits: dict | None) -> dict[str, int]:
        raw_limits = raw_limits or {}
        norm = {}
        valid_values = {a.value for a in TaskAction}
        for k, v in raw_limits.items():
            if isinstance(k, TaskAction):
                norm[k.value] = v
            elif k in valid_values:
                norm[k] = v
            else:
                logger.warning(
                    "Unknown stage key {!r} in concurrency_limits - ignored. "
                    "Valid keys: {}",
                    k,
                    sorted(valid_values),
                )
        return norm
