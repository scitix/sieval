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
import anyio.to_thread
import orjson
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger

from sieval import __version__
from sieval.core.datasets import repeat_index_of
from sieval.core.models import ModelOutput
from sieval.core.tasks.anomaly import TaskAnomalyDetector
from sieval.core.tasks.concurrency import (
    compute_stream_buffer_capacity,
    get_limiter_for,
    min_limit,
    prepare_limiters,
)
from sieval.core.tasks.consts import NON_RETRIABLE_REASON, NonRetriableSampleError
from sieval.core.tasks.context import (
    TaskAction,
    TaskContext,
    TaskManifest,
    TaskStage,
    TaskStageMeta,
    TaskStageOutput,
)
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.meta import EvalMode, get_task_run_identity
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.core.tasks.profiler import TaskProfiler
from sieval.core.tasks.progress import TaskProgress
from sieval.core.tasks.records import iter_grader_outputs
from sieval.core.tasks.run_meta import (
    ABSENT,
    DigestedStage,
    GraderIdentity,
    InferenceIdentity,
    JudgeIdentity,
    RunFrom,
    aggregate_digest,
    call_key,
    digest,
    empty_identity,
    grader_identity,
    grader_key,
    grader_modules,
    inference_identity,
    judge_identities,
    prompt_of,
    request_digest,
    validate_judge_model_ref,
)
from sieval.core.tasks.saver import TaskSaver
from sieval.core.tasks.services import service_identities
from sieval.core.tasks.task import Task
from sieval.core.types import JSONValue
from sieval.core.utils.concurrency import CompositeLimiter
from sieval.core.utils.meta import (
    build_model_call_meta_from_mapping,
    build_stage_meta,
    count_scored_rollouts,
    count_truncated_rollouts,
    report_versions,
)

from .resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_reject_message,
    resume_version_verdict,
)
from .stage_resume import (
    LegacyStageMatch,
    StageResumeError,
    copy_recorded_dir,
    has_graded,
    recorded_identity,
    recorded_sample_digests,
    remove_copied,
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


class UnreadableIntervalError(ValueError):
    """A finished report publishes an interval no reader can use.

    Its own type so the completion path can tell THIS refusal apart from any
    other failure and finish writing the run's remaining artifacts before it
    surfaces -- see :meth:`TaskRunner._save_report_with_diagnostics`. A
    ``ValueError`` subclass because that is what the check raised before it had a
    name, so nothing catching the broader shape changes behaviour.

    Every other exception still aborts finalization where it is raised: this is
    deferred only because the run itself is complete and its artifacts are worth
    having, not because the failure is tolerable.
    """


@dataclass
class TaskRunnerConfig:
    """Configuration for :class:`TaskRunner`.

    Controls execution behavior, persistence, concurrency, profiling, and
    progress reporting for a single task run.

    Attributes:
        result_dir: Output directory for shards, manifest, and report.
            If ``None``, auto-generated under ``outputs/<task_name>/<timestamp>``.
        auto_resume: When ``True``, detect and resume from existing result_dir.
        resume_from: A recorded task directory, possibly another task's and
            read-only, whose shards are copied into an empty ``result_dir`` and
            resumed (see :mod:`sieval.core.runners.stage_resume`). Implies
            ``auto_resume``; ignored once ``result_dir`` holds a run, so the
            same config resumes an interrupted copy.
    """

    result_dir: str | None = None
    auto_resume: bool = False
    resume_from: str | None = None

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


def _read_run_meta(root_dir: Path) -> dict | None:
    """Return the parsed ``root_dir/meta.json``, or ``None`` if unusable.

    Sole reader: missing, unreadable and non-mapping all collapse to ``None``,
    and what that means is the caller's call — fail-closed for
    :func:`gate_resume_version`, unknown identities for a per-stage resume.
    """
    try:
        meta = orjson.loads((root_dir / "meta.json").read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return meta if isinstance(meta, dict) else None


def run_started(root_dir: Path) -> bool:
    """Whether a run has started in *root_dir*, so a new one would resume it.

    Read from ``meta.json``'s ``runs[]``, which a run appends before it does any
    work: a run killed before its first ``manifest.json`` flush still counts. A
    directory written before ``runs[]`` existed falls back to ``manifest.json``,
    the signal those versions used.
    """
    meta = _read_run_meta(root_dir)
    if meta is not None and isinstance(meta.get("runs"), list) and meta["runs"]:
        return True
    return (root_dir / "manifest.json").exists()


def read_run_version(root_dir: Path) -> str | None:
    """Return the sieval version in ``root_dir/meta.json``, or ``None``.

    Public for the CLI, which explains a cross-version resume in an abort it
    raises before any runner exists. One reader, so its message cannot come to
    describe a different file than :func:`gate_resume_version` read.

    ``None`` covers every unusable case: no file, unreadable, not a mapping, no
    ``version``, or a non-string one.
    """
    meta = _read_run_meta(root_dir)
    if meta is None:
        return None
    version = meta.get("version")
    return version if isinstance(version, str) else None


def gate_resume_version(root_dir: Path, current_version: str) -> None:
    """Refuse to resume across an incompatible sieval version.

    Reads the format-stable ``version`` from ``root_dir/meta.json`` via
    :func:`read_run_version` and applies :func:`resume_version_verdict`.
    Missing, unreadable, or version-less ``meta.json`` is fail-closed
    (reject). Raises :class:`ResumeVersionError` on reject; logs a warning on
    a compatible non-exact resume; silent on an exact match.
    """
    v_run = read_run_version(root_dir)
    if v_run is None:
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


def _reuses_inference(ctx: TaskContext) -> bool:
    """Whether a recorded sample's state holds an inference a resume would keep.

    Read from the manifest-level stage, before hydration.
    """
    if ctx.stage is TaskStage.FAILED:
        return ctx.error_action not in (TaskAction.PREPROCESS, TaskAction.INFER)
    return ctx.stage not in (TaskStage.INITIAL, TaskStage.PREPROCESSED)


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
        *,
        inference_recipe: dict[str, JSONValue] | None = None,
        resume_config_diff: list[str] | None = None,
        legacy_match: LegacyStageMatch | None = None,
    ):
        """*inference_recipe* is the serving plan of the model under test,
        without its checkpoint, for the inference identity; ``None`` when the
        model is not served by SiEval. *resume_config_diff* is the config
        change a resume made, for ``runs[].config``. *legacy_match* compares a
        recorded run without identities through its persisted config.
        """
        # A malformed judge reference would leave every grader identity
        # unknown; refuse it before any work.
        validate_judge_model_ref()
        self._task = task
        self._config = config or TaskRunnerConfig()
        self._inference_recipe = inference_recipe
        self._resume_config_diff = list(resume_config_diff or [])
        self._legacy_match = LegacyStageMatch(**(legacy_match or {}))
        # Per-stage resume: what was copied in, and what the comparison chose.
        self._run_from: RunFrom | None = None
        self._copied: list[Path] = []
        self._recorded: dict | None = None
        self._regrade = False
        self._grader_stamp: str | None = None
        self._data_match: bool | None = None
        self._resume_plan: dict[str, int] = {}
        # Run history: actions executed this run, each sample's data and prompt
        # digests as of its first preprocess, and judge versions the API
        # reported, by role.
        self._ran_actions: set[TaskAction] = set()
        self._sample_digests: dict[
            str | int, tuple[int | None, str | None, str | None]
        ] = {}
        self._input_identity_written = False
        self._judge_versions: dict[str, str] = {}
        self._judge_roles: dict[str, list[str]] | None = None

        # States
        self._contexts: dict[str | int, TaskContext] = {}
        self._hydrated_ids: set[str | int] = set()
        self._total_samples: int = 0
        # {sample_id: {iteration: {rule: [indices]}}}
        # populated in the dispatcher as each FINAL context passes through
        self._anomaly_results: dict[str | int, dict[int, dict[str, list[int]]]] = (
            defaultdict(dict)
        )
        self._auto_resume = self._config.auto_resume or bool(self._config.resume_from)
        # Paths
        self._resumed_from_existing, self._root_dir = self._resolve_result_dir(
            self._config.result_dir, task, self._auto_resume
        )
        # Read off the task, not looked up by name: `task.name` is a
        # user-chosen YAML key, not the registered `@sieval_task(name=...)`.
        self._task_identity = get_task_run_identity(task)
        if self._resumed_from_existing:
            gate_resume_version(self._root_dir, __version__)
            self._recorded = _read_run_meta(self._root_dir)
            logger.info("Auto resumed from: {}", self._root_dir)
        elif self._config.resume_from:
            self._prepare_resume_from(Path(self._config.resume_from).expanduser())

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
            task_identity=self._task_identity,
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
            if cached_report is not None and not await self._last_run_ended():
                # The run that wrote this report never ended (`end_run`
                # failed, or a later run stopped short): finalize again
                # through the full path, which reruns `report()` on the
                # shards instead of trusting the cached one.
                logger.info(
                    "report.json found, but the run that wrote it never ended; "
                    "finalizing again"
                )
                cached_report = None
            if cached_report is not None and self._run_from is not None:
                # A copied directory always runs: its report is another task's.
                cached_report = None
            if cached_report is not None and not self._recorded_stages_match():
                logger.info(
                    "report.json found, but its run's stage identities differ "
                    "from this run's; resuming stage by stage"
                )
                cached_report = None
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
                    # Nothing reran, so the identity stays the one that
                    # produced this report; only the run is recorded.
                    await self._saver.begin_run(
                        resumed=True,
                        model=self._model_name(),
                        config_diff=self._resume_config_diff,
                    )
                    await self._saver.end_run(())
                    return cached_report

        try:
            # 0.5. Lifecycle Setup
            await self._task.setup()

            if self._recorded is not None:
                self._seed_judge_versions(self._recorded)
            # Record the run before any work, so an interrupted run is resumed
            # and the resume gates have a meta.json to read. Inference and
            # grader identities are known now; data and prompt once every
            # sample is past preprocess.
            inference = self._inference_identity()
            grader = self._grader_identity()
            self._grader_stamp = grader_key(grader)
            resumed = self._resumed_from_existing or self._run_from is not None
            if resumed:
                # Read the recorded state first: a refusal must leave the
                # recorded meta.json as it was.
                self._contexts = await self._loader.load_initial_state()
                if any(_reuses_inference(c) for c in self._contexts.values()):
                    self._check_inference(inference)
                self._regrade = self._recorded_grader_stamp() != self._grader_stamp
            # A regrade leaves the grader unknown until the run ends: samples it
            # has not reached yet were graded by the recorded grader.
            await self._saver.begin_run(
                resumed=resumed,
                run_from=self._run_from,
                model=self._model_name(),
                config_diff=self._resume_config_diff if resumed else (),
                data=None,
                prompt=None,
                inference=inference,
                grader=None if self._regrade else grader,
            )

            # 1. Load Initial State
            if not resumed:
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
                prepare_retries=self._auto_resume and not resumed,
                record_each_stage=self._record_each_stage,
            )
            # Decide per sample what a resume keeps before failures roll back,
            # so a sample that failed in grading regrades rather than reinfers.
            changed: list[str | int] = []
            if resumed:
                changed = await self._plan_stage_resume()
                self._loader.prepare_failed_retries(
                    self._contexts, self._record_each_stage
                )
            self._total_samples = max(len(self._contexts), dataset_size)

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
            if changed:
                # Rewritten whole as this run's records, which supersede the
                # recorded ones on load.
                await self._saver.write_records(self._contexts[sid] for sid in changed)

            for ctx in self._contexts.values():
                if ctx.preprocess_result is not None or ctx.is_terminal():
                    self._note_sample_digests(ctx)
            await self._maybe_write_input_identity()

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
                reported = await self._save_report_with_diagnostics(
                    report, finals, fails
                )
                await self._end_run(reported=reported)
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
            # Outlives the storage worker's own receiver, so what was sent to
            # storage but never received can still be written.
            storage_leftover = storage_recv.clone()

            try:
                await self._run_streams(
                    compute_send, compute_recv, storage_recv, storage_send, pending_ids
                )
            finally:
                # A stage that raised (a failed `meta.json` write) cancels the
                # storage worker before its final flush; samples already done,
                # whether buffered by it or still in the stream, reach disk. A
                # no-op after a clean exit.
                with anyio.CancelScope(shield=True):
                    try:
                        await self._saver.drain(storage_leftover)
                    except Exception as exc:
                        logger.error("Flush after the run stopped failed: {}", exc)

            # 8. Finalize
            self._profiler.aggregate_stage_timings(self._contexts)
            self._profiler.aggregate_token_usage(self._contexts)
            self._profiler.log_summary()
            await self._profiler.save(self._root_dir)

            finals, fails = self._final_and_failed()
            report = await self._task.report(finals, fails)
            # Always save report on completion
            refusal: UnreadableIntervalError | None = None
            reported = False
            try:
                reported = await self._save_report_with_diagnostics(
                    report, finals, fails
                )
            except UnreadableIntervalError as exc:
                # HELD, not swallowed. `report.json` is already on disk and the run
                # itself finished, so the remaining artifacts are written before
                # this surfaces -- refusing an unreadable declaration must not also
                # cost the run its `anomalies.json`, which is the evidence a reader
                # of the failure needs. Re-raised unchanged below; only this one
                # type is deferred, so every other failure still aborts here.
                refusal = exc
            if refusal is None:
                await self._end_run(reported=reported)
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
            if refusal is not None:
                raise refusal
            return report

        except (KeyboardInterrupt, anyio.get_cancelled_exc_class()):
            if not self._interrupted:
                self._interrupted = True
                logger.warning("Interrupted. Saving progress...")
                try:
                    with anyio.CancelScope(shield=True):
                        await self._saver.flush()
                        await self._saver.interrupt_run()
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

    async def _run_streams(
        self,
        compute_send: MemoryObjectSendStream[TaskContext],
        compute_recv: MemoryObjectReceiveStream[TaskContext],
        storage_recv: MemoryObjectReceiveStream[TaskContext],
        storage_send: MemoryObjectSendStream[TaskContext],
        pending_ids: list[int],
    ) -> None:
        """Run the compute dispatcher and the storage worker until every sample
        is terminal, seeding resumed and new samples."""
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
                # A stage whose value is a judgement record may still have called
                # a model: an LLM grader runs inside `feedback`, which returns a
                # record, not a ModelOutput. Its flattened output is recorded on
                # the record, so read the calls back from there -- otherwise
                # grader spend is on disk but missing from profile.json.
                return build_stage_meta(
                    timing_s=timing_s,
                    model_calls=[
                        call
                        for output in iter_grader_outputs(inner)
                        if (call := build_model_call_meta_from_mapping(output))
                        is not None
                    ],
                )

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
            if stage == TaskStage.PREPROCESSED:
                meta["identity"] = {
                    "data": self._safe_digest(ctx.raw_sample),
                    "prompt": self._safe_digest(prompt_of(inner)),
                }
            elif stage in (TaskStage.POSTPROCESSED, TaskStage.FEEDBACK):
                # Per sample, so a resume can tell which grader graded it.
                meta["identity"] = {"grader": self._grader_stamp}

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
            # Any stage: agieval's extractor is a judge called from `infer`.
            self._observe_judge_versions(model_calls)
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

        self._ran_actions.add(action)
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

        except Exception as e:
            logger.opt(exception=True).error(
                "Stage {} failed for {}", action, ctx.sample_id
            )
            # A task raising `NonRetriableSampleError` has declared the outcome
            # deterministic in this sample's own input, so it earns the one reason
            # the loader will not roll back.
            reason = (
                NON_RETRIABLE_REASON
                if isinstance(e, NonRetriableSampleError)
                else f"exception::{e.__class__.__name__}"
            )
            new_ctx = ctx.to_failed(action, reason, str(e))

        # Outside the handler above: a failed `meta.json` write fails the task,
        # not this sample.
        if action is TaskAction.PREPROCESS:
            self._note_sample_digests(new_ctx)
            await self._maybe_write_input_identity()
        await compute_send.send(new_ctx)

    def _ensure_raw_sample(self, ctx: TaskContext) -> TaskContext:
        if ctx.raw_sample is not None:
            return ctx
        sid = ctx.sample_id
        test_set = self._task.dataset.test_set
        if test_set and isinstance(sid, int) and 0 <= sid < len(test_set):
            raw = test_set[sid]
            # Stamped here too: this is the second seam that attaches a row, and one
            # attaching it without the copy number serializes as never-repeated.
            return replace(ctx, raw_sample=raw, repeat_index=repeat_index_of(raw))
        return ctx

    def _final_and_failed(self) -> tuple[list[TaskContext], list[TaskContext]]:
        finals = [c for c in self._contexts.values() if c.stage == TaskStage.FINAL]
        fails = [c for c in self._contexts.values() if c.stage == TaskStage.FAILED]
        return finals, fails

    async def _save_report_with_diagnostics(
        self,
        report: JSONValue,
        finals: list[TaskContext],
        fails: list[TaskContext],
    ) -> bool:
        """Inject the run-level diagnostics into a dict report, then save.

        All aggregate over the in-memory terminal contexts' ``stage_meta`` at zero
        extra I/O -- :func:`report_versions` owns the ``"unknown"`` sentinel rule,
        the rollout counters :func:`_scored_rollout_indices`. Injected here rather
        than per task because none is a fact about a task's metric: every task
        would have to remember, and the ones that forgot would look clean.

        ``n_truncated`` and its denominator ``n_scored_rollouts`` go in for ``gen``
        tasks only, and are omitted rather than zeroed elsewhere -- ``ppl``/``clp``
        infer at ``max_tokens=1`` and so finish every call with ``length``, which
        would make the count equal the total and mean nothing. The anomaly rule is
        ``gen``-scoped for the same reason. They go in as a **pair or not at all**:
        a count whose base is missing is not a share of anything, and both drop out
        when any scored record is unmeasurable (see the ``record_meta=False`` resume
        case on ``_scored_rollout_indices``).

        Only finals are counted, matching both the anomaly detector (which runs on
        FINAL) and ``n_unextracted``: a FAILED sample has no score for a truncation
        to explain.

        Then the finished dict's interval declarations are checked, and a report
        that publishes an interval a reader cannot use RAISES -- see
        :func:`interval_declaration_problems`. Here because this is the one place
        every report passes through whole: a task assembles its report out of
        fragments, and a plain ``|`` between two of them drops one's ``ci95_units``
        silently, which no source scan can see (the per-metric keys are built from
        metric names, not literals).

        **Saved first, then refused.** The artifacts of a finished run are worth
        keeping for inspection, and the ordering makes the failure loud without
        making it destructive; a log line would be ignorable, which is what let
        every earlier undeclared-key defect ship.

        That reasoning does not stop at ``report.json``, which is why the refusal
        has a type of its own: raising here would abort the rest of :meth:`arun`'s
        finalization, so a declaration defect would cost the finished run its
        ``anomalies.json`` -- the artifact the paragraph above exists to protect.
        The completion path catches :class:`UnreadableIntervalError`, writes what
        is left, and re-raises it unchanged. The early-return path has nothing
        after it and lets it propagate straight out.

        Non-dict reports are saved unchanged; ``None`` skips the save. Returns
        whether ``report.json`` was written.

        Raises:
            UnreadableIntervalError: if the saved report publishes a ``*_ci95``
                key with no ``ci95_units`` entry, an entry naming a metric or a
                population key the report does not write, an entry whose
                population is not a count key, or two metrics on one unit sharing
                one interval while publishing two different numbers.
        """
        if report is None:
            return False
        if isinstance(report, dict):
            as_dict = cast(dict[str, JSONValue], report)
            as_dict["sieval_versions"] = report_versions(
                (c.stage_meta for c in finals),
                (c.stage_meta for c in fails),
            )
            if EvalMode.GEN.value in self._task.tags:
                metas = [c.stage_meta for c in finals]
                truncated = count_truncated_rollouts(metas)
                scored = count_scored_rollouts(metas)
                if truncated is not None and scored is not None:
                    as_dict["n_truncated"] = truncated
                    as_dict["n_scored_rollouts"] = scored
        await self._saver.save_report(report)
        if isinstance(report, dict):
            problems = interval_declaration_problems(cast(dict[str, JSONValue], report))
            if problems:
                raise UnreadableIntervalError(
                    f"{self._task.name}: report.json was saved, but it publishes "
                    "interval(s) no reader can use:\n  - "
                    + "\n  - ".join(problems)
                    + "\nEvery interval travels with a `ci95_units` entry naming a "
                    "population COUNT the same report writes, and one interval is "
                    "shared only by keys publishing one number. Emit intervals "
                    "through `sieval.core.tasks.metrics` and fold two "
                    "interval-bearing fragments with `merge_metrics`, not `|` -- a "
                    "plain merge replaces the whole declaration."
                )
        return True

    # ------------------------------------------------------------------
    # meta.json: stage identities and run history
    # ------------------------------------------------------------------

    async def _last_run_ended(self) -> bool:
        """Whether the latest ``runs[]`` entry ended with a ``reran`` list.

        A directory from before ``runs[]`` has no entry, and its ``report.json``
        is trusted as before.
        """
        meta = await anyio.to_thread.run_sync(_read_run_meta, self._root_dir)
        runs = (meta or {}).get("runs")
        if not isinstance(runs, list) or not runs:
            return True
        last = runs[-1]
        return isinstance(last, dict) and "reran" in last

    # ------------------------------------------------------------------
    # Per-stage resume
    # ------------------------------------------------------------------

    def _prepare_resume_from(self, src: Path) -> None:
        """Copy *src*'s shards into this run's empty ``result_dir``.

        The version-series gate applies to *src*. Its ``meta.json`` is read,
        never written; this run records it in ``runs[].from``.
        """
        if not src.is_dir():
            raise FileNotFoundError(f"resume_from directory not found: {src}")
        if src.resolve() == self._root_dir.resolve():
            raise ValueError(
                f"resume_from is this task's own result_dir ({src}); use "
                "auto_resume to resume it"
            )
        gate_resume_version(src, __version__)
        recorded = _read_run_meta(src)
        identity = recorded_identity(recorded)
        self._copied = copy_recorded_dir(src, self._root_dir)
        self._recorded = recorded
        self._run_from = {
            "dir": str(src),
            "identity": identity if identity is not None else empty_identity(),
        }
        logger.info("Resuming {} from {}", self._task.name, src)

    def _trusts_unrecorded(self) -> bool:
        """Whether a recorded run without identities counts as matching.

        Only for a same-directory resume with no persisted config to compare
        (no :class:`LegacyStageMatch`): the contract before stage identities.
        """
        return (
            self._run_from is None
            and recorded_identity(self._recorded) is None
            and not self._legacy_match
        )

    def _recorded_grader_stamp(self) -> str | None:
        identity = recorded_identity(self._recorded)
        if identity is None:
            return self._grader_stamp if self._trusts_unrecorded() else None
        return grader_key(identity["grader"])

    def _check_inference(self, current: InferenceIdentity | None) -> None:
        """Refuse a resume whose inference differs from the recorded run's."""
        identity = recorded_identity(self._recorded)
        if identity is not None and identity["inference"] is not None:
            recorded = identity["inference"].get("digest")
            if current is not None and recorded == current["digest"]:
                return
            reason = (
                "this run's inference identity is unknown"
                if current is None
                else f"recorded {recorded}, now {current['digest']}"
            )
        elif identity is None and (
            self._trusts_unrecorded() or self._legacy_match.get("inference") is True
        ):
            return
        elif identity is None and self._legacy_match.get("inference") is False:
            reason = "the recorded run's persisted config differs"
        else:
            reason = "the recorded run has no inference identity"
        self._abandon_copy()
        raise StageResumeError(
            "inference",
            f"Resume refused: stage `inference` differs ({reason}).\n"
            f"  recorded: {self._recorded_dir()}\n"
            "  Inference is reused, never rerun, by a resume. Start fresh in a "
            "new result_dir, or match the model, its request arguments, the "
            "task's inference arguments and the serving recipe.",
        )

    def _recorded_dir(self) -> str:
        return self._run_from["dir"] if self._run_from else str(self._root_dir)

    def _abandon_copy(self) -> None:
        """Undo a ``resume_from`` copy that a refusal leaves unused."""
        if self._copied:
            remove_copied(self._copied)
            (self._root_dir / "meta.json").unlink(missing_ok=True)
            (self._root_dir / "manifest.json").unlink(missing_ok=True)
            self._copied = []

    def _recorded_stages_match(self) -> bool:
        """Whether a finished run may be fast-resumed: same data, inference and
        grader. The prompt is covered by the grader: its template is in the
        task module and arguments the grader digests."""
        identity = recorded_identity(self._recorded)
        if identity is None:
            return (
                self._legacy_match.get("data") is not False
                and self._legacy_match.get("inference") is not False
            )
        inference = self._inference_identity()
        recorded_inference = identity["inference"]
        if (
            inference is None
            or recorded_inference is None
            or recorded_inference.get("digest") != inference["digest"]
        ):
            return False
        if grader_key(identity["grader"]) != grader_key(self._grader_identity()):
            return False
        recorded_data = identity["data"]
        return (
            recorded_data is not None
            and recorded_data.get("digest") is not None
            and recorded_data == self._dataset_identity()
        )

    def _dataset_identity(self) -> DigestedStage:
        test_set = self._task.dataset.test_set
        rows = test_set if test_set else []
        return aggregate_digest(
            (i, repeat_index_of(row), self._safe_digest(row))
            for i, row in enumerate(rows)
        )

    def _recorded_data_matches(self) -> bool | None:
        """Whether the recorded run's data is this run's, for the whole set.

        From ``meta.json``'s data identity when it has a digest, else from the
        caller's :class:`LegacyStageMatch`; ``None`` when neither can tell.
        """
        if self._data_match is None:
            identity = recorded_identity(self._recorded)
            recorded = identity["data"] if identity is not None else None
            if recorded is not None and recorded.get("digest") is not None:
                self._data_match = recorded == self._dataset_identity()
            else:
                self._data_match = self._legacy_match.get("data")
        return self._data_match

    @staticmethod
    def _sample_grader_stamp(ctx: TaskContext) -> str | None:
        for stage in (TaskStage.FEEDBACK, TaskStage.POSTPROCESSED):
            for entry in reversed(ctx.stage_meta.get(stage.value, ())):
                if isinstance(entry, dict) and isinstance(entry.get("identity"), dict):
                    stamp = entry["identity"].get("grader")
                    if isinstance(stamp, str):
                        return stamp
        return None

    async def _current_prompt_digest(self, ctx: TaskContext) -> str | None:
        """What this run's ``preprocess`` makes of *ctx*'s sample, digested."""
        fresh = self._task.make_context(ctx.sample_id)
        try:
            result = await self._task.preprocess(fresh.raw_sample, fresh)
        except Exception as exc:
            logger.warning("preprocess failed for {}: {}", ctx.sample_id, exc)
            return None
        inner = result.value if isinstance(result, TaskStageOutput) else result
        return self._safe_digest(prompt_of(inner))

    async def _plan_stage_resume(self) -> list[str | int]:
        """Decide, per recorded sample, what this resume keeps.

        Changed data or prompt restarts a sample. A changed grader rolls it back
        to its recorded inference, or restarts it if it is past iteration 0,
        since that inference saw the old grader's feedback. Samples outside the
        dataset are dropped. Returns the samples to rewrite as this run's.
        """
        test_set = self._task.dataset.test_set
        size = len(test_set) if test_set else 0
        changed: list[str | int] = []
        counts = dict.fromkeys(("kept", "restarted", "regraded", "dropped"), 0)

        for sid, ctx in list(self._contexts.items()):
            if not isinstance(sid, int) or not 0 <= sid < size:
                del self._contexts[sid]
                counts["dropped"] += 1
                continue
            outcome, new_ctx = await self._plan_sample(ctx)
            counts[outcome] += 1
            if new_ctx is not ctx:
                self._contexts[sid] = new_ctx
                changed.append(sid)

        self._resume_plan = counts
        logger.info(
            "Per-stage resume of {}: {} kept, {} regraded, {} restarted, {} dropped",
            self._task.name,
            counts["kept"],
            counts["regraded"],
            counts["restarted"],
            counts["dropped"],
        )
        return changed

    async def _plan_sample(self, ctx: TaskContext) -> tuple[str, TaskContext]:
        """One sample's resume: ``(outcome, context to continue from)``."""
        if self._trusts_unrecorded() or ctx.stage is TaskStage.INITIAL:
            return "kept", ctx
        ctx = self._ensure_raw_sample(ctx)
        restart = self._task.make_context(ctx.sample_id)
        current_data = self._safe_digest(ctx.raw_sample)
        data, prompt, source = recorded_sample_digests(ctx)
        derived = source == "derived"
        if derived:
            # No per-sample identity on record (a run before them, or one with
            # `record_meta` off): the recorded data identity of the whole set,
            # else the caller's reading of the persisted config, decides.
            match = self._recorded_data_matches()
            if match is None:
                self._abandon_copy()
                raise StageResumeError(
                    "data",
                    "Resume refused: stage `data` is unknown: the recorded run "
                    "kept no data identity, and its shards do not hold the raw "
                    f"samples.\n  recorded: {self._recorded_dir()}",
                )
            data = current_data if match else None
        if data is None or data != current_data:
            return "restarted", restart
        if ctx.preprocess_result is not None:
            current_prompt = await self._current_prompt_digest(ctx)
            if prompt is None or prompt != current_prompt:
                return "restarted", restart
            if derived:
                ctx = self._stamp_input(ctx, current_data, current_prompt)
        if not has_graded(ctx):
            return "kept", ctx
        graded_by = self._sample_grader_stamp(ctx) or self._recorded_grader_stamp()
        if graded_by == self._grader_stamp:
            if self._sample_grader_stamp(ctx) is None:
                ctx = self._stamp_grader(ctx, graded_by)
            return "kept", ctx
        if ctx.iteration > 0:
            return "restarted", restart
        return "regraded", self._to_regrade(ctx)

    @staticmethod
    def _stamp_input(ctx: TaskContext, data: str | None, prompt: str | None):
        """Add the data and prompt digests to *ctx*'s latest ``preprocess`` meta."""
        key = TaskStage.PREPROCESSED.value
        history = list(ctx.stage_meta.get(key) or [{}])
        last = dict(history[-1])
        last["identity"] = {"data": data, "prompt": prompt}
        history[-1] = cast(TaskStageMeta, last)
        return replace(ctx, stage_meta={**ctx.stage_meta, key: history})

    @staticmethod
    def _stamp_grader(ctx: TaskContext, stamp: str | None) -> TaskContext:
        """Stamp the grader that graded *ctx* on its grading meta, so a later
        resume can tell it apart once ``meta.json`` names another grader."""
        meta = dict(ctx.stage_meta)
        for stage in (TaskStage.POSTPROCESSED, TaskStage.FEEDBACK):
            history = list(meta.get(stage.value) or ())
            if history:
                last = dict(history[-1])
                last["identity"] = {"grader": stamp}
                history[-1] = cast(TaskStageMeta, last)
                meta[stage.value] = history
        return replace(ctx, stage_meta=meta)

    @staticmethod
    def _to_regrade(ctx: TaskContext) -> TaskContext:
        """Roll *ctx* back to its recorded inference, to grade it again."""
        keep = {TaskStage.PREPROCESSED.value, TaskStage.INFERRED.value}
        return replace(
            ctx,
            stage=TaskStage.INFERRED,
            postprocess_result=None,
            feedback_result=None,
            error_action=None,
            error_reason=None,
            error_msg=None,
            retry_count=0,
            stage_meta={k: v for k, v in ctx.stage_meta.items() if k in keep},
        )

    def _model_name(self) -> str:
        return self._task.model.meta()["model"]

    @staticmethod
    def _safe_digest(value: Any) -> str | None:
        """Digest *value*, or ``None`` when it holds something undigestable."""
        try:
            return digest(value)
        except (TypeError, ValueError) as exc:
            logger.warning("Identity digest unavailable: {}", exc)
            return None

    def _inference_identity(self) -> InferenceIdentity | None:
        try:
            return inference_identity(
                model=self._task.model,
                task_args=self._task.inference_args(),
                runner_keys={
                    "deterministic": self._config.deterministic,
                    "max_iterations": self._config.max_iterations,
                    "max_retries": self._config.max_retries,
                },
                recipe=self._inference_recipe,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Inference identity unavailable: {}", exc)
            return None

    def _grader_identity(self) -> GraderIdentity | None:
        try:
            judges = judge_identities(
                self._task.judge_models(),
                self._task.judge_prompts(),
                versions=self._judge_versions,
            )
            task_cls = type(self._task)
            return grader_identity(
                grader_modules(task_cls),
                judges,
                service_identities(task_cls.services),
                args=self._task.grader_args(),
                task=f"{task_cls.__module__}.{task_cls.__qualname__}",
            )
        except (TypeError, ValueError, AttributeError, SyntaxError, OSError) as exc:
            logger.warning("Grader identity unavailable: {}", exc)
            return None

    def _seed_judge_versions(self, meta: dict | None) -> None:
        """Carry judge versions a resumed run's ``meta.json`` recorded.

        A resume that grades nothing new never hears from the judge API, and
        the version it reported earlier still describes the verdicts on disk.
        Carried only into a role still held by the same model and sampling.
        """
        grader = ((meta or {}).get("identity") or {}).get("grader") or {}
        current = {j["role"]: j for j in self._current_judges()}
        for judge in grader.get("judges") or ():
            if not isinstance(judge, dict):
                continue
            role, version = judge.get("role"), judge.get("version")
            now = current.get(role) if isinstance(role, str) else None
            if (
                now is not None
                and isinstance(version, str)
                and judge.get("model") == now["model"]
                and judge.get("sampling") == now["sampling"]
            ):
                self._judge_versions.setdefault(now["role"], version)

    def _current_judges(self) -> list[JudgeIdentity]:
        try:
            return judge_identities(self._task.judge_models(), {})
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Judge models unavailable: {}", exc)
            return []

    def _observe_judge_versions(self, model_calls: list) -> None:
        """Keep the first model version a judge API reported, by role.

        A call is matched to its judge by :func:`call_key` (model, sampling and
        ``extra``), not by name: a task judged by the model it evaluates calls
        both under one name, and only the configuration tells them apart.
        """
        if self._judge_roles is None:
            # Identity is bookkeeping: a task whose judge lookup raises still
            # runs, with no judge versions, rather than failing every sample.
            self._judge_roles = {}
            try:
                judges = self._task.judge_models()
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Judge models unavailable: {}", exc)
                judges = {}
            candidate = call_key(self._task.model.meta())
            for role, model in judges.items():
                key = call_key(model.meta())
                # A judge configured exactly like the candidate is the same
                # model; its calls cannot be told apart, so none is a judge's.
                if key != candidate:
                    self._judge_roles.setdefault(key, []).append(role)
        for call in model_calls:
            model = call.get("model")
            reported = call.get("response_model")
            if not isinstance(model, dict) or not isinstance(reported, str):
                continue
            for role in self._judge_roles.get(call_key(model), ()):
                if reported:
                    self._judge_versions.setdefault(role, reported)

    def _note_sample_digests(self, ctx: TaskContext) -> None:
        """Record *ctx*'s data and prompt digests once, as of its first preprocess.

        Prefers the digests its ``preprocess`` record carries -- what that run
        saw -- and falls back to digesting the sample and the recorded prompt.
        """
        if ctx.sample_id in self._sample_digests:
            return
        # Shards do not hold the raw sample; a resumed context gets it back here.
        ctx = self._ensure_raw_sample(ctx)
        recorded: dict = {}
        for entry in ctx.stage_meta.get(TaskStage.PREPROCESSED.value, ()):
            if isinstance(entry, dict) and isinstance(entry.get("identity"), dict):
                recorded = entry["identity"]
                break
        data = recorded.get("data") if recorded else None
        if data is None:
            data = self._safe_digest(ctx.raw_sample)
        prompt = recorded.get("prompt") if recorded else None
        if prompt is None:
            # No prompt at all is a fact (preprocess failed or never ran); a
            # prompt that would not digest is unknown.
            prompt = (
                ABSENT
                if ctx.preprocess_result is None
                else self._safe_digest(prompt_of(ctx.preprocess_result))
            )
        self._sample_digests[ctx.sample_id] = (ctx.repeat_index, data, prompt)

    async def _maybe_write_input_identity(self, force: bool = False) -> None:
        """Write the data and prompt identities once every sample is past preprocess.

        Not a barrier: samples already past preprocess keep going while the rest
        catch up. *force* writes whatever is known at the end of a run, so a
        sample that never reached preprocess is digested from its raw content.
        """
        if self._input_identity_written:
            return
        # `_total_samples` is 0 until `arun` has sized the sample universe.
        if not force and not 0 < self._total_samples <= len(self._sample_digests):
            return
        if force:
            for ctx in self._contexts.values():
                self._note_sample_digests(ctx)
        self._input_identity_written = True
        entries = self._sample_digests.items()
        await self._saver.set_identity(
            data=aggregate_digest((sid, rep, d) for sid, (rep, d, _) in entries),
            prompt=aggregate_digest((sid, rep, p) for sid, (rep, _, p) in entries),
        )

    def _requests_identity(self) -> DigestedStage:
        """Fold what the candidate was sent, per sample, from its ``infer`` records.

        A sample never inferred has :data:`ABSENT` requests. One inferred but
        without its ``model_calls`` on record (a resume of a ``record_meta=False``
        run, or a call the dialect did not echo) is unknown, and so is the stage.
        """
        candidate = call_key(self._task.model.meta())
        entries = []
        for ctx in self._contexts.values():
            calls = [
                call
                for entry in ctx.stage_meta.get(TaskStage.INFERRED.value, ())
                if isinstance(entry, dict)
                for call in entry.get("model_calls") or ()
                if isinstance(call, dict)
            ]
            if calls:
                sent = request_digest(calls, candidate)
            else:
                sent = None if self._was_inferred(ctx) else ABSENT
            entries.append((ctx.sample_id, ctx.repeat_index, sent))
        return aggregate_digest(entries)

    @staticmethod
    def _was_inferred(ctx: TaskContext) -> bool:
        """Whether *ctx* got past ``infer``, read from where it stopped."""
        if ctx.stage == TaskStage.FAILED:
            return ctx.error_action not in (TaskAction.PREPROCESS, TaskAction.INFER)
        return ctx.stage not in (TaskStage.INITIAL, TaskStage.PREPROCESSED)

    async def _end_run(self, *, reported: bool) -> None:
        """Finish this run's ``runs[]`` entry once its report is settled.

        *reported* says ``report.json`` was written; ``report()`` returning
        ``None`` writes none.

        Writes the data and prompt identities if a sample never reached
        preprocess, folds what the candidate was sent into the inference
        identity, and rewrites the grader's with the judge versions reported.
        """
        await self._maybe_write_input_identity(force=True)
        inference = self._saver.inference_identity
        if inference is not None:
            await self._saver.set_identity(
                inference={
                    "digest": inference["digest"],
                    "requests": self._requests_identity(),
                }
            )
        # Always: a regrade left it unknown until now.
        await self._saver.set_identity(grader=self._grader_identity())
        ran = self._ran_actions
        reran = set()
        if TaskAction.PREPROCESS in ran:
            reran |= {"data", "prompt"}
        if TaskAction.INFER in ran:
            reran.add("inference")
        # `report()` has no identity: only the stages that grade count.
        if ran & {TaskAction.POSTPROCESS, TaskAction.FEEDBACK}:
            reran.add("grader")
        await self._saver.end_run(reran, reported=reported)

    def _resolve_result_dir(
        self, result_dir: str | None, task: Task, auto_resume: bool
    ) -> tuple[bool, Path]:
        if result_dir:
            p = Path(result_dir).expanduser()
            if p.exists() and p.is_file():
                raise ValueError("result_dir is a file")
            if p.exists() and run_started(p):
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
                if run_started(latest):
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
