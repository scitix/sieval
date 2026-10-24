"""Single-task execution engine with dual-stream compute/storage architecture."""

import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import anyio
import anyio.to_thread
import orjson
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger

from sieval import __version__
from sieval.core.datasets import repeat_index_of
from sieval.core.models import ModelOutput
from sieval.core.models.overflow import (
    CONTEXT_OVERFLOW,
    ContextOverflowError,
    OverflowScope,
    ReplayAmbiguous,
    is_recorded_context_overflow,
    overflow_scope,
)
from sieval.core.models.request_log import StageScope, stage_scope, timeout_phase
from sieval.core.tasks.anomaly import TaskAnomalyDetector, detect_context_overflow
from sieval.core.tasks.concurrency import (
    compute_stream_buffer_capacity,
    get_limiter_for,
    min_limit,
    prepare_limiters,
)
from sieval.core.tasks.consts import (
    CONTEXT_OVERFLOW_REASON,
    NON_RETRIABLE_REASON,
    STAGE_DIRS,
    NonRetriableSampleError,
)
from sieval.core.tasks.context import (
    TaskAction,
    TaskContext,
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
from sieval.core.tasks.requests import REQUESTS_FILE, TaskRequestLog
from sieval.core.tasks.run_meta import (
    ABSENT,
    GRADING_STAGES,
    IDENTITY_ERRORS,
    RESULT_SCHEMA_VERSION,
    DigestedStage,
    GradingIdentity,
    InferenceIdentity,
    JudgeIdentity,
    PreprocessIdentity,
    ResumeOutcomes,
    RunFrom,
    RunRequests,
    aggregate_digest,
    call_key,
    digest,
    empty_identity,
    grader_key,
    grader_task,
    grading_identities,
    inference_identity,
    is_code_identity,
    judge_identities,
    legacy_grader_keys,
    preprocess_key,
    prompt_of,
    request_digest,
    stage_code,
    stage_key,
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
    resume_schema_verdict,
)
from .resume_option import as_resume, resume_source
from .stage_resume import (
    LegacyStageMatch,
    ResumePlan,
    RunPlan,
    StageResumeError,
    abandon_copy,
    copy_pending,
    copy_recorded_dir,
    earliest_grading_change,
    end_copy,
    has_graded,
    iteration_dirs,
    plan_resume,
    recorded_grader,
    recorded_identity,
    recorded_sample_digests,
    task_preprocess_identity,
)

#: Key of the run's own counts in ``report.json``.
SIEVAL_COUNTS_KEY = "sieval"

#: Sentinel for :meth:`TaskRunner._finalize`: no cached report to reuse.
_NOT_REPORTED: Any = object()


class SievalCounts(TypedDict):
    """The ``sieval`` block of ``report.json``.

    ``anomalies`` counts final samples any rule flagged. ``context_overflow``
    counts samples whose prompt the model under test refused as longer than its
    context window: final ones, scored as an empty answer and also counted in
    ``anomalies``, and failed ones of a task that declares ``context_overflow =
    "fail"``. A fast resume needs ``total == final`` and ``failed == 0``.
    """

    total: int
    final: int
    failed: int
    anomalies: int
    context_overflow: int


def _overflowed(ctx: TaskContext) -> bool:
    """Whether *ctx*'s prompt did not fit the model under test's window."""
    if ctx.stage is TaskStage.FAILED:
        return ctx.error_reason == CONTEXT_OVERFLOW_REASON
    return bool(detect_context_overflow(ctx))


def _counts(
    total: int,
    finals: list[TaskContext],
    fails: list[TaskContext],
    anomalies: dict[str | int, dict[int, dict[str, list[int]]]],
) -> SievalCounts:
    final_ids = {c.sample_id for c in finals}
    return {
        "total": total,
        "final": len(finals),
        "failed": len(fails),
        "anomalies": sum(
            1 for sid, its in anomalies.items() if sid in final_ids and its
        ),
        "context_overflow": sum(1 for c in (*finals, *fails) if _overflowed(c)),
    }


def _no_outcomes() -> ResumeOutcomes:
    return {
        "kept": 0,
        "rereported": 0,
        "regraded": 0,
        "repostprocessed": 0,
        "restarted": 0,
        "dropped": 0,
    }


def _reported_complete(report: object) -> bool:
    """Whether a cached report's ``sieval`` counts say every sample is final.

    A report without them cannot say, and is finalized again from the shards.
    """
    if not isinstance(report, dict):
        return False
    counts = cast(dict[str, object], report).get(SIEVAL_COUNTS_KEY)
    if not isinstance(counts, dict):
        return False
    fields = cast(dict[str, object], counts)
    total, final, failed = (fields.get(k) for k in ("total", "final", "failed"))
    return isinstance(total, int) and total > 0 and final == total and failed == 0


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
            "To continue, set `resume=True`; "
            "to start fresh, specify a different `result_dir` "
            "or delete the directory manually."
        )


class UnfinishedCopyError(FileExistsError):
    """``result_dir`` holds an unfinished ``resume=<dir>`` copy, and this run
    names no source to copy again."""

    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        super().__init__(
            f"Resume aborted: {path} holds an unfinished copy of {source}, made "
            "by `resume` and stopped before its run began.\n"
            "Run again with the same `resume` directory to copy it again, or start "
            "fresh in another result_dir."
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
        resume: ``True`` resumes ``result_dir`` in place. A directory is a
            recorded task directory (read-only ok) copied into an empty
            ``result_dir`` and resumed; ``result_dir`` itself resumes in
            place. Once ``result_dir`` holds a run the copy is skipped, so the
            same config resumes an interrupted copy.
        effective_config: The session's ``effective_config.yaml``, recorded in
            ``meta.json`` as ``config``. Not part of any identity.
    """

    result_dir: str | None = None
    resume: bool | Path | None = None
    effective_config: str | None = None

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
    # `requests.jsonl`: one line per model request attempt, rotated at
    # `request_log_max_bytes`, keeping `request_log_max_files` old files.
    record_requests: bool = True
    request_log_max_bytes: int = 64 * 1024 * 1024
    request_log_max_files: int = 8
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

    def __post_init__(self) -> None:
        self.resume = as_resume(self.resume)


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

    Any ``<iteration>/<stage>/*.idx`` counts, as does a non-empty ``runs[]``
    or a ``meta.json`` from before ``runs[]``. Shards are needed because up
    to v0.6 ``meta.json`` was written only when a run ended or was
    interrupted, so a killed run left shards and no ``meta.json``.
    """
    meta = _read_run_meta(root_dir)
    if meta is not None:
        runs = meta.get("runs")
        if not isinstance(runs, list) or runs:
            return True
    return _holds_shards(root_dir)


def _holds_shards(root_dir: Path) -> bool:
    """Whether *root_dir* holds any shard index, ``<iteration>/<stage>/*.idx``,
    in a directory named for a :class:`TaskStage`."""
    return any(
        next(d.glob(f"{stage}/*.idx"), None)
        for d in iteration_dirs(root_dir)
        for stage in STAGE_DIRS
    )


def _unfinished_copy(root_dir: Path) -> str | None:
    """The source of a ``resume=<dir>`` copy into *root_dir* that no run began
    on, or ``None``. A marker beside a recorded run is stale and ignored."""
    source = copy_pending(root_dir)
    if source is None:
        return None
    meta = _read_run_meta(root_dir)
    runs = meta.get("runs") if meta is not None else None
    return None if isinstance(runs, list) and runs else source


def read_run_version(root_dir: Path) -> str | None:
    """Return the sieval version in ``root_dir/meta.json``, or ``None``.

    Public for the CLI, which explains a cross-version resume in an abort it
    raises before any runner exists.

    ``None`` covers every unusable case: no file, unreadable, not a mapping, no
    ``version``, or a non-string one.
    """
    return _recorded_version(_read_run_meta(root_dir))


def _recorded_version(meta: dict | None) -> str | None:
    version = (meta or {}).get("version")
    return version if isinstance(version, str) else None


def gate_resume_version(
    root_dir: Path, current_version: str, *, task: str | None = None
) -> None:
    """Refuse to resume a directory whose result schema version is not this build's.

    Applies :func:`resume_schema_verdict` to ``root_dir/meta.json``, read once.
    Missing, unreadable, or version-less ``meta.json`` is fail-closed (reject).
    Raises :class:`ResumeVersionError` on reject; logs a warning on a resume
    across SiEval versions; silent on an exact match.
    """
    meta = _read_run_meta(root_dir)
    v_run = _recorded_version(meta)
    if meta is None or v_run is None:
        # *root_dir* may be a `resume` source, so never advise removing it.
        raise ResumeVersionError(
            f"Resume aborted: {root_dir} holds a run, but its meta.json is "
            "missing, unreadable, or has no version, so the sieval version "
            "that wrote it is unknown.\n"
            "Start fresh in another result_dir, without resuming it.",
            task=task,
        )

    recorded_schema = meta.get("result_schema_version")
    verdict = resume_schema_verdict(recorded_schema, v_run, current_version)
    if verdict.action is ResumeAction.REJECT:
        raise ResumeVersionError(
            format_reject_message(
                recorded_schema, v_run, current_version, verdict.reason
            ),
            task=task,
        )
    if verdict.action is ResumeAction.COMPATIBLE:
        logger.warning(
            "Resuming across sieval versions (persisted={}, current={}, result "
            "schema version {}); per-record provenance will record the blend.",
            v_run,
            current_version,
            RESULT_SCHEMA_VERSION,
        )


def _reuses_inference(ctx: TaskContext) -> bool:
    """Whether a recorded sample's state holds an inference a resume would keep.

    Read from the manifest-level stage, before hydration.
    """
    if ctx.stage is TaskStage.FAILED:
        return ctx.error_action not in (TaskAction.PREPROCESS, TaskAction.INFER)
    return ctx.stage not in (TaskStage.INITIAL, TaskStage.PREPROCESSED)


def _failure_message(exc: Exception) -> str:
    """``error_msg`` for *exc*; a client timeout names its phase."""
    phase = timeout_phase(exc)
    return str(exc) if phase is None else f"{exc} (timeout phase: {phase})"


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
        unserved_candidate: bool = False,
    ):
        """*inference_recipe* is the serving plan of the model under test,
        without its checkpoint, for the inference identity; ``None`` when the
        model is not served by SiEval. *resume_config_diff* is the config
        change a resume made, for ``runs[].config``. *legacy_match* compares a
        recorded run without identities through its persisted config.
        *unserved_candidate* says no engine serves the model under test, since
        a plan found nothing to infer: a run that would infer is refused.
        """
        # A malformed judge reference would leave every grader identity
        # unknown; refuse it before any work.
        validate_judge_model_ref()
        self._task = task
        self._config = config or TaskRunnerConfig()
        self._inference_recipe = inference_recipe
        self._resume_config_diff = list(resume_config_diff or [])
        self._legacy_match = LegacyStageMatch(**(legacy_match or {}))
        self._unserved_candidate = unserved_candidate
        # Per-stage resume: what was copied in, and what the comparison chose.
        self._run_from: RunFrom | None = None
        self._copied: list[Path] = []
        self._recorded: dict | None = None
        # The earliest grading stage whose recorded identity differs from this
        # run's, and this run's grading identities and their keys.
        self._grading_change: str | None = None
        self._grading: dict[str, GradingIdentity] | None = None
        self._stage_code: dict | None = None
        self._legacy_keys: set[str] | None = None
        # This run's preprocess identity (`False` until computed); whether a
        # recorded sample's preprocess changed; for a schema-1/2 directory,
        # whether every recorded sample still renders its recorded prompt.
        self._preprocess: PreprocessIdentity | None | Literal[False] = False
        self._preprocess_restarted = False
        self._legacy_prompts_match: bool | None = None
        self._stage_stamps: dict[str, str | None] = dict.fromkeys(GRADING_STAGES)
        # Context overflows an earlier run recorded as failed inference, by
        # sample: a resume scores them from the recorded refusal (`_infer`).
        self._overflow_replays: dict[str | int, str] = {}
        self._data_match: bool | None = None
        self._resume_plan: ResumeOutcomes | None = None
        self._unverified_inference = 0
        # Run history: actions executed this run, each sample's data digest as
        # of its first preprocess, and judge versions the API reported, by role.
        self._ran_actions: set[TaskAction] = set()
        self._sample_digests: dict[str | int, tuple[int | None, str | None]] = {}
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
        self._resumes = bool(self._config.resume)
        # Paths
        self._resumed_from_existing, self._root_dir = self._resolve_result_dir(
            self._config.result_dir, task, self._resumes
        )
        source = resume_source(self._config.resume, self._root_dir)
        # Read off the task, not looked up by name: `task.name` is a
        # user-chosen YAML key, not the registered `@sieval_task(name=...)`.
        self._task_identity = get_task_run_identity(task)
        if self._resumed_from_existing:
            gate_resume_version(self._root_dir, __version__, task=task.name)
            self._recorded = _read_run_meta(self._root_dir)
            logger.info("Auto resumed from: {}", self._root_dir)
        elif source is not None:
            self._prepare_resume_from(source)

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
        self._stored_terminal: set[str | int] = set()

        # Anomaly Detection
        self._anomaly_detector = TaskAnomalyDetector(root_dir=self._root_dir)

        # Runtime Context (may be injected by MultiTaskRunner)
        self._shared_global_limiter: anyio.CapacityLimiter | None = None
        self._shared_stage_limiters: dict[str, anyio.CapacityLimiter] = {}
        self._progress_position: int = 0

        # Opened once the run is recorded (`_open_request_log`).
        self._request_log: TaskRequestLog | None = None
        self._run_start = time.monotonic()

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
        try:
            # 0. Lifecycle Setup
            await self._task.setup()

            # Before any identity is written, fast resume's included.
            if self._recorded is not None:
                self._seed_judge_versions(self._recorded)

            # 0.5. Fast Resume: a finished run's report.json is reused.
            cached = await self._fast_resume()
            if cached is not None:
                return cached
            # Copied here, not in `__init__`, so a runner never run copies
            # nothing. The copy stays marked unfinished until `_begin_run`
            # records the run (see `_holds_run`).
            if self._run_from is not None:
                self._copied = copy_recorded_dir(
                    Path(self._run_from["dir"]), self._root_dir
                )
            resumed = self._resumed_from_existing or self._run_from is not None
            try:
                changed = await self._begin_run(resumed)
            except BaseException:
                self._abandon_copy()
                raise
            end_copy(self._root_dir)
            self._open_request_log()

            # 1. Load Initial State
            if not resumed:
                self._contexts = await self._loader.load_initial_state()
                await self._hydrate_initial(resumed=False)

            # 2. Determine sample universe (without materializing all raw_samples)
            dataset_size = (
                len(self._task.dataset.test_set) if self._task.dataset.test_set else 0
            )
            pending_ids: list[int] = [
                i for i in range(dataset_size) if i not in self._contexts
            ]
            self._total_samples = max(len(self._contexts), dataset_size)

            # 4. Prepare Runtime State
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
                    self._resumes
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
                return await self._finalize()

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
            # Outlive the workers' receivers, so what was sent but never
            # received can still be written.
            storage_leftover = storage_recv.clone()
            compute_leftover = compute_recv.clone()
            # Terminal samples on disk or handed to storage.
            self._stored_terminal = {
                sid for sid, ctx in self._contexts.items() if ctx.is_terminal()
            }

            try:
                await self._run_streams(
                    compute_send, compute_recv, storage_recv, storage_send, pending_ids
                )
            finally:
                if self._progress:
                    self._progress.write_in(None)
                # A raised stage or an interrupt cancels the storage worker
                # before its final flush; every finished sample still reaches
                # disk, wherever it was held. A no-op after a clean exit.
                with anyio.CancelScope(shield=True):
                    try:
                        await self._saver.drain(
                            storage_leftover,
                            also=self._unstored_terminal(compute_leftover),
                        )
                    except Exception as exc:
                        logger.error("Flush after the run stopped failed: {}", exc)

            # 8. Finalize
            return await self._finalize()

        except (KeyboardInterrupt, anyio.get_cancelled_exc_class()):
            if not self._interrupted:
                self._interrupted = True
                logger.warning("Interrupted. Saving progress...")
                try:
                    with anyio.CancelScope(shield=True):
                        await self._saver.flush()
                        await self._flush_request_log()
                        # Anomalies of the samples this run saw finish.
                        rules_hash = await self._write_anomalies(self._anomaly_results)
                        await self._saver.interrupt_run(
                            rules_hash=rules_hash, requests=self._request_counts()
                        )
                except Exception as e:
                    logger.error("Flush failed: {}", e)
            if self._progress:
                with anyio.CancelScope(shield=True):
                    await self._progress.aclose()
            return None

        finally:
            # A run that raised; a no-op after `_finalize`.
            with anyio.CancelScope(shield=True):
                await self._flush_request_log()
            # X. Lifecycle Shutdown
            await self._task.shutdown()

    async def _begin_run(self, resumed: bool) -> list[str | int]:
        """Plan a resume, then record the run in ``meta.json``.

        Recorded before any work, so an interrupted run is resumed and the
        resume gates have a ``meta.json`` to read. Inference and grader
        identities are known now, and so is preprocess unless a recorded
        sample's changed (then at the end); data once every sample is past
        preprocess. Returns the samples a resume rewinds.
        """
        inference = self._inference_identity()
        grading = self._grading_identities()
        preprocess = self._preprocess_identity()
        changed: list[str | int] = []
        keeps_recorded_model = False
        if resumed:
            # Plan before `begin_run`: a refusal leaves meta.json as it was.
            changed, keeps_recorded_model = await self._plan_recorded(inference)
        self._refuse_unserved()
        # A changed grading stage, and every one after it, is unknown until the
        # run ends: samples not reached yet hold the recorded verdicts. One
        # that did not change is written now, upgrading a schema-1 block.
        unknown_from = (
            GRADING_STAGES.index(self._grading_change)
            if resumed and self._grading_change is not None
            else len(GRADING_STAGES)
        )
        await self._saver.begin_run(
            resumed=resumed,
            run_from=self._run_from,
            model=self._model_name(),
            keep_recorded_model=keeps_recorded_model,
            config_diff=self._resume_config_diff if resumed else (),
            config_path=self._config_path(),
            identity={
                "data": None,
                # Unknown while a sample stamped with another preprocess may
                # remain: `_sample_preprocess_changed` trusts a recorded one.
                "preprocess": None if self._preprocess_restarted else preprocess,
                "inference": inference,
                **{
                    stage: None
                    if i >= unknown_from or grading is None
                    else grading[stage]
                    for i, stage in enumerate(GRADING_STAGES)
                },
            },
        )
        return changed

    async def _plan_recorded(
        self, inference: InferenceIdentity | None
    ) -> tuple[list[str | int], bool]:
        """Load the recorded contexts and decide what a resume does with each.

        The one resume plan, shared by :meth:`_begin_run` and :meth:`plan_run`
        so that a plan made before launch counts the samples the run infers.
        Writes nothing; raises a refusal before anything is recorded. Returns
        the samples it rewinds, and whether the recorded model is kept.
        """
        self._contexts = await self._loader.load_initial_state()
        if any(_reuses_inference(c) for c in self._contexts.values()):
            self._check_inference(inference)
        self._grading_change = self._recorded_grading_change()
        await self._hydrate_initial(resumed=True)
        # Decide per sample what a resume keeps before failures roll back,
        # so a sample that failed in grading regrades rather than reinfers.
        changed = await self._plan_stage_resume()
        self._collect_overflow_replays(inference)
        self._loader.prepare_failed_retries(self._contexts, self._record_each_stage)
        for sid in self._overflow_replays:
            # Scored from the record, not retried: no retry is spent.
            ctx = self._contexts[sid]
            self._contexts[sid] = replace(ctx, retry_count=ctx.retry_count - 1)
        keeps_recorded_model = self._model_differs() and self._check_model_switch()
        return changed, keeps_recorded_model

    def _collect_overflow_replays(self, inference: InferenceIdentity | None) -> None:
        """Samples an earlier run failed with a context overflow at ``infer``.

        Such a failure is fixed by the model, its serving and the prompt, so a
        resume that keeps all three scores it from the recorded refusal instead
        of asking again: kept when the task scores overflows and the recorded
        inference identity equals this run's for the same model. A sample whose
        data or prompt changed was restarted by the plan already.
        """
        if self._task.context_overflow != "score":
            return
        identity = recorded_identity(self._recorded)
        recorded = identity["inference"] if identity is not None else None
        if (
            inference is None
            or recorded is None
            or recorded.get("digest") != inference["digest"]
            or self._model_differs()
        ):
            return
        for sid, ctx in self._contexts.items():
            if (
                ctx.stage is TaskStage.FAILED
                and ctx.error_action is TaskAction.INFER
                and ctx.error_msg is not None
                and is_recorded_context_overflow(ctx.error_reason, ctx.error_msg)
            ):
                self._overflow_replays[sid] = ctx.error_msg

    async def _hydrate_initial(self, *, resumed: bool) -> None:
        """Hydrate the recorded contexts a run continues from."""
        stages = {
            TaskStage.INITIAL,
            TaskStage.PREPROCESSED,
            TaskStage.INFERRED,
            TaskStage.POSTPROCESSED,
            TaskStage.FEEDBACK,
        }
        if self._resumes:
            stages |= {TaskStage.FINAL, TaskStage.FAILED}
        await self._loader.hydrate(
            self._contexts,
            self._hydrated_ids,
            include_stages=stages,
            prepare_retries=self._resumes and not resumed,
            record_each_stage=self._record_each_stage,
        )

    async def _fast_resume(self) -> JSONValue | None:
        """A finished run's ``report.json``, or ``None`` when the run must go on.

        Runs after ``setup()``, which a task may use to resolve an argument the
        grader identity reads, so even a fast resume pays for ``setup()``: a
        task that contacts a service there needs it up.
        """
        reusable = await self._reusable_report()
        if reusable is None:
            return None
        cached_report, rereport = reusable
        if rereport:
            return await self._rereport()
        # Nothing reruns. The identity is kept, but for grading identities a
        # schema-1 block lacked; the views are rebuilt from the shards like
        # every other exit.
        await self._saver.begin_run(
            resumed=True,
            model=self._model_name(),
            config_diff=self._resume_config_diff,
            config_path=self._config_path(),
            identity=self._resume_identity_update(),
        )
        self._contexts = await self._loader.load_initial_state()
        await self._loader.hydrate(
            self._contexts,
            self._hydrated_ids,
            include_stages={TaskStage.FINAL, TaskStage.FAILED},
            prepare_retries=False,
            record_each_stage=self._record_each_stage,
        )
        # The resume gate pins the dataset, and the report says every sample
        # is final, so the recorded samples are the whole run.
        self._total_samples = len(self._contexts)
        self._resume_plan = _no_outcomes()
        self._resume_plan["kept"] = len(self._contexts)
        return await self._finalize(cached_report=cached_report)

    async def _reusable_report(self) -> tuple[JSONValue, bool] | None:
        """The recorded ``report.json`` a resume finishes on, and whether only
        ``report`` changed so it is recomputed from the records; ``None`` when
        the run must go on. Read-only."""
        if not self._resumes:
            return None
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
        if cached_report is not None and self._model_differs():
            # Finalize through the full path, which records this run's model.
            logger.info(
                "report.json found, but it was recorded for model {!r}; "
                "finalizing again",
                self._recorded_model(),
            )
            cached_report = None
        rereport = False
        if cached_report is not None:
            match = self._recorded_stages_match()
            if match is not None and not await self._recorded_preprocess_matches():
                match = None
            if match is None:
                logger.info(
                    "report.json found, but its run's stage identities differ "
                    "from this run's; resuming stage by stage"
                )
                cached_report = None
            elif match == "report":
                rereport = True
        if cached_report is None:
            return None
        if not _reported_complete(cached_report):
            # Failures to retry, samples left, or a report from before the
            # `sieval` counts: the full path decides.
            return None
        return cached_report, rereport

    async def _rereport(self) -> JSONValue:
        """Recompute ``report.json`` from a finished run's records: re-report.

        Only ``report`` changed, so every sample is kept as recorded and
        nothing reaches ``preprocess``, ``infer`` or a judge; ``report()`` runs
        on the final and failed records, and the views are rebuilt.
        """
        await self._saver.begin_run(
            resumed=True,
            model=self._model_name(),
            config_diff=self._resume_config_diff,
            config_path=self._config_path(),
            identity=self._resume_identity_update(report_unknown=True),
        )
        self._contexts = await self._loader.load_initial_state()
        await self._loader.hydrate(
            self._contexts,
            self._hydrated_ids,
            include_stages={TaskStage.FINAL, TaskStage.FAILED},
            prepare_retries=False,
            record_each_stage=self._record_each_stage,
        )
        self._total_samples = len(self._contexts)
        self._resume_plan = _no_outcomes()
        self._resume_plan["kept"] = len(self._contexts)
        self._resume_plan["rereported"] = len(self._contexts)
        logger.info(
            "{}: only report changed; re-reporting {} recorded sample(s)",
            self._task.name,
            len(self._contexts),
        )
        return await self._finalize()

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
            if self._progress:
                self._progress.write_in(tg)

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
                    self._stored_terminal.add(ctx.sample_id)
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
                with stage_scope(self._stage_scope(ctx, action)):
                    await self._execute_stage_logic(ctx, compute_send, action)
            finally:
                self._active_counts[action.value] -= 1
                if self._progress:
                    self._progress.set_status(
                        self._active_counts, len(self._hydrated_ids)
                    )
        # Outside the limiters: a slow disk must not hold a stage slot.
        await self._flush_request_log(due_only=True)

    def _open_request_log(self) -> None:
        """Start recording model requests unless ``record_requests`` is off."""
        if not self._config.record_requests:
            return
        try:
            judges = self._task.judge_models()
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Judge models unavailable: {}", exc)
            judges = {}
        self._request_log = TaskRequestLog(
            self._root_dir,
            subject=self._task.model,
            judges=judges,
            max_bytes=self._config.request_log_max_bytes,
            max_files=self._config.request_log_max_files,
        )

    def _stage_scope(self, ctx: TaskContext, action: TaskAction) -> StageScope | None:
        """What a model call made in this stage is recorded against."""
        if self._request_log is None:
            return None
        key = action.value
        limiters = {
            "global": self._shared_global_limiter,
            "global_stage": self._shared_stage_limiters.get(key),
            "task": self._local_limiter,
            "task_stage": self._local_stage_limiters.get(key),
        }
        return StageScope(
            sink=self._request_log,
            task=self._task.name,
            run=self._saver.run_id,
            stage=key,
            sample_id=ctx.sample_id,
            iteration=ctx.iteration,
            repeat_index=ctx.repeat_index,
            limiters={k: v for k, v in limiters.items() if v is not None},
            run_start=self._run_start,
        )

    def _request_counts(self) -> RunRequests | None:
        """This run's request counts, for ``report.json`` and ``runs[]``."""
        return self._request_log.counts() if self._request_log is not None else None

    async def _flush_request_log(self, *, due_only: bool = False) -> None:
        """Write buffered request lines; a write error is logged, never raised.

        Lines that failed to write are dropped; the summary still counts them.
        """
        if self._request_log is None:
            return
        try:
            if due_only:
                await self._request_log.maybe_flush()
            else:
                await self._request_log.flush()
        except OSError as exc:
            logger.error("Failed to write {}: {}", REQUESTS_FILE, exc)

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
            overflows: list[str] | None = None,
        ) -> TaskContext:
            # 1. Resolve value and optional user meta (box preserved)
            val, out_meta = _resolve(raw_result)
            timing_s = _finalize_timing()
            inner = val.value if isinstance(val, TaskStageOutput) else val

            # 2. Auto meta always recorded; user meta merges on top
            meta = _build_auto_meta(timing_s, inner)
            if overflows:
                meta[CONTEXT_OVERFLOW] = list(overflows)
            if stage == TaskStage.PREPROCESSED:
                meta["identity"] = self._input_stamp(self._safe_digest(ctx.raw_sample))
            elif stage in (TaskStage.POSTPROCESSED, TaskStage.FEEDBACK):
                # Per sample, so a resume can tell which code graded it.
                meta["identity"] = self._grading_stamps(stage)

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
            self._check_served(action)
            match action:
                case TaskAction.PREPROCESS:
                    r = await self._task.preprocess(ctx.raw_sample, ctx)
                    new_ctx = _run_stage(TaskStage.PREPROCESSED, r)

                case TaskAction.INFER:
                    r, overflows = await self._infer(ctx)
                    new_ctx = _run_stage(TaskStage.INFERRED, r, overflows=overflows)

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
            if isinstance(e, ContextOverflowError):
                reason = CONTEXT_OVERFLOW_REASON
            elif isinstance(e, NonRetriableSampleError):
                reason = NON_RETRIABLE_REASON
            else:
                reason = f"exception::{e.__class__.__name__}"
            new_ctx = ctx.to_failed(action, reason, _failure_message(e))

        # Outside the handler above: a failed `meta.json` write fails the task,
        # not this sample.
        if action is TaskAction.PREPROCESS:
            self._note_sample_digests(new_ctx)
            await self._maybe_write_input_identity()
        try:
            await compute_send.send(new_ctx)
        except BaseException:
            # Stopped while waiting for room: keep a finished sample for the drain.
            if new_ctx.is_terminal():
                self._contexts[new_ctx.sample_id] = new_ctx
            raise

    async def _infer(self, ctx: TaskContext) -> tuple[Any, list[str]]:
        """Run the task's ``infer``; ``(result, context overflows scored)``.

        Run as it always was first, so a task that handles an overflow itself
        keeps doing so. When one escapes, the run that used to fail the sample,
        ``infer`` runs again with refusals returned as empty outputs (see
        :mod:`sieval.core.models.overflow`); the calls already answered are
        replayed, not sent again, but the rest of ``infer`` runs twice. A task
        declaring ``context_overflow = "fail"`` fails the sample instead, as
        non-retriable.
        """
        model = self._task.model
        pre = ctx.preprocess_result
        replay = self._overflow_replays.pop(ctx.sample_id, None)
        if replay is not None:
            scope = OverflowScope(model, convert=True, replay=replay)
            try:
                with overflow_scope(scope):
                    result = await self._task.infer(pre, ctx)
            except ReplayAmbiguous:
                pass
            except Exception:
                if not scope.ambiguous:
                    raise
            else:
                if not scope.ambiguous:
                    return result, scope.messages
            logger.info(
                "{}: the recorded context overflow of sample {} cannot say which "
                "call it answered; inferring it again",
                self._task.name,
                ctx.sample_id,
            )
        observed = OverflowScope(model)
        try:
            with overflow_scope(observed):
                return await self._task.infer(pre, ctx), []
        except Exception as exc:
            message = observed.escaping(exc)
            if message is None:
                raise
            if self._task.context_overflow == "fail":
                raise ContextOverflowError(message) from exc
        scope = OverflowScope(model, convert=True, known=observed.calls)
        with overflow_scope(scope):
            result = await self._task.infer(pre, ctx)
        return result, scope.messages

    def _unstored_terminal(
        self, compute_leftover: MemoryObjectReceiveStream[TaskContext]
    ) -> list[TaskContext]:
        """Finished samples never handed to storage, once the streams stopped.

        Includes those still in *compute_leftover*.
        """
        with compute_leftover:
            while True:
                try:
                    ctx = compute_leftover.receive_nowait()
                except (anyio.WouldBlock, anyio.EndOfStream):
                    break
                if ctx.is_terminal():
                    self._contexts[ctx.sample_id] = ctx
        return [
            ctx
            for sid, ctx in self._contexts.items()
            if ctx.is_terminal() and sid not in self._stored_terminal
        ]

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
        counts: "SievalCounts | None" = None,
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
        :meth:`_finalize`, which every exit path goes through, catches
        :class:`UnreadableIntervalError`, writes what is left, and re-raises it
        unchanged.

        *counts* go in as the ``sieval`` block (see :class:`SievalCounts`).
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
            if counts is not None:
                block = cast(dict[str, JSONValue], dict(counts))
                requests = self._request_counts()
                if requests is not None:
                    block["requests"] = cast(JSONValue, requests)
                as_dict[SIEVAL_COUNTS_KEY] = block
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

    async def _finalize(self, cached_report: Any = _NOT_REPORTED) -> Any:
        """Finish a run the same way on every exit path.

        Expects every terminal sample hydrated in ``_contexts``. Writes the
        profile, ``report.json`` (a fast resume passes *cached_report* and
        reuses it), ``anomalies.json``, the ended ``runs[]`` entry and
        ``progress.json``.
        """
        finals, fails = self._final_and_failed()
        anomalies = self._anomalies_of(finals)

        await self._flush_request_log()
        if self._request_log is not None:
            self._profiler.set_requests(self._request_log.summary())
        await self._profile(finals + fails)

        refusal: UnreadableIntervalError | None = None
        reused = cached_report is not _NOT_REPORTED
        reported = False
        if reused:
            report = cached_report
        else:
            report = await self._task.report(finals, fails)
            counts = _counts(self._total_samples, finals, fails, anomalies)
            try:
                reported = await self._save_report_with_diagnostics(
                    report, finals, fails, counts
                )
            except UnreadableIntervalError as exc:
                # Held, not swallowed: `report.json` is on disk, so write the
                # remaining artifacts first, then re-raise below.
                refusal = exc

        rules_hash = None
        if not reused or self._config.detect_anomalies_on_resume:
            rules_hash = await self._write_anomalies(anomalies)
            if reused and rules_hash is not None:
                await self._refresh_anomaly_count(report, finals, fails, anomalies)
        if refusal is None:
            if reused:
                await self._saver.end_run(
                    (), rules_hash=rules_hash, resume=self._resume_plan
                )
            else:
                await self._end_run(reported=reported, rules_hash=rules_hash)
        await self._finish_progress(finals, fails, anomalies)
        if refusal is not None:
            raise refusal
        return report

    async def _refresh_anomaly_count(
        self,
        report: dict,
        finals: list[TaskContext],
        fails: list[TaskContext],
        anomalies: dict[str | int, dict[int, dict[str, list[int]]]],
    ) -> None:
        """Match a reused report's ``sieval`` anomaly counts to the re-detection."""
        counts = report[SIEVAL_COUNTS_KEY]
        current = _counts(self._total_samples, finals, fails, anomalies)
        stale = {
            key: current[key]
            for key in ("anomalies", "context_overflow")
            if counts.get(key) != current[key]
        }
        if stale:
            counts.update(stale)
            await self._saver.save_report(report)

    def _anomalies_of(
        self, finals: list[TaskContext]
    ) -> dict[str | int, dict[int, dict[str, list[int]]]]:
        """This run's per-iteration anomalies, plus detection on the final
        samples an earlier run recorded."""
        results: dict[str | int, dict[int, dict[str, list[int]]]] = {
            sid: dict(iterations) for sid, iterations in self._anomaly_results.items()
        }
        if not self._config.detect_anomalies:
            return results
        for ctx in finals:
            if ctx.iteration in results.get(ctx.sample_id, {}):
                continue
            found = self._anomaly_detector.detect(ctx, task_tags=self._task.tags)
            if found:
                results.setdefault(ctx.sample_id, {})[ctx.iteration] = {
                    rule: sorted(indices) for rule, indices in found.items()
                }
        return results

    async def _write_anomalies(
        self, results: dict[str | int, dict[int, dict[str, list[int]]]]
    ) -> str | None:
        """Write ``anomalies.json``; the rules hash it used, else ``None``."""
        if not self._config.detect_anomalies:
            return None
        return await self._anomaly_detector.write(results, self._task.name)

    async def _profile(self, contexts: list[TaskContext]) -> None:
        """Rebuild ``profile.json`` from every terminal sample's ``stage_meta``."""
        by_id = {ctx.sample_id: ctx for ctx in contexts}
        if self._config.record_meta:
            self._profiler.include_stage_meta()
        self._profiler.aggregate_stage_timings(by_id)
        self._profiler.aggregate_token_usage(by_id)
        self._profiler.log_summary()
        await self._profiler.save(self._root_dir)

    async def _finish_progress(
        self,
        finals: list[TaskContext],
        fails: list[TaskContext],
        anomalies: dict[str | int, dict[int, dict[str, list[int]]]],
    ) -> None:
        """Write ``progress.json`` with ``finished: true``."""
        progress = self._progress
        if progress is None:
            progress = TaskProgress(
                total=self._total_samples,
                desc=f"Running {self._task.name}",
                show_progress=False,
                root_dir=self._root_dir,
                dump_progress=self._config.dump_progress,
            )
            details: dict[str, int] = defaultdict(int)
            for iterations in anomalies.values():
                for rule in set().union(*iterations.values()):
                    details[rule] += 1
            progress.init_state(
                [c.sample_id for c in finals + fails],
                failed_ids=[c.sample_id for c in fails],
                anomaly_ids=[sid for sid, its in anomalies.items() if its],
                anomaly_details=details,
            )
        progress.finish()
        await progress.aclose()

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
        """Gate *src* and read its ``meta.json``; ``arun`` copies its shards.

        *src* is read, never written; this run records it in ``runs[].from``.
        """
        if not src.is_dir():
            raise FileNotFoundError(f"resume directory not found: {src}")
        gate_resume_version(src, __version__, task=self._task.name)
        recorded = _read_run_meta(src)
        block = (recorded or {}).get("identity")
        self._recorded = recorded
        # As that directory recorded it, a schema-1 block included.
        self._run_from = {
            "dir": str(src),
            "identity": block if isinstance(block, dict) else empty_identity(),
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

    def _resumes_own_identity(self) -> bool:
        """Whether this resumes its own directory, whose run recorded identities.

        Then a sample with no digests is rerun rather than refused.
        """
        return self._run_from is None and recorded_identity(self._recorded) is not None

    def _recorded_grading_change(self) -> str | None:
        """The earliest grading stage the recorded run graded differently.

        A run without identities is trusted only as :meth:`_trusts_unrecorded`
        says. A schema-1 ``grader`` matches when its key is one an earlier
        release would compute for this task now (:func:`legacy_grader_keys`).
        """
        identity = recorded_identity(self._recorded)
        if identity is None:
            return None if self._trusts_unrecorded() else GRADING_STAGES[0]
        current = self._grading_identities() or {}
        grader = recorded_grader(self._recorded)
        legacy = None
        if grader is not None:
            legacy = grader_key(grader) in self._legacy_grader_keys()
        return earliest_grading_change(identity, current, legacy)

    def _legacy_grader_keys(self) -> set[str]:
        """:func:`legacy_grader_keys` for this task, walked once per runner."""
        if self._legacy_keys is None:
            self._legacy_keys = self._compute_legacy_grader_keys()
        return self._legacy_keys

    def _compute_legacy_grader_keys(self) -> set[str]:
        try:
            judges = judge_identities(
                self._task.judge_models(), self._task.judge_prompts()
            )
            task_cls = type(self._task)
            return legacy_grader_keys(
                task_cls,
                judges,
                service_identities(task_cls.services),
                self._task.grader_args(),
            )
        except IDENTITY_ERRORS as exc:
            logger.warning("Legacy grader identity unavailable: {}", exc)
            return set()

    def _resume_identity_update(self, *, report_unknown: bool = False) -> dict:
        """This run's preprocess and grading identities, which a fast resume or
        re-report writes at its start to upgrade an older block in place.
        ``report`` stays unknown until a re-report writes it."""
        update: dict = {"preprocess": self._preprocess_identity()}
        grading = self._grading_identities()
        if grading is not None:
            update.update(grading)
            if report_unknown:
                update["report"] = None
        return update

    async def _recorded_preprocess_matches(self) -> bool:
        """Whether a finished run's ``preprocess`` is this run's: by
        :func:`preprocess_key`, or for rendered prompts (schemas 1 and 2) by
        rendering every recorded sample once."""
        identity = recorded_identity(self._recorded)
        if identity is None:
            return True
        recorded = identity["preprocess"]
        if is_code_identity(recorded):
            mine = preprocess_key(self._preprocess_identity())
            return mine is not None and mine == preprocess_key(recorded)
        if self._legacy_prompts_match is not None:
            return self._legacy_prompts_match
        contexts = await self._loader.load_initial_state()
        await self._loader.hydrate(
            contexts,
            set(),
            include_stages={TaskStage.FINAL, TaskStage.FAILED},
            prepare_retries=False,
            record_each_stage=self._record_each_stage,
        )
        self._legacy_prompts_match = await self._legacy_prompts_render(
            contexts.values()
        )
        return self._legacy_prompts_match

    async def _legacy_prompts_render(self, contexts: Iterable[TaskContext]) -> bool:
        """Whether every unstamped recorded sample of *contexts* renders the
        prompt it recorded: one ``preprocess`` call each. Samples whose data
        changed rerun anyway and are skipped."""
        checked = 0
        for ctx in contexts:
            if ctx.preprocess_result is None:
                continue
            ctx = self._ensure_raw_sample(ctx)
            stamps = recorded_sample_digests(ctx)
            if stamps["preprocess"] is not None:
                continue
            data = stamps["data"]
            if stamps["source"] == "derived":
                data = (
                    self._safe_digest(ctx.raw_sample)
                    if self._recorded_data_matches()
                    else None
                )
            if data is None or data != self._safe_digest(ctx.raw_sample):
                continue
            checked += 1
            if stamps["prompt"] is None or (
                stamps["prompt"] != await self._current_prompt_digest(ctx)
            ):
                logger.info(
                    "{}: sample {} renders another prompt than it recorded; "
                    "preprocess changed, every sample reruns",
                    self._task.name,
                    ctx.sample_id,
                )
                return False
        if checked:
            logger.info(
                "{}: {} recorded prompt(s) re-rendered unchanged; preprocess "
                "identity upgraded in place",
                self._task.name,
                checked,
            )
        return True

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
        raise StageResumeError(
            "inference",
            f"Resume refused: stage `inference` differs ({reason}).\n"
            f"  recorded: {self._recorded_dir()}\n"
            "  Inference is reused, never rerun, by a resume. Start fresh in a "
            "new result_dir, or match the model, its request arguments, the "
            "task's inference arguments and the serving recipe.",
            task=self._task.name,
        )

    def _recorded_dir(self) -> str:
        return self._run_from["dir"] if self._run_from else str(self._root_dir)

    async def plan_run(self) -> RunPlan:
        """Decide what :meth:`arun` would do, without doing or writing any of it.

        Runs the resume planning :meth:`arun` runs (:meth:`_plan_recorded`)
        against the recorded directory, read where it lies: ``resume=<dir>`` is
        not copied. Raises what :meth:`arun` would raise before its first
        sample, such as a :class:`StageResumeError`. Calls the task's
        ``setup()`` and ``shutdown()``, as :meth:`arun` does. A runner that
        planned is not run.
        """
        await self._task.setup()
        try:
            return await self._plan_run()
        finally:
            await self._task.shutdown()

    async def _plan_run(self) -> RunPlan:
        test_set = self._task.dataset.test_set
        size = len(test_set) if test_set else 0
        if not (self._resumed_from_existing or self._run_from is not None):
            return {
                "start": "fresh",
                "recorded": None,
                "samples": size,
                "outcomes": None,
                "stages": None,
                "new": size,
                "to_infer": size,
                "to_grade": size,
            }
        recorded = self._recorded_dir()
        reusable = (
            await self._reusable_report() if self._resumed_from_existing else None
        )
        if reusable is not None:
            outcomes = _no_outcomes()
            outcomes["kept"] = size
            if reusable[1]:
                outcomes["rereported"] = size
            return {
                "start": "reuse",
                "recorded": recorded,
                "samples": size,
                "outcomes": outcomes,
                "stages": self._stage_plan(),
                "new": 0,
                "to_infer": 0,
                "to_grade": 0,
            }
        if self._run_from is not None:
            # `arun` copies the directory first; the plan reads it in place.
            self._loader = TaskLoader(
                task=self._task,
                root_dir=Path(self._run_from["dir"]),
                shard_read_concurrency=self._config.shard_read_concurrency,
                profiler=self._profiler,
            )
        await self._plan_recorded(self._inference_identity())
        to_infer, to_grade = self._pending_work()
        return {
            "start": "resume",
            "recorded": recorded,
            "samples": size,
            "outcomes": self._resume_plan,
            "stages": self._stage_plan(),
            "new": sum(1 for sid in range(size) if sid not in self._contexts),
            "to_infer": to_infer,
            "to_grade": to_grade,
        }

    def _stage_plan(self) -> ResumePlan:
        """:func:`plan_resume` over this runner's identities, with its decisions.

        ``preprocess`` is compared as code; a schema-1/2 one takes the verdict
        of the rendering pass planning made. ``rerun_from`` is the runner's,
        which trusts a directory with no identities; ``refused`` is ``None``,
        as a refusal raises before this.
        """
        current: dict[str, object] = {
            "data": self._dataset_identity(),
            "preprocess": self._preprocess_identity(),
            "inference": self._inference_identity(),
            **(self._grading_identities() or {}),
        }
        plan = plan_resume(self._recorded, current, self._legacy_grader_keys())
        if plan["legacy_preprocess"] and self._legacy_prompts_match is not None:
            plan["stages"]["preprocess"] = (
                "same" if self._legacy_prompts_match else "differs"
            )
        changed = plan["stages"]["preprocess"] == "differs"
        plan["rerun_from"] = (
            "preprocess"
            if changed or self._preprocess_restarted
            else self._recorded_grading_change()
        )
        plan["refused"] = None
        return plan

    def _pending_work(self) -> tuple[int, int]:
        """``(to_infer, to_grade)``: the samples a run starting now would send
        to the model under test, and those it would grade.

        Read after :meth:`_plan_recorded`. Counts samples with no record, and
        recorded ones that are not terminal and stay within ``max_retries``. A
        recorded context overflow the run scores from its record still counts
        as inferred, since a task that calls the model more than once infers
        it again. A sample ``feedback`` sends to another iteration after a
        regrade is not counted: the plan cannot know it (see
        :meth:`_check_served`).
        """
        test_set = self._task.dataset.test_set
        size = len(test_set) if test_set else 0
        to_infer = to_grade = 0
        for sid in range(size):
            ctx = self._contexts.get(sid)
            if ctx is not None and (
                ctx.is_terminal()
                or (
                    self._resumes
                    and self._max_retries is not None
                    and ctx.retry_count > self._max_retries
                )
            ):
                continue
            to_grade += 1
            if ctx is None or ctx.stage in (TaskStage.INITIAL, TaskStage.PREPROCESSED):
                to_infer += 1
        return to_infer, to_grade

    def _refuse_unserved(self) -> None:
        """Refuse, before recording anything, a run that would infer with no
        engine serving the model under test.

        :meth:`plan_run` counted the same samples through the same
        :meth:`_plan_recorded`, so this only fires when the directory changed
        between the plan and the run.
        """
        if not self._unserved_candidate:
            return
        to_infer, _ = self._pending_work()
        if not to_infer:
            return
        raise RuntimeError(
            f"Task '{self._task.name}' now has {to_infer} sample(s) to infer, but "
            "no engine was launched for its model: the plan made before launch "
            f"found none. Did {self._recorded_dir()} change since? Run again."
        )

    def _check_served(self, action: TaskAction) -> None:
        """Fail a sample about to infer with no engine serving the model.

        Reached when a rerun ``feedback`` sends a sample to another iteration,
        which no plan can foresee. The sample fails before any request, and a
        resume that launches the engine retries it.
        """
        if action is TaskAction.INFER and self._unserved_candidate:
            raise RuntimeError(
                "infer needs an engine for the model under test, and none was "
                "launched: the plan made before launch found nothing to infer. "
                "Resume to retry this sample."
            )

    def _abandon_copy(self) -> None:
        """Undo a ``resume=<dir>`` copy no run began on."""
        if self._run_from is None:
            return
        if self._copied:
            (self._root_dir / "meta.json").unlink(missing_ok=True)
        abandon_copy(self._root_dir, self._copied)
        self._copied = []

    def _recorded_stages_match(self) -> Literal["all", "report"] | None:
        """Whether a finished run may be reused: ``all`` its stages match,
        ``report`` all but the report, ``None`` otherwise.

        Same data, inference, postprocess and feedback reuse every sample.
        Preprocess is compared after, by :meth:`_recorded_preprocess_matches`,
        which may render the recorded samples.
        """
        identity = recorded_identity(self._recorded)
        if identity is None:
            trusted = (
                self._legacy_match.get("data") is not False
                and self._legacy_match.get("inference") is not False
            )
            return "all" if trusted else None
        inference = self._inference_identity()
        recorded_inference = identity["inference"]
        if (
            inference is None
            or recorded_inference is None
            or recorded_inference.get("digest") != inference["digest"]
        ):
            return None
        recorded_data = identity["data"]
        if (
            recorded_data is None
            or recorded_data.get("digest") is None
            or recorded_data != self._dataset_identity()
        ):
            return None
        change = self._recorded_grading_change()
        if change is None:
            return "all"
        return "report" if change == "report" else None

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
    def _sample_stamps(ctx: TaskContext) -> dict[str, str] | None:
        """The grading stage keys *ctx*'s grading records carry, or ``None``.

        ``{"grader": key}`` for a record a schema-1 run stamped, else
        ``{stage: key}`` per grading stage. ``None`` when no record says.
        """
        for stage in (TaskStage.FEEDBACK, TaskStage.POSTPROCESSED):
            for entry in reversed(ctx.stage_meta.get(stage.value, ())):
                if not isinstance(entry, dict):
                    continue
                identity = entry.get("identity")
                if not isinstance(identity, dict):
                    continue
                stamps = {k: v for k, v in identity.items() if isinstance(v, str)}
                if stamps:
                    return stamps
        return None

    def _sample_grading_change(self, ctx: TaskContext) -> str | None:
        """The earliest grading stage that graded *ctx* differently.

        Read from its own stamps when it has them: a sample an interrupted
        regrade already redid is current while the directory's grading
        identity is still unknown. Otherwise the directory's comparison
        decides.
        """
        stamps = self._sample_stamps(ctx)
        if stamps is None:
            return self._grading_change
        if "grader" in stamps:
            if recorded_grader(self._recorded) is None:
                # Upgraded to schema 2 since: that run found the stamp current,
                # so the directory's identity describes this sample.
                return self._grading_change
            return (
                None
                if stamps["grader"] in self._legacy_grader_keys()
                else GRADING_STAGES[0]
            )
        for stage in ("postprocess", "feedback"):
            # A stage the sample has not reached carries no stamp.
            if stage in stamps and stamps[stage] != self._stage_stamps[stage]:
                return stage
        # `report` runs over every sample at once: the directory says.
        return "report" if self._grading_change == "report" else None

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

        Each sample rolls back to the earliest stage that changed for it:
        changed data or preprocess restarts it (a preprocess change restarts
        every sample: all were stamped alike); a changed postprocess rolls it
        back to its recorded inference, a changed feedback to its recorded
        postprocess; a sample past iteration 0 restarts instead, since its
        later inference saw the old feedback. A changed report keeps every
        sample and recomputes ``report.json``. Samples outside the dataset are
        dropped. Returns the samples to rewrite as this run's.
        """
        test_set = self._task.dataset.test_set
        size = len(test_set) if test_set else 0
        changed: list[str | int] = []
        counts = _no_outcomes()
        if (
            self._legacy_prompts_match is None
            and not self._trusts_unrecorded()
            and not is_code_identity(
                (recorded_identity(self._recorded) or {}).get("preprocess")
            )
        ):
            in_range = [
                c
                for sid, c in self._contexts.items()
                if isinstance(sid, int) and 0 <= sid < size
            ]
            self._legacy_prompts_match = await self._legacy_prompts_render(in_range)

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

        if self._grading_change == "report":
            counts["rereported"] = counts["kept"]
        self._resume_plan = counts
        if self._unverified_inference:
            logger.warning(
                "{}: {} recorded sample(s) with inference rerun from the start: "
                "the run kept no per-sample data digests (`record_meta` off), so "
                "their data cannot be checked",
                self._task.name,
                self._unverified_inference,
            )
        logger.info(
            "Per-stage resume of {}: {} kept ({} re-reported), {} regraded, "
            "{} re-postprocessed, {} restarted, {} dropped",
            self._task.name,
            counts["kept"],
            counts["rereported"],
            counts["regraded"],
            counts["repostprocessed"],
            counts["restarted"],
            counts["dropped"],
        )
        return changed

    async def _plan_sample(
        self, ctx: TaskContext
    ) -> tuple[
        Literal["kept", "regraded", "repostprocessed", "restarted"], TaskContext
    ]:
        """One sample's resume: ``(outcome, context to continue from)``."""
        if self._trusts_unrecorded() or ctx.stage is TaskStage.INITIAL:
            return "kept", ctx
        ctx = self._ensure_raw_sample(ctx)
        restart = self._task.make_context(ctx.sample_id)
        current_data = self._safe_digest(ctx.raw_sample)
        stamps = recorded_sample_digests(ctx)
        data = stamps["data"]
        derived = stamps["source"] == "derived"
        if derived:
            # No per-sample identity on record (a run before them, or one with
            # `record_meta` off): the recorded data identity of the whole set,
            # else the caller's reading of the persisted config, decides.
            match = self._recorded_data_matches()
            if match is None and self._resumes_own_identity():
                # Unverifiable, so it reruns: a sample never preprocessed, or,
                # under `record_meta` off, any sample.
                if ctx.infer_result is not None:
                    self._unverified_inference += 1
                return "restarted", restart
            if match is None:
                raise StageResumeError(
                    "data",
                    "Resume refused: stage `data` is unknown: the recorded run "
                    "kept no data identity, and its shards do not hold the raw "
                    f"samples.\n  recorded: {self._recorded_dir()}",
                    task=self._task.name,
                )
            data = current_data if match else None
        if data is None or data != current_data:
            return "restarted", restart
        if ctx.preprocess_result is not None:
            if self._sample_preprocess_changed(stamps["preprocess"]):
                self._preprocess_restarted = True
                return "restarted", restart
            if derived:
                ctx = self._stamp_input(ctx, current_data)
        if not has_graded(ctx):
            return "kept", ctx
        change = self._sample_grading_change(ctx)
        if change in (None, "report"):
            return "kept", ctx
        if ctx.iteration > 0:
            return "restarted", restart
        if change == "feedback" and ctx.postprocess_result is not None:
            return "regraded", self._to_refeed(ctx)
        return "repostprocessed", self._to_regrade(ctx)

    def _sample_preprocess_changed(self, stamp: str | None) -> bool:
        """Whether a recorded sample's ``preprocess`` differs from this run's.

        *stamp* is the :func:`preprocess_key` its record carries. Without one,
        the directory's code identity decides if it has one (an upgrade
        verified the sample), else the rendering pass. Unknown differs.
        """
        current = preprocess_key(self._preprocess_identity())
        if stamp is not None:
            return current is None or stamp != current
        identity = recorded_identity(self._recorded)
        recorded = identity["preprocess"] if identity is not None else None
        if is_code_identity(recorded):
            return current is None or preprocess_key(recorded) != current
        return self._legacy_prompts_match is not True

    def _stamp_input(self, ctx: TaskContext, data: str | None):
        """Add the data digest and this run's preprocess key to *ctx*'s latest
        ``preprocess`` meta."""
        key = TaskStage.PREPROCESSED.value
        history = list(ctx.stage_meta.get(key) or [{}])
        last = dict(history[-1])
        last["identity"] = self._input_stamp(data)
        history[-1] = cast(TaskStageMeta, last)
        return replace(ctx, stage_meta={**ctx.stage_meta, key: history})

    def _grading_stamps(self, upto: TaskStage) -> dict[str, str | None]:
        """The grading stage keys a record of stage *upto* is stamped with.

        A ``postprocess`` record carries the ``postprocess`` key, a
        ``feedback`` record ``postprocess`` and ``feedback``: what produced it.
        """
        stages = ["postprocess"]
        if upto is TaskStage.FEEDBACK:
            stages.append("feedback")
        return {stage: self._stage_stamps[stage] for stage in stages}

    @staticmethod
    def _to_refeed(ctx: TaskContext) -> TaskContext:
        """Roll *ctx* back to its recorded ``postprocess``, to judge it again."""
        keep = {
            TaskStage.PREPROCESSED.value,
            TaskStage.INFERRED.value,
            TaskStage.POSTPROCESSED.value,
        }
        return replace(
            ctx,
            stage=TaskStage.POSTPROCESSED,
            feedback_result=None,
            error_action=None,
            error_reason=None,
            error_msg=None,
            retry_count=0,
            stage_meta={k: v for k, v in ctx.stage_meta.items() if k in keep},
        )

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

    def _config_path(self) -> str | None:
        """``effective_config`` relative to this directory."""
        configured = self._config.effective_config
        if not configured:
            return None
        return os.path.relpath(
            Path(configured).expanduser().resolve(), self._root_dir.resolve()
        )

    def _model_name(self) -> str:
        return self._task.model.meta()["model"]

    def _recorded_model(self) -> str | None:
        model = (self._recorded or {}).get("model")
        return model if isinstance(model, str) else None

    def _model_differs(self) -> bool:
        """Whether the recorded run was for another model under test."""
        recorded = self._recorded_model()
        return recorded is not None and recorded != self._model_name()

    def _check_model_switch(self) -> bool:
        """Refuse a resume with another model that would mix two models' samples.

        Returns whether recorded inference is kept, so the directory stays the
        recorded model's. Keeping none (every sample failed before inference)
        makes it this model's run.
        """
        if not any(c.infer_result is not None for c in self._contexts.values()):
            return False
        test_set = self._task.dataset.test_set
        size = len(test_set) if test_set else 0
        to_infer = sum(
            1
            for sid in range(size)
            if (ctx := self._contexts.get(sid)) is None
            or (not ctx.is_terminal() and ctx.infer_result is None)
        )
        if not to_infer:
            return True
        raise StageResumeError(
            "inference",
            f"Resume refused: {to_infer} sample(s) would be inferred by "
            f"{self._model_name()!r}, but the recorded samples are "
            f"{self._recorded_model()!r}'s.\n"
            f"  recorded: {self._recorded_dir()}\n"
            "  Start fresh in a new result_dir, or resume with the recorded model.",
            task=self._task.name,
        )

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

    def _grading_identities(self) -> dict[str, GradingIdentity] | None:
        """This run's ``postprocess``, ``feedback`` and ``report`` identities.

        The stage code is walked once per runner; judge versions are read
        each time, as the API reports them. Sets the keys each grading record
        is stamped with.
        """
        try:
            judges = judge_identities(
                self._task.judge_models(),
                self._task.judge_prompts(),
                versions=self._judge_versions,
            )
            task_cls = type(self._task)
            code = self._stage_code_once([j["role"] for j in judges])
            grading = grading_identities(
                code,
                judges,
                service_identities(task_cls.services),
                self._task.grader_args(),
                grader_task(task_cls),
            )
        except IDENTITY_ERRORS as exc:
            logger.warning("Grading identities unavailable: {}", exc)
            self._stage_stamps = dict.fromkeys(GRADING_STAGES)
            return None
        self._stage_stamps = {s: stage_key(grading[s]) for s in GRADING_STAGES}
        self._grading = grading
        return grading

    def _stage_code_once(self, judge_roles: list[str]) -> dict:
        """:func:`stage_code` for this task, walked once per runner."""
        if self._stage_code is None:
            task_cls = type(self._task)
            self._stage_code = stage_code(
                task_cls,
                judge_roles,
                task_cls.services,
                list(self._task.grader_args()),
            )
            for stage, code in self._stage_code.items():
                if "fallback" in code:
                    logger.info(
                        "{}: {} counts all of its task code ({})",
                        self._task.name,
                        stage,
                        code["fallback"],
                    )
        return self._stage_code

    def _preprocess_identity(self) -> PreprocessIdentity | None:
        """This run's ``preprocess`` identity, computed once: no sample is
        preprocessed for it. ``None`` when it cannot be computed."""
        if self._preprocess is False:
            try:
                roles = [j["role"] for j in self._current_judges()]
                code = self._stage_code_once(roles)["preprocess"]
                self._preprocess = task_preprocess_identity(self._task, code)
            except IDENTITY_ERRORS as exc:
                logger.warning("Preprocess identity unavailable: {}", exc)
                self._preprocess = None
        return self._preprocess

    def _input_stamp(self, data: str | None) -> dict[str, str | None]:
        """What a ``preprocess`` record is stamped with: its sample's data
        digest and this run's :func:`preprocess_key`."""
        return {
            "data": data,
            "preprocess": preprocess_key(self._preprocess_identity()),
        }

    def _seed_judge_versions(self, meta: dict | None) -> None:
        """Carry judge versions a resumed run's ``meta.json`` recorded.

        A resume that grades nothing new never hears from the judge API, and
        the version it reported earlier still describes the verdicts on disk.
        Carried only into a role still held by the same model and sampling.
        """
        identity = (meta or {}).get("identity") or {}
        # The schema-1 `grader`, else schema 2's `feedback`, which holds every judge.
        grader = identity.get("grader") or identity.get("feedback") or {}
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

    def judge_roles(self) -> set[str]:
        """The judge roles this task grades with; empty when they cannot be read."""
        return {judge["role"] for judge in self._current_judges()}

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
        """Record *ctx*'s data digest once, as of its first preprocess.

        Prefers the digest its ``preprocess`` record carries -- what that run
        saw -- and falls back to digesting the sample.
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
        self._sample_digests[ctx.sample_id] = (ctx.repeat_index, data)

    async def _maybe_write_input_identity(self, force: bool = False) -> None:
        """Write the data identity once every sample is past preprocess.

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
            data=aggregate_digest((sid, rep, d) for sid, (rep, d) in entries)
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

    async def _end_run(self, *, reported: bool, rules_hash: str | None = None) -> None:
        """Finish this run's ``runs[]`` entry once its report is settled.

        *reported* says ``report.json`` was written; ``report()`` returning
        ``None`` writes none.

        Writes the data identity if a sample never reached preprocess, and the
        preprocess one if a resume left it unknown; folds what the candidate
        was sent into the inference identity, and rewrites the grading ones
        with the judge versions reported.
        """
        await self._maybe_write_input_identity(force=True)
        await self._saver.set_identity(preprocess=self._preprocess_identity())
        inference = self._saver.inference_identity
        if inference is not None:
            await self._saver.set_identity(
                inference={
                    "digest": inference["digest"],
                    "requests": self._requests_identity(),
                }
            )
        # Always: a regrade left them unknown until now.
        grading = self._grading_identities()
        await self._saver.set_identity(
            **{s: (grading or {}).get(s) for s in GRADING_STAGES}
        )
        ran = self._ran_actions
        reran = set()
        if TaskAction.PREPROCESS in ran:
            reran |= {"data", "preprocess"}
        if TaskAction.INFER in ran:
            reran.add("inference")
        if TaskAction.POSTPROCESS in ran:
            reran.add("postprocess")
        if TaskAction.FEEDBACK in ran:
            reran.add("feedback")
        if reported:
            reran.add("report")
        await self._saver.end_run(
            reran,
            reported=reported,
            rules_hash=rules_hash,
            requests=self._request_counts(),
            resume=self._resume_plan,
        )

    def _resolve_result_dir(
        self, result_dir: str | None, task: Task, resumes: bool
    ) -> tuple[bool, Path]:
        if result_dir:
            p = Path(result_dir).expanduser()
            if p.exists() and p.is_file():
                raise ValueError("result_dir is a file")
            if self._holds_run(p):
                if not resumes:
                    raise ResultDirExistsError(p)
                return True, p
            return False, p
        root = Path("outputs") / task.name
        if resumes and root.exists():
            candidates = [
                d
                for d in root.iterdir()
                if d.is_dir() and re.fullmatch(r"\d{14}", d.name)
            ]
            if candidates:
                latest = max(candidates, key=lambda x: x.name)
                if self._holds_run(latest):
                    return True, latest
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return False, root / ts

    def _holds_run(self, p: Path) -> bool:
        """Whether *p* holds a run this one would resume or refuse.

        An unfinished ``resume=<dir>`` copy is checked first, since it may not
        have reached an ``.idx`` yet: this run copies again if it has a
        ``resume`` directory, else :class:`UnfinishedCopyError` names the source.
        """
        if not p.exists():
            return False
        source = _unfinished_copy(p)
        if source is None:
            return run_started(p)
        if resume_source(self._config.resume, p) is None:
            raise UnfinishedCopyError(p, source)
        logger.info("{} holds an unfinished copy of {}; copying again", p, source)
        return False

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
