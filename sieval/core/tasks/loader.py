"""Persistence recovery: the ``.idx`` scan and shard hydration.

A sample's state is its latest shard record, found through ``.idx``. The saver
appends a record before its ``.idx`` line, so a record no line points at was
never committed: that stage reruns. A line past its shard's end is dropped.
"""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import anyio
import orjson
from anyio.to_thread import run_sync
from loguru import logger

from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    register_types,
)

from .consts import (
    DEPENDENCY_STAGE_RANKS,
    ERROR_ACTION_CLEAR_FIELDS,
    ERROR_ACTION_PREV_STAGE,
    ERROR_REASONS_NON_RETRIABLE,
    STAGE_DIRS,
    STAGE_RANK,
    STAGE_TO_RESULT_FIELD,
    TaskAction,
    TaskStage,
)
from .context import TaskContext
from .profiler import TaskProfiler, TaskProfilerContext
from .task import Task


class ShardOffset(TypedDict):
    """Shard file location for a single sample: file path + byte range."""

    shard: Path
    offset: int
    length: int


class ShardOffsetWithIteration(TypedDict):
    """Shard file location tagged with the iteration and run it was written in."""

    shard: Path
    offset: int
    length: int
    iteration: int
    run: int


# (run, iteration, stage_rank, shard_path, byte_offset, byte_length)
# stage_rank uses -1 when comparing stage-only offsets (for dependency loading)
OffsetMeta = tuple[int, int, int, Path, int, int]
"""Tuple ``(run, iteration, stage_rank, shard_path, byte_offset, byte_length)``
used for deterministic latest-offset selection.

``run`` is the ordinal of the run that wrote the record, ``0`` for a record
written before the column existed: a later run's record supersedes an earlier
one's even at a lower stage, which is how a per-stage resume rewinds a sample.
``stage_rank`` is set to ``-1`` for per-stage comparisons (dependency loading)
where only iteration and file position matter."""


def _hydrate_model_call_provenance(meta: Any, registry: dict[str, type]) -> Any:
    """Rehydrate typed provenance without interpreting arbitrary stage metadata.

    ``TaskStageMeta`` otherwise stays a JSON-shaped mapping after loading.  Walk
    only its declared ``model_calls[*].model.provenance`` seam so a user-owned
    ``extra`` mapping containing a coincidental ``__sieval_cls__`` key remains
    untouched.
    """

    if not isinstance(meta, dict):
        return meta
    model_calls = meta.get("model_calls")
    if not isinstance(model_calls, list):
        return meta

    hydrated_calls: list[Any] = []
    changed = False
    for call in model_calls:
        if not isinstance(call, dict):
            hydrated_calls.append(call)
            continue
        model = call.get("model")
        if not isinstance(model, dict) or "provenance" not in model:
            hydrated_calls.append(call)
            continue
        hydrated_model = dict(model)
        hydrated_model["provenance"] = dict_to_obj(model["provenance"], registry)
        hydrated_call = dict(call)
        hydrated_call["model"] = hydrated_model
        hydrated_calls.append(hydrated_call)
        changed = True

    if not changed:
        return meta
    hydrated_meta = dict(meta)
    hydrated_meta["model_calls"] = hydrated_calls
    return hydrated_meta


def _hydrate_stage_meta_history(value: Any, registry: dict[str, type]) -> Any:
    """Rehydrate provenance in a stage-name -> metadata-history mapping."""

    if not isinstance(value, dict):
        return value
    return {
        stage: (
            [_hydrate_model_call_provenance(meta, registry) for meta in history]
            if isinstance(history, list)
            else history
        )
        for stage, history in value.items()
    }


class IdxRecord(TypedDict):
    """Parsed representation of a single line from a ``.idx`` shard index file.

    Required fields map to the first five TSV columns; optional fields are
    present only when the idx line contains error/retry information.
    """

    sample_id: str | int
    iteration: int
    stage: TaskStage
    offset: int
    length: int
    error_action: NotRequired[str]
    error_reason: NotRequired[str | None]
    retry_count: NotRequired[int]
    run: NotRequired[int]


def _should_replace_offset(current: OffsetMeta | None, candidate: OffsetMeta) -> bool:
    """
    Choose the latest offset deterministically for stage and best hydration records.

    Priority:
    1) later run
    2) higher iteration
    3) higher stage rank (use -1 for stage-only comparisons)
    4) for same shard, larger offset (later append)
    5) deterministic fallback by (shard path, offset, length)
    """
    if current is None:
        return True

    cand_run, cand_it, cand_rank, cand_shard, cand_offset, cand_length = candidate
    cur_run, cur_it, cur_rank, cur_shard, cur_offset, cur_length = current

    if cand_run != cur_run:
        return cand_run > cur_run

    if cand_it != cur_it:
        return cand_it > cur_it

    if cand_rank != cur_rank:
        return cand_rank > cur_rank

    if cand_shard == cur_shard:
        return cand_offset > cur_offset

    cand_key = (str(cand_shard), cand_offset, cand_length)
    cur_key = (str(cur_shard), cur_offset, cur_length)
    return cand_key > cur_key


def _try_task_action(val: str | None) -> TaskAction | None:
    if not val:
        return None
    try:
        return TaskAction(val)
    except ValueError:
        return None


def _parse_idx_line(parts: list[str]) -> IdxRecord | None:
    """Parse a single TSV-split ``.idx`` line into an ``IdxRecord``."""
    if len(parts) < 5:
        return None
    try:
        try:
            sid: str | int = int(parts[0])
        except ValueError:
            sid = parts[0]

        rec: IdxRecord = {
            "sample_id": sid,
            "iteration": int(parts[1]),
            "stage": TaskStage(parts[2]),
            "offset": int(parts[3]),
            "length": int(parts[4]),
        }
        if len(parts) > 5 and parts[5]:
            rec["error_action"] = parts[5]
        if len(parts) > 6 and parts[6]:
            rec["error_reason"] = parts[6]
        if len(parts) > 7 and parts[7]:
            rec["retry_count"] = int(parts[7])
        if len(parts) > 8 and parts[8]:
            rec["run"] = int(parts[8])
        return rec
    except (ValueError, KeyError):
        return None


def _parse_idx_file(path: Path) -> list[IdxRecord]:
    """Parse an ``.idx`` file, skipping and logging malformed lines."""
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out: list[IdxRecord] = []
    for line in lines:
        if not line:
            continue
        parts = line.split("\t")
        rec = _parse_idx_line(parts)
        if rec is not None:
            out.append(rec)
        else:
            logger.warning("Skipping malformed idx line in {}: {!r}", path, line)
    return out


def _within_shard(
    records: list[IdxRecord], idx_path: Path, shard: Path
) -> list[IdxRecord]:
    """The records of *idx_path* that lie inside *shard*; the rest are dropped."""
    try:
        size = shard.stat().st_size
    except OSError:
        size = 0
    kept = [r for r in records if r["offset"] + r["length"] <= size]
    if len(kept) < len(records):
        logger.warning(
            "Dropped {} line(s) of {} pointing past the end of {} ({} bytes)",
            len(records) - len(kept),
            idx_path,
            shard,
            size,
        )
    return kept


class TaskLoader:
    """Persistence recovery engine.

    Scans the shard offset indices (``.idx`` files) into skeleton contexts and
    hydrates :class:`TaskContext` objects from the shard records they point at.
    """

    def __init__(
        self,
        task: Task,
        root_dir: Path,
        shard_read_concurrency: int = 8,
        profiler: TaskProfiler | None = None,
    ):
        """Set up type registry by merging global registry with task-module types."""
        self._task = task
        self._latest: dict[str | int, IdxRecord] = {}
        self._sample_offsets: dict[str | int, ShardOffset] = {}
        self._all_stage_offsets: dict[
            str | int, dict[TaskStage, ShardOffsetWithIteration]
        ] = {}

        self._root_dir = root_dir
        self._report_path = self._root_dir / "report.json"

        self._shard_read_concurrency = max(1, shard_read_concurrency)

        self._profiler = profiler or TaskProfiler()

        # Type registry setup
        self._type_registry: dict[str, type] = dict(global_type_registry)
        task_mod = sys.modules.get(task.__class__.__module__)
        if task_mod:
            register_types(
                self._type_registry,
                [v for v in vars(task_mod).values() if isinstance(v, type)],
                task.__class__.__module__,
            )

    async def load_initial_state(self) -> dict[str | int, TaskContext]:
        """Scan the ``.idx`` files into a skeleton context per sample."""
        async with TaskProfilerContext(self._profiler, "load_or_compensate"):
            await anyio.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
            await self._scan_idx()

            contexts = {}
            for sid, r in self._latest.items():
                base = self._task.make_context(sid, None)
                ctx = replace(
                    base,
                    iteration=r["iteration"],
                    stage=r["stage"],
                    error_action=_try_task_action(r.get("error_action")),
                    error_reason=r.get("error_reason"),
                    retry_count=r.get("retry_count", 0),
                )
                contexts[sid] = ctx
            return contexts

    async def load_cached_report(self) -> Any | None:
        """Return the parsed ``report.json`` if it exists, or ``None``.

        Used for fast resume: if the report is already on disk the pipeline
        can skip hydration and reporting entirely.
        """
        if not await anyio.Path(self._report_path).exists():
            return None
        try:
            logger.info("Loading cached report from: {}", self._report_path)
            async with await anyio.open_file(self._report_path, "rb") as f:
                content = await f.read()
                return orjson.loads(content)
        except (OSError, orjson.JSONDecodeError) as e:
            logger.warning(
                "Failed to load cached report: {}. Proceeding with normal hydration.",
                e,
            )
            return None

    async def hydrate(
        self,
        contexts: dict[str | int, TaskContext],
        hydrated_ids: set[str | int],
        include_stages: set[TaskStage],
        prepare_retries: bool = False,
        record_each_stage: bool = True,
    ) -> None:
        """Read shard data to fill result fields on the given contexts.

        Contexts whose ``stage`` is in *include_stages* and whose
        ``sample_id`` is not already in *hydrated_ids* are targeted.  When
        *record_each_stage* is ``True``, dependency stage data (earlier
        pipeline stages from the same iteration) is also loaded so that
        downstream stages have access to prior results.

        If *prepare_retries* is set, failed contexts are reset for retry
        after hydration completes.
        """
        async with TaskProfilerContext(self._profiler, "hydrate"):
            targets = [
                c
                for c in contexts.values()
                if c.stage in include_stages and c.sample_id not in hydrated_ids
            ]
            if not targets:
                if prepare_retries:
                    self._prepare_failed_retries(contexts, record_each_stage)
                return

            # Build shard map for main stage
            shard_map: dict[Path, list[tuple[TaskContext, int, int]]] = {}
            for c in targets:
                off = self._sample_offsets.get(c.sample_id)
                if not off:
                    continue
                shard_map.setdefault(off["shard"], []).append(
                    (c, off["offset"], off["length"])
                )

            # Build shard map for dependency stages
            dependency_shard_map: dict[
                Path, list[tuple[TaskContext, TaskStage, int, int]]
            ] = {}
            if record_each_stage:
                for c in targets:
                    current_rank = STAGE_RANK[c.stage]
                    stage_offsets = self._all_stage_offsets.get(c.sample_id, {})

                    for dep_stage, dep_rank in DEPENDENCY_STAGE_RANKS:
                        if dep_rank >= current_rank:
                            continue

                        dep_off = stage_offsets.get(dep_stage)
                        if not dep_off:
                            continue

                        if dep_off["iteration"] == c.iteration:
                            dependency_shard_map.setdefault(
                                dep_off["shard"], []
                            ).append(
                                (c, dep_stage, dep_off["offset"], dep_off["length"])
                            )

            # Read main stage data
            if shard_map:
                limiter = anyio.CapacityLimiter(self._shard_read_concurrency)

                async def _read(shard: Path, items: list[tuple[TaskContext, int, int]]):
                    async with limiter:
                        await self._read_shard_items(
                            shard, items, contexts, hydrated_ids
                        )

                async with anyio.create_task_group() as tg:
                    for shard, lst in shard_map.items():
                        tg.start_soon(_read, shard, lst)

            # Read dependency stage data
            if dependency_shard_map:
                limiter = anyio.CapacityLimiter(self._shard_read_concurrency)

                async def _read_dep(
                    shard: Path, items: list[tuple[TaskContext, TaskStage, int, int]]
                ):
                    async with limiter:
                        await self._read_dependency_shard_items(shard, items, contexts)

                async with anyio.create_task_group() as tg:
                    for shard, lst in dependency_shard_map.items():
                        tg.start_soon(_read_dep, shard, lst)

            if prepare_retries:
                self._prepare_failed_retries(contexts, record_each_stage)

    async def _read_shard_items(
        self,
        shard: Path,
        items: list[tuple[TaskContext, int, int]],
        contexts: dict[str | int, TaskContext],
        hydrated_ids: set[str | int],
    ):
        """Async seek-and-read of main-stage records from a single shard file.

        Items are sorted by offset to allow sequential reads. Each read blob
        is forwarded to :meth:`_parse_and_hydrate` for deserialization.
        """
        try:
            sorted_items = sorted(items, key=lambda x: x[1])
            async with await anyio.open_file(shard, "rb") as f:
                for ctx, off, length in sorted_items:
                    if ctx.sample_id in hydrated_ids:
                        continue
                    try:
                        await f.seek(off)
                        blob = await f.read(length)
                        self._parse_and_hydrate(ctx, blob, contexts, hydrated_ids)
                    except Exception as e:
                        logger.warning("Failed to hydrate {}: {}", ctx.sample_id, e)
        except OSError as e:
            logger.opt(exception=True).error("Hydration error for {}: {}", shard, e)

    async def _read_dependency_shard_items(
        self,
        shard: Path,
        items: list[tuple[TaskContext, TaskStage, int, int]],
        contexts: dict[str | int, TaskContext],
    ):
        """Async seek-and-read of dependency-stage records from a single shard.

        Similar to :meth:`_read_shard_items` but each entry also carries the
        dependency ``TaskStage``, and blobs are forwarded to
        :meth:`_parse_and_merge_dependency`.
        """
        try:
            sorted_items = sorted(items, key=lambda x: x[2])
            async with await anyio.open_file(shard, "rb") as f:
                for ctx, dep_stage, off, length in sorted_items:
                    try:
                        await f.seek(off)
                        blob = await f.read(length)
                        self._parse_and_merge_dependency(ctx, dep_stage, blob, contexts)
                    except Exception as e:
                        logger.warning(
                            "Failed to load dependency {} for {}: {}",
                            dep_stage,
                            ctx.sample_id,
                            e,
                        )
        except OSError as e:
            logger.opt(exception=True).error(
                "Dependency hydration error for {}: {}", shard, e
            )

    def _parse_and_merge_dependency(
        self,
        ctx: TaskContext,
        dep_stage: TaskStage,
        blob: bytes,
        contexts: dict[str | int, TaskContext],
    ):
        """Deserialise a dependency-stage JSON blob and merge its result field
        and ``stage_meta`` into the context.

        Only the result field mapped by ``STAGE_TO_RESULT_FIELD`` and the
        ``meta_last`` entry are merged; other fields are ignored.
        """
        try:
            obj = orjson.loads(blob)
        except orjson.JSONDecodeError as e:
            logger.warning(
                "Failed to parse dependency JSON for {} (stage={}): {}",
                ctx.sample_id,
                dep_stage,
                e,
            )
            return

        result_field = STAGE_TO_RESULT_FIELD.get(dep_stage)
        if not result_field:
            return

        current_ctx = contexts.get(ctx.sample_id)
        if not current_ctx:
            return

        updates: dict[str, Any] = {}

        if result_field in obj:
            try:
                updates[result_field] = dict_to_obj(
                    obj[result_field], self._type_registry
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse {} for {}: {}",
                    result_field,
                    ctx.sample_id,
                    e,
                )

        # Merge stage_meta from dependency snapshot
        if "meta_last" in obj:
            stage_key = dep_stage.value
            if stage_key not in current_ctx.stage_meta:
                existing_meta = dict(current_ctx.stage_meta)
                existing_meta[stage_key] = [
                    _hydrate_model_call_provenance(
                        obj["meta_last"], self._type_registry
                    )
                ]
                updates["stage_meta"] = existing_meta

        if updates:
            contexts[ctx.sample_id] = replace(current_ctx, **updates)

    def _parse_and_hydrate(
        self,
        ctx: TaskContext,
        blob: bytes,
        contexts: dict[str | int, TaskContext],
        hydrated_ids: set[str | int],
    ):
        """Deserialise a main-stage JSON blob and update the context's result
        fields, error state, and stage metadata.

        On success the ``sample_id`` is added to *hydrated_ids* to prevent
        duplicate hydration.
        """
        try:
            obj = orjson.loads(blob)
        except orjson.JSONDecodeError as e:
            logger.warning(
                "Failed to parse hydration JSON for {}: {}",
                ctx.sample_id,
                e,
            )
            return

        updates: dict[str, Any] = {}

        if "raw_sample" in obj:
            updates["raw_sample"] = obj["raw_sample"]

        for field in STAGE_TO_RESULT_FIELD.values():
            if field in obj:
                try:
                    updates[field] = dict_to_obj(obj[field], self._type_registry)
                except Exception as e:
                    logger.warning(
                        "Failed to deserialize '{}' for {}: {}",
                        field,
                        ctx.sample_id,
                        e,
                    )

        if obj.get("error_action"):
            updates["error_action"] = _try_task_action(obj.get("error_action"))
        if obj.get("error_reason"):
            updates["error_reason"] = obj.get("error_reason")
        if obj.get("error_msg"):
            updates["error_msg"] = obj.get("error_msg")
        if obj.get("retry_count"):
            updates["retry_count"] = obj.get("retry_count", 0)

        if "stage_meta" in obj:
            updates["stage_meta"] = _hydrate_stage_meta_history(
                obj["stage_meta"], self._type_registry
            )
        elif "meta_last" in obj and "stage" in obj:
            updates["stage_meta"] = {
                obj["stage"]: [
                    _hydrate_model_call_provenance(
                        obj["meta_last"], self._type_registry
                    )
                ]
            }

        new_ctx = replace(ctx, **updates)
        contexts[ctx.sample_id] = new_ctx
        hydrated_ids.add(ctx.sample_id)

    # idx functions (sync I/O) are dispatched to a worker thread by the async methods
    def _collect_idx_files(self) -> list[tuple[Path, Path]]:
        idx_files: list[tuple[Path, Path]] = []
        if not self._root_dir.exists():
            return idx_files
        for iter_dir in self._root_dir.iterdir():
            if iter_dir.is_dir() and iter_dir.name.isdigit():
                for stage_dir in iter_dir.iterdir():
                    if stage_dir.is_dir() and stage_dir.name in STAGE_DIRS:
                        for idx_file in stage_dir.glob("*.idx"):
                            idx_files.append((idx_file, idx_file.with_suffix(".jsonl")))
        return idx_files

    async def _scan_idx(self) -> None:
        """Rebuild the latest record and offsets per sample from every ``.idx``.

        Lines past their shard's end are dropped; the latest record is chosen
        by :func:`_should_replace_offset`.
        """
        self._latest = {}
        self._sample_offsets = {}
        self._all_stage_offsets = {}
        idx_files = await run_sync(self._collect_idx_files)

        if not idx_files:
            return

        limiter = anyio.CapacityLimiter(self._shard_read_concurrency)
        best_records: dict[str | int, tuple[OffsetMeta, IdxRecord]] = {}
        all_stage_offsets: dict[
            str | int, dict[TaskStage, ShardOffsetWithIteration]
        ] = {}

        async def _scan_one(idx_path: Path, shard: Path) -> None:
            async with limiter:
                records = await run_sync(_parse_idx_file, idx_path)
                records = await run_sync(_within_shard, records, idx_path, shard)
                for r in records:
                    sid = r["sample_id"]
                    it_val = r["iteration"]
                    stg_val = r["stage"]
                    rank = STAGE_RANK[stg_val]
                    run_val = r.get("run", 0)

                    # Keep the latest offset per (sample, stage) for dependency loading.
                    off_with_it: ShardOffsetWithIteration = {
                        "shard": shard,
                        "offset": r["offset"],
                        "length": r["length"],
                        "iteration": it_val,
                        "run": run_val,
                    }
                    stage_map = all_stage_offsets.setdefault(sid, {})
                    existing = stage_map.get(stg_val)
                    stage_current_meta: OffsetMeta | None = None
                    if existing is not None:
                        stage_current_meta = (
                            existing["run"],
                            existing["iteration"],
                            -1,
                            existing["shard"],
                            existing["offset"],
                            existing["length"],
                        )
                    stage_candidate_meta: OffsetMeta = (
                        run_val,
                        it_val,
                        -1,
                        shard,
                        r["offset"],
                        r["length"],
                    )
                    if _should_replace_offset(stage_current_meta, stage_candidate_meta):
                        stage_map[stg_val] = off_with_it

                    candidate_meta: OffsetMeta = (
                        run_val,
                        it_val,
                        rank,
                        shard,
                        r["offset"],
                        r["length"],
                    )
                    prev = best_records.get(sid)
                    if prev and not _should_replace_offset(prev[0], candidate_meta):
                        continue

                    best_records[sid] = (candidate_meta, r)

        async with anyio.create_task_group() as tg:
            for idx, shard in idx_files:
                tg.start_soon(_scan_one, idx, shard)

        for sid, (meta, r) in best_records.items():
            _run, _it, _rank, shard, offset, length = meta
            self._latest[sid] = r
            self._sample_offsets[sid] = ShardOffset(
                shard=shard, offset=offset, length=length
            )

        self._all_stage_offsets = all_stage_offsets
        logger.info("Loaded {} samples from the shard indices.", len(self._latest))

    def prepare_failed_retries(
        self, contexts: dict[str | int, TaskContext], record_each_stage: bool
    ) -> None:
        """Reset failed contexts for retry; see :meth:`_prepare_failed_retries`.

        Public for a per-stage resume, which decides what to keep from the
        recorded contexts before they are rolled back.
        """
        self._prepare_failed_retries(contexts, record_each_stage)

    def _prepare_failed_retries(
        self, contexts: dict[str | int, TaskContext], record_each_stage: bool
    ) -> None:
        """Reset failed contexts for retry.

        For each retriable failed context the ``retry_count`` is
        pre-incremented (so the runner can compare ``retry_count > max_retries``
        immediately) and the context is rolled back to the appropriate stage
        based on its ``error_action``.

        When *record_each_stage* is ``True`` the rollback target is determined
        by ``ERROR_ACTION_PREV_STAGE`` and only downstream result fields are
        cleared.  When ``False``, the context is reset to ``INITIAL`` —
        either preserving prior-iteration data (``iteration > 0``) or fully
        clearing everything.
        """
        failed_ctxs = [c for c in contexts.values() if c.stage == TaskStage.FAILED]
        retried = 0
        for c in failed_ctxs:
            if c.error_reason in ERROR_REASONS_NON_RETRIABLE or not c.error_action:
                continue

            updates = {
                "error_action": None,
                "error_reason": None,
                "error_msg": None,
                # Pre-increment retry_count so that runner can compare
                # retry_count > max_retries: retry_count=1 means "on retry #1",
                # so max_retries=1 allows exactly one retry.
                "retry_count": c.retry_count + 1,
            }

            if record_each_stage:
                prev_stage = ERROR_ACTION_PREV_STAGE.get(
                    c.error_action, TaskStage.INITIAL
                )
                updates["stage"] = prev_stage
                for attr in ERROR_ACTION_CLEAR_FIELDS.get(c.error_action.value, []):
                    updates[attr] = None
                # Filter meta
                try:
                    prev_rank = STAGE_RANK[prev_stage]
                    updates["stage_meta"] = {
                        k: v
                        for k, v in c.stage_meta.items()
                        if STAGE_RANK[TaskStage(k)] <= prev_rank
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(
                        "Failed to filter stage_meta for {}: {}",
                        c.sample_id,
                        e,
                    )
            else:
                # record_each_stage=False: iteration boundary snapshots
                # are saved as full contexts at INITIAL stage with
                # iteration > 0.  If the hydrated context has iteration
                # data we can resume from the last completed iteration
                # instead of restarting from scratch.
                if c.iteration > 0:
                    # Resume from the start of the last recorded iteration.
                    # The INITIAL snapshot already contains all accumulated
                    # results and meta from prior iterations, so we only
                    # need to clear stage results for the current (failed)
                    # iteration and reset to INITIAL.
                    updates.update(
                        {
                            "stage": TaskStage.INITIAL,
                            "preprocess_result": None,
                            "infer_result": None,
                            "postprocess_result": None,
                            "feedback_result": None,
                        }
                    )
                else:
                    updates.update(
                        {
                            "stage": TaskStage.INITIAL,
                            "iteration": 0,
                            "preprocess_result": None,
                            "infer_result": None,
                            "postprocess_result": None,
                            "feedback_result": None,
                            "stage_meta": {},
                        }
                    )

            contexts[c.sample_id] = replace(c, **updates)
            retried += 1
        if retried:
            logger.info("Retry prepared: {} samples.", retried)
