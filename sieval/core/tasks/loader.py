"""Persistence recovery: manifest loading, shard hydration, and compensation scan."""

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
    STAGE_RANK,
    STAGE_TO_RESULT_FIELD,
    TaskAction,
    TaskStage,
)
from .context import TaskContext, TaskManifest
from .profiler import TaskProfiler, TaskProfilerContext
from .task import Task


class ShardOffset(TypedDict):
    """Shard file location for a single sample: file path + byte range."""

    shard: Path
    offset: int
    length: int


class ShardOffsetWithIteration(TypedDict):
    """Shard file location tagged with the iteration it was written in."""

    shard: Path
    offset: int
    length: int
    iteration: int


# (iteration, stage_rank, shard_path, byte_offset, byte_length)
# stage_rank uses -1 when comparing stage-only offsets (for dependency loading)
OffsetMeta = tuple[int, int, Path, int, int]
"""Tuple ``(iteration, stage_rank, shard_path, byte_offset, byte_length)``
used for deterministic latest-offset selection.

``stage_rank`` is set to ``-1`` for per-stage comparisons (dependency loading)
where only iteration and file position matter."""


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


def _should_replace_offset(current: OffsetMeta | None, candidate: OffsetMeta) -> bool:
    """
    Choose the latest offset deterministically for stage and best hydration records.

    Priority:
    1) higher iteration
    2) higher stage rank (use -1 for stage-only comparisons)
    3) for same shard, larger offset (later append)
    4) deterministic fallback by (shard path, offset, length)
    """
    if current is None:
        return True

    cand_it, cand_rank, cand_shard, cand_offset, cand_length = candidate
    cur_it, cur_rank, cur_shard, cur_offset, cur_length = current

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


class TaskLoader:
    """Persistence recovery engine.

    Loads ``manifest.json`` and shard offset indices (``.idx`` files),
    hydrates in-memory :class:`TaskContext` objects from on-disk shard data,
    and handles compensation scan when the manifest is missing or stale.
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
        self._manifest: dict[str | int, TaskManifest] = {}
        self._sample_offsets: dict[str | int, ShardOffset] = {}
        self._all_stage_offsets: dict[
            str | int, dict[TaskStage, ShardOffsetWithIteration]
        ] = {}

        self._root_dir = root_dir
        self._manifest_path = self._root_dir / "manifest.json"
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
        """Load manifest and idx files to produce skeleton contexts.

        When a completed report already exists alongside a trusted manifest the
        offsets are loaded directly from idx files; otherwise a full
        compensation scan is performed to rebuild both manifest and offsets
        from scratch.
        """
        async with TaskProfilerContext(self._profiler, "load_or_compensate"):
            await anyio.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
            await self._load_manifest()

            if await anyio.Path(self._report_path).exists() and self._manifest:
                await self._load_offsets_from_idx()
                logger.info("Trusted completed manifest; offsets loaded.")
            else:
                await self._compensate_scan()

            contexts = {}
            for sid, m in self._manifest.items():
                base = self._task.make_context(sid, None)
                ctx = replace(
                    base,
                    iteration=m["iteration"],
                    stage=TaskStage(m["stage"]),
                    error_action=_try_task_action(m.get("error_action")),
                    error_reason=m.get("error_reason"),
                    retry_count=m.get("retry_count", 0),
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

    async def get_manifest_status(self) -> tuple[bool, bool]:
        """Return ``(has_failures, all_final)`` from the current manifest.

        Loads the manifest from disk if it has not been loaded yet.
        """
        await self._load_manifest()
        if not self._manifest:
            return False, False

        has_failures = False
        all_final = True
        for m in self._manifest.values():
            if m.get("failed"):
                has_failures = True
            if not m.get("final"):
                all_final = False
        return has_failures, all_final

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
                existing_meta[stage_key] = [obj["meta_last"]]
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
            updates["stage_meta"] = obj["stage_meta"]
        elif "meta_last" in obj and "stage" in obj:
            updates["stage_meta"] = {obj["stage"]: [obj["meta_last"]]}

        new_ctx = replace(ctx, **updates)
        contexts[ctx.sample_id] = new_ctx
        hydrated_ids.add(ctx.sample_id)

    async def _load_manifest(self) -> None:
        """Read ``manifest.json`` from disk into ``self._manifest``.

        No-op if the manifest is already loaded. Tolerates a missing or
        corrupt file by falling back to an empty manifest.
        """
        if self._manifest:
            return
        if not await anyio.Path(self._manifest_path).exists():
            self._manifest = {}
            return
        async with await anyio.open_file(self._manifest_path, "r") as f:
            raw = await f.read()
        try:
            arr = orjson.loads(raw)
        except orjson.JSONDecodeError:
            arr = []
        self._manifest = {
            item["sample_id"]: item
            for item in arr
            if isinstance(item, dict) and "sample_id" in item
        }

    # idx functions (sync I/O) are dispatched to a worker thread by the async methods
    def _collect_idx_files(self) -> list[tuple[Path, Path]]:
        idx_files: list[tuple[Path, Path]] = []
        if not self._root_dir.exists():
            return idx_files
        for iter_dir in self._root_dir.iterdir():
            if iter_dir.is_dir() and iter_dir.name.isdigit():
                for stage_dir in iter_dir.iterdir():
                    if stage_dir.is_dir():
                        for idx_file in stage_dir.glob("*.idx"):
                            idx_files.append((idx_file, idx_file.with_suffix(".jsonl")))
        return idx_files

    async def _load_offsets_from_idx(self) -> None:
        """Scan all ``.idx`` files in parallel to build shard-offset maps.

        Used when the manifest is trusted (completed run): each idx record
        is evaluated via :func:`_should_replace_offset` so that only the
        latest offset per sample (and per sample+stage) is retained.
        """
        self._sample_offsets = {}
        idx_files = await run_sync(self._collect_idx_files)

        limiter = anyio.CapacityLimiter(self._shard_read_concurrency)
        best_meta: dict[str | int, OffsetMeta] = {}
        all_stage_offsets: dict[
            str | int, dict[TaskStage, ShardOffsetWithIteration]
        ] = {}

        async def _one(idx_path: Path, shard: Path):
            async with limiter:
                records = await run_sync(_parse_idx_file, idx_path)
                for r in records:
                    sid = r["sample_id"]
                    it_val = r["iteration"]
                    stg_val = r["stage"]
                    rank = STAGE_RANK[stg_val]

                    off_with_it: ShardOffsetWithIteration = {
                        "shard": shard,
                        "offset": r["offset"],
                        "length": r["length"],
                        "iteration": it_val,
                    }

                    # Keep the latest offset per (sample, stage).
                    stage_map = all_stage_offsets.setdefault(sid, {})
                    existing = stage_map.get(stg_val)
                    stage_current_meta: OffsetMeta | None = None
                    if existing is not None:
                        stage_current_meta = (
                            existing["iteration"],
                            -1,
                            existing["shard"],
                            existing["offset"],
                            existing["length"],
                        )
                    stage_candidate_meta: OffsetMeta = (
                        it_val,
                        -1,
                        shard,
                        r["offset"],
                        r["length"],
                    )
                    if _should_replace_offset(stage_current_meta, stage_candidate_meta):
                        stage_map[stg_val] = off_with_it

                    candidate_meta: OffsetMeta = (
                        it_val,
                        rank,
                        shard,
                        r["offset"],
                        r["length"],
                    )
                    if not _should_replace_offset(best_meta.get(sid), candidate_meta):
                        continue

                    best_meta[sid] = candidate_meta
                    self._sample_offsets[sid] = ShardOffset(
                        shard=shard, offset=r["offset"], length=r["length"]
                    )

        async with anyio.create_task_group() as tg:
            for idx, shard in idx_files:
                tg.start_soon(_one, idx, shard)

        self._all_stage_offsets = all_stage_offsets

    async def _compensate_scan(self) -> None:
        """Full parallel scan of all idx files — rebuilds manifest and offsets
        from scratch.

        Invoked when ``manifest.json`` is missing or stale. Every idx record
        is parsed, the latest offset per sample is selected deterministically,
        and ``self._manifest`` / ``self._sample_offsets`` /
        ``self._all_stage_offsets`` are rebuilt.
        """
        logger.info("Compensation scan (parallel).")

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
                for r in records:
                    sid = r["sample_id"]
                    it_val = r["iteration"]
                    stg_val = r["stage"]
                    rank = STAGE_RANK[stg_val]

                    # Keep the latest offset per (sample, stage) for dependency loading.
                    off_with_it: ShardOffsetWithIteration = {
                        "shard": shard,
                        "offset": r["offset"],
                        "length": r["length"],
                        "iteration": it_val,
                    }
                    stage_map = all_stage_offsets.setdefault(sid, {})
                    existing = stage_map.get(stg_val)
                    stage_current_meta: OffsetMeta | None = None
                    if existing is not None:
                        stage_current_meta = (
                            existing["iteration"],
                            -1,
                            existing["shard"],
                            existing["offset"],
                            existing["length"],
                        )
                    stage_candidate_meta: OffsetMeta = (
                        it_val,
                        -1,
                        shard,
                        r["offset"],
                        r["length"],
                    )
                    if _should_replace_offset(stage_current_meta, stage_candidate_meta):
                        stage_map[stg_val] = off_with_it

                    candidate_meta: OffsetMeta = (
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

        self._manifest = {}
        self._sample_offsets = {}

        for sid, (meta, r) in best_records.items():
            it_val, _rank, shard, offset, length = meta
            # Reconstruct manifest entry
            entry: TaskManifest = {
                "sample_id": sid,
                "stage": r["stage"].value,
                "iteration": it_val,
                "final": r["stage"] == TaskStage.FINAL,
                "failed": r["stage"] == TaskStage.FAILED,
            }
            if r.get("error_action"):
                entry["error_action"] = r["error_action"]
            if r.get("error_reason"):
                entry["error_reason"] = r["error_reason"]
            if r.get("retry_count"):
                entry["retry_count"] = r["retry_count"]

            self._manifest[sid] = entry

            self._sample_offsets[sid] = ShardOffset(
                shard=shard, offset=offset, length=length
            )

        self._all_stage_offsets = all_stage_offsets
        logger.info("Compensated manifest with {} entries.", len(self._manifest))

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
