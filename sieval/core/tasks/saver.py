"""Buffered async shard writer with atomic manifest and report persistence."""

import contextlib
import time
from pathlib import Path

import anyio
import orjson
import xxhash
from anyio.streams.memory import MemoryObjectReceiveStream
from loguru import logger

from sieval.core.types import JSONValue

from .consts import TaskStage
from .context import TaskContext, TaskManifest, TaskRunMeta
from .profiler import TaskProfiler, TaskProfilerContext


class TaskSaver:
    """Buffered async shard writer.

    Receives :class:`TaskContext` objects via a memory stream, buffers them,
    and flushes periodically to append-only ``.jsonl`` shards with companion
    ``.idx`` index files.  Manifest updates are atomic (write-tmp + replace).
    """

    def __init__(
        self,
        root_dir: Path,
        shard_samples: int = 1024,
        shard_write_concurrency: int = 8,
        write_buffer_size: int = 256,
        write_buffer_flush_interval: float = 16.0,
        record_type_metadata: bool = True,
        record_meta: bool = True,
        profiler: TaskProfiler | None = None,
        deterministic: bool = False,
    ):
        """Initialise the saver.

        *record_type_metadata* includes ``__type__`` markers for polymorphic
        deserialization.
        """
        self._manifest: dict[str | int, TaskManifest] = {}
        self._stage_queue: list[TaskContext] = []

        self._root_dir = root_dir
        self._manifest_path = self._root_dir / "manifest.json"
        self._report_path = self._root_dir / "report.json"

        self._shard_samples = shard_samples
        self._shard_write_concurrency = max(1, shard_write_concurrency)
        self._write_buffer_size = write_buffer_size
        self._write_buffer_flush_interval = write_buffer_flush_interval
        self._record_type_metadata = record_type_metadata
        self._record_meta = record_meta
        self._last_flush_time = time.perf_counter()

        self._profiler = profiler or TaskProfiler()
        self._deterministic = deterministic

    async def consume_stream(self, stream: MemoryObjectReceiveStream[TaskContext]):
        """Read contexts from a ``MemoryObjectReceiveStream``, buffer them,
        and flush when the buffer-size or time-interval threshold is reached.

        A final flush is performed after the stream is closed to ensure no
        buffered contexts are lost.
        """
        async with stream:
            async for ctx in stream:
                self._update_manifest_entry(ctx)
                self._stage_queue.append(ctx)

                now = time.perf_counter()
                time_due = (
                    now - self._last_flush_time
                ) >= self._write_buffer_flush_interval
                if len(self._stage_queue) >= self._write_buffer_size or time_due:
                    await self.flush()

        # Final flush on stream close
        await self.flush()

    async def flush(self) -> None:
        """Flush the current buffer to disk.

        Buffered contexts are grouped by ``(iteration, stage, shard_id)`` and
        each group is written in parallel.  After all shard writes complete
        the manifest is atomically updated.
        """
        if not self._stage_queue:
            return

        async with TaskProfilerContext(self._profiler, "flush"):
            batch = self._stage_queue
            self._stage_queue = []

            groups = {}
            for c in batch:
                shard_id = self._shard_id_for(c.sample_id)
                groups.setdefault((c.iteration, c.stage, shard_id), []).append(c)

            limiter = anyio.CapacityLimiter(self._shard_write_concurrency)

            async def _write_group(iteration, stage, shard_id, ctxs):
                async with limiter:
                    await self._write_shard(iteration, stage, shard_id, ctxs)

            async with anyio.create_task_group() as tg:
                for (iteration, stage, shard_id), ctxs in groups.items():
                    tg.start_soon(_write_group, iteration, stage, shard_id, ctxs)

            await self._write_manifest()
            self._last_flush_time = time.perf_counter()

    async def save_report(self, report: JSONValue) -> None:
        """Atomically write ``report.json`` via a temporary file and replace.

        The report is serialised with ``orjson`` (supporting non-string keys
        and numpy types).  A ``meta.json`` file is written alongside
        for reproducibility tracking.
        """
        tmp_path = self._report_path.with_suffix(".tmp")
        try:
            async with await anyio.open_file(tmp_path, "wb") as f:
                await f.write(
                    orjson.dumps(
                        report,
                        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
                    )
                )
            await anyio.Path(tmp_path).replace(self._report_path)
            logger.info("Saved report to: {}", self._report_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                await anyio.Path(tmp_path).unlink(missing_ok=True)
            logger.error("Failed to save report.json: {}", e)

    async def write_run_meta(self) -> None:
        """Write ``meta.json`` create-if-absent to the result directory.

        Written at run start so the format-stable version handshake exists
        for interrupted-then-resumed runs (resume is detected via
        ``manifest.json``, which may exist without a completed run). Never
        overwrites an existing ``meta.json`` — the file records the run's
        originating version and stays stable across compatible resumes.
        Self-creates ``_root_dir`` since at run start no shard exists yet.
        All failures are non-fatal: logged, never raised.
        """
        meta_path = self._root_dir / "meta.json"
        tmp_path = meta_path.with_suffix(".tmp")
        try:
            if await anyio.Path(meta_path).exists():
                return
            # Lazy: keeps the version lookup inside this try/except so an
            # unresolvable __version__ degrades gracefully, not a hard import fail.
            from sieval import __version__

            meta: TaskRunMeta = {
                "version": __version__,
                "deterministic": self._deterministic,
            }
            await anyio.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(tmp_path, "wb") as f:
                await f.write(orjson.dumps(meta))
            await anyio.Path(tmp_path).replace(meta_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                await anyio.Path(tmp_path).unlink(missing_ok=True)
            logger.error("Failed to write meta.json: {}", e)

    def _shard_id_for(self, sample_id: int | str) -> int:
        """Map a sample ID to a shard bucket index."""
        if isinstance(sample_id, int):
            return sample_id // self._shard_samples
        h = xxhash.xxh3_64(str(sample_id)).intdigest()
        return h % self._shard_samples

    async def _write_shard(
        self, iteration: int, stage: TaskStage, shard_id: int, ctxs: list[TaskContext]
    ):
        """Append serialised contexts to a ``.jsonl`` shard and write the
        corresponding offset index to the companion ``.idx`` file.

        Each context is serialised to a single JSON line.  Byte offsets are
        computed relative to the shard's pre-existing size so that the idx
        entries remain valid even after multiple appends.
        """
        shard_path = self._root_dir / str(iteration) / stage.value / f"{shard_id}.jsonl"
        idx_path = self._root_dir / str(iteration) / stage.value / f"{shard_id}.idx"

        await anyio.Path(shard_path.parent).mkdir(parents=True, exist_ok=True)

        p = anyio.Path(shard_path)
        if await p.exists():
            st = await p.stat()
            base_offset = st.st_size
        else:
            base_offset = 0

        serialized = []
        for c in ctxs:
            line_str = orjson.dumps(
                c.serialize(self._record_type_metadata, include_meta=self._record_meta),
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
            serialized.append((c, line_str + b"\n"))

        async with await anyio.open_file(shard_path, "ab") as f:
            await f.write(b"".join(b for _, b in serialized))

        current = base_offset
        idx_lines = []
        for c, line_bytes in serialized:
            length = len(line_bytes)
            idx_lines.append(
                f"{c.sample_id}\t{c.iteration}\t{c.stage.value}\t{current}\t{length}"
                f"\t{(c.error_action.value if c.error_action else '')}"
                f"\t{(c.error_reason or '')}"
                f"\t{c.retry_count}\n"
            )
            current += length

        async with await anyio.open_file(idx_path, "a") as idx_f:
            await idx_f.write("".join(idx_lines))

    def sync_manifest(self, initial_manifest: dict[str | int, TaskManifest]):
        """Import pre-existing manifest entries from the loader.

        Called once during startup so that the saver's in-memory manifest
        reflects the on-disk state before any new writes occur.
        """
        self._manifest = initial_manifest.copy()

    def _update_manifest_entry(self, ctx: TaskContext):
        """Update the in-memory manifest entry for a single context."""
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
        if ctx.retry_count > 0:
            entry["retry_count"] = ctx.retry_count
        self._manifest[ctx.sample_id] = entry

    async def _write_manifest(self):
        """Atomically write ``manifest.json`` (sorted entries, via tmp + replace).

        Entries are sorted by ``sample_id`` (integers first, then strings) to
        ensure deterministic output.
        """
        async with TaskProfilerContext(self._profiler, "write_manifest"):
            entries = list(self._manifest.values())
            entries_sorted = sorted(
                entries,
                key=lambda e: (
                    (0, e["sample_id"])
                    if isinstance(e["sample_id"], int)
                    else (1, str(e["sample_id"]))
                ),
            )

            tmp = self._manifest_path.with_suffix(".tmp")
            data = orjson.dumps(entries_sorted)

            await anyio.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
            try:
                async with await anyio.open_file(tmp, "wb") as f:
                    await f.write(data)
                await anyio.Path(tmp).replace(self._manifest_path)
            except Exception:
                with contextlib.suppress(OSError):
                    await anyio.Path(tmp).unlink(missing_ok=True)
                raise
