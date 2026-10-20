"""Buffered async shard writer with atomic manifest, report and meta persistence."""

import contextlib
import os
import stat
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import anyio.to_thread
import orjson
import xxhash
from anyio.streams.memory import MemoryObjectReceiveStream
from loguru import logger

from sieval.core.types import JSONValue

from .consts import TaskStage
from .context import TaskContext, TaskManifest, TaskRunIdentity, TaskRunMeta
from .profiler import TaskProfiler, TaskProfilerContext
from .run_meta import (
    DigestedStage,
    GraderIdentity,
    InferenceIdentity,
    RunFrom,
    carried_identity,
    carried_runs,
    new_run_record,
    ordered_stages,
    utc_now,
)


class _Unset:
    """A ``set_identity`` stage not given, apart from ``None`` (unknown)."""


_UNSET = _Unset()


def _process_umask() -> int:
    """The process umask, read once: ``os.umask`` can only be read by setting it."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _process_umask()


def _replacement_mode(path: Path) -> int:
    """*path*'s mode, else ``0o666 & ~umask`` as ``open()`` gives.

    ``mkstemp`` alone creates ``0o600``, unreadable to another uid.
    """
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return 0o666 & ~_UMASK


def _atomic_write_sync(path: Path, data: bytes) -> None:
    mode = _replacement_mode(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


async def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to a unique temporary file beside *path*, fsync, then replace.

    The temporary name is unique per call, so two writers never share one, and
    the data is on disk before the rename makes it visible.
    """
    await anyio.to_thread.run_sync(_atomic_write_sync, path, data)


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
        task_identity: TaskRunIdentity | None = None,
    ):
        """Initialise the saver.

        *record_type_metadata* includes ``__type__`` markers for polymorphic
        deserialization. *task_identity* is the registry identity of the task
        being run, persisted into ``meta.json``; ``None`` for a ``Task``
        subclass that carries no ``@sieval_task`` metadata.

        ``meta.json`` is rewritten through :meth:`begin_run`,
        :meth:`set_identity` and :meth:`end_run`.
        """
        self._manifest: dict[str | int, TaskManifest] = {}
        self._stage_queue: list[TaskContext] = []

        self._root_dir = root_dir
        self._manifest_path = self._root_dir / "manifest.json"
        self._report_path = self._root_dir / "report.json"
        self._meta_path = self._root_dir / "meta.json"
        self._meta: TaskRunMeta | None = None
        self._meta_lock = anyio.Lock()
        # Ordinal of this run in `runs[]`, written into each `.idx` line so a
        # later run's record supersedes an earlier one's (see the loader).
        self._run_index = 0

        self._shard_samples = shard_samples
        self._shard_write_concurrency = max(1, shard_write_concurrency)
        self._write_buffer_size = write_buffer_size
        self._write_buffer_flush_interval = write_buffer_flush_interval
        self._record_type_metadata = record_type_metadata
        self._record_meta = record_meta
        self._last_flush_time = time.perf_counter()

        self._profiler = profiler or TaskProfiler()
        self._deterministic = deterministic
        self._task_identity = task_identity

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

    async def drain(
        self,
        stream: MemoryObjectReceiveStream[TaskContext],
        also: Iterable[TaskContext] = (),
    ) -> None:
        """Take every context *stream* still holds, then *also*, flush, and close.

        For a run that stopped before :meth:`consume_stream` reached the end of
        its stream: *stream* is a clone of that receiver, so the contexts sent
        but never received are still in it. *also* are finished contexts that
        never reached the stream.
        """
        with stream:
            while True:
                try:
                    ctx = stream.receive_nowait()
                except (anyio.WouldBlock, anyio.EndOfStream):
                    break
                self._update_manifest_entry(ctx)
                self._stage_queue.append(ctx)
        for ctx in also:
            self._update_manifest_entry(ctx)
            self._stage_queue.append(ctx)
        await self.flush()

    async def flush(self) -> None:
        """Flush the current buffer to disk.

        Buffered contexts are grouped by ``(iteration, stage, shard_id)`` and
        each group is written in parallel.  After all shard writes complete
        the manifest is atomically updated. A flush that fails or is cancelled
        puts its batch back in the queue for the next one; a shard group it had
        already written is appended again, and the loader keeps the later line.
        """
        if not self._stage_queue:
            return

        async with TaskProfilerContext(self._profiler, "flush"):
            batch = self._stage_queue
            self._stage_queue = []
            try:
                await self._write_batch(batch)
            except BaseException:
                self._stage_queue = batch + self._stage_queue
                raise
            self._last_flush_time = time.perf_counter()

    async def _write_batch(self, batch: list[TaskContext]) -> None:
        """Write *batch* to its shards, then the manifest."""
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

    async def save_report(self, report: JSONValue) -> None:
        """Atomically write ``report.json`` via a temporary file and replace.

        The report is serialised with ``orjson`` (supporting non-string keys
        and numpy types). A failed write raises: ``report.json`` is the result
        and what decides a fast resume, so a task whose report did not land
        has not finished.
        """
        await _atomic_write(
            self._report_path,
            orjson.dumps(
                report,
                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
            ),
        )
        logger.info("Saved report to: {}", self._report_path)

    async def begin_run(
        self,
        *,
        resumed: bool,
        run_from: RunFrom | None = None,
        model: str | None = None,
        keep_recorded_model: bool = False,
        config_diff: Sequence[str] = (),
        data: DigestedStage | None | _Unset = _UNSET,
        prompt: DigestedStage | None | _Unset = _UNSET,
        inference: InferenceIdentity | None | _Unset = _UNSET,
        grader: GraderIdentity | None | _Unset = _UNSET,
    ) -> None:
        """Rewrite ``meta.json`` for a run that is starting, appending its ``runs[]``.

        ``version``, ``deterministic`` and ``task`` describe the run that
        created the directory: a resume keeps the persisted values and never
        backfills a missing ``deterministic`` or ``task``, since either would
        describe the resuming process rather than the run that produced the
        samples. A
        fresh run writes its own. A stage identity given here replaces the
        carried one (``None`` sets it unknown); a stage not given is carried
        from a resumed ``meta.json`` of the same identity schema.

        *run_from* starts a resume across tasks: the directory's history begins
        with this run (``start: "from"``) and its origin fields are this run's,
        since this run created the directory; the copied directory and its
        identity are recorded in ``runs[].from``. Self-creates ``_root_dir``. A
        failed write raises.

        *keep_recorded_model* keeps a resumed directory's ``model`` when
        *model* differs, since the inference on disk is the recorded model's;
        *model* then goes in this run's ``runs[]`` entry.
        """
        from sieval import __version__

        existing = await self._read_meta() if resumed and run_from is None else None
        meta: TaskRunMeta
        if existing is not None and isinstance(existing.get("version"), str):
            meta = cast(TaskRunMeta, {"version": existing["version"]})
            # Carried only when recorded: a pre-feature run never said.
            if isinstance(existing.get("deterministic"), bool):
                meta["deterministic"] = existing["deterministic"]
            if isinstance(existing.get("task"), dict):
                meta["task"] = existing["task"]
        else:
            meta = {"version": __version__, "deterministic": self._deterministic}
            if self._task_identity is not None:
                meta["task"] = self._task_identity
        recorded_model = (existing or {}).get("model")
        if keep_recorded_model and isinstance(recorded_model, str):
            meta["model"] = recorded_model
        elif model is not None:
            meta["model"] = model
        meta["identity"] = carried_identity(existing)
        if run_from is not None:
            run = new_run_record(__version__, "from", config=config_diff)
            run["from"] = run_from
        else:
            run = new_run_record(
                __version__, "resume" if resumed else "fresh", config=config_diff
            )
        if model is not None and meta.get("model") != model:
            run["model"] = model
        meta["runs"] = [*carried_runs(existing), run]
        self._run_index = len(meta["runs"])
        self._meta = meta
        self._apply_identity(data, prompt, inference, grader)
        await self._write_meta()

    async def set_identity(
        self,
        *,
        data: DigestedStage | None | _Unset = _UNSET,
        prompt: DigestedStage | None | _Unset = _UNSET,
        inference: InferenceIdentity | None | _Unset = _UNSET,
        grader: GraderIdentity | None | _Unset = _UNSET,
    ) -> None:
        """Replace the given stage identities and rewrite ``meta.json``.

        A stage left out keeps its value; ``None`` sets it unknown.
        """
        self._require_meta()
        self._apply_identity(data, prompt, inference, grader)
        await self._write_meta()

    def _apply_identity(
        self,
        data: DigestedStage | None | _Unset,
        prompt: DigestedStage | None | _Unset,
        inference: InferenceIdentity | None | _Unset,
        grader: GraderIdentity | None | _Unset,
    ) -> None:
        identity = self._require_meta()["identity"]
        if not isinstance(data, _Unset):
            identity["data"] = data
        if not isinstance(prompt, _Unset):
            identity["prompt"] = prompt
        if not isinstance(inference, _Unset):
            identity["inference"] = inference
        if not isinstance(grader, _Unset):
            identity["grader"] = grader

    @property
    def inference_identity(self) -> InferenceIdentity | None:
        """This run's ``inference`` identity; ``None`` before :meth:`begin_run`."""
        return None if self._meta is None else self._meta["identity"]["inference"]

    async def end_run(self, reran: Iterable[str], *, reported: bool = False) -> None:
        """Mark the current ``runs[]`` entry ended, with the stages it reran.

        *reported* is set by a run that wrote ``report.json``.
        """
        meta = self._require_meta()
        run = meta["runs"][-1]
        run["ended"] = utc_now()
        run["reran"] = ordered_stages(reran)
        if reported:
            run["reported"] = True
        await self._write_meta()

    async def interrupt_run(self) -> None:
        """Mark the current ``runs[]`` entry ended by an interrupt, without ``reran``.

        A no-op before :meth:`begin_run`: an interrupt that early left nothing
        to record.
        """
        if self._meta is None:
            return
        run = self._meta["runs"][-1]
        run["ended"] = utc_now()
        run["interrupted"] = True
        await self._write_meta()

    @property
    def run_meta(self) -> TaskRunMeta | None:
        """The ``meta.json`` document this run is writing, once begun."""
        return self._meta

    def _require_meta(self) -> TaskRunMeta:
        if self._meta is None:
            raise RuntimeError("begin_run() must be called before meta.json updates")
        return self._meta

    async def _read_meta(self) -> dict[str, Any] | None:
        try:
            data = orjson.loads(await anyio.Path(self._meta_path).read_bytes())
        except (OSError, orjson.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    async def _write_meta(self) -> None:
        async with self._meta_lock:
            await anyio.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
            await _atomic_write(
                self._meta_path,
                orjson.dumps(self._meta, option=orjson.OPT_SERIALIZE_NUMPY),
            )

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
                f"\t{c.retry_count}"
                f"\t{self._run_index}\n"
            )
            current += length

        async with await anyio.open_file(idx_path, "a") as idx_f:
            await idx_f.write("".join(idx_lines))

    async def write_records(self, contexts: Iterable[TaskContext]) -> None:
        """Write *contexts* as full records and flush, with the manifest.

        For records a run writes outside the compute stream: the samples a
        per-stage resume keeps, rewritten so no superseded record can outrank
        them on load.
        """
        for ctx in contexts:
            self._update_manifest_entry(ctx)
            self._stage_queue.append(ctx)
        await self.flush()
        await self._write_manifest()

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
