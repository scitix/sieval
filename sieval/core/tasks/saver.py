"""Buffered async shard writer, with atomic ``report.json`` and ``meta.json``.

A shard's ``.idx`` lines are written after the records they point at.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import anyio.to_thread
import orjson
import xxhash
from anyio.streams.memory import MemoryObjectReceiveStream
from loguru import logger

from sieval.core.types import JSONValue
from sieval.core.utils.atomic import aatomic_write

from .consts import TaskStage
from .context import TaskContext, TaskRunIdentity, TaskRunMeta
from .profiler import TaskProfiler, TaskProfilerContext
from .run_meta import (
    RESULT_SCHEMA_VERSION,
    STAGES,
    InferenceIdentity,
    ResumeOutcomes,
    RunFrom,
    RunRequests,
    ServingFingerprint,
    carried_identity,
    carried_runs,
    new_run_record,
    ordered_stages,
    utc_now,
)


class TaskSaver:
    """Buffered async shard writer.

    Receives :class:`TaskContext` objects via a memory stream, buffers them,
    and flushes periodically to append-only ``.jsonl`` shards with companion
    ``.idx`` index files.
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
        self._stage_queue: list[TaskContext] = []

        self._root_dir = root_dir
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
                self._stage_queue.append(ctx)
        for ctx in also:
            self._stage_queue.append(ctx)
        await self.flush()

    async def flush(self) -> None:
        """Flush the current buffer to disk.

        Buffered contexts are grouped by ``(iteration, stage, shard_id)`` and
        each group is written in parallel. A flush that fails or is cancelled
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
        """Write *batch* to its shards."""
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

    async def save_report(self, report: JSONValue) -> None:
        """Atomically write ``report.json`` via a temporary file and replace.

        The report is serialised with ``orjson`` (supporting non-string keys
        and numpy types). A failed write raises: ``report.json`` is the result
        and what decides a fast resume, so a task whose report did not land
        has not finished.
        """
        await aatomic_write(
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
        weights: str | None = None,
        keep_recorded_model: bool = False,
        serving: ServingFingerprint | None = None,
        config_diff: Sequence[str] = (),
        config_path: str | None = None,
        identity: Mapping[str, object] | None = None,
    ) -> None:
        """Rewrite ``meta.json`` for a run that is starting, appending its ``runs[]``.

        ``result_schema_version`` is always this build's: the resume gate
        admits no other. ``version``, ``deterministic`` and ``task`` describe
        the run that created the directory: a resume keeps the persisted values
        and never backfills a missing ``deterministic`` or ``task``, since
        either would describe the resuming process rather than the run that
        produced the samples. A fresh run writes its own. A stage in
        *identity* replaces the carried one (``None`` sets it unknown); a stage
        not given is carried from a resumed ``meta.json``, a schema-1 block
        read as schema 2 (:func:`carried_identity`).

        *run_from* starts a resume across tasks: the directory's history begins
        with this run (``start: "from"``) and its origin fields are this run's,
        since this run created the directory; the copied directory and its
        identity are recorded in ``runs[].from``. *config_path* is recorded as
        ``config``. Self-creates ``_root_dir``. A failed write raises.

        *keep_recorded_model* keeps a resumed directory's ``model`` when
        *model* differs, since the inference on disk is the recorded model's;
        *model* then goes in this run's ``runs[]`` entry. *weights* names the
        weights *model* serves; ``None`` keeps the recorded ones for the same
        model. *serving* is ``runs[].serving``.
        """
        from sieval import __version__

        existing = await self._read_meta() if resumed and run_from is None else None
        meta: TaskRunMeta
        if existing is not None and isinstance(existing.get("version"), str):
            # The gate admitted only this schema; a pre-field directory records it now.
            meta = cast(
                TaskRunMeta,
                {
                    "version": existing["version"],
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                },
            )
            # Carried only when recorded: a pre-feature run never said.
            if isinstance(existing.get("deterministic"), bool):
                meta["deterministic"] = existing["deterministic"]
            if isinstance(existing.get("task"), dict):
                meta["task"] = existing["task"]
        else:
            meta = {
                "version": __version__,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "deterministic": self._deterministic,
            }
            if self._task_identity is not None:
                meta["task"] = self._task_identity
        recorded_model = (existing or {}).get("model")
        recorded_weights = (existing or {}).get("weights")
        if keep_recorded_model and isinstance(recorded_model, str):
            meta["model"] = recorded_model
        elif model is not None:
            meta["model"] = model
        if not keep_recorded_model and weights is not None:
            meta["weights"] = weights
        elif isinstance(recorded_weights, str) and meta.get("model") == recorded_model:
            # Unstated weights keep the recorded ones, or a later resume by
            # other weights under the same name would pass.
            meta["weights"] = recorded_weights
        if config_path is not None:
            meta["config"] = config_path
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
        if serving is not None:
            run["serving"] = serving
        meta["runs"] = [*carried_runs(existing), run]
        self._run_index = len(meta["runs"])
        self._meta = meta
        self._apply_identity(identity or {})
        await self._write_meta()

    async def set_identity(self, **stages: object) -> None:
        """Replace the given stage identities, by stage name, and rewrite ``meta.json``.

        A stage left out keeps its value; ``None`` sets it unknown.
        """
        self._require_meta()
        self._apply_identity(stages)
        await self._write_meta()

    def _apply_identity(self, stages: Mapping[str, object]) -> None:
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown identity stage(s): {sorted(unknown)}")
        identity = cast(dict[str, object], self._require_meta()["identity"])
        identity.update(stages)

    @property
    def inference_identity(self) -> InferenceIdentity | None:
        """This run's ``inference`` identity; ``None`` before :meth:`begin_run`."""
        return None if self._meta is None else self._meta["identity"]["inference"]

    async def end_run(
        self,
        reran: Iterable[str],
        *,
        reported: bool = False,
        rules_hash: str | None = None,
        requests: RunRequests | None = None,
        resume: ResumeOutcomes | None = None,
        stopped_after: str | None = None,
    ) -> None:
        """Mark the current ``runs[]`` entry ended, with the stages it reran.

        *reported* is set by a run that wrote ``report.json``; *rules_hash* by
        one that wrote ``anomalies.json``; *requests* by one that recorded any;
        *resume* by a resume that planned per sample; *stopped_after* by a run
        that ended at its ``stop_after`` barrier.
        """
        meta = self._require_meta()
        run = meta["runs"][-1]
        run["ended"] = utc_now()
        run["reran"] = ordered_stages(reran)
        if resume is not None:
            run["resume"] = resume
        if reported:
            run["reported"] = True
        if stopped_after is not None:
            run["stopped_after"] = stopped_after
        if rules_hash is not None:
            run["rules_hash"] = rules_hash
        if requests:
            run["requests"] = requests
        await self._write_meta()

    async def interrupt_run(
        self,
        *,
        rules_hash: str | None = None,
        requests: RunRequests | None = None,
    ) -> None:
        """Mark the current ``runs[]`` entry ended by an interrupt, without ``reran``.

        A no-op before :meth:`begin_run`: an interrupt that early left nothing
        to record.
        """
        if self._meta is None:
            return
        run = self._meta["runs"][-1]
        run["ended"] = utc_now()
        run["interrupted"] = True
        if rules_hash is not None:
            run["rules_hash"] = rules_hash
        if requests:
            run["requests"] = requests
        await self._write_meta()

    @property
    def run_id(self) -> str | None:
        """This run's ``runs[].id``; ``None`` before :meth:`begin_run`."""
        return None if self._meta is None else self._meta["runs"][-1].get("id")

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
            await aatomic_write(
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
        """Write *contexts* as full records and flush.

        For records a run writes outside the compute stream: the samples a
        per-stage resume keeps, rewritten so no superseded record can outrank
        them on load.
        """
        for ctx in contexts:
            self._stage_queue.append(ctx)
        await self.flush()
