"""Read HuggingFace datasets into memory, writing no Arrow cache.

``load_dataset`` writes every split of a source as Arrow files under
``HF_DATASETS_CACHE``, and each later ``map``/``filter`` writes another. A
caller that reads a dataset once (``sieval run --dry-run``) reads none of them
back.

Inside :func:`uncached_reads`, an Arrow-based builder (parquet, json, csv,
arrow, text) streams its splits from the same ``_generate_tables`` the cached
path uses: the requested ones into an in-memory table, the others only counted.
Every split is parsed and HF's split-size check runs over all of them, as in
the cached path. A load keeps the cached path for a loading script (its cached
path also checks duplicate keys), a split asked for by slice (``"test[:10]"``),
a split not on local disk or over :data:`MAX_IN_MEMORY_BYTES`, any failure to
stream, and an empty requested split; the cached path then raises its own error.

The module factory is memoized within the block, so a loader calling
``load_dataset`` once per config (C-Eval's 52) parses the card once.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import copy
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import pyarrow as pa
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import IterableDataset
from datasets import load as hf_load
from datasets.builder import ArrowBasedBuilder, DatasetBuilder
from datasets.splits import NamedSplit, Split, SplitDict, SplitInfo
from datasets.table import InMemoryTable
from datasets.utils.info_utils import VerificationMode, verify_splits

#: Above this many bytes of source files in a requested split, read through the cache.
MAX_IN_MEMORY_BYTES = 2 * 1024**3

_BATCH_ROWS = 10_000
_lock = threading.Lock()
_PREPARE_ARGS = "_sieval_uncached_prepare"

type ReadPath = Literal["memory", "cache"]


@dataclass(frozen=True, slots=True)
class BuilderRead:
    """How one ``load_dataset`` call inside the block was read."""

    builder: str
    path: ReadPath
    reason: str = ""


@dataclass(slots=True)
class ReadLog:
    """Each builder read inside an :func:`uncached_reads` block, in order."""

    reads: list[BuilderRead] = field(default_factory=list)


def uncached_reads() -> contextlib.AbstractContextManager[ReadLog]:
    """Within the block, :func:`datasets.load_dataset` reads in memory where it can.

    Patches ``DatasetBuilder`` and the module factory process-wide while open,
    so one block at a time: a second raises ``RuntimeError``.
    """
    return _uncached_reads()


@contextlib.contextmanager
def _uncached_reads() -> Iterator[ReadLog]:
    if not _lock.acquire(blocking=False):
        raise RuntimeError(
            "uncached_reads() is already open; it patches `datasets` "
            "process-wide, so only one block may be open at a time"
        )
    log = ReadLog()
    prepare = DatasetBuilder.download_and_prepare
    as_dataset = DatasetBuilder.as_dataset
    factory = hf_load.dataset_module_factory
    modules: dict[str, Any] = {}

    def memo_factory(path: str, *args: Any, **kwargs: Any) -> Any:
        # `load_dataset_builder` passes paths, strings, data-file specs and a
        # `DownloadConfig`, a dataclass: each repr shows every field, so equal
        # reprs mean equal arguments.
        key = repr((path, args, sorted(kwargs.items(), key=lambda kv: kv[0])))
        if key not in modules:
            modules[key] = factory(path, *args, **kwargs)
        # `load_dataset_builder` pops keys from the module's builder kwargs.
        return copy.deepcopy(modules[key])

    def deferred_prepare(self: DatasetBuilder, *args: Any, **kwargs: Any) -> None:
        # Decided at `as_dataset`, which knows the split asked for.
        setattr(self, _PREPARE_ARGS, (args, kwargs))

    def in_memory_as_dataset(
        self: DatasetBuilder, split: Any = None, **options: Any
    ) -> HFDataset | HFDatasetDict:
        # `datasets` 4 passes `verification_mode` here, 5 only to
        # `download_and_prepare`; a value here wins. `options` reach the real
        # method as given.
        prepare_args, prepare_kwargs = getattr(self, _PREPARE_ARGS, ((), {}))
        verification_mode = options.get(
            "verification_mode", prepare_kwargs.get("verification_mode")
        )
        names, reason = _memory_splits(self, split)
        built: dict[str, HFDataset] = {}
        counts: dict[str, int] = {}
        if names is not None:
            try:
                for name in map(str, cast(dict, self.config.data_files)):
                    if name in names:
                        built[name] = _read_split(self, name)
                        counts[name] = len(built[name])
                    else:
                        counts[name] = _count_split(self, name)
            except _EmptySplit as exc:
                names, reason = None, f"split {exc} has no rows"
            except Exception as exc:
                # Batches whose schemas differ (the cache writer casts them to
                # the first), or a file that does not parse: the cached path
                # reads it, or raises its own error.
                names, reason = None, f"split did not stream ({exc})"
        if names is None:
            log.reads.append(BuilderRead(_label(self), "cache", reason))
            prepare(self, *prepare_args, **prepare_kwargs)
            return as_dataset(self, split=split, **options)
        mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        if mode != VerificationMode.NO_CHECKS:
            recorded = SplitDict(dataset_name=self.dataset_name)
            for name, rows in counts.items():
                recorded.add(SplitInfo(name=name, num_examples=rows))
            verify_splits(self.info.splits, recorded)
        log.reads.append(BuilderRead(_label(self), "memory"))
        if split is None:
            return HFDatasetDict(dict(built))
        return built[names[0]]

    DatasetBuilder.download_and_prepare = deferred_prepare  # ty: ignore[invalid-assignment]
    DatasetBuilder.as_dataset = in_memory_as_dataset  # ty: ignore[invalid-assignment]
    hf_load.dataset_module_factory = memo_factory  # ty: ignore[invalid-assignment]
    try:
        yield log
    finally:
        DatasetBuilder.download_and_prepare = prepare
        DatasetBuilder.as_dataset = as_dataset
        hf_load.dataset_module_factory = factory
        _lock.release()


def _label(builder: DatasetBuilder) -> str:
    name = builder.dataset_name or type(builder).__name__
    config = builder.config.name
    return f"{name}/{config}" if config and config != "default" else name


def _memory_splits(
    builder: DatasetBuilder, split: object
) -> tuple[list[str] | None, str]:
    """The requested split names to hold in memory, or ``None`` and why not."""
    if not isinstance(builder, ArrowBasedBuilder):
        return None, "loading script"
    files = builder.config.data_files
    if files is None:
        return None, "no data files"
    names_all = [str(name) for name in files]
    if split is None:
        names = names_all
    elif isinstance(split, str) and split in files:
        names = [split]
    else:
        return None, f"split {split!s} is not a split name"
    for name in names_all:
        size = _local_size(files[name])
        if size is None:
            return None, f"split {name} is not on local disk"
        if name in names and size > MAX_IN_MEMORY_BYTES:
            return None, f"split {name} is {size} bytes"
    return names, ""


def _local_size(paths: list[str]) -> int | None:
    """Total bytes of *paths*; ``None`` when one is not a local file."""
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except (OSError, TypeError):
            return None
    return total


class _EmptySplit(Exception):
    """A requested split that streamed no rows."""


def _batches(builder: DatasetBuilder, name: str) -> Iterator[pa.Table]:
    streamed = builder.as_streaming_dataset(split=name)
    if not isinstance(streamed, IterableDataset):
        raise TypeError(f"split {name!r} streamed as {type(streamed).__name__}")
    return streamed.with_format("arrow").iter(batch_size=_BATCH_ROWS)


def _count_split(builder: DatasetBuilder, name: str) -> int:
    """Rows in split *name*, parsed and dropped batch by batch."""
    return sum(len(table) for table in _batches(builder, name))


def _read_split(builder: DatasetBuilder, name: str) -> HFDataset:
    tables = list(_batches(builder, name))
    if not tables:
        raise _EmptySplit(name)
    split = Split(name)
    # The cached path's fingerprint: without one, HF hashes every row of an
    # in-memory table on each later transform.
    return HFDataset(
        InMemoryTable(pa.concat_tables(tables)),
        info=builder.info.copy(),
        split=cast(NamedSplit, split),
        fingerprint=builder._get_dataset_fingerprint(split),
    )
