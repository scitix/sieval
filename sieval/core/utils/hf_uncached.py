"""Read HuggingFace datasets into memory, writing no Arrow cache.

``load_dataset`` first converts every split of a source into Arrow files under
``HF_DATASETS_CACHE``, then memory-maps them; each ``map`` and ``filter`` a
loader applies writes another file. A caller that reads a dataset once, such as
``sieval run --dry-run`` in a job with a fresh cache, pays for writing all of
them and reads none back.

Inside :func:`uncached_reads`, an Arrow-based builder (parquet, json, csv,
arrow, text) streams each requested split's files into one in-memory table,
and transforms on it stay in memory. The rows, their order and the features are
those ``load_dataset`` gives, since the same ``_generate_tables`` produces
them, and HF's split-size check still runs. A load keeps the cached path when:

- the builder is generator-based (a loading script): its cached path also
  checks for duplicate keys, which streaming does not;
- the split is asked for by slice or instruction (``"test[:10]"``) rather than
  by name;
- a split's source files are not local, or exceed :data:`MAX_IN_MEMORY_BYTES`,
  where a memory-mapped file is the cheaper reading;
- its batches infer different schemas (JSON lines whose column is null in one
  batch and a string in the next), which the cache writer casts to the first;
- a split has no rows, which the cached path refuses.

The module factory is memoized within the block too: a loader calling
``load_dataset`` once per config (C-Eval's 52 subjects) otherwise parses the
dataset card each time.

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

#: Above this many bytes of source files in one split, read through the cache.
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
    """Within the block, :func:`datasets.load_dataset` writes no Arrow cache
    where it can read in memory; the :class:`ReadLog` says which loads did.

    Process-wide while open: it patches ``DatasetBuilder`` and the module
    factory. Not reentrant; a second open block raises ``RuntimeError``.
    """
    return _uncached_reads()


@contextlib.contextmanager
def _uncached_reads() -> Iterator[ReadLog]:
    if not _lock.acquire(blocking=False):
        raise RuntimeError("uncached_reads() is already open")
    log = ReadLog()
    prepare = DatasetBuilder.download_and_prepare
    as_dataset = DatasetBuilder.as_dataset
    factory = hf_load.dataset_module_factory
    modules: dict[str, Any] = {}

    def memo_factory(path: str, *args: Any, **kwargs: Any) -> Any:
        key = repr((path, args, sorted(kwargs.items(), key=lambda kv: kv[0])))
        if key not in modules:
            modules[key] = factory(path, *args, **kwargs)
        # `load_dataset_builder` pops keys from the module's builder kwargs.
        return copy.deepcopy(modules[key])

    def deferred_prepare(self: DatasetBuilder, *args: Any, **kwargs: Any) -> None:
        # Decided at `as_dataset`, which knows the split asked for.
        setattr(self, _PREPARE_ARGS, (args, kwargs))

    def in_memory_as_dataset(
        self: DatasetBuilder,
        split: Any = None,
        run_post_process: bool = True,
        verification_mode: VerificationMode | str | None = None,
        in_memory: bool = False,
    ) -> HFDataset | HFDatasetDict:
        names, reason = _memory_splits(self, split)
        built: dict[str, HFDataset] = {}
        if names is not None:
            try:
                built = {name: _read_split(self, name) for name in names}
            except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                # Batches whose inferred schemas differ: the cache writer casts
                # each to the first one's, which concatenation does not.
                names, reason = None, f"batches differ in schema ({exc})"
            except _EmptySplit as exc:
                # The cached path refuses it, in its own words.
                names, reason = None, f"split {exc} has no rows"
        if names is None:
            log.reads.append(BuilderRead(_label(self), "cache", reason))
            args, kwargs = getattr(self, _PREPARE_ARGS, ((), {}))
            prepare(self, *args, **kwargs)
            return as_dataset(
                self,
                split=split,
                run_post_process=run_post_process,
                verification_mode=verification_mode,
                in_memory=in_memory,
            )
        mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        if mode != VerificationMode.NO_CHECKS and self.info.splits:
            expected = {n: s for n, s in self.info.splits.items() if n in built}
            recorded = SplitDict(dataset_name=self.dataset_name)
            for name, rows in built.items():
                recorded.add(SplitInfo(name=name, num_examples=len(rows)))
            verify_splits(expected, recorded)
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
    """The split names to read in memory, or ``None`` and why not."""
    if not isinstance(builder, ArrowBasedBuilder):
        return None, "loading script"
    files = builder.config.data_files
    if files is None:
        return None, "no data files"
    if split is None:
        names = [str(name) for name in files]
    elif isinstance(split, str) and split in files:
        names = [split]
    else:
        return None, f"split {split!s} is not a split name"
    for name in names:
        size = _local_size(files[name])
        if size is None:
            return None, f"split {name} is not on local disk"
        if size > MAX_IN_MEMORY_BYTES:
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
    """A split that streamed no rows."""


def _read_split(builder: DatasetBuilder, name: str) -> HFDataset:
    streamed = builder.as_streaming_dataset(split=name)
    if not isinstance(streamed, IterableDataset):
        raise TypeError(f"split {name!r} streamed as {type(streamed).__name__}")
    tables = list(streamed.with_format("arrow").iter(batch_size=_BATCH_ROWS))
    if not tables:
        raise _EmptySplit(name)
    table = pa.concat_tables(tables)
    # `Split(name)` is how the cached path names it; at runtime a NamedSplit.
    split = Split(name)
    # The fingerprint the cached path gives this split. Without one, HF hashes
    # every row of an in-memory table on each later transform.
    return HFDataset(
        InMemoryTable(table),
        info=builder.info.copy(),
        split=cast(NamedSplit, split),
        fingerprint=builder._get_dataset_fingerprint(split),
    )
