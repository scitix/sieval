"""Abstract Dataset base class backed by HuggingFace DatasetDict."""

import contextlib
import copy
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Literal, Self, cast, overload

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from loguru import logger

from sieval.core.utils.hf import maybe_resolve_hf_path

TRetrieveStrategy = Literal["random", "fixed"]

#: How :meth:`Dataset.filter` reads a row's key: a column name, a list of column
#: names forming a composite key, or a callable deriving one from the whole row.
TFilterKey = str | list[str] | Callable[[Mapping[str, object]], object]

#: Column :meth:`Dataset.repeat` stamps on every row, naming which copy the row
#: belongs to. Plainly named rather than dunder-prefixed because it is meant to be
#: read: a task may branch on it, and anything reporting a spread across repeats has
#: to group by it.
REPEAT_INDEX_COLUMN = "repeat_index"


def repeat_index_of(raw: object) -> int | None:
    """Read a repeated row's copy number, or ``None`` if the split was not repeated.

    Tolerant on purpose: a raw sample is whatever the dataset yields, so anything
    that is not a mapping carrying an integer under :data:`REPEAT_INDEX_COLUMN` reads
    as "not repeated" rather than raising. A bool is rejected explicitly — ``True``
    would otherwise pass ``isinstance(..., int)`` and record copy 1.

    Public because both seams that attach a row to a context share it: ``make_context``
    and the runner's resume backfill.
    """
    if not isinstance(raw, Mapping):
        return None
    # The cast only says "keys are strings"; the value is still checked below, so a
    # row carrying something other than an int under this key cannot slip through.
    value = cast("Mapping[str, object]", raw).get(REPEAT_INDEX_COLUMN)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class Dataset[TSample](ABC):
    """Abstract evaluation dataset backed by a HuggingFace DatasetDict.

    Transformations (repeat/slice/shuffle/filter/stratified_sample) return
    immutable shallow copies.
    """

    def __init__(
        self,
        name_or_path: str | None = None,
        *,
        _hf_dict: HFDatasetDict | None = None,
        **load_kwargs,
    ):
        """Initialize from *name_or_path* or a pre-built *_hf_dict*.

        ``_hf_dict`` takes priority; at least one is required.
        """
        if _hf_dict is not None:
            self._dataset_dict = _hf_dict
        elif name_or_path is not None:
            name_or_path = maybe_resolve_hf_path(name_or_path)
            self._dataset_dict = self.load(name_or_path, **load_kwargs)
        else:
            raise ValueError("Either name_or_path or _hf_dict must be provided.")

    @property
    def dataset_dict(self) -> HFDatasetDict:
        """The underlying HuggingFace DatasetDict."""
        return self._dataset_dict

    @property
    def train_set(self) -> HFDataset | None:
        """The ``"train"`` split, or ``None`` if it does not exist."""
        return self._dataset_dict.get("train")

    @property
    def test_set(self) -> HFDataset | None:
        """The ``"test"`` split, or ``None`` if it does not exist."""
        return self._dataset_dict.get("test")

    @abstractmethod
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        """Load a HuggingFace DatasetDict from *name_or_path*.

        Must return a DatasetDict containing at least a ``"test"`` split.
        """
        ...

    def repeat(self, times: int, split: str = "test") -> Self:
        """Return a shallow clone with *split* repeated *times* times.

        Every row is stamped with :data:`REPEAT_INDEX_COLUMN` — a 0-based copy
        number — because the copies are otherwise indistinguishable, and the spread
        across them is the only thing repeating a split measures. Without it the
        index has to be recovered from row position (``i // n_rows``), which is
        right only while the rows stay in the order this method emitted them:
        :meth:`shuffle` permutes them and leaves that arithmetic returning a
        confident wrong answer instead of raising. The stamp travels with the row,
        so it survives any later reordering, filtering or slicing.

        Stamped even when *times* is 1, so that the column's presence is a property
        of "this split was repeated" rather than of how many times — a column that
        appeared only sometimes would have to be guarded at every read.

        Returns ``self`` unchanged if *split* is absent, warning that it did.
        *times* is checked first, so a bad count is reported whichever split it named.

        Raises:
            ValueError: if *times* is less than 1 — HuggingFace answers zero and
                negatives with an empty split, which surfaces later as a run that
                silently scored zero samples, the failure :meth:`filter` also
                refuses. Or if *split* already carries a ``repeat_index`` column:
                overwriting it redefines a column the caller is presumably reading,
                and repeating twice needs a composite index one column cannot hold.
        """
        if times < 1:
            raise ValueError(
                f"repeat: 'times' must be at least 1; got {times}. Zero or a "
                f"negative would empty the split and score no samples."
            )
        if split not in self._dataset_dict:
            _report_no_op("repeat", _absent_split(split, self._dataset_dict))
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        original = new_dict[split]
        if REPEAT_INDEX_COLUMN in original.column_names:
            raise ValueError(
                f"split {split!r} already has a {REPEAT_INDEX_COLUMN!r} column; "
                f"repeating twice needs a composite index one column cannot carry. "
                f"Repeat once: if the task repeats this split itself (those taking "
                f"an 'n_repeats' argument), drop the repeat around it or set that to "
                f"1; otherwise rename the column, or drop an earlier repeat()'s."
            )
        n_rows = len(original)
        # Copy-major, matching what HuggingFace's own `repeat` concatenates: copy 0
        # in full, then copy 1. Built from `times`/`n_rows` rather than read back off
        # the result so the stamp cannot agree with a reordering that already
        # happened.
        new_dict[split] = original.repeat(times).add_column(
            REPEAT_INDEX_COLUMN, [i for i in range(times) for _ in range(n_rows)]
        )
        return self._clone_with_new_dict(new_dict)

    def slice(self, num: int, split: str = "test") -> Self:
        """Return a shallow clone with only the first *num* samples of *split*.

        Positional, deterministic prefix. Keeps all samples if *num* exceeds
        split length. Returns ``self`` unchanged if *split* is absent, warning
        that it did — an unnarrowed split is the failure that looks like a
        selection, since every row survives and scores as if *num* had applied.
        """
        if split not in self._dataset_dict:
            _report_no_op("slice", _absent_split(split, self._dataset_dict))
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        num_to_keep = min(num, len(new_dict[split]))
        new_dict[split] = new_dict[split].select(range(num_to_keep))
        return self._clone_with_new_dict(new_dict)

    def shuffle(self, seed: int = 0, split: str = "test") -> Self:
        """Return a shallow clone with *split* shuffled (deterministic via *seed*).

        Returns ``self`` unchanged if *split* is absent, warning that it did.
        """
        if split not in self._dataset_dict:
            _report_no_op("shuffle", _absent_split(split, self._dataset_dict))
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = new_dict[split].shuffle(seed=seed)
        return self._clone_with_new_dict(new_dict)

    def filter(
        self,
        by: TFilterKey,
        value: object,
        *,
        require_all: bool = False,
        split: str = "test",
    ) -> Self:
        """Return a shallow clone of *split* keeping rows whose key is accepted.

        *by* says how to read a row's key, in three forms:

        * a **column name** — the key is that column's value;
        * a **list of column names** — the key is the tuple of their values,
          selecting on a composite identity no single column carries;
        * a **callable** ``row -> key`` — the key is derived rather than stored,
          which is what a content hash, a normalised id or a concatenation of
          fields needs. Driven from a config the key must be a **scalar**:
          neither YAML nor JSON has a tuple, and the list each writes instead is
          unhashable. From Python a tuple is fine.

        *value* is one accepted key, or a list/tuple/set of them (membership
        test). A string is always one key, never a set of characters. Under a
        composite *by* each accepted key must itself be a sequence of the same
        length — ``value: [[a, b]]`` is one two-column key, ``value: [a, b]`` is
        two one-column keys.

        Relative order among the kept rows is preserved, so a caller that
        narrows to one category gets the same sequence — and therefore the same
        sample ids — it would have got by loading that category alone.

        *require_all* raises on a requested key that matches no row, where the
        default only warns; a request may legitimately over-cover a split. It
        checks the *keys*, not the row count — one key matching many rows is
        not an error.

        Returns ``self`` unchanged if *split* is absent or empty, warning that
        it did, as every transform here does. *require_all* escalates that to a
        raise: asking for every key to land is not a question about one split,
        and the flag would otherwise be defeated by a typo in it.

        Raises if *by* names a column that does not exist, or if **nothing**
        matches: an empty result is a misspelled *value* far more often than an
        intended selection, and it would otherwise surface as a run that
        silently scores zero samples.
        """
        if split not in self._dataset_dict:
            _report_no_op(
                "filter", _absent_split(split, self._dataset_dict), strict=require_all
            )
            return self
        hf = self._dataset_dict[split]
        if len(hf) == 0:
            _report_no_op("filter", f"split {split!r} is empty", strict=require_all)
            return self

        cols = _filter_columns(by, hf.column_names)
        label = _filter_key_label(by)
        keys = _derive_filter_keys(hf, by, cols)
        raw = list(value) if isinstance(value, list | tuple | set) else [value]
        if isinstance(value, set):
            # A set has no order of its own; give it one, so the messages below
            # quote the same keys in the same order across runs of the same
            # selection. Mixed-type keys are not orderable — those keep
            # iteration order rather than failing here.
            with contextlib.suppress(TypeError):
                raw = sorted(raw)
        # Composite keys are tupled *before* the dedup below, which hashes them.
        if cols is not None and len(cols) > 1:
            raw = [_filter_composite_key(v, len(cols), label) for v in raw]
        # Deduped but ordered: the set drives membership, the list fixes the
        # order the messages below quote.
        try:
            requested = list(dict.fromkeys(raw))
        except TypeError as exc:
            # The one shape that reaches here from a config: a callable key
            # whose accepted values were written as JSON/YAML lists.
            hint = (
                " (a callable key driven from a config file must take scalar "
                "accepted values — see the 'by' callable form in the docstring)"
                if cols is None and any(isinstance(v, list) for v in raw)
                else ""
            )
            raise ValueError(
                f"filter: accepted values must be hashable; {exc}{hint}"
            ) from exc
        if not requested:
            raise ValueError(
                f"filter: no accepted values given for {label}; an empty "
                "selection would keep no rows at all"
            )
        accepted = set(requested)

        try:
            kept_indices = [i for i, key in enumerate(keys) if key in accepted]
        except TypeError as exc:
            raise ValueError(
                f"filter: {label} produced a key that cannot be compared for "
                f"membership ({exc}); keys must be hashable"
            ) from exc

        if not kept_indices:
            present = list(dict.fromkeys(keys))
            shown = present[:10]
            suffix = f", ... ({len(present)} distinct)" if len(present) > 10 else ""
            # A `values_file` puts thousands of keys in `value`, and no overlap
            # at all is exactly what a stale id list produces — truncate it the
            # way the unmatched warning below already truncates its own.
            asked = (
                f"{requested[:10]}, ... ({len(requested)} requested)"
                if len(requested) > 10
                else repr(value)
            )
            raise ValueError(
                f"filter: no row of split {split!r} has {label}={asked}; "
                f"present values: {shown}{suffix}"
            )

        matched = {keys[i] for i in kept_indices}
        unmatched = [key for key in requested if key not in matched]
        if unmatched:
            shown = unmatched[:10]
            suffix = ", ..." if len(unmatched) > 10 else ""
            detail = (
                f"filter: {len(unmatched)} of {len(requested)} requested keys "
                f"match no row of split {split!r} on {label}; "
                f"unmatched: {shown}{suffix}"
            )
            if require_all:
                raise ValueError(detail)
            logger.warning(detail)

        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = hf.select(kept_indices)
        return self._clone_with_new_dict(new_dict)

    def stratified_sample(
        self,
        by: str | list[str],
        *,
        num: int | None = None,
        per_group: int | None = None,
        fraction: float | None = None,
        min_per_group: int | None = None,
        seed: int = 0,
        split: str = "test",
    ) -> Self:
        """Return a clone keeping a group-balanced subsample of *split*.

        Rows are grouped into strata by the column(s) named in *by* (a single
        name, or a list whose values form a composite key). Exactly one budget
        must be given:

        * ``num`` — **proportional** allocation toward a *total*. Each stratum
          gets a floor of ``min(min_per_group, stratum_size)`` (``min_per_group``
          defaults to 1); the remaining budget toward *num* is distributed
          proportionally to stratum size (capped by availability). If the floors
          already sum above *num*, the total rises to honour them and a warning
          is logged.
        * ``per_group`` — **equal** allocation. Each stratum keeps exactly
          ``min(per_group, stratum_size)`` rows; strata smaller than *per_group*
          keep all their rows and a single summary warning is logged.
        * ``fraction`` — **share of each stratum**, in ``(0, 1]``. Each stratum
          keeps ``ceil(stratum_size * fraction)`` rows, never fewer than the
          floor. Unlike ``num`` this needs no total, so it states a protocol
          ("10% of every locale") directly and stays correct when the dataset
          grows. The resulting total is whatever the per-stratum ceilings sum to.

        ``min_per_group`` applies to the ``num`` and ``fraction`` paths, which
        both allocate per stratum, and may not be combined with ``per_group``
        (which already fixes the count). It behaves differently on the two,
        because rounding up already keeps every non-empty stratum: on ``num`` the
        floor decides whether a small stratum survives at all (``min_per_group=0``
        lets one drop to zero), while on ``fraction`` it can only *raise* a
        stratum's share above its ceiling — a stratum there always keeps at least
        one row. Within each stratum rows are chosen by a deterministic
        *seed*-driven shuffle, so the selection reproduces across runs and
        processes.

        Returns ``self`` unchanged if *split* is absent or empty, warning that
        it did — like :meth:`slice`, an unsubsampled split keeps every row and
        scores as if the budget had applied.
        """
        budgets = [num, per_group, fraction]
        if sum(budget is not None for budget in budgets) != 1:
            raise ValueError(
                "stratified_sample: provide exactly one of 'num', 'per_group' "
                "or 'fraction'"
            )
        if per_group is not None and min_per_group is not None:
            raise ValueError(
                "stratified_sample: 'min_per_group' applies to the per-stratum "
                "('num' / 'fraction') paths and cannot be combined with "
                "'per_group'"
            )
        # Budgets arrive from YAML, so the annotation is not a guarantee. `bool`
        # is rejected explicitly because it is an `int` subclass: `fraction: true`
        # would otherwise pass `0 < f <= 1` and silently keep every row. The
        # message matches the one `EvalSession` raises, so the same bad value
        # reads the same whether it came from YAML or a direct call.
        if fraction is not None and (
            isinstance(fraction, bool)
            or not isinstance(fraction, int | float)
            or not 0 < fraction <= 1
        ):
            raise ValueError(
                "stratified_sample: 'fraction' must be a number in the interval "
                f"(0, 1]; got {fraction!r}"
            )

        if split not in self._dataset_dict:
            _report_no_op("stratified_sample", _absent_split(split, self._dataset_dict))
            return self
        hf = self._dataset_dict[split]
        if len(hf) == 0:
            _report_no_op("stratified_sample", f"split {split!r} is empty")
            return self

        cols = [by] if isinstance(by, str) else list(by)
        if not cols:
            raise ValueError("stratified_sample: 'by' must name at least one column")
        missing = [c for c in cols if c not in hf.column_names]
        if missing:
            raise ValueError(
                f"stratified_sample: column(s) {missing!r} not found; "
                f"available columns: {hf.column_names}"
            )

        # Group row indices by composite key. A single column keeps a scalar key
        # (not a 1-tuple) so the within-stratum seed string stays byte-identical
        # with the pre-multikey behaviour.
        column_data = [hf[c] for c in cols]
        single = len(cols) == 1
        groups: dict[object, list[int]] = {}
        for index in range(len(hf)):
            values = tuple(col[index] for col in column_data)
            key = values[0] if single else values
            groups.setdefault(key, []).append(index)

        keys = sorted(groups, key=str)
        sizes = {key: len(groups[key]) for key in keys}

        if per_group is not None:
            # Equal allocation: K per stratum, capped at availability.
            alloc = {key: min(per_group, sizes[key]) for key in keys}
            short = [key for key in keys if sizes[key] < per_group]
            if short:
                logger.warning(
                    "stratified_sample: per_group={} unmet for {} of {} strata "
                    "(short {} rows total); kept all available in those",
                    per_group,
                    len(short),
                    len(keys),
                    sum(per_group - sizes[key] for key in short),
                )
        elif fraction is not None:
            # Share of each stratum, rounded up so a small stratum still
            # contributes; the floor applies as it does on the `num` path. No
            # total is involved, so nothing has to be redistributed.
            floor = 1 if min_per_group is None else min_per_group
            alloc = {
                key: min(sizes[key], max(floor, math.ceil(sizes[key] * fraction)))
                for key in keys
            }
        else:
            # Proportional allocation toward num, honouring the floor.
            # num is non-None here: it is the only budget left once per_group and
            # fraction are ruled out, and exactly one was required.
            assert num is not None
            floor = 1 if min_per_group is None else min_per_group
            total = len(hf)
            alloc = {key: min(floor, sizes[key]) for key in keys}
            target = min(max(num, sum(alloc.values())), total)
            if target > num:
                logger.warning(
                    "stratified_sample: min_per_group={} across {} groups requires "
                    "{} rows, exceeding the requested num={}",
                    floor,
                    len(keys),
                    target,
                    num,
                )
            while sum(alloc.values()) < target:
                candidates = [key for key in keys if alloc[key] < sizes[key]]
                if not candidates:
                    break
                # Group furthest below its proportional quota; ties → smallest key.
                chosen = max(
                    candidates,
                    key=lambda key: sizes[key] * target / total - alloc[key],
                )
                alloc[chosen] += 1

        # Deterministic within-group selection.
        selected: list[int] = []
        for key in keys:
            indices = list(groups[key])
            random.Random(f"{seed}:{key}").shuffle(indices)
            selected.extend(indices[: alloc[key]])
        selected.sort()

        new_dict = HFDatasetDict(self._dataset_dict)
        new_dict[split] = hf.select(selected)
        return self._clone_with_new_dict(new_dict)

    def _clone_with_new_dict(self, new_dict: HFDatasetDict) -> Self:
        """Shallow-copy this Dataset with a replacement DatasetDict."""
        new_instance = copy.copy(self)
        new_instance._dataset_dict = new_dict
        return new_instance

    @overload
    def retrieve_samples(
        self,
        k: int,
        split: str = "train",
        *,
        mode: Literal["random"] = "random",
        seed: int = 0,
        lazy: Literal[False] = False,
    ) -> list[TSample]: ...

    @overload
    def retrieve_samples(
        self,
        k: int,
        split: str = "train",
        *,
        mode: Literal["random"] = "random",
        seed: int = 0,
        lazy: Literal[True],
    ) -> Iterator[TSample]: ...

    @overload
    def retrieve_samples(
        self,
        k: int,
        split: str = "train",
        *,
        mode: Literal["fixed"],
        indices: list[int] | None = None,
        lazy: Literal[False] = False,
    ) -> list[TSample]: ...

    @overload
    def retrieve_samples(
        self,
        k: int,
        split: str = "train",
        *,
        mode: Literal["fixed"],
        indices: list[int] | None = None,
        lazy: Literal[True],
    ) -> Iterator[TSample]: ...

    def retrieve_samples(
        self,
        k: int,
        split: str = "train",
        *,
        mode: TRetrieveStrategy = "random",
        seed: int = 0,
        indices: list[int] | None = None,
        lazy: bool = False,
    ) -> list[TSample] | Iterator[TSample]:
        """Retrieve *k* samples from *split*.

        Modes:

        * ``"random"`` — shuffle with *seed*, take first *k*.
        * ``"fixed"`` — select by *indices* (default ``0..k-1``); out-of-range dropped.

        Returns a list or, if *lazy*, an iterator. Empty if *split* is missing
        or empty, warning that it was: returning nothing serves a k-shot prompt
        as 0-shot, which the run reports as a plausible number. Every caller
        guards the split and raises first, so this is the net under them.
        """
        ds = self._dataset_dict.get(split)
        if ds is None or len(ds) == 0:
            why = (
                _absent_split(split, self._dataset_dict)
                if ds is None
                else f"split {split!r} is empty"
            )
            _report_no_op("retrieve_samples", why, outcome="no samples are returned")
            return iter([]) if lazy else []

        k = min(k, len(ds))
        if mode == "random":
            selected_ds = ds.shuffle(seed=seed).select(range(k))
        elif mode == "fixed":
            if indices is None:
                # Default: first k samples
                indices = list(range(k))
            else:
                # Validate and clip indices
                indices = [i for i in indices if 0 <= i < len(ds)][:k]
            selected_ds = ds.select(indices)
        else:
            raise ValueError(f"Unknown mode: {mode}.")

        return iter(selected_ds) if lazy else list(selected_ds)


def _absent_split(split: str, have: Iterable) -> str:
    """Why an operation had nothing to act on, when the split is not there.

    *have* is bare rather than ``Iterable[str]`` because every caller passes a
    ``DatasetDict``, whose upstream stub does not parameterise ``dict`` — so the
    element type would be a claim the checker cannot see, not one it verifies.
    """
    return f"split {split!r} is not in this dataset (have: {sorted(have)})"


def _report_no_op(
    op: str,
    why: str,
    *,
    outcome: str = "the dataset is unchanged",
    strict: bool = False,
) -> None:
    """Report an operation a missing or empty split left with nothing to do.

    Returning early is right — a config may name a split only some datasets
    carry — but silence is not, and it costs differently per caller:

    * ``slice``, ``filter`` and ``stratified_sample`` exist to *narrow*, so a
      skipped narrowing keeps every row and the run reports a plausible number.
    * ``repeat`` and ``shuffle`` fail toward fewer rows or unchanged order,
      which surfaces downstream. They report for the contract, not the danger.
    * ``retrieve_samples`` returns nothing, serving a k-shot prompt as 0-shot.
      It overrides *outcome*: nothing was "unchanged", the caller got nothing.

    Only :meth:`Dataset.filter` passes *strict*, because only it has a flag
    (``require_all``) that already promised the selection landed.
    """
    detail = f"{op}: {why}, so {outcome}"
    if strict:
        raise ValueError(detail)
    logger.warning(detail)


def _filter_columns(by: TFilterKey, column_names: list[str]) -> list[str] | None:
    """The columns *by* reads, or ``None`` when it is a callable.

    Callables are handed the whole row, so there is nothing to check up front —
    the columns they touch are not knowable without running them.
    """
    if callable(by):
        return None
    cols = [by] if isinstance(by, str) else list(by)
    if not cols:
        raise ValueError("filter: 'by' must name at least one column")
    if not all(isinstance(col, str) for col in cols):
        raise ValueError(f"filter: 'by' must name columns as strings; got {by!r}")
    missing = [col for col in cols if col not in column_names]
    if missing:
        # Unchanged wording for the single-column case, which is most calls.
        if isinstance(by, str):
            raise ValueError(
                f"filter: column {by!r} not found; available columns: {column_names}"
            )
        raise ValueError(
            f"filter: column(s) {missing!r} not found; "
            f"available columns: {column_names}"
        )
    return cols


def _filter_key_label(by: TFilterKey) -> str:
    """How *by* is named in diagnostics."""
    if callable(by):
        return f"{getattr(by, '__qualname__', None) or type(by).__name__}()"
    if isinstance(by, str):
        return by
    return f"({', '.join(by)})"


def _derive_filter_keys(
    hf: HFDataset, by: TFilterKey, cols: list[str] | None
) -> list[object]:
    """One key per row of *hf*, in row order."""
    if cols is None:
        # A callable can read anything, so the rows must be materialised; the
        # column paths below read only the columns named. `cols is None` is the
        # callable form and nothing else — but `callable(by)` would not narrow
        # to it, since a str subclass could carry `__call__`.
        assert not isinstance(by, str | list)
        keys: list[object] = []
        for index, row in enumerate(hf):
            try:
                keys.append(by(row))
            except Exception as exc:
                # `by` may name any importable function, so its failure is a
                # config error — name the row and the function rather than
                # letting a bare KeyError surface.
                raise ValueError(
                    f"filter: key function {_filter_key_label(by)} raised on "
                    f"row {index} of {len(hf)}: {type(exc).__name__}: {exc}"
                ) from exc
        return keys
    if len(cols) == 1:
        return list(hf[cols[0]])
    # zip over whole columns: building each tuple from a per-row generator
    # instead costs ~4.5x on a 200k-row split for an identical result.
    return list(zip(*(hf[col] for col in cols), strict=True))


def _filter_composite_key(value: object, n_cols: int, label: str) -> tuple:
    """*value* as a composite key of *n_cols* parts.

    Rejects a scalar rather than promoting it: under a two-column *by*,
    ``[a, b]`` is two keys and ``[[a, b]]`` is one, so a scalar here means the
    caller wrote the first while meaning the second.
    """
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(
            f"filter: {label} is a composite key, so each accepted value must "
            f"be a sequence of its {n_cols} parts (e.g. value: [[a, b]] for one "
            f"key, not value: [a, b]); got {value!r}"
        )
    key = tuple(value)
    if len(key) != n_cols:
        raise ValueError(
            f"filter: {label} has {n_cols} parts but the accepted value "
            f"{value!r} has {len(key)}"
        )
    return key
