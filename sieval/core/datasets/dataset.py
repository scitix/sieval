"""Abstract Dataset base class backed by HuggingFace DatasetDict."""

import copy
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal, Self, overload

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from loguru import logger

from sieval.core.utils.hf import maybe_resolve_hf_path

TRetrieveStrategy = Literal["random", "fixed"]


class Dataset[TSample](ABC):
    """Abstract evaluation dataset backed by a HuggingFace DatasetDict.

    Transformations (repeat/slice/shuffle/stratified_sample) return immutable
    shallow copies.
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

        Returns ``self`` unchanged if *split* is absent.
        """
        if split not in self._dataset_dict:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = new_dict[split].repeat(times)
        return self._clone_with_new_dict(new_dict)

    def slice(self, num: int, split: str = "test") -> Self:
        """Return a shallow clone with only the first *num* samples of *split*.

        Positional, deterministic prefix. Keeps all samples if *num* exceeds
        split length. Returns ``self`` unchanged if *split* is absent.
        """
        if split not in self._dataset_dict:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        num_to_keep = min(num, len(new_dict[split]))
        new_dict[split] = new_dict[split].select(range(num_to_keep))
        return self._clone_with_new_dict(new_dict)

    def shuffle(self, seed: int = 0, split: str = "test") -> Self:
        """Return a shallow clone with *split* shuffled (deterministic via *seed*).

        Returns ``self`` unchanged if *split* is absent.
        """
        if split not in self._dataset_dict:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = new_dict[split].shuffle(seed=seed)
        return self._clone_with_new_dict(new_dict)

    def stratified_sample(
        self,
        by: str | list[str],
        *,
        num: int | None = None,
        per_group: int | None = None,
        min_per_group: int | None = None,
        seed: int = 0,
        split: str = "test",
    ) -> Self:
        """Return a clone keeping a group-balanced subsample of *split*.

        Rows are grouped into strata by the column(s) named in *by* (a single
        name, or a list whose values form a composite key). Exactly one budget
        must be given:

        * ``num`` — **proportional** allocation. Each stratum gets a floor of
          ``min(min_per_group, stratum_size)`` (``min_per_group`` defaults to 1);
          the remaining budget toward *num* is distributed proportionally to
          stratum size (capped by availability). If the floors already sum above
          *num*, the total rises to honour them and a warning is logged.
        * ``per_group`` — **equal** allocation. Each stratum keeps exactly
          ``min(per_group, stratum_size)`` rows; strata smaller than *per_group*
          keep all their rows and a single summary warning is logged.

        ``min_per_group`` applies only to the proportional (``num``) path and may
        not be combined with ``per_group``. Within each stratum rows are chosen by
        a deterministic *seed*-driven shuffle, so the selection reproduces across
        runs and processes.

        Returns ``self`` unchanged if *split* is absent or empty.
        """
        if (num is None) == (per_group is None):
            raise ValueError(
                "stratified_sample: provide exactly one of 'num' or 'per_group'"
            )
        if per_group is not None and min_per_group is not None:
            raise ValueError(
                "stratified_sample: 'min_per_group' applies only to proportional "
                "('num') sampling and cannot be combined with 'per_group'"
            )

        if split not in self._dataset_dict:
            return self
        hf = self._dataset_dict[split]
        if len(hf) == 0:
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
        else:
            # Proportional allocation toward num, honouring the floor.
            # num is non-None here (the per_group is None branch guarantees it).
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

        Returns a list or, if *lazy*, an iterator.  Empty if the split is missing.
        """
        ds = self._dataset_dict.get(split)
        if ds is None or len(ds) == 0:
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
