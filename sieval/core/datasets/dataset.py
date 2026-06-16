"""Abstract Dataset base class backed by HuggingFace DatasetDict."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal, Self, overload

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from sieval.core.utils.hf import maybe_resolve_hf_path

TRetrieveStrategy = Literal["random", "fixed"]


class Dataset[TSample](ABC):
    """Abstract evaluation dataset backed by a HuggingFace DatasetDict.

    Transformations (repeat/select/shuffle) return immutable shallow copies.
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

        Returns ``self`` unchanged if the test set is absent.
        """
        if self.test_set is None:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = new_dict[split].repeat(times)
        return self._clone_with_new_dict(new_dict)

    def select(self, num: int, split: str = "test") -> Self:
        """Return a shallow clone with only the first *num* samples of *split*.

        Keeps all samples if *num* exceeds split length.
        """
        if self.test_set is None:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        num_to_select = min(num, len(new_dict[split]))
        new_dict[split] = new_dict[split].select(range(num_to_select))
        return self._clone_with_new_dict(new_dict)

    def shuffle(self, seed: int = 0, split: str = "test") -> Self:
        """Return a shallow clone with *split* shuffled (deterministic via *seed*)."""
        if self.test_set is None:
            return self
        new_dict = HFDatasetDict(self.dataset_dict)
        new_dict[split] = new_dict[split].shuffle(seed=seed)
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
