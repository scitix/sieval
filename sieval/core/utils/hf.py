import re

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import IterableDataset as HFIterableDataset
from datasets import IterableDatasetDict as HFIterableDatasetDict
from sieval.core.utils.paths import resolve_data_dir

# HF repo_id shape: `<org>/<name>`, each segment alphanumeric/underscore/dot/dash,
# leading char must be alphanumeric or underscore (rejects `../foo`, `./.`, etc.).
_HF_REPO_ID_RE = re.compile(r"^\w[\w.-]*/[\w.-]+$")


def maybe_resolve_hf_path(name_or_path: str) -> str:
    """Resolve a bare HF repo_id to its on-disk staging dir; pass-through otherwise.

    YAML convention writes hf-scheme datasets as bare repo_id (e.g.
    ``HuggingFaceH4/aime_2024``); ``sieval dataset download`` stages them at
    ``{SIEVAL_DATA_DIR}/<org>/<name>/`` as plain files. This helper closes the
    loop so ``Dataset(name_or_path="HuggingFaceH4/aime_2024")`` ends up calling
    ``load_dataset("{data_dir}/HuggingFaceH4/aime_2024", ...)`` via
    ``LocalDatasetModuleFactory`` — no online/offline split.

    Pass-through rule: anything that doesn't match the strict repo_id shape
    above. Concretely, this means absolute paths (leading ``/``), ``./``-
    prefixed relatives, unexpanded ``${VAR}`` forms, single-segment names,
    and paths with three or more segments. A bare two-segment relative path
    like ``data/foo.csv`` *does* match the pattern and gets rewritten — sieval
    YAML never produces that shape (callers either supply an HF repo_id or
    an absolute ``${SIEVAL_DATA_DIR}/...`` path after env expansion), so the
    collision is theoretical.

    Pattern matches but the staging dir doesn't exist → returned path triggers
    a downstream ``FileNotFoundError`` on ``load_dataset``, which the
    eval-session setup wraps with the standard "run ``sieval dataset download
    <name>`` first" hint.
    """
    if not _HF_REPO_ID_RE.match(name_or_path):
        return name_or_path
    return str(resolve_data_dir(None) / name_or_path)


def apply_eval_split(dataset: HFDatasetDict, eval_split: str | None) -> HFDatasetDict:
    """Remap *eval_split* to ``"test"`` so downstream code can access it uniformly."""
    if eval_split and eval_split in dataset and eval_split != "test":
        dataset = HFDatasetDict(dataset)
        dataset["test"] = dataset[eval_split]
    return dataset


def ensure_dataset_dict(
    dataset: HFDataset | HFIterableDataset | HFDatasetDict | HFIterableDatasetDict,
) -> HFDatasetDict:
    """Ensure the dataset is a DatasetDict, otherwise raise."""
    if isinstance(dataset, HFDatasetDict):
        return dataset
    if isinstance(dataset, HFIterableDatasetDict):
        raise TypeError(
            "IterableDatasetDict is not supported by current dataset interfaces."
        )
    raise TypeError(f"Expected DatasetDict, got {type(dataset).__name__}")


def ensure_dataset(
    dataset: HFDataset | HFIterableDataset | HFDatasetDict | HFIterableDatasetDict,
) -> HFDataset:
    """Ensure the dataset is a Dataset, otherwise raise."""
    if isinstance(dataset, HFDataset):
        return dataset
    if isinstance(dataset, HFIterableDataset):
        raise TypeError(
            "IterableDataset is not supported by current dataset interfaces."
        )
    raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")


def ensure_dataset_list(
    datasets: list[
        HFDataset | HFIterableDataset | HFDatasetDict | HFIterableDatasetDict
    ],
) -> list[HFDataset]:
    """Ensure every entry is a Dataset, otherwise raise."""
    normalized: list[HFDataset] = []
    for item in datasets:
        normalized.append(ensure_dataset(item))
    return normalized
