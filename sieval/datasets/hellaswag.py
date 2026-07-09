"""
HellaSwag dataset — commonsense sentence completion (4-way multiple choice).

Mirrors the native ``Rowan/hellaswag`` parquet schema as-is. The lm-eval-harness
query/choice text construction lives in the task layer
(``sieval.community.hellaswag``), not here, so the loader stays a faithful source
mirror (per the dataset-layer rule "load() must mirror the source as-is").

Default eval split is ``validation``: HellaSwag withholds the ``test`` labels
(the native ``test`` split ships with empty ``label`` strings), so ``test`` cannot
be scored. Override via the ``eval_split`` load kwarg.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


class HellaSwagDatasetSample(TypedDict):
    ind: int
    activity_label: str
    ctx_a: str
    ctx_b: str
    ctx: str
    endings: list[str]
    source_id: str
    split: str
    split_type: str
    label: str


@sieval_dataset(
    name="hellaswag",
    display_name="HellaSwag",
    description="Commonsense sentence completion — pick the most plausible ending.",
    source=f"hf:Rowan/hellaswag@{HELLASWAG_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "CommonSense"),),
    tags=("english", "multiple-choice", "commonsense"),
    license="MIT",
)
class HellaSwagDataset(Dataset[HellaSwagDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        eval_split: str | None = "validation",
        **kwargs,
    ) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return apply_eval_split(dataset, eval_split)
