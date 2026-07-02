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

# Pinned to the snapshot behind current results. HF main has advanced since
# (an eval.yaml-only change; sample data is byte-identical) — don't bump blindly.
MMLU_PRO_REVISION = "54611cde22c74cca43dd78732198de6abe971398"


class MMLUProDatasetSample(TypedDict):
    question: str
    options: list[str]
    answer: str
    category: str


@sieval_dataset(
    name="mmlu_pro",
    display_name="MMLU-Pro",
    description="MMLU-Pro — harder MCQ with 10 options, filtered for reasoning.",
    source=f"hf:TIGER-Lab/MMLU-Pro@{MMLU_PRO_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "multiple-choice"),
    license="MIT",
)
class MMLUProDataset(Dataset[MMLUProDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        dataset = apply_eval_split(dataset, eval_split)
        return dataset
