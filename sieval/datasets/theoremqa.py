"""
TheoremQA dataset wrapper.

AI-Generated Code - GPT-5 (OpenAI)
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
from sieval.core.utils.hf import ensure_dataset_dict

THEOREMQA_HF_REVISION = "a340b1782960a712843aae3ed25f1e013cc008a5"


class TheoremQADatasetSample(TypedDict):
    Question: str
    Answer: str
    Answer_type: str


@sieval_dataset(
    name="theoremqa",
    display_name="TheoremQA",
    description="Theorem-driven STEM QA benchmark with 800 expert-written questions.",
    source=f"hf:TIGER-Lab/TheoremQA@{THEOREMQA_HF_REVISION}",
    categories=(
        Category(Level1Category.KNOWLEDGE, "STEM"),
        Category(Level1Category.MATHEMATICS, "AppliedMath"),
    ),
    tags=("english", "open-ended", "theorem-driven"),
    license="MIT",
)
class TheoremQADataset(Dataset[TheoremQADatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        for split_name, split in dataset.items():
            if "Picture" in split.column_names:
                dataset[split_name] = split.remove_columns("Picture")
        return dataset
