import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.community.math import strip_string
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict


class MATH500DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="math_500",
    display_name="MATH-500",
    description="Hendrycks MATH-500 subset — 500 problems across difficulty levels.",
    source="hf:HuggingFaceH4/MATH-500",
    categories=(Category(Level1Category.MATHEMATICS, "AdvancedMath"),),
    tags=("english", "open-ended"),
    license="MIT",
)
class MATH500Dataset(Dataset[MATH500DatasetSample]):
    def _strip_sample(self, sample: MATH500DatasetSample) -> MATH500DatasetSample:
        sample["problem"] = strip_string(sample["problem"])
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return dataset.map(self._strip_sample, num_proc=os.cpu_count())
