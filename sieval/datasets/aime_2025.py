import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.community.math import strip_string
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_list

AIME_2025_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"


class AIME2025DatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="aime_2025",
    display_name="AIME 2025",
    description="American Invitational Mathematics Examination 2025, 30 problems.",
    source=f"hf:opencompass/AIME2025@{AIME_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="MIT",
)
class AIME2025Dataset(Dataset[AIME2025DatasetSample]):
    def _strip_sample(self, sample: AIME2025DatasetSample) -> AIME2025DatasetSample:
        sample["question"] = strip_string(sample["question"])
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        subsets_names = ["AIME2025-I", "AIME2025-II"]
        subsets = [
            load_dataset(name_or_path, name=subset_name, split="test", **kwargs)
            for subset_name in subsets_names
        ]
        subsets = ensure_dataset_list(subsets)
        dataset = concatenate_datasets(subsets)
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
