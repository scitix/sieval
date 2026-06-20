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
from sieval.core.utils.hf import ensure_dataset

AIME_2024_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"


class AIME2024DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="aime_2024",
    display_name="AIME 2024",
    description="American Invitational Mathematics Examination 2024, 30 problems.",
    source=f"hf:HuggingFaceH4/aime_2024@{AIME_2024_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    # Inherited from upstream AI-MO/aimo-validation-aime (Apache-2.0 on HF).
    # Covers the packaged artifact; AIME problem text remains MAA-copyrighted.
    license="Apache-2.0",
)
class AIME2024Dataset(Dataset[AIME2024DatasetSample]):
    def _strip_sample(self, sample: AIME2024DatasetSample) -> AIME2024DatasetSample:
        sample["problem"] = strip_string(sample["problem"])
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, split="train", **kwargs)
        dataset = ensure_dataset(dataset)
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
