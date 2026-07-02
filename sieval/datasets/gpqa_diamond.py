import os
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

GPQADiamondDatasetSample = TypedDict(
    "GPQADiamondDatasetSample",
    {
        "Question": str,
        "Correct Answer": str,
        "Incorrect Answer 1": str,
        "Incorrect Answer 2": str,
        "Incorrect Answer 3": str,
    },
)


@sieval_dataset(
    name="gpqa_diamond",
    display_name="GPQA Diamond",
    description="Graduate-level science MCQ — diamond subset, 198 questions.",
    source="url:https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
    checksums={
        "gpqa_diamond.csv": "sha256:41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",  # noqa: E501
    },
    categories=(
        Category(Level1Category.LOGIC, "ComplexLogic"),
        Category(Level1Category.KNOWLEDGE, "STEM"),
    ),
    tags=("english", "multiple-choice", "graduate-level"),
    license="CC-BY-4.0",
)
class GPQADiamondDataset(Dataset[GPQADiamondDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        csv_path = (
            os.path.join(name_or_path, "gpqa_diamond.csv")
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        data_files = {"test": csv_path}
        dataset = load_dataset("csv", data_files=data_files, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return dataset
