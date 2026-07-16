"""
ARC-Easy dataset loader.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from typing import Any, TypedDict, override

from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._arc import AI2_ARC_REVISION, load_arc


class ARCEasyDatasetSample(TypedDict):
    question: str
    choices: list[str]
    answer: int


@sieval_dataset(
    name="arc_easy",
    display_name="ARC-Easy",
    description="ARC-Easy science MCQ split from AI2 ARC, normalized for evaluation.",
    source=f"hf:allenai/ai2_arc@{AI2_ARC_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "STEM"),),
    tags=("english", "science", "multiple-choice"),
    license="cc-by-sa-4.0",
)
class ARCEasyDataset(Dataset[ARCEasyDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        eval_split: str | None = None,
        **kwargs: Any,
    ) -> HFDatasetDict:
        return load_arc(name_or_path, "ARC-Easy", eval_split, **kwargs)
