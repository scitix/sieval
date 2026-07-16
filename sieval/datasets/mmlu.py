"""
MMLU dataset loader — mirrors the official cais/mmlu HF dataset as-is.

The native schema (``question`` / ``subject`` / ``choices`` / ``answer`` as a
0-3 ClassLabel over A-D) is preserved without coercion; per-task formatting
lives in the task layer. The ``all`` config exposes ``test`` (14042),
``validation``, ``dev`` (57 subjects x 5, the official few-shot source), and
``auxiliary_train`` splits.

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

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


class MMLUDatasetSample(TypedDict):
    question: str
    subject: str
    choices: list[str]
    answer: int


@sieval_dataset(
    name="mmlu",
    display_name="MMLU",
    description="Massive Multitask Language Understanding — 57 academic subjects, MCQ.",
    source=f"hf:cais/mmlu@{MMLU_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "multiple-choice"),
    license="MIT",
)
class MMLUDataset(Dataset[MMLUDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = "all",
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, config, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return apply_eval_split(dataset, eval_split)
