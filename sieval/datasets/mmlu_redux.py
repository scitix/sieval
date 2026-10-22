"""
MMLU-Redux dataset loader — ``edinburgh-dawg/mmlu-redux``.

3,000 MMLU questions (30 subjects x 100) re-annotated for label errors.
``load`` merges the per-subject configs into one ``test`` split and stamps each
row with its ``subject``. The annotation columns (``error_type``,
``correct_answer``, ``potential_reason``) are kept; dropping non-``ok`` rows is
the task's call.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

MMLU_REDUX_REVISION = "3720db6aeb3d019de48bf37916c1a54074ff4997"

# Config names at the pin, merged in sorted order so sample ids are stable.
MMLU_REDUX_SUBJECTS = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "high_school_us_history",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "public_relations",
    "virology",
)


class MMLUReduxDatasetSample(TypedDict):
    question: str
    choices: list[str]
    # No cast needed: upstream ships `answer` as an int64 index into `choices`.
    answer: int
    error_type: str
    source: str
    correct_answer: str | None
    potential_reason: str | None
    subject: str


@sieval_dataset(
    name="mmlu_redux",
    display_name="MMLU-Redux",
    description="MMLU-Redux — 3,000 re-annotated MMLU questions, 30 subjects, MCQ.",
    source=f"hf:edinburgh-dawg/mmlu-redux@{MMLU_REDUX_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "multiple-choice"),
    license="CC-BY-4.0",
)
class MMLUReduxDataset(Dataset[MMLUReduxDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        parts = []
        for subject in MMLU_REDUX_SUBJECTS:
            rows = ensure_dataset(
                load_dataset(name_or_path, subject, split="test", **kwargs)
            )
            if len(rows) == 0:
                raise ValueError(
                    f"MMLU-Redux subject {subject!r} is empty; expected revision "
                    f"{MMLU_REDUX_REVISION}."
                )
            parts.append(rows.map(lambda _row, s=subject: {"subject": s}))
        return HFDatasetDict({"test": concatenate_datasets(parts)})
