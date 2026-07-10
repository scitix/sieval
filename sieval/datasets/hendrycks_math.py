"""
Full Hendrycks MATH dataset (all 7 subjects, train + test).

Distinct from ``math_500`` (the 500-problem MATH-500 subset): this is the
complete benchmark — 7,500 train + 5,000 test problems. The source schema has
no ``answer`` column; the answer is the ``\\boxed{...}`` value inside
``solution`` (extract task-side, e.g. via ``sieval.community.deepseek_math``).

Data-source note: DeepSeek-Math's ``math-cot-test`` loads its bundled
``datasets/math/test.jsonl``; this loads ``EleutherAI/hendrycks_math`` instead —
verified equivalent (5,000 test rows, same 7 subjects).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
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
from sieval.core.utils.hf import ensure_dataset_dict

HENDRYCKS_MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"

# HF config names (one per subject); concatenated in this fixed order so the
# combined splits are reproducible.
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


class HendrycksMathDatasetSample(TypedDict):
    problem: str
    level: str
    type: str
    solution: str


@sieval_dataset(
    name="hendrycks_math",
    display_name="Hendrycks MATH",
    description="Full Hendrycks MATH — 12,500 competition problems across 7 subjects.",
    source=f"hf:EleutherAI/hendrycks_math@{HENDRYCKS_MATH_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "AdvancedMath"),),
    tags=("english", "open-ended"),
    license="MIT",
)
class HendrycksMathDataset(Dataset[HendrycksMathDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        train_subsets = []
        test_subsets = []
        for subject in SUBJECTS:
            subset = ensure_dataset_dict(
                load_dataset(name_or_path, name=subject, **kwargs)
            )
            train_subsets.append(subset["train"])
            test_subsets.append(subset["test"])
        return HFDatasetDict(
            {
                "train": concatenate_datasets(train_subsets),
                "test": concatenate_datasets(test_subsets),
            }
        )
