"""BRUMO 2025 dataset loader (MathArena source).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

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

# Pin the MathArena HF snapshot for reproducibility (see check_datasets / #8).
BRUMO_2025_REVISION = "6de9bc0987f5710c6376ee980e1483dbeb55cfda"


class BRUMO2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="brumo_2025",
    display_name="BRUMO 2025",
    description="Brown University Math Olympiad, 2025, 30 problems.",
    source=f"hf:MathArena/brumo_2025@{BRUMO_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class BRUMO2025Dataset(Dataset[BRUMO2025DatasetSample]):
    def _strip_sample(self, sample: BRUMO2025DatasetSample) -> BRUMO2025DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX (f\left(\frac{1}{x}\right) -> f(\frac{1}{x}), 6 of 30 problems).
        # DEVIATION: matharena does not normalize golds; sieval does, so
        # math-verify compares canonical forms. Score-neutral here: 0 of 30 golds
        # change, so this is a no-op on the pinned snapshot.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / problem / answer / problem_type, kept under their
        # upstream names.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        # No dtype cast: `answer` is already `string` in the pinned snapshot and
        # `.map` preserves it. MathArena's answer dtype varies per competition, so a
        # sibling loader's cast is a fact about its own pin, not a family rule.
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
