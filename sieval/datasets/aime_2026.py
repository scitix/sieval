"""AIME 2026 dataset loader (MathArena source).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Value, load_dataset

from sieval.community.math import strip_string
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

# Pin the MathArena HF snapshot for reproducibility (see check_datasets / #8).
AIME_2026_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"


class AIME2026DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="aime_2026",
    display_name="AIME 2026",
    description="American Invitational Mathematics Examination 2026, 30 problems.",
    source=f"hf:MathArena/aime_2026@{AIME_2026_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class AIME2026Dataset(Dataset[AIME2026DatasetSample]):
    def _strip_sample(self, sample: AIME2026DatasetSample) -> AIME2026DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX (\sqrt[20]{x} -> \sqrt{[}20]{x}). DEVIATION: matharena does not
        # normalize golds; sieval does, so math-verify compares canonical forms.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem / answer / problem_idx, kept under their upstream names.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        # MathArena ships AIME answers as int64; cast to string up front so the
        # `answer: str` contract holds (otherwise `.map` re-infers against the
        # existing int64 feature and silently casts the stripped string back).
        dataset = dataset.cast_column("answer", Value("string"))
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
