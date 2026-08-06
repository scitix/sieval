"""SMT 2025 dataset loader (MathArena source).

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
SMT_2025_REVISION = "d8e99d83f1583842f46d4953cd960eea5d887e45"


class SMT2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="smt_2025",
    display_name="SMT 2025",
    description="Stanford Math Tournament, 2025, 53 problems.",
    source=f"hf:MathArena/smt_2025@{SMT_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class SMT2025Dataset(Dataset[SMT2025DatasetSample]):
    def _strip_sample(self, sample: SMT2025DatasetSample) -> SMT2025DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX (collapses the `\\` row break inside aligned environments, 13 of 53
        # problems). DEVIATION: matharena does not normalize golds; sieval does, so
        # math-verify compares canonical forms. Score-neutral here: 0 of 53 golds
        # change, so this is a no-op on the pinned snapshot.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / answer / problem_type / problem, kept under their
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
