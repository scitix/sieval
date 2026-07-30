"""HMMT November 2025 dataset loader (MathArena source).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
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
HMMT_NOV_2025_REVISION = "118dbfb45c4c9467c672268ed55166642897aa46"


class HMMTNov2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="hmmt_nov_2025",
    display_name="HMMT Nov 2025",
    description="Harvard-MIT Mathematics Tournament, November 2025, 30 problems.",
    source=f"hf:MathArena/hmmt_nov_2025@{HMMT_NOV_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class HMMTNov2025Dataset(Dataset[HMMTNov2025DatasetSample]):
    def _strip_sample(
        self, sample: HMMTNov2025DatasetSample
    ) -> HMMTNov2025DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX (\sqrt[20]{x} -> \sqrt{[}20]{x}). DEVIATION: matharena does not
        # normalize golds; sieval does, so math-verify compares canonical forms.
        # Score-neutral here: 3 of 30 golds change (1/91 -> \frac{1}{91}, 91/6,
        # 199/8), all already equivalent under math-verify.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / problem / answer, kept under their upstream names.
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
