"""HMMT February 2025 dataset loader (MathArena source).

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
HMMT_FEB_2025_REVISION = "6fdc4277120810ff75aa22d2d5489b91f7a262a1"


class HMMTFeb2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="hmmt_feb_2025",
    display_name="HMMT Feb 2025",
    description="Harvard-MIT Mathematics Tournament, February 2025, 30 problems.",
    source=f"hf:MathArena/hmmt_feb_2025@{HMMT_FEB_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class HMMTFeb2025Dataset(Dataset[HMMTFeb2025DatasetSample]):
    def _strip_sample(
        self, sample: HMMTFeb2025DatasetSample
    ) -> HMMTFeb2025DatasetSample:
        # Normalize the gold answer only; the problem text stays verbatim.
        # strip_string is an answer normalizer (rewrites \frac/\sqrt, drops
        # \left/\right) and mangles full problem LaTeX — e.g. \sqrt[20]{x} becomes
        # \sqrt{[}20]{x}. DEVIATION: matharena does not normalize golds at all;
        # sieval does so math-verify compares canonical forms.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / problem / answer / problem_type, kept under their
        # upstream names.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        # No dtype cast: `answer` is `string` in the pinned snapshot, so the
        # `answer: str` contract holds from upstream and `.map` preserves it across
        # _strip_sample. MathArena's answer dtype varies per competition, so a
        # sibling loader casting is a fact about its own pin, not a family rule.
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
