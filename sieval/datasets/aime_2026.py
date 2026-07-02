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
    question: str
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
        # Normalize the answer only; leave the problem text verbatim. strip_string
        # is an answer normalizer (it rewrites \frac/\sqrt, drops \left/\right) and
        # mangles full problem LaTeX if applied to the question — e.g. \sqrt[20]{x}
        # becomes \sqrt{[}20]{x} and \tfrac pq becomes \frac{ }{p}q. Matches the
        # aime_2024 / math_500 loaders.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem / answer / problem_idx. Rename `problem` -> `question` to
        # match the shared AIME sample schema.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        dataset = dataset.rename_column("problem", "question")
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
