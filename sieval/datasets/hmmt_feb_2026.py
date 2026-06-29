"""HMMT February 2026 dataset loader (MathArena source).

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
HMMT_FEB_2026_REVISION = "02fba4f74d8e68e73e66a02d540fd979c05c274c"


class HMMTFeb2026DatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="hmmt_feb_2026",
    display_name="HMMT Feb 2026",
    description="Harvard-MIT Mathematics Tournament, February 2026, 33 problems.",
    source=f"hf:MathArena/hmmt_feb_2026@{HMMT_FEB_2026_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class HMMTFeb2026Dataset(Dataset[HMMTFeb2026DatasetSample]):
    def _strip_sample(
        self, sample: HMMTFeb2026DatasetSample
    ) -> HMMTFeb2026DatasetSample:
        # Normalize the answer only; leave the problem text verbatim. strip_string
        # is an answer normalizer and mangles full problem LaTeX if applied to the
        # question — e.g. "1, 2, ..., n" becomes "1, 2, 0..., n". Matches the
        # aime_2024 / math_500 loaders.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem / answer / problem_idx. Rename `problem` -> `question` to
        # match the shared math sample schema.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        dataset = dataset.rename_column("problem", "question")
        # HMMT answers are already strings (symbolic + some plain integers); the
        # cast is a harmless no-op that keeps both math loaders uniform and the
        # `answer: str` contract explicit.
        dataset = dataset.cast_column("answer", Value("string"))
        dataset = dataset.map(self._strip_sample, num_proc=os.cpu_count())
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
