"""MathArena Apex Shortlist 2025 dataset loader (MathArena source).

The shortlist Apex was drawn from: 47 problems from 2025 contests on which
state-of-the-art models score around 50%, so its `source` column carries each
problem's originating contest. Three overlap sibling loaders byte-for-byte —
`brumo_2025` problem 30 and `hmmt_feb_2025` problems 19 and 20 — so evaluating
those datasets in one run scores them twice. Upstream keeps the HF repo year-less
(`MathArena/apex-shortlist`); the `_2025` here follows the competition config
name (`apex/shortlist_2025.yaml`) and the dataset card's `pretty_name`.

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
APEX_SHORTLIST_2025_REVISION = "f3efdf224ef665f129ddaae37699f6098c65781b"


class ApexShortlist2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="apex_shortlist_2025",
    display_name="MathArena Apex Shortlist 2025",
    description=(
        "MathArena Apex Shortlist 2025 — 47 problems where frontier models "
        "score around 50%."
    ),
    source=f"hf:MathArena/apex-shortlist@{APEX_SHORTLIST_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class ApexShortlist2025Dataset(Dataset[ApexShortlist2025DatasetSample]):
    def _strip_sample(
        self, sample: ApexShortlist2025DatasetSample
    ) -> ApexShortlist2025DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX (\rightarrow -> " arrow", 4 of 47 problems). DEVIATION: matharena
        # does not normalize golds; sieval does, so math-verify compares canonical
        # forms. Score-neutral here: 2 of 47 golds change (3/2 -> \frac{3}{2},
        # 317/3528), both already equivalent under math-verify.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / answer / problem / source, kept under their upstream
        # names.
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
