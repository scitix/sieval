"""MathArena Apex 2025 dataset loader (MathArena source).

Apex is not a competition: MathArena curates it from 2025 contest problems that
frontier models still fail — of ~100 competitions reviewed, only these 12 went
unsolved by Grok 4, GPT-5 (High), Gemini 2.5 Pro and GLM 4.5 across 4 attempts
each. So its `source` column carries each problem's originating contest, and the
set is deliberately biased against those four models. Three of the twelve are
byte-identical to `smt_2025` problems 8, 42 and 43 (upstream's own "the dataset
now contains the remaining samples from SMT 2025" note) — evaluating both
datasets in one run scores those problems twice. Note `apex_shortlist_2025` is a
sibling tier, not a superset: it shares no problem with this set.

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
APEX_2025_REVISION = "ac8a641db12cc87be39e61ea89f2e04c80c5f2e7"


class Apex2025DatasetSample(TypedDict):
    problem: str
    answer: str


@sieval_dataset(
    name="apex_2025",
    display_name="MathArena Apex 2025",
    description=(
        "MathArena Apex 2025 — 12 problems curated from 2025 competitions to be "
        "very hard for models."
    ),
    source=f"hf:MathArena/apex_2025@{APEX_2025_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-NC-SA-4.0",
)
class Apex2025Dataset(Dataset[Apex2025DatasetSample]):
    def _strip_sample(self, sample: Apex2025DatasetSample) -> Apex2025DatasetSample:
        # Gold answer only; problem text stays verbatim — strip_string normalizes
        # answers (rewrites \frac/\sqrt, drops \left/\right) and mangles problem
        # LaTeX elsewhere in the family (none of these 12 problems, as it happens).
        # DEVIATION: matharena does not normalize golds; sieval does, so math-verify
        # compares canonical forms. Score-neutral here: 3 of 12 golds change (1/2 ->
        # \frac{1}{2}, 248/517, 14/5), all already equivalent under math-verify.
        sample["answer"] = strip_string(sample["answer"])
        return sample

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # MathArena exposes a single `default` config under the `train` split with
        # columns problem_idx / answer / source / problem, kept under their upstream
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
