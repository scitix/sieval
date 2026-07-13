"""SimpleQA Verified dataset loader (Google DeepMind).

SimpleQA Verified (Haas et al., 2025, arXiv:2509.07968) is a 1,000-prompt
short-form factuality benchmark: a de-duplicated, topic-balanced, label-checked
subset of OpenAI's SimpleQA (Wei et al., 2024). The Hub repo ships a single
``simpleqa_verified.csv``; this loader mirrors it as-is into a ``test`` split.

Numeric golds embed their tolerance directly in the ``answer`` text, e.g.
``"120k (acceptable range: anything between 118k and 122k)"`` — the autorater
consumes that verbatim, so the loader keeps ``answer`` untouched.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (current `main` at integration time).
SIMPLEQA_VERIFIED_REVISION = "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"


class SimpleQAVerifiedDatasetSample(TypedDict):
    original_index: int
    problem: str
    answer: str
    topic: str
    answer_type: str
    multi_step: bool
    requires_reasoning: bool
    urls: str


@sieval_dataset(
    name="simpleqa_verified",
    display_name="SimpleQA Verified",
    description="SimpleQA Verified — 1,000-prompt short-form factuality benchmark.",
    source=f"hf:google/simpleqa-verified@{SIMPLEQA_VERIFIED_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "factuality", "open-ended"),
    license="MIT",
)
class SimpleQAVerifiedDataset(Dataset[SimpleQAVerifiedDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        csv_path = (
            os.path.join(name_or_path, "simpleqa_verified.csv")
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        dataset = load_dataset("csv", data_files={"test": csv_path}, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        if len(dataset["test"]) == 0:
            raise ValueError(
                f"SimpleQA Verified produced an empty 'test' split from "
                f"{csv_path!r}; check that the dataset has been downloaded via "
                "`sieval dataset download simpleqa_verified`."
            )
        return dataset
