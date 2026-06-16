"""
AI-Generated Code - GPT-5.5 (OpenAI)
"""

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

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


class GSM8KDatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="gsm8k",
    display_name="GSM8K",
    description="Grade School Math 8K - grade-school arithmetic word problems.",
    source=f"hf:openai/gsm8k@{GSM8K_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "math-word-problems", "open-ended"),
    license="MIT",
)
class GSM8KDataset(Dataset[GSM8KDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = "main",
        **kwargs,
    ) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, config, **kwargs)
        return ensure_dataset_dict(dataset)
