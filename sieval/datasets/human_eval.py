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

HUMAN_EVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"


class HumanEvalDatasetSample(TypedDict):
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@sieval_dataset(
    name="human_eval",
    display_name="HumanEval",
    description="OpenAI HumanEval — 164 Python function-synthesis problems.",
    source=f"hf:openai/openai_humaneval@{HUMAN_EVAL_REVISION}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "python", "code-exec"),
    license="MIT",
)
class HumanEvalDataset(Dataset[HumanEvalDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return dataset
