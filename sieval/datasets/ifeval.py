from typing import Any, TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"


class IFEvalDatasetSample(TypedDict):
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@sieval_dataset(
    name="ifeval",
    display_name="IFEval",
    description="Instruction-Following Eval — 541 prompts with verifiable constraints.",
    source=f"hf:google/IFEval@{IFEVAL_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "open-ended"),
    license="Apache-2.0",
)
class IFEvalDataset(Dataset[IFEvalDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, split="train", **kwargs)
        dataset = ensure_dataset(dataset)
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
