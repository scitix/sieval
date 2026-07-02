"""
MBPP dataset loader (Mostly Basic Python Problems).

Loads ``google-research-datasets/mbpp`` config ``full`` — the same repo and
config lm-evaluation-harness uses. The repo natively ships the four official
splits prompt (10), test (500), validation (90), and train (374), so no
task_id-range split rebuild is needed here.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
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


class MBPPDatasetSample(TypedDict):
    task_id: int
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]


@sieval_dataset(
    name="mbpp",
    display_name="MBPP",
    description="Mostly Basic Python Problems: 974 entry-level Python tasks.",
    source="hf:google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "python", "code-exec"),
    license="CC-BY-4.0",
)
class MBPPDataset(Dataset[MBPPDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = "full",
        **kwargs,
    ) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, config, **kwargs)
        return ensure_dataset_dict(dataset)
