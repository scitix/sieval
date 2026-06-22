"""
MBPP dataset loader (Mostly Basic Python Problems).

Loads the upstream ``mbpp.jsonl`` and rebuilds the four official splits by
``task_id`` range: prompt (1-10), test (11-510), validation (511-600), and
train (601-974).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from pathlib import Path
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

MBPP_JSONL_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "2529b9bcfb930399929b047731804d40dc9a9e2a/mbpp/mbpp.jsonl"
)
_MBPP_FILENAME = "mbpp.jsonl"


class MBPPDatasetSample(TypedDict):
    task_id: int
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]


def _process_sample(sample: dict) -> MBPPDatasetSample:
    return {
        "task_id": sample["task_id"],
        "text": sample.get("text") or sample.get("prompt", ""),
        "code": sample["code"],
        "test_list": sample["test_list"],
        "test_setup_code": sample.get("test_setup_code") or "",
        "challenge_test_list": sample.get("challenge_test_list") or [],
    }


def _resolve_data_file(name_or_path: str) -> str:
    if name_or_path.startswith(("http://", "https://")):
        return name_or_path

    path = Path(name_or_path)
    if path.is_dir():
        path = path / _MBPP_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"MBPP data file not found: {path}\n"
            "Tip: run `sieval dataset download mbpp` to fetch the dataset."
        )
    return str(path)


@sieval_dataset(
    name="mbpp",
    display_name="MBPP",
    description="Mostly Basic Python Problems: 974 entry-level Python tasks.",
    source=f"url:{MBPP_JSONL_URL}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "python", "code-exec"),
    license="CC-BY-4.0",
)
class MBPPDataset(Dataset[MBPPDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        data_file = _resolve_data_file(name_or_path)
        dataset = load_dataset(
            "json",
            config,
            data_files={"full": data_file},
            **kwargs,
        )
        dataset = ensure_dataset_dict(dataset)
        full = dataset["full"].map(_process_sample)

        splits = HFDatasetDict(
            {
                "prompt": full.filter(lambda sample: 1 <= sample["task_id"] <= 10),
                "test": full.filter(lambda sample: 11 <= sample["task_id"] <= 510),
                "validation": full.filter(
                    lambda sample: 511 <= sample["task_id"] <= 600
                ),
                "train": full.filter(lambda sample: 601 <= sample["task_id"] <= 974),
            }
        )

        empty = [name for name, split in splits.items() if len(split) == 0]
        if empty:
            raise ValueError(
                f"MBPP produced empty split(s) {empty} from {len(full)} rows. "
                "Expected task_id ranges prompt(1-10)/test(11-510)/"
                "validation(511-600)/train(601-974); the data file may be "
                "truncated or use unexpected task_id values."
            )
        return splits
