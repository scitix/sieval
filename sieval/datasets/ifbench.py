"""IFBench dataset loader.

AI-Generated Code - GPT-5 (OpenAI)
"""

from pathlib import Path
from typing import Any, TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

IFBENCH_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"


class IFBenchDatasetSample(TypedDict):
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@sieval_dataset(
    name="ifbench",
    display_name="IFBench",
    description=(
        "Precise instruction-following benchmark with verifiable OOD constraints."
    ),
    source=f"hf:allenai/IFBench_test@{IFBENCH_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "open-ended"),
    license="ODC-BY-1.0",
)
class IFBenchDataset(Dataset[IFBenchDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        path = Path(name_or_path).expanduser()

        if path.is_file():
            return self._load_jsonl(path, **kwargs)

        if path.is_dir():
            jsonl_path = path / "IFBench_test.jsonl"
            if jsonl_path.is_file():
                return self._load_jsonl(jsonl_path, **kwargs)

        load_source = str(path) if path.exists() else name_or_path
        dataset = ensure_dataset_dict(load_dataset(load_source, **kwargs))
        return apply_eval_split(dataset, "train")

    def _load_jsonl(self, path: Path, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(
            "json",
            data_files={"train": str(path), "test": str(path)},
            **kwargs,
        )
        return ensure_dataset_dict(dataset)
