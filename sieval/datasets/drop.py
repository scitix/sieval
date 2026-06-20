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


class DROPDatasetSample(TypedDict):
    context: str
    completion: str
    ref_text: str


@sieval_dataset(
    name="drop",
    display_name="DROP",
    description="Discrete Reasoning Over Paragraphs — reading-comprehension benchmark.",
    source=(
        "url:https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_train.jsonl.gz",
        "url:https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_dev.jsonl.gz",
    ),
    checksums={
        "drop_v0_train.jsonl.gz": "sha256:d4a3a00ea2cfbe69d11f0bd24f5ba069731c645489bd39248b480d8eb34e1fb6",  # noqa: E501
        "drop_v0_dev.jsonl.gz": "sha256:7a58d35552dc476699d87ee8af0254f63e9f00f5a7f6b8b033e11c933157e186",  # noqa: E501
    },
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended"),
    license="CC-BY-SA-4.0",
)
class DROPDataset(Dataset[DROPDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        train_path = os.path.join(name_or_path, "drop_v0_train.jsonl.gz")
        dev_path = os.path.join(name_or_path, "drop_v0_dev.jsonl.gz")
        dataset = load_dataset(
            "json",
            data_files={"train": train_path, "test": dev_path},
            **kwargs,
        )
        dataset = ensure_dataset_dict(dataset)
        return dataset.map(self._process_sample)

    def _process_sample(self, sample: dict) -> DROPDatasetSample:
        context = str(sample.get("context", ""))
        completion = str(sample.get("completion", ""))
        ref_text = str(sample.get("ref_text", ""))
        return {
            "context": context,
            "completion": completion,
            "ref_text": ref_text,
        }
