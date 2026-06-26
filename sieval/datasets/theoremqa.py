"""
TheoremQA dataset wrapper.

AI-Generated Code - GPT-5 (OpenAI)
"""

from typing import NotRequired, TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Image as HFImage
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

THEOREMQA_HF_REVISION = "a340b1782960a712843aae3ed25f1e013cc008a5"


class TheoremQADatasetSample(TypedDict):
    Question: str
    Answer: str
    Answer_type: str
    Picture: NotRequired[object]


@sieval_dataset(
    name="theoremqa",
    display_name="TheoremQA",
    description="Theorem-driven STEM QA benchmark with 800 expert-written questions.",
    source=f"hf:TIGER-Lab/TheoremQA@{THEOREMQA_HF_REVISION}",
    categories=(
        Category(Level1Category.KNOWLEDGE, "STEM"),
        Category(Level1Category.MATHEMATICS, "AppliedMath"),
    ),
    tags=("english", "open-ended", "theorem-driven"),
    license="MIT",
)
class TheoremQADataset(Dataset[TheoremQADatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # Mirror the source as-is, but disable eager image decoding. ``Picture``
        # is an HF ``Image`` feature (``decode=True`` by default); materializing
        # the 53/800 picture rows would decode them into PIL objects and require
        # Pillow, which ``datasets`` only ships under its ``vision`` extra and is
        # outside ``sieval[math]``. This task is text-only and never reads
        # ``Picture``, so we keep the column as raw bytes without pulling Pillow
        # into the install closure. The decode default is made an explicit choice
        # rather than left implicit.
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        for split_name, split in dataset.items():
            picture = split.features.get("Picture")
            if isinstance(picture, HFImage) and picture.decode:
                dataset[split_name] = split.cast_column(
                    "Picture", HFImage(decode=False)
                )
        return dataset
