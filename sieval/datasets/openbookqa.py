"""OpenBookQA dataset loader.

Source note: OpenCompass's ``obqa_gen_9069e4`` (the reference this benchmark's
prompt + extractor are vendored from) runs against its own mirror
``opencompass/openbookqa_test``; this loader points at the original
``allenai/openbookqa`` (``main``) instead. The two resolve to the same
500-example ``main`` test split — both position-map ``choices.text[0..3]`` → A–D
and every record's ``choices.label`` is ordered ``[A, B, C, D]`` so ``answerKey``
aligns (the ``" what?"``-style stem rewrite lives in OpenCompass's
``OBQADatasetV2``, not the config targeted here).

AI-Generated Code - Opus 4.8 (Anthropic)
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
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

OPENBOOKQA_REVISION = "388097ea7776314e93a529163e0fea805b8a6454"


class OpenBookQADatasetSample(TypedDict):
    id: str
    question_stem: str
    choices: dict[str, list[str]]
    answerKey: str


@sieval_dataset(
    name="openbookqa",
    display_name="OpenBookQA",
    description="OpenBookQA elementary-science open-book multiple-choice QA.",
    source=f"hf:allenai/openbookqa@{OPENBOOKQA_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "STEM"),),
    tags=("english", "science", "multiple-choice"),
    license="Apache-2.0",
)
class OpenBookQADataset(Dataset[OpenBookQADatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        name: str = "main",
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        # `name` is HF's subset selector (load_dataset's 2nd positional arg);
        # pass it as a keyword so a config `args: {name: main}` can't collide
        # with the positional and raise "multiple values for argument 'name'".
        dataset = load_dataset(name_or_path, name=name, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        return apply_eval_split(dataset, eval_split)
