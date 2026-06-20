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
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict


class MMLUDatasetSample(TypedDict):
    Question: str
    A: str
    B: str
    C: str
    D: str
    Answer: str
    Subject: str


def _normalize_answer(answer, choices: list[str]) -> str:
    if answer is None:
        return ""
    if isinstance(answer, int):
        if 0 <= answer < len(choices):
            return chr(65 + answer)
        return ""
    text = str(answer).strip()
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    if text in choices:
        return chr(65 + choices.index(text))
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(choices):
            return chr(65 + idx)
    return ""


def _process_sample(sample: dict) -> MMLUDatasetSample:
    question = sample.get("question", sample.get("Question", ""))
    subject = sample.get("subject", sample.get("Subject", ""))
    choices = sample.get("choices")
    if choices is None:
        choices = [
            sample.get("A", ""),
            sample.get("B", ""),
            sample.get("C", ""),
            sample.get("D", ""),
        ]
    if not isinstance(choices, list):
        choices = list(choices)
    choices = [str(c) for c in choices]
    while len(choices) < 4:
        choices.append("")
    answer = sample.get("answer", sample.get("Answer"))
    return {
        "Question": str(question),
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
        "Answer": _normalize_answer(answer, choices),
        "Subject": str(subject),
    }


@sieval_dataset(
    name="mmlu",
    display_name="MMLU",
    description="Massive Multitask Language Understanding — 57 academic subjects, MCQ.",
    source="url:https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    checksums={
        "mmlu.csv": "sha256:15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb",  # noqa: E501
    },
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "multiple-choice"),
    license="MIT",
)
class MMLUDataset(Dataset[MMLUDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = "all",
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        if name_or_path.endswith(".csv") or os.path.isdir(name_or_path):
            csv_path = name_or_path
            if os.path.isdir(name_or_path):
                csv_path = os.path.join(name_or_path, "mmlu.csv")
            dataset = load_dataset("csv", data_files={"test": csv_path}, **kwargs)
        else:
            dataset = load_dataset(name_or_path, config, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        dataset = apply_eval_split(dataset, eval_split)
        return dataset.map(_process_sample)
