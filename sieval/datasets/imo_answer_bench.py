"""IMO-AnswerBench dataset loader (Google DeepMind IMO-Bench suite).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Value, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

# Pin the HF snapshot for reproducibility (see check_datasets / #8).
IMO_ANSWER_BENCH_REVISION = "0258becbd00fc07d34862bc8539e61c8742f0d14"


class IMOAnswerBenchDatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="imo_answer_bench",
    display_name="IMO-AnswerBench",
    description=(
        "IMO-Bench AnswerBench (Google DeepMind) — 400 short-answer olympiad problems."
    ),
    source=f"hf:Hwilner/imo-answerbench@{IMO_ANSWER_BENCH_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-4.0",
)
class IMOAnswerBenchDataset(Dataset[IMOAnswerBenchDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # Columns: "Problem ID" / "Problem" / "Short Answer" / "Category" /
        # "Subcategory" / "Source". Map Problem -> question, Short Answer -> answer
        # to match the shared math sample schema; other columns are kept as-is.
        dataset = ensure_dataset(load_dataset(name_or_path, split="train", **kwargs))
        dataset = dataset.rename_column("Problem", "question")
        dataset = dataset.rename_column("Short Answer", "answer")
        # Golds are short answers (integers, LaTeX expressions, small answer sets);
        # kept verbatim — IMO-Bench grades via math-verify, not string normalization.
        dataset = dataset.cast_column("answer", Value("string"))
        # the test split is the same as the train split
        return HFDatasetDict(
            {
                "train": dataset,
                "test": dataset,
            }
        )
