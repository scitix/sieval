"""IMO-AnswerBench dataset loader (Google DeepMind IMO-Bench suite).

Authoritative source: google-deepmind/superhuman (path ``imobench/``) +
imobench.github.io + arXiv 2511.01846. We pin the CURRENT ``answerbench_v2.csv``
(released 2026-02-12, which "fix[ed] some problems that had ambiguous problem
statements or incorrect answers"; the previous ``answerbench.csv`` is now
deprecated). The earlier HF mirror ``hf:Hwilner/imo-answerbench`` is the
deprecated v1 (byte-for-byte the old ``answerbench.csv``) and is NOT used.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Value, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the official v2 CSV to an immutable commit blob (a bare `main` URL is not
# immutable and would break the checksum). google-deepmind/superhuman @ this
# commit; answerbench_v2.csv sha256 verified below.
IMO_ANSWER_BENCH_COMMIT = "96fa6c4cc3a9bb7450ee7b6773b659d3a030dace"
IMO_ANSWER_BENCH_URL = (
    "url:https://raw.githubusercontent.com/google-deepmind/superhuman/"
    f"{IMO_ANSWER_BENCH_COMMIT}/imobench/answerbench_v2.csv"
)


class IMOAnswerBenchDatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="imo_answer_bench",
    display_name="IMO-AnswerBench",
    description=(
        "IMO-Bench AnswerBench (Google DeepMind) — 400 short-answer olympiad problems."
    ),
    source=IMO_ANSWER_BENCH_URL,
    checksums={
        "answerbench_v2.csv": "sha256:275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9",  # noqa: E501
    },
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended"),
    license="CC-BY-4.0",
)
class IMOAnswerBenchDataset(Dataset[IMOAnswerBenchDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # Columns: "Problem ID" / "Problem" / "Short Answer" / "Category" /
        # "Subcategory" / "Source" (unchanged v1 -> v2). Map Problem -> question,
        # Short Answer -> answer to match the shared math sample schema; other
        # columns are kept as-is. NOTE: v2 carries two known upstream spreadsheet
        # artifacts kept VERBATIM (faithful to the official CSV, checksummed):
        # imo-bench-algebra-036 (answer corrupted to the Category "Algebra") and
        # imo-bench-geometry-004 (answer Excel-autoformatted to the date serial
        # "45752"). They grade wrong for ~any model (score impact <=0.5%); see the
        # task's reference_impl.notes.
        csv_path = (
            os.path.join(name_or_path, "answerbench_v2.csv")
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        dataset = load_dataset(
            "csv", data_files={"train": csv_path, "test": csv_path}, **kwargs
        )
        dataset = ensure_dataset_dict(dataset)
        dataset = dataset.rename_column("Problem", "question")
        dataset = dataset.rename_column("Short Answer", "answer")
        # Golds are short answers (integers, LaTeX expressions, small answer sets);
        # kept verbatim — IMO-Bench grades via math-verify, not string normalization.
        dataset = dataset.cast_column("answer", Value("string"))
        # the test split is the same as the train split (mirrored above)
        return dataset
