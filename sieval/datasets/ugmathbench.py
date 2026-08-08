"""
UGMathBench dataset loader — 5,061 undergraduate problems x 3 randomized versions.

The HF repo ships one config per subject, and each row packs all three
randomized versions of a problem side by side (``problem_v1`` / ``answer_v1``
/ ... / ``options_v3``). This loader concatenates the 16 subjects and unpacks
each row into three samples carrying a ``version`` field, the same shape the
upstream generation script materializes before inference.

Ordering is **problem-major**: a problem's three versions are adjacent, so
``slice(n)`` keeps whole problems and the effective-accuracy metric stays
defined on a truncated run. (Upstream emits version-major, which only matters
for file layout — the score is order-independent.)

Rows are mirrored as-is, including two upstream-corrupt versions whose problem
text is an error message and whose answer sequence is empty
(``Financial_mathematics_0132`` v2, ``Linear_algebra_0306`` v3). Dropping them
would quietly inflate effective accuracy for those two problems, which can
never satisfy "correct in all three versions" upstream either.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.community.ugmathbench import SUBJECTS, VERSIONS
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset, ensure_dataset_dict

UGMATHBENCH_REVISION = "8ab16f0c131a9b3b52195e64175cb5a8f3881bbf"


class UGMathBenchDatasetSample(TypedDict):
    id: str
    subject: str
    topic: str
    subtopic: str
    level: str
    keywords: list[str]
    version: int
    problem: str
    answer: list[str]
    answer_type: list[str]
    options: list[list[str]]


@sieval_dataset(
    name="ugmathbench",
    display_name="UGMathBench",
    description="Undergraduate math, 16 subjects, 3 randomized versions per problem.",
    source=f"hf:UGMathBench/ugmathbench@{UGMATHBENCH_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "AdvancedMath"),),
    tags=("english", "open-ended"),
    license="GPL-3.0",
)
class UGMathBenchDataset(Dataset[UGMathBenchDatasetSample]):
    SUBJECTS = SUBJECTS

    @override
    def load(
        self,
        name_or_path: str,
        subjects: list[str] | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        # `None` means "unspecified, load everything"; `[]` is a caller asking
        # for nothing, which silently loading all 16 subjects would misread.
        selected = self.SUBJECTS if subjects is None else tuple(subjects)
        if not selected:
            raise ValueError(
                "UGMathBench `subjects` is empty; omit it to load all "
                f"{len(self.SUBJECTS)} subjects."
            )
        unknown = [subject for subject in selected if subject not in self.SUBJECTS]
        if unknown:
            raise ValueError(
                f"Unknown UGMathBench subject(s) {unknown}; "
                f"expected a subset of {list(self.SUBJECTS)}."
            )

        rows: list[UGMathBenchDatasetSample] = []
        for subject in selected:
            split = ensure_dataset(
                load_dataset(name_or_path, subject, split="test", **kwargs)
            )
            for row in split:
                rows.extend(_unpack_versions(row))
        return ensure_dataset_dict(
            HFDatasetDict({"test": HFDataset.from_list([dict(row) for row in rows])})
        )


def _unpack_versions(row: dict) -> list[UGMathBenchDatasetSample]:
    """Split one packed row into one sample per randomized version."""
    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "topic": row["topic"],
            "subtopic": row["subtopic"],
            "level": row["level"],
            "keywords": list(row["keywords"]),
            "version": version,
            "problem": row[f"problem_v{version}"],
            "answer": list(row[f"answer_v{version}"]),
            "answer_type": list(row[f"answer_type_v{version}"]),
            "options": [list(entry) for entry in row[f"options_v{version}"]],
        }
        for version in range(1, VERSIONS + 1)
    ]
