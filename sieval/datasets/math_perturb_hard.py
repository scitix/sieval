"""
MATH-P-Hard — 279 MATH level-5 problems under fundamental perturbations.

The treatment arm of MATH-Perturb (arXiv:2502.06453, ICML 2025): each problem is
edited so that the original solution method *no longer applies*, so a model
carrying the seed problem's reasoning pattern forward scores wrong. Every model
the paper benchmarks drops 10–25 points here relative to the unmodified
problems. Row-aligned with :mod:`sieval.datasets.math_perturb_simple` by
``problem_id``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._math_perturb import data_file, data_url, load_math_perturb


class MATHPerturbHardDatasetSample(TypedDict):
    #: Shared with the MATH-P-Simple row perturbed from the same seed problem.
    problem_id: int
    problem: str
    #: Always a string here; upstream's column is mixed int/float/str and is cast
    #: at load. See `sieval.datasets._math_perturb`.
    answer: str
    #: ``"Level 5"`` on every row — the seed set is MATH's hardest level only.
    level: str
    #: MATH subject, e.g. ``"Algebra"``, ``"Precalculus"``.
    type: str
    #: Which MATH split the SEED problem came from (``"train"`` / ``"test"``).
    #: Provenance, not a split of these rows — the paper reads the two separately
    #: because a train-seeded problem may have been memorized.
    original_split: str


@sieval_dataset(
    name="math_perturb_hard",
    display_name="MATH-P-Hard",
    description=(
        "279 MATH level-5 problems perturbed so the original method no longer works."
    ),
    source=f"url:{data_url('hard')}",
    checksums={
        data_file(
            "hard"
        ): "sha256:418ca51e02f3b639a919dda72b5201ce35ac6c6d7d31c3231735beb4ea012dd7",  # noqa: E501
    },
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended", "robustness"),
    license="Apache-2.0",
)
class MATHPerturbHardDataset(Dataset[MATHPerturbHardDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        return load_math_perturb(name_or_path, "hard")
