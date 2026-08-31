"""
MATH-P-Simple — 279 MATH level-5 problems under non-essential perturbations.

The control arm of MATH-Perturb (arXiv:2502.06453, ICML 2025): each problem is
edited so that the *same* solution method still applies, which is what makes a
drop against MATH-P-Hard attributable to the perturbation being fundamental
rather than merely being an edit. Row-aligned with
:mod:`sieval.datasets.math_perturb_hard` by ``problem_id``.

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


class MATHPerturbSimpleDatasetSample(TypedDict):
    #: Shared with the MATH-P-Hard row perturbed from the same seed problem.
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
    name="math_perturb_simple",
    display_name="MATH-P-Simple",
    description=(
        "279 MATH level-5 problems perturbed so the original method still solves them."
    ),
    source=f"url:{data_url('simple')}",
    checksums={
        data_file(
            "simple"
        ): "sha256:2130821d4235dc1f5d962bee7f84a096651bf98fcb17b6dcfd3d0a48875ba5bb",  # noqa: E501
    },
    categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    tags=("english", "open-ended", "robustness"),
    license="Apache-2.0",
)
class MATHPerturbSimpleDataset(Dataset[MATHPerturbSimpleDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        return load_math_perturb(name_or_path, "simple")
