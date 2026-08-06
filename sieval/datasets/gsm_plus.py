"""
GSM-Plus — GSM8K's test set under 8 adversarial math perturbations.

Every one of GSM8K's 1319 test problems is rewritten eight ways (numerical
substitution, digit expansion, integer-decimal-fraction conversion, adding
operation, reversing operation, problem understanding, distraction insertion,
critical thinking), so a model's GSM8K score can be compared against its score
on the same problems perturbed. Each row keeps the ``seed_*`` fields of the
GSM8K problem it came from, which is what makes that pairing possible.

AI-Generated Code - Claude Opus 5 (Anthropic)
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

GSM_PLUS_REVISION = "3b708db57b96a16e8e3368ed2956990c0809440e"


class GSMPlusDatasetSample(TypedDict):
    question: str
    solution: str
    # The bare final answer. `"None"` (a string, not null) on `critical thinking`
    # rows, where the perturbation deletes a quantity the question needs and the
    # right answer is that it cannot be solved.
    answer: str
    perturbation_type: str
    # The GSM8K problem this row was perturbed from, for paired GSM8K-vs-GSM-Plus
    # comparison.
    seed_question: str
    seed_solution: str
    seed_answer: str


@sieval_dataset(
    name="gsm_plus",
    display_name="GSM-Plus",
    description=(
        "GSM-Plus - GSM8K test problems rewritten under 8 adversarial "
        "math perturbations."
    ),
    source=f"hf:qintongli/GSM-Plus@{GSM_PLUS_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "math-word-problems", "open-ended", "robustness"),
    license="CC-BY-SA-4.0",
)
class GSMPlusDataset(Dataset[GSMPlusDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        eval_split: str | None = "test",
        **kwargs,
    ) -> HFDatasetDict:
        """Load *eval_split* as the eval set.

        Upstream ships two: ``"test"`` (10552 rows = 1319 GSM8K test problems x 8
        perturbations, the paper's headline set) and ``"testmini"`` (2400 = 300 x
        8, for cheap iteration). Whichever is requested is remapped to ``"test"``,
        because that is the split ``Dataset.test_set`` hands to a run.

        Both splits are laid out as 8 consecutive rows per seed problem, one per
        perturbation type in a fixed order, so a ``slice`` of a multiple of 8
        stays perturbation-balanced. ``stratified_sample`` by ``perturbation_type``
        is the explicit route for any other size.

        All seven columns are strings at the pinned revision — no cast needed,
        upstream ships these dtypes.
        """
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        # `apply_eval_split` no-ops on an unknown name, which for this dataset
        # would silently fall through to the 10552-row `test` split — an
        # expensive way to learn that "testmini" was misspelled.
        if eval_split is not None and eval_split not in dataset:
            raise ValueError(
                f"GSM-Plus has no split {eval_split!r}; available: {sorted(dataset)}."
            )
        dataset = apply_eval_split(dataset, eval_split)
        if len(dataset["test"]) == 0:
            raise ValueError(
                f"GSM-Plus loaded 0 samples for eval split {eval_split!r} from "
                f"{name_or_path!r}; re-run 'sieval dataset download gsm_plus'."
            )
        return dataset
