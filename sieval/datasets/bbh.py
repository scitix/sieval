"""
BIG-Bench Hard (BBH) dataset loader — the ``lukaemon/bbh`` Hub mirror.

``load`` merges the 27 per-subset configs into one ``test`` split (6,511 rows,
row-identical to suzgunmirac/BIG-Bench-Hard ``bbh/*.json``) and stamps each row
with its ``subset``; narrow it with a ``filter``. The CoT prompts are not in
the mirror and belong to the task.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

BBH_REVISION = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"

# Config names at the pin, merged in a fixed order so sample ids are stable.
BBH_SUBSETS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)


class BBHDatasetSample(TypedDict):
    # No cast needed: upstream ships `target` as a string, numeric subsets too.
    input: str
    target: str
    subset: str


@sieval_dataset(
    name="bbh",
    display_name="BIG-Bench Hard",
    description="BIG-Bench Hard — 27 challenging BIG-Bench subsets, 6,511 problems.",
    source=f"hf:lukaemon/bbh@{BBH_REVISION}",
    categories=(Category(Level1Category.LOGIC, "ComplexLogic"),),
    tags=("english", "reasoning", "open-ended"),
    # The mirror's card declares none; this is suzgunmirac/BIG-Bench-Hard's.
    license="MIT",
)
class BBHDataset(Dataset[BBHDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        parts = []
        for subset in BBH_SUBSETS:
            rows = ensure_dataset(
                load_dataset(name_or_path, subset, split="test", **kwargs)
            )
            if len(rows) == 0:
                raise ValueError(
                    f"BBH subset {subset!r} is empty; expected revision {BBH_REVISION}."
                )
            parts.append(rows.map(lambda _row, s=subset: {"subset": s}))
        return HFDatasetDict({"test": concatenate_datasets(parts)})
