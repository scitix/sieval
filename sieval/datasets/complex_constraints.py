"""ComplexConstraints dataset loader (Surge AI).

ComplexConstraints (Mehta et al., 2026, arXiv:2606.09118) is a 75-prompt
multi-constraint instruction-following benchmark (``CIF-001``-``CIF-075``). Each
row is one realistic prompt plus 10-40 atomic rubric criteria (1,559 in total)
describing what a correct response must satisfy.

The Hub repo ships a single wide CSV: five item columns and 40 sparse
``criterion_{i}`` columns, of which a row uses the first 10-40. This loader
collapses those 40 columns into one ``criteria`` list and drops them. That is a
**reshape, not a rename for uniformity**: a 40-key ``TypedDict`` of mostly-absent
columns is unusable as a sample type, and every consumer wants the list. The
other five columns keep their upstream names, and no dtype cast is applied --
the pinned revision already ships all 45 columns as strings.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

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
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (current `main` at integration time).
COMPLEX_CONSTRAINTS_REVISION = "e9625c6f635f42b72cb85a04c2be64746f945126"

#: The repo's one data file. The dataset card's ``configs.data_files.path`` spells
#: it ``ComplexConstraints_Benchmark_Set.csv``, which does not exist -- so
#: ``load_dataset("surgeai/ComplexConstraints")`` cannot resolve the file at all.
#: Reading the staged snapshot by its real name sidesteps the card's typo.
CSV_FILENAME = "ComplexConstraints_benchmark_set.csv"

#: Widest criterion column upstream ships; a row fills the first 10-40.
MAX_CRITERIA = 40

_CRITERION_COLUMNS = tuple(f"criterion_{i}" for i in range(1, MAX_CRITERIA + 1))


class ComplexConstraintsDatasetSample(TypedDict):
    benchmark_id: str
    prompt: str
    use_case: str
    instruction_type: str
    prompt_style: str
    criteria: list[str]


def _collapse_criteria(row: dict) -> dict:
    """Gather a row's non-empty ``criterion_{i}`` cells into one ordered list.

    Every non-empty cell is kept, rather than stopping at the first empty one.
    On the pinned revision the filled cells are a contiguous prefix (verified:
    0 of 75 rows have a gap), so the two readings agree there -- but stopping
    early would silently drop criteria if a later revision ever left a hole,
    and a dropped criterion inflates the score.
    """
    criteria = [
        text
        for column in _CRITERION_COLUMNS
        if (value := row[column]) is not None and (text := str(value).strip())
    ]
    return {"criteria": criteria}


@sieval_dataset(
    name="complex_constraints",
    display_name="ComplexConstraints",
    description="75 multi-constraint prompts with 1,559 rubric criteria (Surge AI).",
    source=f"hf:surgeai/ComplexConstraints@{COMPLEX_CONSTRAINTS_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "instruction-following", "open-ended"),
    license="CC-BY-4.0",
)
class ComplexConstraintsDataset(Dataset[ComplexConstraintsDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        csv_path = (
            os.path.join(name_or_path, CSV_FILENAME)
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        dataset = load_dataset("csv", data_files={"test": csv_path}, **kwargs)
        dataset = ensure_dataset_dict(dataset)
        split = dataset["test"]
        if len(split) == 0:
            raise ValueError(
                f"ComplexConstraints produced an empty 'test' split from "
                f"{csv_path!r}; check that the dataset has been downloaded via "
                "`sieval dataset download complex_constraints`."
            )

        missing = [c for c in _CRITERION_COLUMNS if c not in split.column_names]
        if missing:
            raise ValueError(
                f"ComplexConstraints is missing criterion column(s) {missing} in "
                f"{csv_path!r}; the loader expects the wide format of revision "
                f"{COMPLEX_CONSTRAINTS_REVISION} (criterion_1..criterion_"
                f"{MAX_CRITERIA})."
            )

        # The width is checked from BOTH sides. A revision that widened the
        # rubric past criterion_40 would not fail anywhere downstream: the extra
        # columns are absent from `_CRITERION_COLUMNS`, so they are neither
        # collected into `criteria` nor removed, and the sample would carry the
        # unknown column while its rubric quietly lost criteria -- fewer criteria
        # to satisfy is an INFLATED task pass rate, scored against a rubric that
        # no longer matches the data it was pinned to.
        extra = sorted(
            c
            for c in split.column_names
            if c.startswith("criterion_") and c not in _CRITERION_COLUMNS
        )
        if extra:
            raise ValueError(
                f"ComplexConstraints has unexpected criterion column(s) {extra} "
                f"in {csv_path!r}; the loader expects the wide format of revision "
                f"{COMPLEX_CONSTRAINTS_REVISION} (criterion_1..criterion_"
                f"{MAX_CRITERIA}). Widen MAX_CRITERIA and re-pin the revision "
                "rather than dropping them -- criteria dropped from the rubric "
                "inflate the score."
            )

        return HFDatasetDict(
            {
                "test": split.map(
                    _collapse_criteria, remove_columns=list(_CRITERION_COLUMNS)
                )
            }
        )
