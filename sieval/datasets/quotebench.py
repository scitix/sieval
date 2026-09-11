"""QuoteBench frozen core: 56 one-shot Bash tasks over 14 operation families.

The rows are code-generated, not downloaded -- upstream constructs each task in
Python. `scripts/gen_quotebench_snapshot.py` renders the model-facing half into
`<data_dir>/quotebench/quotebench-core.json`, which the `local:` scheme treats
as bring-your-own; the oracle, the naive probe and the fixture builders stay on
the evaluator's side.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
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
from sieval.core.utils.hf import ensure_dataset_dict

#: Upstream pin the vendored task definitions were taken from.
QUOTEBENCH_COMMIT = "693325a671e65f889e5cd9d83965db9cc3b26dc2"

_BASENAME = "quotebench-core.json"


class QuoteBenchDatasetSample(TypedDict):
    task_id: str
    scenario: str
    tier: int
    hazards: list[str]
    instruction: str


@sieval_dataset(
    name="quotebench",
    display_name="QuoteBench",
    description=(
        "56 one-shot Bash tasks over 14 families, graded by exact final state."
    ),
    # Code-generated, so there is no remote origin to fetch. Staged flat as
    # <data_dir>/quotebench/<basename>, the local: convention.
    source=f"local:{_BASENAME}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "shell", "code-exec"),
    license="Apache-2.0",
)
class QuoteBenchDataset(Dataset[QuoteBenchDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        **kwargs,
    ) -> HFDatasetDict:
        # `name_or_path` is the base data dir; the staging subdir is appended
        # here so every entry point resolves the same path (cf. ruler).
        path = f"{name_or_path}/quotebench/{_BASENAME}"
        # No dtype cast: the snapshot is written by our own generator, so
        # tier is already int64 and hazards already list<string>.
        dataset = load_dataset("json", data_files={"test": path}, **kwargs)
        loaded = ensure_dataset_dict(dataset)
        if len(loaded["test"]) == 0:
            raise ValueError(
                f"QuoteBench snapshot at {path} is empty; regenerate it with "
                f"scripts/gen_quotebench_snapshot.py"
            )
        return loaded
