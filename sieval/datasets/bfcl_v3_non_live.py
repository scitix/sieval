"""BFCL v3 non-live categories -- the hand-written half of the benchmark.

Seven categories, 1390 rows: four Python AST categories, Java and JavaScript
simple-call categories, and an irrelevance-detection set with no gold.

`function` and `ground_truth` are json strings because pyarrow cannot type
their heterogeneous nesting -- the same reason `t_eval.py` stores its
`ground_truth` that way.

References:

* Blog: <https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html>
* Harness: <https://github.com/ShishirPatil/gorilla/tree/v1.3/berkeley-function-call-leaderboard>
* Dataset: <https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard>

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict

from sieval.community.bfcl_v3 import NON_LIVE_COUNTS
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._bfcl_v3 import BFCL_V3_REVISION, load_categories


class BfclV3NonLiveDatasetSample(TypedDict):
    """One BFCL v3 non-live row.

    `ground_truth` is None for `irrelevance`, which has no gold: the correct
    outcome there is that no call is produced at all.
    """

    id: str
    category: str
    language: str
    question: list[dict]
    function: str
    ground_truth: str | None


@sieval_dataset(
    name="bfcl_v3_non_live",
    display_name="BFCL v3 (non-live)",
    description=(
        "Berkeley Function Calling Leaderboard v3 non-live: "
        "Python/Java/JS AST plus irrelevance."
    ),
    source=f"hf:gorilla-llm/Berkeley-Function-Calling-Leaderboard@{BFCL_V3_REVISION}",
    categories=(Category(Level1Category.AGENT, "ToolUseSimple"),),
    tags=("english", "function-calling"),
    license="Apache-2.0",
)
class BfclV3NonLiveDataset(Dataset[BfclV3NonLiveDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        return load_categories(name_or_path, NON_LIVE_COUNTS)
