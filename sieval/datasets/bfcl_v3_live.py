"""BFCL v3 live categories -- the user-contributed half of the benchmark.

Six categories, 2251 rows: four AST categories built from real user queries,
plus irrelevance and relevance detection sets with no gold.

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

from sieval.community.bfcl_v3 import LIVE_COUNTS
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._bfcl_v3 import BFCL_V3_REVISION, load_categories


class BfclV3LiveDatasetSample(TypedDict):
    """One BFCL v3 live row.

    `ground_truth` is None for `live_irrelevance` and `live_relevance`, which
    have no gold: the correct outcome there is about whether a call is
    produced, not about which call it is.
    """

    id: str
    category: str
    language: str
    question: list[dict]
    function: str
    ground_truth: str | None


@sieval_dataset(
    name="bfcl_v3_live",
    display_name="BFCL v3 (live)",
    description=(
        "Berkeley Function Calling Leaderboard v3 live: "
        "user-contributed AST, irrelevance, relevance."
    ),
    source=f"hf:gorilla-llm/Berkeley-Function-Calling-Leaderboard@{BFCL_V3_REVISION}",
    categories=(Category(Level1Category.AGENT, "ToolUseSimple"),),
    tags=("english", "function-calling"),
    license="Apache-2.0",
)
class BfclV3LiveDataset(Dataset[BfclV3LiveDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        _ = kwargs
        return load_categories(name_or_path, LIVE_COUNTS)
