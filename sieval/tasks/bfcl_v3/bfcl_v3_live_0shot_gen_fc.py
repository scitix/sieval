"""BFCL v3 live categories, 0-shot, upstream's function-calling protocol.

Six user-contributed categories, 2251 rows: four Python AST categories plus
irrelevance and relevance detection. The headline is upstream's `Live Overall
Acc` -- the SAMPLE-COUNT-WEIGHTED mean of the six, which is algebraically the
pooled rate over all 2251 rows. That is why this group publishes an interval
where the non-live group cannot: a weighted mean of category rates over their
own row counts is the rate over their union, so a problem-clustered interval
brackets the number printed beside it. `ast_summary`, weighted over the four AST
categories, carries one on `n_ast` for the same reason.

This is the `(FC)` column: the same schemas reach the model through the
provider's tools API, and the calls are read back off structured tool calls
rather than parsed out of the reply. The sibling without the suffix is the
`(Prompt)` column, and it holds the unqualified name because every model on the
leaderboard has a Prompt row, while an FC row exists only where the provider
exposes a tools API -- and because this task gates on the `function_tools`
capability, which would make the default name the narrow one.

One scoring divergence follows from the transport and it is upstream's. A native
tool name cannot carry a dot in the OpenAI dialects, so `convert_to_tool`
rewrites `geometry.triangle_area` to `geometry_triangle_area` on the way out;
gold still spells the dot, so the comparison rewrites gold to match. That is
`underscore_to_dot=True`, and it is the only behavioural difference from the
Prompt sibling beyond how the schemas and the calls travel.

References:

* Blog: <https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html>
* Harness: <https://github.com/ShishirPatil/gorilla/tree/ea13468e4423454d0c213704fb87cf7cb3990433/berkeley-function-call-leaderboard>

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import BfclV3LiveDatasetSample

from ._base import (
    BFCL_V3_HARNESS_URL,
    BFCL_V3_SHARED_NOTES,
    BfclV3FCMixin,
    BfclV3LiveTask,
)


@sieval_task(
    name="bfcl_v3_live_0shot_gen_fc",
    display_name="BFCL v3 Live (0-shot, FC)",
    description=(
        "Berkeley Function Calling Leaderboard v3, live categories, "
        "upstream's FC (tools API) protocol."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "function-calling"),
    model_type="chat",
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="gorilla/berkeley-function-call-leaderboard",
        url=BFCL_V3_HARNESS_URL,
        notes=(
            "FC protocol -- upstream's `(FC)` leaderboard column. Headline is "
            "`Live Overall Acc`: the sample-count-weighted mean of six "
            "categories, which is algebraically the pooled rate over all 2251 "
            "live rows, so it carries a problem-clustered interval; "
            "`ast_summary` (weighted over the four AST categories) carries one "
            "on `n_ast` for the same reason. underscore_to_dot=True: "
            "convert_to_tool rewrote `.` to `_` in every function name on the "
            "way out, because a native tool name cannot carry a dot, so gold is "
            "rewritten to match before comparison. Requires the "
            "`function_tools` capability; a binding that does not offer it is "
            "rejected when the task is constructed, which prelaunch "
            "reconciliation reaches before the first request and before a "
            "result directory exists. Replaying upstream's released "
            "gpt-4.1-2025-04-14-FC rollouts reproduces all six published "
            "category accuracies exactly and agrees with upstream's own "
            "verdict on 2251/2251 rows: live_simple 80.23, live_multiple "
            "78.35, live_parallel 68.75, live_parallel_multiple 66.67, "
            "live_irrelevance 82.31, live_relevance 77.78 -- so ast_summary "
            f"78.39 and Live Overall Acc 79.92. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3LiveZeroShotGenFCTask(
    BfclV3FCMixin, BfclV3LiveTask[BfclV3LiveDatasetSample]
):
    pass
