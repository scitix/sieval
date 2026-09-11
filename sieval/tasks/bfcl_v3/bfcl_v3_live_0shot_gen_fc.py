"""BFCL v3 live categories, 0-shot, upstream's function-calling protocol.

Six user-contributed categories, 2251 rows: four Python AST categories plus
irrelevance and relevance detection. Unlike the non-live group this one carries
intervals, because its weighted rollup IS the pooled rate over the union of its
rows -- see `BfclV3LiveTask`.

Upstream's `(FC)` column: the same schemas reach the model through the provider's
tools API and the calls are read off structured `tool_calls` rather than parsed
out of the reply. Beyond that transport, the only behavioural difference from
the Prompt sibling is `underscore_to_dot=True` -- upstream's own reversal of the
`.`->`_` rewrite a native tool name forces, explained in `_base`. Headline,
interval policy and the published numbers are in `reference_impl.notes`.

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
            "rejected at task construction, which prelaunch reconciliation "
            "reaches before the first request and before a result directory "
            "exists. Replay of upstream's gpt-4.1-2025-04-14-FC rollouts agrees "
            "on 2251/2251 rows and reproduces every published cell: live_simple "
            "80.23, live_multiple 78.35, live_parallel 68.75, "
            "live_parallel_multiple 66.67, live_irrelevance 82.31, "
            "live_relevance 77.78; ast_summary 78.39, Live Overall Acc "
            f"79.92. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3LiveZeroShotGenFCTask(
    BfclV3FCMixin, BfclV3LiveTask[BfclV3LiveDatasetSample]
):
    pass
