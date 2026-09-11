"""BFCL v3 non-live categories, 0-shot, upstream's function-calling protocol.

Seven hand-written categories, 1390 rows: four Python AST categories, Java and
JavaScript simple-call categories, and irrelevance detection. The headline is
upstream's `Non_Live Overall Acc` -- the UNWEIGHTED mean of five numbers, one of
which (`simple_ast`) is itself the unweighted mean of three. It publishes no
confidence interval for that reason: it is a mean of rates, not the pooled rate
over 1390 rows, and an interval computed over those rows would bracket a
different statistic than the one printed beside it. Every per-category rate does
carry one.

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
from sieval.datasets import BfclV3NonLiveDatasetSample

from ._base import (
    BFCL_V3_HARNESS_URL,
    BFCL_V3_SHARED_NOTES,
    BfclV3FCMixin,
    BfclV3NonLiveTask,
)


@sieval_task(
    name="bfcl_v3_non_live_0shot_gen_fc",
    display_name="BFCL v3 Non-Live (0-shot, FC)",
    description=(
        "Berkeley Function Calling Leaderboard v3, non-live categories, "
        "upstream's FC (tools API) protocol."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "function-calling"),
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="gorilla/berkeley-function-call-leaderboard",
        url=BFCL_V3_HARNESS_URL,
        notes=(
            "FC protocol -- upstream's `(FC)` leaderboard column. Headline is "
            "`Non_Live Overall Acc`: unweighted mean of simple_ast (itself the "
            "unweighted mean of simple/java/javascript), multiple, parallel, "
            "parallel_multiple and irrelevance. NO interval is published on the "
            "headline, nor on simple_ast or ast_summary: an unweighted mean of "
            "category rates is not the pooled rate over the rows (javascript's "
            "50 weigh as much as simple's 400), so a sample-clustered CI would "
            "bracket a different statistic. Every per-category rate carries its "
            "own. underscore_to_dot=True: convert_to_tool rewrote `.` to `_` in "
            "every function name on the way out, because a native tool name "
            "cannot carry a dot, so gold is rewritten to match before "
            "comparison. Requires the `function_tools` capability; a binding "
            "that does not offer it is rejected when the task is constructed, "
            "which prelaunch reconciliation reaches before the first request "
            "and before a result directory exists. experimental until an "
            f"alignment run against a published BFCL v3 number lands. "
            f"{BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3NonLiveZeroShotGenFCTask(
    BfclV3FCMixin, BfclV3NonLiveTask[BfclV3NonLiveDatasetSample]
):
    pass
