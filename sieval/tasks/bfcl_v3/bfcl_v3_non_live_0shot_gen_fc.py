"""BFCL v3 non-live categories, 0-shot, upstream's function-calling protocol.

Seven hand-written categories, 1390 rows: four Python AST categories, Java and
JavaScript simple-call categories, and irrelevance detection.

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
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="gorilla/berkeley-function-call-leaderboard",
        url=BFCL_V3_HARNESS_URL,
        notes=(
            "FC protocol -- upstream's `(FC)` leaderboard column. Headline is "
            "`Non_Live Overall Acc`: unweighted mean of simple_ast (itself the "
            "unweighted mean of simple/java/javascript), multiple, parallel, "
            "parallel_multiple and irrelevance. No interval on the headline, "
            "simple_ast or ast_summary: an unweighted mean of category rates is "
            "not the pooled rate (javascript's 50 rows weigh as much as simple's "
            "400), so a sample-clustered CI would bracket a different statistic. "
            "Every per-category rate carries its own. underscore_to_dot=True: "
            "convert_to_tool rewrote `.` to `_` in every function name on the "
            "way out, because a native tool name cannot carry a dot, so gold is "
            "rewritten to match before comparison. Requires the `function_tools` "
            "capability; a binding that does not offer it is rejected at task "
            "construction, which prelaunch reconciliation reaches before the "
            "first request and before a result directory exists. Replay of "
            "upstream's gpt-4.1-2025-04-14-FC rollouts agrees on 1390/1390 rows "
            "and reproduces every published cell: simple 93.50, multiple 90.50, "
            "parallel 91.00, parallel_multiple 86.00, java 61.00, javascript "
            "68.00, irrelevance 89.58; simple_ast 74.17, Non_Live Overall Acc "
            f"86.25. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3NonLiveZeroShotGenFCTask(
    BfclV3FCMixin, BfclV3NonLiveTask[BfclV3NonLiveDatasetSample]
):
    pass
