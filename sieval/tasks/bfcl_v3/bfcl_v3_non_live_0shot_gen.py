"""BFCL v3 non-live categories, 0-shot, upstream's prompting protocol.

Seven hand-written categories, 1390 rows: four Python AST categories, Java and
JavaScript simple-call categories, and irrelevance detection. The headline is
upstream's `Non_Live Overall Acc` -- the UNWEIGHTED mean of five numbers, one of
which (`simple_ast`) is itself the unweighted mean of three. It publishes no
confidence interval for that reason: it is a mean of rates, not the pooled rate
over 1390 rows, and an interval computed over those rows would bracket a
different statistic than the one printed beside it. Every per-category rate does
carry one.

This is the `(Prompt)` column: the function schemas arrive in the system turn
and the reply is parsed with `ast_parse`. The `_fc` sibling is the `(FC)`
column, where the same schemas arrive as tools. The unqualified name is this one
because every model on the leaderboard has a Prompt row, while an FC row exists
only where the provider exposes a tools API -- and because FC gates on the
`function_tools` capability, which would make the default name the narrow one.

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
    BfclV3NonLiveTask,
    BfclV3PromptMixin,
)


@sieval_task(
    name="bfcl_v3_non_live_0shot_gen",
    display_name="BFCL v3 Non-Live (0-shot, prompting)",
    description=(
        "Berkeley Function Calling Leaderboard v3, non-live categories, "
        "upstream's prompting protocol."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "function-calling"),
    # Declared here and NOT on the `_fc` sibling: `_decode` calls `ast_parse`,
    # whose module imports both tree-sitter source parsers at module scope, so
    # the Prompt protocol needs the extra for every category rather than just
    # `java` and `javascript`. FC never parses source text. Without this the
    # import is deferred far enough that readiness reports `yes`, the run bills
    # for inference, and then every sample dies at postprocess.
    deps_group="bfcl-v3",
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="gorilla/berkeley-function-call-leaderboard",
        url=BFCL_V3_HARNESS_URL,
        notes=(
            "PROMPT protocol -- upstream's `(Prompt)` leaderboard column. "
            "Headline is `Non_Live Overall Acc`: unweighted mean of simple_ast "
            "(itself the unweighted mean of simple/java/javascript), multiple, "
            "parallel, parallel_multiple and irrelevance. NO interval is "
            "published on the headline, nor on simple_ast or ast_summary: an "
            "unweighted mean of category rates is not the pooled rate over the "
            "rows (javascript's 50 weigh as much as simple's 400), so a "
            "sample-clustered CI would bracket a different statistic. Every "
            "per-category rate carries its own. underscore_to_dot=False: "
            "schemas reach the model verbatim, so dotted function names survive "
            "and gold needs no rewrite. experimental until an alignment run "
            f"against a published BFCL v3 number lands. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3NonLiveZeroShotGenTask(
    BfclV3PromptMixin, BfclV3NonLiveTask[BfclV3NonLiveDatasetSample]
):
    pass
