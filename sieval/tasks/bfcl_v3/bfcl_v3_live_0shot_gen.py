"""BFCL v3 live categories, 0-shot, upstream's prompting protocol.

Six user-contributed categories, 2251 rows: four Python AST categories plus
irrelevance and relevance detection. The headline is upstream's `Live Overall
Acc` -- the SAMPLE-COUNT-WEIGHTED mean of the six, which is algebraically the
pooled rate over all 2251 rows. That is why this group publishes an interval
where the non-live group cannot: a weighted mean of category rates over their
own row counts is the rate over their union, so a problem-clustered interval
brackets the number printed beside it. `ast_summary`, weighted over the four AST
categories, carries one on `n_ast` for the same reason.

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
from sieval.datasets import BfclV3LiveDatasetSample

from ._base import (
    BFCL_V3_HARNESS_URL,
    BFCL_V3_SHARED_NOTES,
    BfclV3LiveTask,
    BfclV3PromptMixin,
)


@sieval_task(
    name="bfcl_v3_live_0shot_gen",
    display_name="BFCL v3 Live (0-shot, prompting)",
    description=(
        "Berkeley Function Calling Leaderboard v3, live categories, "
        "upstream's prompting protocol."
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
            "PROMPT protocol -- upstream's `(Prompt)` leaderboard column. "
            "Headline is `Live Overall Acc`: the sample-count-weighted mean of "
            "six categories, which is algebraically the pooled rate over all "
            "2251 live rows, so it carries a problem-clustered interval; "
            "`ast_summary` (weighted over the four AST categories) carries one "
            "on `n_ast` for the same reason. underscore_to_dot=False: schemas "
            "reach the model verbatim, so dotted function names survive and "
            "gold needs no rewrite. experimental until an alignment run "
            f"against a published BFCL v3 number lands. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3LiveZeroShotGenTask(
    BfclV3PromptMixin, BfclV3LiveTask[BfclV3LiveDatasetSample]
):
    pass
