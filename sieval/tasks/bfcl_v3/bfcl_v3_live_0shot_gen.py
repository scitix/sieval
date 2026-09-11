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
    # Needed even though every live category is Python, which is the part that
    # reads wrong: `_decode` calls `ast_parse`, and its module imports both
    # tree-sitter source parsers at module scope, so the import is paid before
    # the language dispatch ever runs. Not on the `_fc` sibling -- FC parses no
    # source text. Without this, readiness reports `yes`, the run bills for
    # inference, and then every sample dies at postprocess.
    deps_group="bfcl-v3",
    model_type="chat",
    status="stable",
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
            "gold needs no rewrite. Replaying upstream's released "
            "gpt-4.1-2025-04-14 (Prompt) rollouts reproduces all six published "
            "category accuracies exactly and agrees with upstream's own "
            "verdict on 2251/2251 rows: live_simple 85.66, live_multiple "
            "76.54, live_parallel 93.75, live_parallel_multiple 75.00, "
            "live_irrelevance 77.89, live_relevance 88.89 -- so ast_summary "
            f"78.46 and Live Overall Acc 78.32. {BFCL_V3_SHARED_NOTES}"
        ),
    ),
)
class BfclV3LiveZeroShotGenTask(
    BfclV3PromptMixin, BfclV3LiveTask[BfclV3LiveDatasetSample]
):
    pass
