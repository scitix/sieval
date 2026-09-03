"""WikiSQL — 0-shot text-to-SQL, scored by upstream's own two metrics.

The model sees one table's schema and a question, and answers with a **logical
form** — ``{"sel": int, "agg": int, "conds": [[col, op, value], …]}`` — which is
upstream's own prediction format, the thing its ``evaluate.py`` reads off a
``.jsonl``. Both published columns follow from it, computed by upstream's
unmodified code: ``ex_accuracy`` runs gold and prediction through ``DBEngine``
and compares result lists; ``lf_accuracy`` is ``Query.__eq__``, comparing
conditions as an unordered set (upstream's default — the leaderboard marks the
ordered reading ``*``). ``ex_accuracy`` is the headline, being
``evaluate.py``'s first key and the only column *every* leaderboard row reports.

**Nothing the model writes is ever executed as SQL** — upstream's protocol, not
a hardening choice, since a prediction is three integers and a value list bound
as parameters into a fixed template. Two guards sit on top of it, both raising
into the ``except Exception`` that ``evaluate.py`` already wraps every prediction
in, so a malformed prediction scores wrong exactly as upstream scores it: index
validation before the SQL template (argued in
``sieval.community.wikisql.dbengine``) and the logical-form comparison, which
upstream computes *outside* its own guard (argued in ``feedback``).

**The prompt is sieval's, and it has to be.** Upstream ships no LLM path at all:
no prompt, no API client, no chat template. Its inputs came from ``annotate.py``
(Stanza-tokenised streams feeding a trained decoder), which its own README
declares unreproducible since Stanza's deprecation, and every leaderboard row is
a fine-tuned model. So no published number fits a 0-shot chat run, and
``status="experimental"`` says so: the *grader* is upstream's and anchored
(below), the *protocol* is new. What the prompt carries is upstream's own
constraints, not invented ones:

* **Schema only — no table content.** Upstream's leaderboard rule, which marks
  the rows that break it with ``^``. The sample carries the rows regardless —
  the grader needs them to build the table.
* **Condition values appear verbatim in the question.** A measured property of
  the data, asserted by upstream's own ``test/check.py`` on every row of every
  split, so stating it describes the benchmark rather than leaking an answer.
* **``op`` 3 (``OP``) is omitted from the operator list.** It is in upstream's
  ``cond_ops`` but appears in no gold row and renders as invalid SQL, so offering
  it would only invite a guaranteed-wrong answer. The omission is in the prompt,
  not in the scorer: the grader still accepts it and lets SQLite reject it.

**Grader anchored on upstream's own artefact.** ``test/example.pred.dev.jsonl.bz2``
ships 8,421 predictions in upstream's prediction format, precisely so a harness
can be checked. Replayed through this whole task — reply text → extractor →
``feedback`` → ``report``, not merely through the engine — it scores **53.81
ex_accuracy / 45.21 lf_accuracy** against ``0.538060`` / ``0.452084`` from a
faithful transcription of ``evaluate.py`` over the shipped ``dev.db``, and
matches its ``0.441159`` for the ordered reading. That replay's
``n_unextracted``/``n_execution_errors`` counts look like defects and are
upstream's own behaviour; ``reference_impl.notes`` has both.

``n=1`` is this task's default rather than a ported one — upstream publishes no
sampling protocol, its models being argmax decoders. Decoding params are
model-layer, via ``models:``/``infer_args``, never this task.

Reference points, for orientation only — supervised fine-tuned models, **not**
an anchor for a 0-shot chat run (test-split lf / ex): SeaD+EG 87.5 / 93.0,
SQLova 80.7 / 86.2, Seq2SQL 48.3 / 59.4, the paper's baseline 23.4 / 35.9.

References:

* Paper: <https://arxiv.org/abs/1709.00103>
* Harness: <https://github.com/salesforce/WikiSQL>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from typing import override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    NonRetriableSampleError,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
)
from sieval.datasets import WikiSQLDatasetSample

#: Upstream's `Query.agg_ops`, rendered for the prompt. Index 0 is upstream's
#: empty string for "no aggregation", spelled `none` so the list reads.
_AGG_OPS_HELP = "0 = none, 1 = MAX, 2 = MIN, 3 = COUNT, 4 = SUM, 5 = AVG"

#: Upstream's `Query.cond_ops` minus index 3 (`OP`) — see the module docstring.
_COND_OPS_HELP = "0 for =, 1 for >, 2 for <"

_PROMPT_TEMPLATE = """\
You are given the schema of a single SQL table and a question about it. \
Translate the question into a query over that table.

Table columns, as `index: name [type]`:
{schema}

Question: {question}

The query must have this form, over the table above:

    SELECT <agg>(col<sel>) FROM table WHERE col<i> <op> <value> [AND ...]

Reply with a single JSON object and nothing else:

    {{"sel": <int>, "agg": <int>, "conds": [[<column index>, <operator index>, \
<value>], ...]}}

- `sel` is the index of the column being selected.
- `agg` is the aggregation applied to it: {agg_ops}.
- `conds` is the list of WHERE conditions, each a
  `[column index, operator index, value]` triple, where the operator index is
  {cond_ops}. Use an empty list if the question needs no condition.
- Every condition value appears verbatim in the question.\
"""


def render_schema(header: list[str], types: list[str]) -> str:
    """Render the schema block: one indented ``index: name [type]`` per column.

    Indices lead each line because a prediction refers to columns by integer,
    not by name.
    """
    return "\n".join(
        f"    {i}: {name} [{type_}]"
        for i, (name, type_) in enumerate(zip(header, types, strict=True))
    )


def build_prompt(question: str, header: list[str], types: list[str]) -> str:
    """Render the full user turn for one question."""
    return _PROMPT_TEMPLATE.format(
        schema=render_schema(header, types),
        question=question,
        agg_ops=_AGG_OPS_HELP,
        cond_ops=_COND_OPS_HELP,
    )


#: The three keys a logical form must carry to be a candidate. Upstream's
#: `Query.from_dict` reads exactly these.
_REQUIRED_KEYS = frozenset({"sel", "agg", "conds"})

#: One reusable decoder. `raw_decode` parses a JSON value *at* a given offset,
#: ignoring whatever follows it.
_DECODER = json.JSONDecoder()


def extract_logical_form(text: str) -> dict | None:
    """Pull the predicted logical form out of a reply, or ``None``.

    Tries to decode a JSON value at every ``{`` in the reply and returns the one
    starting **last** among those parsing as an object with all of
    ``sel``/``agg``/``conds``. *Last* because a chat model reasons before it
    answers, so an earlier run is a draft it corrected; *all three keys* so
    neither a stray fragment nor a wrapper is mistaken for an answer (given
    ``{"answer": {"sel": …}}`` the outer object fails and the inner is returned);
    *by start position*, which is what lands the nested case on the inner object,
    since it opens later even though it closes first.

    ``raw_decode`` rather than brace counting, because only the parser knows a
    brace inside a string literal is not structure: an unbalanced ``{`` or ``}``
    in a condition value would otherwise unbalance the scan and lose a
    well-formed answer. Tracking quote state instead would only move the
    failure, an unmatched ``"`` in the surrounding prose then swallowing the
    answer; decoding *at* a brace reads nothing before it and has neither
    problem.

    The prompt's own format spec cannot be matched here: it holds ``<int>``
    placeholders and does not parse as JSON. Deliberate, so a reply quoting the
    instructions back is never scored as if it had answered them.
    """
    best: dict | None = None
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            candidate, _ = _DECODER.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.keys() >= _REQUIRED_KEYS:
            best = candidate
    return best


@sieval_task(
    name="wikisql_0shot_gen",
    display_name="WikiSQL (0-shot, generative)",
    description=(
        "Text-to-SQL over one Wikipedia table, scored by execution and "
        "logical-form accuracy."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "sql", "tabular", "text-to-sql"),
    deps_group="wikisql",
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="WikiSQL",
        # The repo root: upstream keeps evaluate.py at the top level over
        # lib/query.py + lib/dbengine.py, so the root tree is what this mirrors.
        url="https://github.com/salesforce/WikiSQL/tree/cffb423077756d04c1bac5bcd45167c86903fbcb/",
        notes=(
            "0-shot generative port of WikiSQL (Salesforce; Zhong et al. 2017). "
            "SCORING IS UPSTREAM'S, UNMODIFIED: predictions are emitted in "
            "upstream's own logical-form format ({sel, agg, conds}), so "
            "evaluate.py's two metrics are computed by its own code, vendored in "
            "sieval.community.wikisql -- lf_accuracy via Query.__eq__ "
            "(conditions compared as an UNORDERED set of (col, op, "
            "str(val).lower()), upstream's default; the leaderboard marks the "
            "ordered reading with *), and ex_accuracy by running both sides "
            "through DBEngine and comparing result lists. Headline is "
            "ex_accuracy: evaluate.py's first key, and the only column every "
            "leaderboard row reports. "
            "KEY NAMES ARE UPSTREAM'S, NOT THE FAMILY'S, and deliberately so: "
            "ex_accuracy/lf_accuracy are evaluate.py's own two JSON keys, so "
            "they point back at the code that defines them. The Spider tasks "
            "spell the same two concepts execution_accuracy/exact_match, after "
            "their own upstream -- same measurements, different published "
            "vocabulary, so read ex_accuracy as execution accuracy and "
            "lf_accuracy as the structural (non-execution) match. Renaming "
            "either to match a sibling would cost the pointer back to upstream, "
            "the more load-bearing property for a port. "
            "NO MODEL-AUTHORED SQL IS EXECUTED -- a property of upstream's "
            "protocol, not a local hardening: a prediction is three integers "
            "plus a value list, so the executed statement is upstream's fixed "
            "template with values bound as parameters. "
            "THREE DELIBERATE DIVERGENCES. (1) records/SQLAlchemy -> stdlib "
            "sqlite3, and tables are rebuilt in memory by upstream's own "
            "create_table from the dataset's own types/rows, instead of read "
            "from the shipped {split}.db. Verified, not assumed: all 15,878 "
            "test gold queries return identical results from the rebuild and "
            "from the shipped test.db (0 mismatches, 0 exceptions); the declared "
            "schema matches the types column for all 5,230 test and 2,716 dev "
            "tables; and upstream's example predictions score identically both "
            "ways to six decimals. Drops ~120 MB of binary SQLite and leaves "
            "the engine no filesystem reach. Score impact: 0.00pp, measured. "
            "(2) sel/agg/col/op are validated as ints in range before being "
            "formatted into the SQL text. Upstream interpolates them unchecked, "
            "which is safe for a decoder over a closed output space and not for "
            "a chat model: a non-int sel is an injection point, a bool renders "
            "as colTrue, and a NEGATIVE index silently wraps (agg_ops[-1] is "
            "AVG), scoring a query nobody asked for. An execution-safety stop, "
            "so no _fixed variant: a rejected prediction raises into the same "
            "`except Exception` evaluate.py already wraps every prediction in, "
            "and scores wrong exactly as upstream scores it. op=3 (OP) is "
            "deliberately NOT rejected -- it is in upstream's cond_ops, appears "
            "in no gold row, and renders as invalid SQL, so SQLite rejects it as "
            "upstream lets it. (3) The logical-form comparison is computed "
            "INSIDE that same guard; evaluate.py computes it outside. "
            "Query.__eq__ unpacks every condition into a 3-tuple and hashes it, "
            "so a conds that is not a list of 3-element hashable triples "
            "([[0, 0]], a dict of condition fields, a nested list) raises past "
            "the guard -- which upstream never meets (closed-vocabulary "
            "decoder) and a chat model emits readily, failing the SAMPLE rather "
            "than scoring wrong, burning retries on a deterministic outcome and "
            "filing a malformed answer under pipeline faults. Same argument and "
            "resolution as (2): a raise reads as lf=False. Score impact is "
            "identically zero rather than measured small, since lf is true only "
            "of a form structurally equal to a gold that executes; confirmed on "
            "the anchor below at 0 occurrences in 8,421 predictions, both "
            "metrics unchanged to the reported precision. "
            "PROMPT IS SIEVAL'S, NECESSARILY: upstream ships no LLM path at all "
            "-- no prompt, no API client, no chat template. Its inputs came from "
            "annotate.py's Stanza-tokenised streams feeding a trained decoder, "
            "and its README declares that path unreproducible since Stanza's "
            "deprecation; every leaderboard row is a fine-tuned model. The "
            "prompt does carry upstream's constraints: schema only and NO table "
            "content (upstream's leaderboard rule, which marks violations with "
            "^), and the documented property that condition values appear "
            "verbatim in the question (asserted by upstream's own test/check.py "
            "on every row). cond_op 3 (OP) is omitted from the prompt's "
            "operator list only. "
            "ANCHOR: no published number exists for a 0-shot chat protocol, so "
            "the GRADER is anchored instead, on upstream's own shipped "
            "test/example.pred.dev.jsonl.bz2 (8,421 predictions in its "
            "prediction format). Replayed through the WHOLE task -- reply text, "
            "extractor, feedback, report -- this port scores 53.81 ex_accuracy / "
            "45.21 lf_accuracy against 0.538060 / 0.452084 from a faithful "
            "transcription of evaluate.py over the shipped dev.db, and matches "
            "its 0.441159 for the ordered reading (2026-08-22). Two counts from "
            "that replay look like defects and are not: n_unextracted=606 is "
            'upstream\'s own {"query": null, "error": ...} lines, which '
            "evaluate.py also scores wrong on both metrics (its error branch "
            "leaves qp None) -- the same state this port reaches through "
            "prediction is None, which is why the totals agree rather than "
            "merely approximate; and n_execution_errors=78 is well-formed "
            "predictions that do not run (mostly a column index past the table "
            "width), which upstream also scores wrong through its "
            "except-Exception branch. Hence status=experimental: the grader is "
            "upstream's and verified, the protocol is new. "
            "EX_ACCURACY'S OWN LIMIT, upstream's and not this port's: 1,279 of "
            "15,878 test gold queries (8.1%) execute to a degenerate empty or "
            "[None] result, so any prediction that also returns nothing is "
            "credited. lf_accuracy has no such slack, which is why both are "
            "reported side by side. "
            "SAMPLING: n=1 is this task's default, not a ported protocol -- "
            "upstream publishes none (its models are argmax decoders). "
            "Decoding params are model-layer, via models:/infer_args. "
            "SPLITS: test (15,878) is the default; validation (8,421) is the "
            "other published column. train is not materialised by the loader."
        ),
    ),
)
class WikiSQLZeroShotGenTask(
    Task[
        WikiSQLDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        # One user turn. The table's `rows` travel on the sample but stay out of
        # the prompt (upstream's leaderboard rule) -- `feedback` needs them to
        # build the table the query runs against.
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": build_prompt(
                        raw["question"], raw["header"], raw["types"]
                    ),
                }
            ],
            reference=raw["sql_json"],
            extra={"table_id": raw["table_id"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # `None` means no JSON object carrying all three keys was found, which is
        # what `n_unextracted` counts -- distinct from a form that parsed but
        # will not execute, which reaches the engine and is scored.
        return build_prediction_record(
            [extract_logical_form(text) for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Score every rollout the way ``evaluate.py``'s loop does.

        One in-memory engine per sample, scoped to this call: the table is on
        the sample, so nothing is cached and no state crosses the stage
        boundary. Gold is executed once and both metrics are read against it.
        A prediction that parsed but will not execute (out-of-range column,
        ``OP`` as an operator, a value SQLite cannot bind) lands in the same
        ``except Exception`` upstream wraps every prediction in, so its "result"
        becomes the exception's ``repr`` and cannot equal the gold list. The
        reason is kept in ``extra``, since "a well-formed query that does not
        run" is a different diagnosis from "nothing".

        The engine import is deferred to here rather than module scope: it
        reaches ``babel`` through upstream's ``dbengine``, and registering a
        task must not pull its optional grading dependency in — otherwise the
        ``wikisql`` group stops being optional and ``sieval task list`` needs
        it. Pinned by ``tests/unit/tasks/test_import_discipline_family.py``.
        """
        from sieval.community.wikisql import DBEngine, Query

        raw = ctx.raw_sample
        table_id = raw["table_id"]
        sql = json.loads(raw["sql_json"])
        table_rows = json.loads(raw["rows_json"])

        with DBEngine.from_table(table_id, raw["types"], table_rows) as engine:
            gold_query = Query.from_dict(sql)
            try:
                gold = engine.execute_query(table_id, gold_query, lower=True)
            except Exception as exc:
                # Upstream's own check.py asserts every gold query executes on
                # every row of every split, so this is a broken sample rather
                # than a recoverable fault.
                raise NonRetriableSampleError(
                    f"WikiSQL gold query failed to execute for table "
                    f"{table_id!r}: {exc!r}"
                ) from exc

            rollouts = []
            for rollout in post["rollouts"]:
                predicted = rollout.get("prediction")
                if predicted is None:
                    # No `unextracted` flag here: the prediction record's own
                    # `extracted` is the durable companion `health_metrics`
                    # counts `n_unextracted` from. One fact, one place.
                    rollouts.append(
                        build_rollout_judgement(
                            rollout["index"],
                            False,
                            score=0.0,
                            metrics={"ex": False, "lf": False},
                        )
                    )
                    continue

                error: str | None = None
                pred_query = None
                try:
                    pred_query = Query.from_dict(predicted)
                    result = engine.execute_query(table_id, pred_query, lower=True)
                except Exception as exc:  # noqa: BLE001 - upstream's guard
                    result = repr(exc)
                    error = repr(exc)
                # `pred_query == gold_query` with a None left operand falls back
                # to identity and is False, exactly as it does upstream.
                ex = result == gold
                try:
                    lf = pred_query == gold_query
                except Exception:  # noqa: BLE001 - widens upstream's own guard
                    # `Query.__eq__` unpacks each condition into a 3-tuple and
                    # hashes it, so a `conds` that is not a list of 3-element
                    # hashable triples raises. Upstream computes this outside its
                    # guard and never meets one (closed-output-space decoder); a
                    # chat model emits them readily, and left there the sample
                    # fails instead of scoring wrong. Same argument as the index
                    # guard in `dbengine`, same resolution -- and it cannot move
                    # a published number, since `lf` is True only for a form
                    # structurally equal to a gold that executes.
                    lf = False
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"],
                        ex,
                        score=float(ex),
                        metrics={"ex": ex, "lf": lf},
                        # Only when there is one -- an absent key reads as "ran".
                        extra={"execution_error": error} if error else None,
                    )
                )

        return True, build_judgement_record(raw["sql_json"], rollouts)

    @override
    async def report(self, finals, fails):
        n_ex = 0
        n_lf = 0
        n_exec_errors = 0
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                metrics = rollout.get("metrics") or {}
                if metrics.get("ex"):
                    n_ex += 1
                if metrics.get("lf"):
                    n_lf += 1
                if (rollout.get("extra") or {}).get("execution_error"):
                    n_exec_errors += 1

        # Upstream's denominator is every line of the prediction file, so a
        # pipeline failure counts as wrong rather than being excluded.
        n = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / n, 2)) if n else (lambda c: 0.0)
        return {
            "score": rate(n_ex),
            "ex_accuracy": rate(n_ex),
            "lf_accuracy": rate(n_lf),
            "n": float(n),
            "fails": float(len(fails)),
            # Predictions that parsed but would not run. Both this and
            # `n_unextracted` (nothing to parse at all) score zero, so without
            # the pair a model emitting well-formed but invalid queries is
            # indistinguishable from one emitting prose.
            "n_execution_errors": float(n_exec_errors),
            SCORE_KEY_FIELD: "ex_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        } | health_metrics(finals)
