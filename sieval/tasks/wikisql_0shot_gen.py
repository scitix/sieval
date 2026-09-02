"""WikiSQL — 0-shot text-to-SQL, scored by upstream's own two metrics.

The model sees one table's schema and a question, and answers with the query as
a **logical form** — ``{"sel": int, "agg": int, "conds": [[col, op, value], …]}``
— which is upstream's own prediction format, the thing its ``evaluate.py`` reads
off a ``.jsonl``. Both published columns follow from it directly, computed by
upstream's unmodified code:

* ``ex_accuracy`` — run gold and prediction through ``DBEngine`` and compare the
  result lists. The headline, because it is ``evaluate.py``'s first key and the
  one column *every* leaderboard row reports (the weakly-supervised table
  publishes nothing else).
* ``lf_accuracy`` — ``Query.__eq__``: ``sel`` and ``agg`` equal, and conditions
  equal as an unordered set of ``(col, op, str(value).lower())``. Unordered is
  upstream's default; the leaderboard marks the ordered reading with ``*``.

**Nothing the model writes is ever executed as SQL.** That is a property of
upstream's protocol rather than a hardening choice: a prediction is three
integers and a value list, so the query that runs is upstream's fixed template
with the values bound as parameters. The one thing added is validation that the
integers *are* integers in range before they are formatted into the template —
argued in ``sieval.community.wikisql.dbengine``, whose docstring also explains
why a negative index is the dangerous case rather than an obviously invalid one.
A rejected prediction raises into the same guard ``evaluate.py`` already wraps
every prediction in, so it scores wrong exactly as upstream scores it.

That guard is widened by one line here, for the same reason and no other.
Upstream computes its logical-form comparison *outside* its own
``except Exception``; ``Query.__eq__`` unpacks each condition into a 3-tuple and
hashes it, so a ``conds`` that is not a list of 3-element hashable triples
raises past the guard. Upstream never meets one — its predictions come from a
decoder over a closed output space — and a chat model produces them readily
(``[[0, 0]]``, a dict of condition fields, a nested list). Left alone, such a
prediction fails the *sample* rather than scoring wrong, which burns retries on
a deterministic outcome and files a model's malformed answer under pipeline
faults. So ``lf`` is computed inside the guard and a raise reads as ``False``.
No published number can move: ``lf`` is true only of a form structurally equal
to a gold that executes, so a form that raises here was never going to score.

**The prompt is sieval's, and it has to be.** Upstream ships no LLM path at all:
no prompt, no API client, no chat template — its inputs came from ``annotate.py``
(Stanza-tokenised column/question token streams feeding a trained decoder), and
its own README declares that path unreproducible since Stanza's deprecation.
Every leaderboard row is a fine-tuned model. So there is no published number a
0-shot chat run can be compared against, and ``status="experimental"`` says so:
the *grader* is upstream's and anchored (below), the *protocol* is new.

What the prompt does carry is upstream's own constraints, not invented ones:

* **Schema only — no table content.** Upstream's leaderboard rule is explicit:
  "your models only use the table schema and question during inference. That is
  they do *not* use the table content", with ``^`` marking the rows that break
  it. So column names and types go in; the rows do not, even though the sample
  carries them (the grader needs them to build the table).
* **Condition values appear verbatim in the question.** A measured property of
  the data, asserted by upstream's own ``test/check.py`` on every row of every
  split, so stating it describes the benchmark rather than leaking an answer.
* **``op`` 3 (``OP``) is omitted from the operator list.** It is in upstream's
  ``cond_ops`` but appears in no gold row of either split and renders as invalid
  SQL, so offering it would only invite a guaranteed-wrong answer. The *grader*
  still accepts it and lets SQLite reject it, which is upstream's behaviour — the
  omission is in the prompt, not in the scorer.

The format spec in the prompt uses ``<int>`` placeholders, so the example is not
itself parseable JSON. That is deliberate: an extractor that scans for a JSON
object would otherwise happily read the template back out of a reply that quoted
it, which is the failure mode where a task scores its own prompt.

**Grader anchored on upstream's own artefact.** Upstream ships
``test/example.pred.dev.jsonl.bz2`` — 8,421 predictions in its prediction format
— precisely so a harness can be checked. Replayed through this whole task
(reply text → extractor → ``feedback`` → ``report``, not merely through the
engine): **53.81 ex_accuracy / 45.21 lf_accuracy**, against ``0.538060`` /
``0.452084`` from a faithful transcription of ``evaluate.py`` over upstream's
shipped ``dev.db``. The ``--ordered`` reading matches too, at ``0.441159``.

Two counts from that replay are worth reading, because both look like defects
and are neither:

* ``n_unextracted = 606`` — upstream's own file carries ``{"query": null,
  "error": …}`` on exactly those 606 lines: predictions its example model
  declined to make. ``evaluate.py`` takes its ``error`` branch and scores them
  wrong on *both* metrics, leaving ``qp`` as ``None``. This port reaches the same
  verdict through ``prediction is None``, so upstream's ``error`` line and an
  unextracted rollout here are the same state — which is why the totals agree
  rather than merely being close.
* ``n_execution_errors = 78`` — well-formed predictions from upstream's model
  that do not run (a column index past the table's width, most of them). Upstream
  scores these wrong too, via the ``except Exception`` that turns the exception's
  ``repr`` into the compared "result".

Decoding params are model-layer, via ``models:``/``infer_args`` — never this
task. Upstream publishes no sampling protocol (its models are argmax decoders),
so ``n=1`` is this task's default rather than a ported one.

Reference points, for orientation only — these are supervised fine-tuned models
and **not** an anchor for a 0-shot chat run: SeaD+EG 87.5 lf / 93.0 ex, SQLova
80.7 / 86.2, Seq2SQL 48.3 / 59.4, and the paper's own baseline 23.4 / 35.9
(test-split columns).

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

#: Upstream's `Query.agg_ops`, rendered for the prompt. Index 0 is the empty
#: string upstream uses for "no aggregation"; spelling it as `none` keeps the
#: list readable without renumbering anything.
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

    Column *indices* are what a prediction refers to, so they lead each line —
    the model has to produce the integer, not the name.
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


def extract_logical_form(text: str) -> dict | None:
    """Pull the predicted logical form out of a reply, or ``None``.

    Scans every balanced ``{...}`` run at any nesting depth and returns the one
    that starts **last** among those parsing as a JSON object carrying all of
    ``sel``/``agg``/``conds``. Three properties come out of that rule:

    * *Last* rather than first, because a chat model reasons before it answers —
      an earlier brace run is a draft it then corrected.
    * *All three keys required*, so a stray JSON-looking fragment is not mistaken
      for an answer, and neither is a wrapper: given
      ``{"answer": {"sel": …}}`` the outer object fails the key test and the
      inner one is returned, so a model that labels its answer still scores.
    * *By start position*, which is what makes the nested case land on the inner
      object — it opens later even though it closes first.

    The prompt's own format spec cannot be matched here: it holds ``<int>``
    placeholders and does not parse as JSON. That is deliberate, so a reply
    quoting the instructions back is never scored as if it had answered them.
    """
    best: dict | None = None
    best_start = -1
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if start <= best_start:
                continue
            try:
                candidate = json.loads(text[start : i + 1])
            except ValueError:
                continue
            if isinstance(candidate, dict) and candidate.keys() >= _REQUIRED_KEYS:
                best = candidate
                best_start = start
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
            "their upstream. Same measurements, different published vocabulary "
            "-- a reader comparing text-to-SQL columns across benchmarks should "
            "read ex_accuracy as execution accuracy and lf_accuracy as the "
            "structural (non-execution) match. Renaming either to match a "
            "sibling would cost the pointer back to upstream, which is the more "
            "load-bearing property for a port. "
            "NO MODEL-AUTHORED SQL IS EXECUTED -- a property of upstream's "
            "protocol, not a local hardening: a prediction is three integers "
            "plus a value list, so the executed statement is upstream's fixed "
            "template with values bound as parameters. "
            "THREE DELIBERATE DIVERGENCES. (1) records/SQLAlchemy -> stdlib "
            "sqlite3, and tables are rebuilt in memory from the dataset's own "
            "types/rows instead of read from the shipped {split}.db. Upstream's "
            "own create_table does the rebuilding. Verified, not assumed: all "
            "15,878 test gold queries return identical results from the rebuild "
            "and from the shipped test.db (0 mismatches, 0 exceptions); the "
            "declared schema matches the types column for all 5,230 test and "
            "2,716 dev tables; and upstream's example predictions score "
            "identically both ways to six decimals. Drops ~120 MB of binary "
            "SQLite and leaves the engine no filesystem reach. Score impact: "
            "0.00pp, measured. (2) sel/agg/col/op are validated as ints in "
            "range before being formatted into the SQL text. Upstream "
            "interpolates them unchecked, which is safe for a decoder over a "
            "closed output space and not for a chat model: a non-int sel is an "
            "injection point, a bool renders as colTrue, and a NEGATIVE index "
            "silently wraps (agg_ops[-1] is AVG), scoring a query nobody asked "
            "for. An execution-safety stop, so no _fixed variant: a rejected "
            "prediction raises into the same `except Exception` evaluate.py "
            "already wraps every prediction in, and scores wrong exactly as "
            "upstream scores it. op=3 (OP) is deliberately NOT rejected -- it is "
            "in upstream's cond_ops, appears in no gold row, and renders as "
            "invalid SQL, so SQLite rejects it as upstream lets it. (3) The "
            "logical-form comparison is computed INSIDE that same guard. "
            "evaluate.py computes it outside its own except-Exception, and "
            "Query.__eq__ unpacks every condition into a 3-tuple and hashes it, "
            "so a conds that is not a list of 3-element hashable triples "
            "([[0, 0]], a dict of condition fields, a nested list) raises past "
            "the guard. Upstream never meets one (closed-vocabulary decoder); a "
            "chat model emits them readily, and left alone such a prediction "
            "fails the SAMPLE rather than scoring wrong -- burning retries on a "
            "deterministic outcome and filing a malformed answer under pipeline "
            "faults. Same argument as (2), same resolution: a raise reads as "
            "lf=False. Score impact is identically zero, not merely measured "
            "small: lf is true only of a form structurally equal to a gold that "
            "executes, so a form that raises here could never have scored. "
            "Confirmed on the anchor below -- 0 occurrences in 8,421 "
            "predictions, both metrics unchanged to the reported precision. "
            "PROMPT IS SIEVAL'S, NECESSARILY: upstream ships no LLM path at all "
            "-- no prompt, no API client, no chat template. Its inputs came from "
            "annotate.py's Stanza-tokenised streams feeding a trained decoder, "
            "and its README declares that path unreproducible since Stanza's "
            "deprecation; every leaderboard row is a fine-tuned model. The "
            "prompt does carry upstream's constraints: schema only and NO table "
            "content (upstream's leaderboard rule, which marks violations with "
            "^), and the documented property that condition values appear "
            "verbatim in the question (asserted by upstream's own "
            "test/check.py on every row). cond_op 3 (OP) is omitted from the "
            "prompt's operator list only. "
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
            "leaves qp None) -- so upstream's error line and an unextracted "
            "rollout here are the same state, which is why the totals agree "
            "rather than merely approximate; and n_execution_errors=78 is "
            "well-formed predictions that do not run (mostly a column index past "
            "the table width), which upstream also scores wrong through its "
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
        # One user turn. The table's `rows` are deliberately absent from the
        # prompt (upstream's leaderboard rule) though they travel on the sample
        # -- `feedback` needs them to build the table the query runs against.
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
        # `None` means no JSON object carrying all three keys was found, which
        # is what `n_unextracted` counts. It is distinct from a form that parsed
        # but will not execute -- that one reaches the engine and is scored.
        return build_prediction_record(
            [extract_logical_form(text) for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Score every rollout the way ``evaluate.py``'s loop does.

        One in-memory engine per sample, scoped to this call: the table is on
        the sample, so nothing is cached across samples and no state crosses the
        stage boundary. Gold is executed once and both metrics are read against
        it.

        A prediction that parsed but will not execute (out-of-range column,
        ``OP`` as an operator, a value SQLite cannot bind) lands in the same
        ``except Exception`` upstream wraps every prediction in: its "result"
        becomes the exception's ``repr``, which cannot equal the gold list, so it
        scores wrong. The failure reason is kept in ``extra`` rather than
        discarded, since "the model emitted a well-formed query that does not
        run" is a different diagnosis from "the model emitted nothing".

        The engine import is deferred to here, not module scope: it reaches
        ``babel`` through upstream's ``dbengine``, and registering a task must
        not pull its optional grading dependency in — otherwise the ``wikisql``
        group stops being optional and ``sieval task list`` needs it. Pinned by
        ``tests/unit/tasks/test_import_discipline_family.py``.
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
                # than a recoverable fault -- and scoring against a gold we
                # could not compute would be worse than failing the sample.
                raise NonRetriableSampleError(
                    f"WikiSQL gold query failed to execute for table "
                    f"{table_id!r}: {exc!r}"
                ) from exc

            rollouts = []
            for rollout in post["rollouts"]:
                predicted = rollout.get("prediction")
                if predicted is None:
                    # No `unextracted` flag here: the prediction record's own
                    # `extracted` is the durable companion, and it is what
                    # `health_metrics` counts `n_unextracted` from. Restating it
                    # on the judgement would be one fact in two places.
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
                except Exception:  # noqa: BLE001 - see below
                    # Upstream computes this comparison outside its own guard and
                    # would raise here too -- but its predictions come from a
                    # decoder over a closed output space, and these come from a
                    # chat model. `Query.__eq__` unpacks each condition into a
                    # 3-tuple and hashes it, so a `conds` that is not a list of
                    # 3-element hashable triples (`[[0, 0]]`, a dict, a nested
                    # list) raises out of `feedback` and fails the sample --
                    # burning retries on a deterministic outcome and reporting a
                    # model's malformed answer as a pipeline fault. The same
                    # argument that licenses the index guard in `dbengine`, and
                    # the same resolution: score it wrong, which is what
                    # upstream's `except Exception` means to do with any
                    # prediction it cannot grade. Cannot change a published
                    # number -- `lf` is True only for a form structurally equal
                    # to a gold that executes, so a form that raises here was
                    # never going to score.
                    lf = False
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"],
                        ex,
                        score=float(ex),
                        metrics={"ex": ex, "lf": lf},
                        # Only when there is one -- an absent key reads as "ran",
                        # which is the common case and needs no record.
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
            # Predictions that parsed but would not run. Distinct from
            # `n_unextracted` (nothing to parse at all): both score zero, and
            # without the pair a model emitting well-formed but invalid queries
            # is indistinguishable from one emitting prose.
            "n_execution_errors": float(n_exec_errors),
            SCORE_KEY_FIELD: "ex_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        } | health_metrics(finals)
