"""Spider 1.0 — 0-shot generative text-to-SQL, execution- and match-scored.

Spider (Yu et al., EMNLP 2018) is the reference cross-domain text-to-SQL
benchmark. This task evaluates its **dev** split — 1,034 questions over 20
databases, the split the literature reports, because the real test set was held
out for years. Three metrics, all upstream's, and the headline is the one
upstream itself moved to:

* **Test-suite accuracy** (headline) — upstream's official metric since October
  2020 (``taoyds/test-suite-sql-eval``). The prediction and the gold are run
  against ~39 *distilled* databases per question, generated to distinguish
  neighbouring queries, and the prediction must return the same rows on **every**
  one of them. Results are compared as raw result sets under bag semantics, with
  no parsing involved.
* **Execution accuracy** — the pre-2020 metric, on the single shipped database,
  compared through ``eval_exec_match``'s column-keyed projection.
* **Exact set match** — upstream's clause-by-clause set comparison, at its
  ``DISABLE_VALUE = True`` default, so literal values are not compared.

The last two are reference columns, reported because papers still quote them.
Neither is the number to rank on, and the reason is not merely that upstream
deprecated them — **both are scored through a parser**, and it rejects most of
what a chat model writes. The parser is a hand-written tokeniser over Spider's
own gold dialect; a prediction it rejects is compared as an *empty* projection
and scores 0 on both columns no matter what SQLite returned for it. On the
pinned dev data it accepts 100% of the golds and 30–59% of real model
predictions, depending on the model. So the gap between the headline and those
two columns is mostly dialect, not correctness, and ``n_parser_rejected`` is
published beside them to make the size of that gate visible rather than leave
two rates to be read as if they measured answers. ``score`` is deliberately not
one of them.

*Measured over two full dev passes* (2026-09-03), which is where that band and
this warning come from:

===================  ==================  =========  ==========  =================
model                test_suite (score)  execution  exact_set   n_parser_rejected
===================  ==================  =========  ==========  =================
gpt-5.4-mini         66.05               24.47      21.76       724 / 1,034
Qwen3.5-397B-A17B    78.05               53.87      51.74       419 / 1,034
===================  ==================  =========  ==========  =================

The reference columns do not merely run low, they **rank differently**: they put
Qwen ahead by 29.4 pp where the headline puts it ahead by 12.0 pp, because the
weaker model is also the one whose dialect the parser likes less (70% rejected
against 41%). A leaderboard built on either column would be reporting
conformance about as much as correctness, which is the whole reason ``score``
is the headline and these two are published beside their gate.

**Prompt: Rajkumar et al. 2022**, the "Create Table + Select 3" format from
*Evaluating the Text-to-SQL Capabilities of Large Language Models*
(arXiv:2204.00498) — every table's ``CREATE TABLE`` plus three example rows,
then the question. Spider predates LLM prompting and has no canonical prompt of
its own; this is the most-cited LLM-era convention, which is what makes the
number comparable to published work. **One divergence**: upstream's prompt ends
in a bare ``SELECT`` for a completion model, and a chat turn cannot end
mid-token, so the prompt asks for a fenced ``sql`` block instead. That is the
one reason a chat-mode score is not bit-comparable to the paper's Codex figures;
a completion-faithful ``spider_0shot_base_gen`` sibling is where those belong.

**Execution safety.** Both of upstream's evaluators open a *read-write*
``sqlite3.connect`` and run model-generated SQL with no timeout behind a bare
``except:``. Grading is synchronous on one shared event loop, so an unbounded
query stalls the session rather than one sample. This task therefore carries the
hardened reading on both paths — read-only immutable connection,
``ATTACH``/``DETACH`` denied, and a progress-handler deadline that aborts inside
SQLite. Per ``sieval/tasks/CLAUDE.md`` this is the one divergence that does
**not** earn a ``_fixed`` variant: a variant exists so two readings can be
compared, and the unsafe reading is not one we will run. Details and the measured
bounds live in ``sieval.tasks._spider_exec``; what the test-suite path adds on
top of them is in ``sieval.tasks._spider_test_suite``.

Only *execution* is ours. Every comparison is upstream's own bytes — ``result_eq``
for the headline, ``eval_exec_match``'s projection and ``eval_exact_match`` for
the reference columns — and upstream is preserved everywhere safety does not
object, including where it is wrong: an unparseable prediction is still scored
against upstream's empty parse rather than skipped, and exact match runs *after*
execution because ``eval_exact_match`` mutates the parse trees in place.

**Safety delta, measured.** All three obligations the hardening owes are
discharged on both paths, none of them needing a model.

*No bound binds* (test-suite path, 2026-09-03). All 20 dev databases are present
in the distilled archive, 25–60 variants each, 38.8 per sample and 40,167 gold
executions for a full pass. Every one of them succeeds — **zero gold failures** —
and gold-vs-gold scores 1,034/1,034. The largest gold result is 92,450 rows and
the slowest 0.359 s, against a 500,000-row cap and a 5 s deadline; only four dev
golds exceed 10,000 rows at all. The cap was *raised* from 100,000 for this
measurement, because 92,450 clears that by 7.5% and a bound that close is not
evidence of anything — see ``DEFAULT_MAX_ROWS`` for why the change cannot move a
verdict. A whole gold pass takes 20 s, and the worst single sample 1.08 s, so the
38.8x execution fan-out costs wall clock and no accuracy.

*No bound binds* (pre-2020 path, 2026-08-22). Over all 1,034 dev golds the
largest result is 20,662 rows and the slowest query 0.486 s. A gold-vs-gold pass
scores 1,034/1,034 on both metrics with zero errors, and its hardness split
(248 easy / 446 medium / 174 hard / 166 extra) reproduces Spider's published dev
distribution.

*Quantified score impact: 99.903% verdict parity* (2026-08-22). Over 1,033
comparable pairs — each dev gold graded against a sibling row's gold from the
same database, so the mix is realistic and executable without an API — the
hardened executor and upstream's own ``eval_exec_match`` (called with its own
module globals, not a reimplementation) agree 1,032 times and differ once.
Upstream additionally crashed outright on one further pair.

*Safety, not repair.* Both of those two cases are ``wta_1`` and both trace to the
same cause, the UTF-8 text factory: upstream's decode error surfaces as ``False``
when it hits the prediction (caught by its bare ``except:``) and as a crash when
it hits the gold. **The read-only connection, the ATTACH denial, the deadline and
the row cap produced zero verdict differences.** Worst-case headline impact is
2 of 1,034 samples, 0.19 pp, and only on models that get those two questions
right.

*The headline path does not inherit that divergence.* It runs on the same
connection, but its upstream is the other repo, and that one **does** set a text
factory — ``b.decode(errors="ignore")`` — so there is no crash to diverge from,
only a lossier decode. Lossier is the direction that can move a verdict: dropping
bytes can fold two distinct stored values onto one string and make result sets
compare equal that ours separates. *Measured, 2026-09-03: zero effect on the
pinned data.* Across all 715 databases the graded dev set reaches, exactly two
values are not valid UTF-8 — one ``first_name`` and one ``last_name`` in
``wta_1.players`` — and ``ignore`` is injective over every distinct value present
in those two columns (0 collisions in 41,324), so the two factories induce the
same equality relation. Details in ``_spider_sqlite.open_readonly``.

So on the headline path every *known* cause of divergence now measures zero, each
by its own measurement rather than by inheritance from the pre-2020 one.

Target: published Spider dev test-suite accuracy for the model under test, and
the two reference columns against papers that quote them.

*Port vs upstream, end to end: 2,387 pairs, zero divergences* (2026-09-03).
Every verdict this port reaches on the headline path was compared against
upstream's own ``eval_exec_match`` — called with its own module globals over the
same ``.sqlite`` set, at the same two flags — across three independent
prediction sets: the two full dev passes above (1,031 + 1,034) and upstream's
**own** shipped example predictions (``evaluation_examples/predict.txt``, 322
pairs over 4 db_ids). All 2,387 agree, with zero upstream crashes and zero gold
failures, the last of which the port asserts by construction: it *raises* on a
gold it cannot run, so a clean pass over ~39 databases per sample is itself the
evidence. This is the anchor the named-divergence measurements above could not
supply — they show every cause we can *name* measures zero, which is not the
same as having compared the two implementations.

**Why this still ships ``experimental``.** One anchor is left, and it is the
weaker of the two: *this task against a published number*. Spider's own
leaderboard is almost entirely fine-tuned systems, so it does not compare to a
0-shot chat model, and the one published figure using **this** prompt
(Rajkumar et al.'s ``code-davinci-002``) is a retired completion model, which is
exactly the divergence this task documents. So the remaining gap is not a
measurement anyone has skipped; it is a comparison Spider does not currently
offer, and the honest reading is that the harness is anchored and the score is
not. A ``spider_0shot_base_gen`` sibling is where a completion-faithful,
paper-comparable number belongs.

References:

* Paper: <https://arxiv.org/abs/1809.08887>
* Harness: <https://github.com/taoyds/spider>
* Test-suite harness: <https://github.com/taoyds/test-suite-sql-eval>
* Test-suite paper: <https://arxiv.org/abs/2010.02840>
* Prompt: <https://arxiv.org/abs/2204.00498>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import re
from functools import cache
from typing import override

from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
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
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import SpiderDatasetSample

from ._spider_schema import build_prompt

#: Upstream's four difficulty buckets, from `Evaluator.eval_hardness`.
HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")

#: Any fence label, not just ``sql``. The prompt asks for "valid SQLite", so
#: ```sqlite is a label models reach for; a pattern matching only ```sql falls
#: through to the raw reply and drags the closing backticks into the statement.
_FENCE = re.compile(r"```[^\s`]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_STATEMENT = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def _mask_comments(sql: str) -> str:
    """Blank out *sql*'s comment bodies, preserving every offset.

    So a match on the mask can slice the original. Comments are masked rather
    than removed for exactly that reason: the statement keeps whatever
    commentary the model wrote (SQLite accepts it), and only the *search* stops
    seeing it.

    A ``--`` inside a string literal opens no comment, so quoting is tracked in
    the same left-to-right pass — ``WHERE note = 'a--b'`` holds no comment.
    """
    masked = list(sql)
    quote: str | None = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
        elif char in "'\"":
            quote = char
            index += 1
        elif sql.startswith("--", index):
            end = sql.find("\n", index)
            end = len(sql) if end == -1 else end
            masked[index:end] = " " * (end - index)
            index = end
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            # An unterminated block comment runs to the end, which is how SQLite
            # reads it too.
            end = len(sql) if end == -1 else end + 2
            masked[index:end] = " " * (end - index)
            index = end
        else:
            index += 1
    return "".join(masked)


def _terminator_index(masked: str) -> int | None:
    """Offset of the first statement-terminating semicolon, or ``None``.

    Takes the *masked* text, so a semicolon inside a comment terminates
    nothing. A semicolon inside a string literal does not either, so quoting is
    tracked rather than the text being split on ``;`` — ``WHERE name = 'a;b'``
    is one statement. A doubled quote, SQLite's escape, falls out of the same
    toggle: it closes and reopens across the pair.
    """
    quote: str | None = None
    for index, char in enumerate(masked):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ";":
            return index
    return None


def _statement_from(candidate: str) -> str | None:
    """Slice one statement out of *candidate*, or ``None`` if it holds no SQL.

    The keyword search runs over the comment-masked text. A model that opens
    with ``-- Find all cars with more than 4 cylinders`` would otherwise have
    the statement start at that ``with``, and the slice — comment prose and all
    — reaches SQLite as ``with more than 4 cylinders SELECT ...``, one syntax
    error scored as a wrong answer and counted against `n_execution_errors`.
    ``with`` and ``select`` are ordinary English words, so a leading comment is
    enough on its own; no model in the two dev runs opened one, which is what
    made this reachable but unobserved.
    """
    masked = _mask_comments(candidate)
    match = _STATEMENT.search(masked)
    if match is None:
        return None
    statement, masked = candidate[match.start() :], masked[match.start() :]
    # A fence marker ends the statement. Reached only on the unfenced fallback,
    # where the reply can still carry a ``` the fence pattern did not consume;
    # leaving it in fails the query on `unrecognized token: "```"`.
    fence = masked.find("```")
    if fence != -1:
        statement, masked = statement[:fence], masked[:fence]
    end = _terminator_index(masked)
    if end is not None:
        statement = statement[:end]
    return statement.strip() or None


def extract_sql(text: str) -> str | None:
    """Pull one SQL statement out of a chat reply.

    Prefers the **last fenced block that holds SQL**: models routinely show
    working and give the answer last, and the block they close with is not
    always the answer — a trailing ```json note is common. Falls back to the
    text from the first ``SELECT``/``WITH`` keyword onward. Returns ``None``
    when nothing looks like SQL, which is what marks the rollout unextracted.
    """
    for candidate in reversed(_FENCE.findall(text)):
        statement = _statement_from(candidate)
        if statement is not None:
            return statement
    return _statement_from(text)


@cache
def _download_punkt_tab_once() -> None:
    import nltk

    nltk.download("punkt_tab", quiet=True)


def _ensure_punkt_tab() -> None:
    """Stage NLTK's ``punkt_tab`` before grading; the vendored parser needs it.

    ``process_sql.py`` tokenises every query with ``nltk.word_tokenize``, which
    resolves through **punkt_tab** on nltk >= 3.9. Without it every sample dies
    one ``LookupError`` at a time inside the grader — a wrong score rather than
    a loud stop. Cached so an offline run pays one network timeout, not 1,034.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass
    _download_punkt_tab_once()


@sieval_task(
    name="spider_0shot_gen",
    display_name="Spider 1.0 (0-shot, generative)",
    description="Cross-domain text-to-SQL, scored by test-suite execution accuracy.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "text-to-sql", "code-exec"),
    model_type="chat",
    # Flipped to "stable" once the safety delta and an alignment run exist.
    status="experimental",
    deps_group="spider",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="taoyds/test-suite-sql-eval",
        url=(
            "https://github.com/taoyds/test-suite-sql-eval/tree/"
            "e97acc546ecbee8fa27fa8dbf025ef61493a876c/"
        ),
        notes=(
            "TWO upstream trees, because Spider's metric moved in October 2020 "
            "and the fork does not carry the old one. Named above is the "
            "headline's: test-suite execution accuracy, vendored byte-identical "
            "in sieval.community.spider_test_suite except exec_eval.py:11, a "
            "flat import that cannot resolve inside a package. The pre-2020 tree "
            "supplies the two reference columns and the hardness buckets — "
            "taoyds/spider at "
            "https://github.com/taoyds/spider/tree/"
            "b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c/, vendored the same way in "
            "sieval.community.spider except evaluation.py:29. Exact set match is "
            "upstream's DISABLE_VALUE=True default; exact match runs AFTER "
            "execution because eval_exact_match mutates the parse trees in "
            "place. Both pre-2020 columns are scored through a parser that "
            "accepts 100% of golds and 30-59% of model predictions, so they are "
            "reported beside n_parser_rejected and are not the number to rank "
            "on -- measured over two full dev passes (2026-09-03), they also "
            "RANK differently from the headline: gpt-5.4-mini scores 66.05 "
            "test_suite / 24.47 execution / 21.76 exact_set with 724 of 1,034 "
            "parser-rejected, and Qwen3.5-397B-A17B 78.05 / 53.87 / 51.74 with "
            "419, so the reference columns put Qwen ahead by 29.4pp where the "
            "headline puts it ahead by 12.0pp. "
            "Upstream's two test-suite flags are pinned to its CLI defaults "
            "(plug_value=False, keep_distinct=False); a run configured either "
            "way is not comparable to a published Spider score. "
            "Both paths diverge for SAFETY ONLY: read-only immutable "
            "connection, ATTACH/DETACH denied, a progress-handler deadline and a "
            "row cap, because both upstreams run model SQL on a read-write "
            "connection with no timeout behind a bare except. Every comparison "
            "stays upstream's own bytes — result_eq for the headline, the "
            "column-keyed res_map for the pre-2020 column. Upstream's unseeded "
            "RNG in get_constraint_permutation is left unseeded: every "
            "permutation it discards is discarded on a sound necessary "
            "condition, so the draw changes how long the search runs and never "
            "what it concludes (measured over 300 draws). Three further "
            "verdict-preserving divergences: get_schema is reproduced read-only "
            "(upstream opens read-write; dict equality asserted in tests); a "
            "surrogateescape text factory, because one first_name and one "
            "last_name in wta_1.players are not valid UTF-8 — the pre-2020 tree "
            "sets no factory and fetches gold outside its except, so it dies on "
            "two dev examples rather than scoring them, while the test-suite "
            "tree sets the lossy decode(errors='ignore'); and a BLANK prediction "
            "is scored False rather than executed, which is the one place with "
            "no upstream behaviour to preserve — upstream reads predictions from "
            "a file in which a blank line is a session boundary, so it cannot "
            "receive one, and passing it through would score an unextracted "
            "answer CORRECT against any gold returning no rows, since SQLite "
            "returns [] for empty SQL. TEXT-FACTORY DELTA MEASURED (2026-09-03, "
            "headline path): zero. Over all 715 databases the dev set reaches, "
            "only those two columns hold invalid bytes, and ignore is injective "
            "over every distinct value in them (0 collisions in 41,324), so both "
            "factories induce the same equality relation. SAFETY DELTA MEASURED "
            "(pre-2020 path): 99.903% verdict parity against upstream's own "
            "eval_exec_match over 1,033 comparable pairs (1,032 agree, 1 "
            "differs, plus 1 upstream crash); all three cases are wta_1 and all "
            "trace to the text factory — the read-only connection, ATTACH "
            "denial, deadline and row cap produced zero verdict differences. "
            "Worst-case headline impact 2/1,034 = 0.19pp. PORT VS UPSTREAM "
            "ANCHORED END TO END (headline path): 2,387 pairs, ZERO "
            "divergences, against upstream's own eval_exec_match called with "
            "its own module globals over the same .sqlite set and the same two "
            "flags -- the two full dev passes (1,031 + 1,034) plus upstream's "
            "own shipped evaluation_examples/predict.txt (322 pairs, 4 db_ids). "
            "Zero upstream crashes and zero gold failures, the latter asserted "
            "by construction since this port raises on a gold it cannot run. "
            "STILL OWED, and the only reason this is experimental: a "
            "published-anchor run. That one is structurally weak for Spider -- "
            "its leaderboard is almost all fine-tuned systems, and the one "
            "published figure using this prompt (Rajkumar's code-davinci-002) "
            "is a retired completion model, which is the divergence this task "
            "documents. Bounds measured on both paths and no bound binds: 1,034 dev "
            "golds on the shipped databases, largest result 20,662 rows and "
            "slowest 0.486s; 40,167 gold executions across the distilled suite "
            "(38.8 databases per sample), zero gold failures, gold-vs-gold "
            "1,034/1,034, largest 92,450 rows and slowest 0.359s. Against a 5s "
            "deadline and a 500,000-row cap, raised from 100,000 because the "
            "distilled suite put a real gold within 7.5% of the old one. Prompt "
            "follows Rajkumar et al. 2022 (arXiv:2204.00498) CREATE TABLE + 3 "
            "example rows; upstream's trailing bare SELECT becomes a "
            "fenced-block instruction because a chat turn cannot end mid-token. "
            "Upstream runs one deterministic pass per question; n=1 is the "
            "protocol, not just this task's default."
        ),
    ),
)
class SpiderZeroShotGenTask(
    Task[
        SpiderDatasetSample,
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
        db_dir: str | None = None,
        tables_json_path: str | None = None,
        test_suite_db_dir: str | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._db_dir = db_dir
        self._tables_json_path = tables_json_path
        self._test_suite_db_dir = test_suite_db_dir

    def _staged(self, attribute: str, override_value: str | None) -> str:
        """Resolve a staged path from the constructor or the dataset.

        Mirrors ``SciCodeZeroShotGenTask``'s handling of its h5: the override
        exists for tests and out-of-tree wiring, the dataset is the normal
        source, and an unresolvable path is a loud stop rather than a run that
        grades nothing.
        """
        resolved = override_value or getattr(self.dataset, attribute, None)
        if not resolved or not os.path.exists(resolved):
            raise ValueError(
                f"Spider needs {attribute!r} but it did not resolve to an "
                f"existing path (got {resolved!r}). Stage the data with "
                "'sieval dataset download spider', or pass it to the task."
            )
        return resolved

    @property
    def db_dir(self) -> str:
        return self._staged("db_dir", self._db_dir)

    @property
    def tables_json_path(self) -> str:
        return self._staged("tables_json_path", self._tables_json_path)

    @property
    def test_suite_db_dir(self) -> str:
        """The distilled databases behind the headline metric.

        A second staged directory rather than a subdirectory of ``db_dir``: it
        ships as its own 1.3 GB archive from its own repository, and it holds a
        *different* database per name — same schema, different rows — so putting
        it beside the shipped one under a single root would make the two easy to
        confuse in exactly the place where confusing them silently changes the
        score.
        """
        return self._staged("test_suite_db_dir", self._test_suite_db_dir)

    def _db_path(self, db_id: str) -> str:
        return os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")

    @override
    async def preprocess(self, raw, ctx):
        prompt = build_prompt(self._db_path(raw["db_id"]), raw["question"])
        return build_prompt_record(
            [{"role": "user", "content": prompt}],
            reference=raw["query"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        return build_prediction_record([extract_sql(text) for text in inf.texts])

    @override
    async def feedback(self, post, ctx):
        # Imported here, not at module scope: this pulls the vendored parser and
        # with it nltk and sqlparse, both behind the optional `spider` group.
        # Importing a task module registers it, and registration is paid by
        # `sieval task list`, the meta index, and any run that fails before
        # grading -- none of which should need the group installed.
        from ._spider_test_suite import grade_one

        _ensure_punkt_tab()
        raw = ctx.raw_sample
        db_id = raw["db_id"]
        gold = raw["query"]
        db_path = self._db_path(db_id)
        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            prediction = rollout.get("prediction")
            try:
                # An unextracted answer is graded as the empty string rather
                # than skipped: upstream scores a prediction it cannot parse
                # against an empty parse, so a miss is a wrong answer.
                graded = await run_cpu_bound(
                    grade_one,
                    db_path,
                    self.tables_json_path,
                    self.test_suite_db_dir,
                    db_id,
                    prediction or "",
                    gold,
                    timeout=GRADE_TIMEOUT,
                )
            except TimeoutError:
                # A grade that could not be computed IN TIME stays a wrong
                # answer -- the prediction is a shape the grader cannot bound,
                # which is the model's problem, and `report` charges fails to
                # the denominator either way. Every OTHER exception propagates
                # and the sample lands in `fails` as `exception::<class>`, which
                # `grade_one` depends on rather than merely tolerates: it
                # *raises* on a gold it cannot parse, our bug and not a model
                # failure, so swallowing here would record that as the model
                # answering wrongly. SQL that will not run never reaches this
                # path -- `grade_one` scores it `False` and names the reason in
                # `error`, which is what `n_execution_errors` counts.
                #
                # This is the one place the test-suite fan-out is visible as a
                # risk rather than a cost. Every statement is individually
                # bounded, but a sample now runs ~78 of them (gold and
                # prediction against up to 60 distilled databases), so their sum
                # can exceed the per-sample budget where two statements never
                # could -- and a prediction slow enough to do that is scored
                # wrong even if it is right. The slowest sample of the gold pass
                # takes 1.08 s against a 30 s budget, so the exposure is real but
                # far from the measured range. It cannot be reduced without
                # changing the metric -- a shorter per-statement deadline would
                # bind on a real gold -- so it is stated and counted rather than
                # designed away.
                logger.warning(
                    "Grading sample {} exceeded {}s and was scored wrong; every "
                    "statement is individually bounded, so the cost is either in "
                    "parsing the prediction or in the ~78 executions the "
                    "test-suite metric fans out to.",
                    ctx.sample_id,
                    GRADE_TIMEOUT,
                )
                graded = {
                    "test_suite": False,
                    "exact_match": False,
                    "execution": False,
                    "hardness": None,
                    # `None`, not `False`: on this path the parser may never
                    # have run, and `False` would charge a timeout to
                    # `n_parser_rejected` -- the one number that makes the two
                    # parse-gated columns readable. Measured at 30.3% (a
                    # gpt-5.4-mini dev pass) and 57.2% (Qwen3.5-397B), it is
                    # what explains a 43pp gap between the headline and
                    # `execution_accuracy`, so a counter that quietly absorbs
                    # unrelated failures is worse than one that omits them.
                    "parsed": None,
                    "error": f"TimeoutError: grading exceeded {GRADE_TIMEOUT}s",
                    "test_suite_error": (
                        f"TimeoutError: grading exceeded {GRADE_TIMEOUT}s"
                    ),
                }
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    # `correct` is the headline metric, so it is the test-suite
                    # verdict -- the only axis comparable across tasks, and the
                    # one upstream ranks on.
                    bool(graded["test_suite"]),
                    metrics={
                        "test_suite": bool(graded["test_suite"]),
                        "execution": bool(graded["execution"]),
                        "exact_match": bool(graded["exact_match"]),
                    },
                    extra={
                        "hardness": graded["hardness"],
                        "error": graded["error"],
                        "test_suite_error": graded["test_suite_error"],
                        # Not a metric: the flag that says whether the two
                        # reference metrics above scored this prediction's
                        # answer or only its syntax. `None` when the parser did
                        # not run at all, so it is passed through rather than
                        # coerced -- `bool(None)` is exactly the misreport the
                        # timeout branch above avoids.
                        "parsed": graded["parsed"],
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        n_suite = 0
        n_exec = 0
        n_exact = 0
        n_execution_errors = 0
        n_parser_rejected = 0
        by_hardness: dict[str, list[int]] = {level: [0, 0] for level in HARDNESS_LEVELS}
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                metrics = rollout.get("metrics") or {}
                extra = rollout.get("extra") or {}
                passed = bool(metrics.get("test_suite"))
                n_suite += passed
                n_exec += bool(metrics.get("execution"))
                n_exact += bool(metrics.get("exact_match"))
                if extra.get("error") or extra.get("test_suite_error"):
                    n_execution_errors += 1
                # An explicit `False` only. A missing or `None` flag means the
                # parser never ran (a grading timeout), which is not a
                # rejection -- and that rollout is already counted by
                # `n_execution_errors`, so counting it here too would put one
                # event in two diagnostics.
                if extra.get("parsed") is False:
                    n_parser_rejected += 1
                bucket = by_hardness.get(extra.get("hardness") or "")
                if bucket is not None:
                    bucket[0] += passed
                    bucket[1] += 1

        # Denominator spans the full requested set: a pipeline failure produced
        # no gradeable answer and counts as wrong, matching upstream (whose
        # total is every dev example) and the *_gen family.
        total = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / total, 2)) if total else (lambda c: 0.0)
        metrics: dict[str, float | str] = {
            "score": rate(n_suite),
            "test_suite_accuracy": rate(n_suite),
            # Reference columns, both parse-gated -- read them next to
            # `n_parser_rejected`, never on their own. See the module docstring.
            "execution_accuracy": rate(n_exec),
            "exact_match": rate(n_exact),
            "n": float(total),
            "fails": float(len(fails)),
            # Predictions that would not run at all (syntax error, deadline,
            # row cap), on either path. They score 0 either way; the count
            # separates "wrong answer" from "no answer", which the headline
            # cannot.
            "n_execution_errors": float(n_execution_errors),
            # Predictions the pre-2020 parser refused. These are scored 0 by
            # both reference columns whatever SQLite returns for them, so the
            # count is what makes those two rates readable -- it is the size of
            # the gate, not a second error metric, and it does not touch
            # `score`. Named for the actor, since this task has two graders and
            # only one of them parses.
            "n_parser_rejected": float(n_parser_rejected),
            SCORE_KEY_FIELD: "test_suite_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Per-hardness rates are over rollouts actually GRADED in each bucket,
        # not the requested set: a failed sample never reveals which bucket its
        # gold belongs to. The paired count makes each denominator visible
        # rather than leaving four rates to be read as if they shared one.
        #
        # Split on the HEADLINE, matching upstream, which prints the breakdown
        # for whichever `--etype` it ran. One split rather than three: the
        # buckets exist to say where a model loses, and repeating them for two
        # parse-gated columns would mostly report where the parser loses.
        for level, (correct, seen) in by_hardness.items():
            metrics[f"test_suite_accuracy_{level}"] = (
                round(100 * correct / seen, 2) if seen else 0.0
            )
            metrics[f"n_{level}"] = float(seen)
        return metrics | health_metrics(finals)
