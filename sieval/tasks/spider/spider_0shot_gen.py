"""Spider 1.0 — 0-shot generative text-to-SQL, execution-graded.

Spider (Yu et al., EMNLP 2018) is the reference cross-domain text-to-SQL
benchmark. This task evaluates its **dev** split — 1,034 questions over 20
databases, the split the literature reports, because the real test set was held
out for years. Three metrics, all upstream's, and the headline is the one
upstream itself moved to in October 2020:

* **Test-suite accuracy** (headline) — prediction and gold are run against ~39
  *distilled* databases per question, generated to distinguish neighbouring
  queries, and the prediction must return the same rows on **every** one. Raw
  result sets under bag semantics; nothing is parsed.
* **Execution accuracy** — the pre-2020 metric, on the single shipped database,
  through ``eval_exec_match``'s column-keyed projection.
* **Exact set match** — upstream's clause-by-clause set comparison, at its
  ``DISABLE_VALUE = True`` default, so literal values are not compared.

The last two are reference columns, reported because papers still quote them.
**Both are scored through a parser** — a hand-written tokeniser over Spider's own
gold dialect — and a prediction it rejects is compared as an *empty* projection,
scoring 0 whatever SQLite returned for it. On the pinned dev data it accepts 100%
of golds and 30–59% of model predictions, so the gap between those two columns
and the headline is mostly dialect, not correctness. ``n_parser_rejected`` is
published beside them to make the size of that gate visible; ``score`` is
deliberately not one of them.

*Measured over two full dev passes* (2026-09-03):

===================  ==================  =========  ==========  =================
model                test_suite (score)  execution  exact_set   n_parser_rejected
===================  ==================  =========  ==========  =================
gpt-5.4-mini         66.05               24.47      21.76       724 / 1,034
Qwen3.5-397B-A17B    78.05               53.87      51.74       419 / 1,034
===================  ==================  =========  ==========  =================

They do not merely run low, they **rank differently**: they put Qwen ahead by
29.4 pp where the headline puts it ahead by 12.0 pp, because the weaker model is
also the one whose dialect the parser likes less (70% rejected against 41%). A
leaderboard built on either would report conformance as much as correctness.

**Prompt: Rajkumar et al. 2022**, the "Create Table + Select 3" format from
*Evaluating the Text-to-SQL Capabilities of Large Language Models*
(arXiv:2204.00498) — every table's ``CREATE TABLE`` plus three example rows,
then the question. Spider predates LLM prompting and has no canonical prompt of
its own; this is the most-cited LLM-era convention, which is what makes the
number comparable to published work. **One divergence**: upstream's prompt ends
in a bare ``SELECT`` for a completion model, and a chat turn cannot end
mid-token, so the prompt asks for a fenced ``sql`` block instead — the one reason
a chat-mode score is not bit-comparable to the paper's Codex figures. A
completion-faithful ``spider_0shot_base_gen`` sibling is where those belong.

**Execution safety.** Both upstreams open a *read-write* ``sqlite3.connect`` and
run model SQL with no timeout behind a bare ``except:``. Grading is synchronous
on one shared event loop, so an unbounded query stalls the session rather than
one sample. This task therefore carries the hardened reading on both paths —
read-only immutable connection, ``ATTACH``/``DETACH`` denied, a progress-handler
deadline and a row cap. Per ``sieval/tasks/CLAUDE.md`` this is the one divergence
that does **not** earn a ``_fixed`` variant: a variant exists so two readings can
be compared, and the unsafe reading is not one we will run. The bounds live in
``_spider_sqlite``; what the test-suite path adds is in ``_spider_test_suite``.

Only *execution* is ours. Every comparison is upstream's own bytes, and upstream
is preserved everywhere safety does not object, including where it is wrong: an
unparseable prediction is still scored against upstream's empty parse, and exact
match runs *after* execution because ``eval_exact_match`` mutates the parse trees
in place.

**What the hardening costs, measured.** All three obligations it owes are
discharged on both paths, none of them needing a model.

*No bound binds.* All 1,034 dev golds run on the shipped databases (largest
20,662 rows, slowest 0.486 s), and all 40,167 gold executions across the
distilled suite succeed — 38.8 databases per sample, **zero gold failures**,
largest 92,450 rows, slowest 0.359 s — against a 5 s deadline and a 500,000-row
cap (raised from 100,000 because a real gold sat 7.5% under it; see
``DEFAULT_MAX_ROWS``). Gold-vs-gold scores 1,034/1,034 on every metric, and its
hardness split (248 easy / 446 medium / 174 hard / 166 extra) reproduces Spider's
published dev distribution.

*Safety, not repair* (pre-2020 path, 2026-08-22). Against upstream's own
``eval_exec_match`` over 1,033 comparable gold pairs the hardened executor agrees
1,032 times, differs once, and survives one outright upstream crash. All three
are ``wta_1`` and trace to the UTF-8 text factory: **the read-only connection,
the ATTACH denial, the deadline and the row cap produced zero verdict
differences.** Worst case 2 of 1,034 samples, 0.19 pp.

*The headline path does not inherit that* (2026-09-03). Its upstream sets a lossy
``b.decode(errors="ignore")``, and lossy is the direction that can move a
verdict. Measured: across all 715 databases the dev set reaches, exactly two
values are not valid UTF-8, and ``ignore`` is injective over every distinct value
in their two columns (0 collisions in 41,324), so both factories induce the same
equality relation. Details in ``_spider_sqlite.open_readonly``.

Target: published Spider dev test-suite accuracy for the model under test, and
the two reference columns against papers that quote them.

*Port vs upstream, end to end: 2,387 pairs, zero divergences* (2026-09-03).
Every headline verdict was compared against upstream's own ``eval_exec_match`` —
called with its own module globals over the same ``.sqlite`` set, at the same two
flags — across three independent prediction sets: one full dev pass per model
above (1,034 + 1,031 graded rollouts; the Qwen pass is an earlier one than the
row tabulated) and upstream's **own** shipped example predictions
(``evaluation_examples/predict.txt``, 322 pairs over 4 db_ids). All 2,387 agree,
with zero upstream crashes and zero gold failures, the last by construction: the
port *raises* on a gold it cannot run, so a clean pass over ~39 databases per
sample is itself the evidence. This is the anchor the named-divergence
measurements above cannot supply — they show every cause we can *name* measures
zero, which is not the same as having compared the two implementations. The shape
of that comparison is pinned hermetically in the tests, on both paths.

``status="experimental"`` is the **terminal** status here rather than a
placeholder: there is **no published number to compare against**, and that is a
property of Spider, not a gap in this work. Its leaderboard is almost entirely
fine-tuned systems, so it does not compare to a 0-shot chat model, and the one
published figure using **this** prompt (Rajkumar's ``code-davinci-002``) is a
retired completion model — exactly the divergence documented above. Only a
completion-faithful ``spider_0shot_base_gen`` sibling could carry a
``Target:``/``Measured:`` block. Not "unvalidated"; validated against the only
reference that exists, with a stated limit — the reading
``gsm1k_kshot_base_gen`` uses for the same status. The two dev passes above are
reported as measurements, not as an alignment claim.

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
    interval_metrics,
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
#: ``WITH`` opening a real CTE: a name (bare or quoted), an optional column
#: list, then ``AS``. Matched at the keyword to tell a CTE from the English word.
_CTE = re.compile(
    r"""WITH\s+(?:RECURSIVE\s+)?      # the keyword, and SQLite's one modifier
        (?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[\w$]+)   # the CTE's name, however quoted
        \s*(?:\([^()]*\))?            # its optional column list
        \s+AS\b""",
    re.IGNORECASE | re.VERBOSE,
)


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


def _statement_start(masked: str) -> int | None:
    """Offset where the SQL begins in *masked*, or ``None`` if none does.

    ``with`` and ``select`` are ordinary English words, so the first occurrence
    of either is not necessarily the statement. Two things follow, and both are
    reachable but were unobserved across the two dev runs:

    * The search runs over the **comment-masked** text, so a model opening with
      ``-- Find all cars with more than 4 cylinders`` does not start the
      statement at that ``with``.
    * A ``WITH`` is accepted only where a CTE can actually follow it. Masking
      cannot help on the unfenced fallback, where the prose is not a comment:
      ``Here is a query with a join: SELECT ...`` would otherwise reach SQLite
      as ``with a join: SELECT ...``, one syntax error scored as a wrong answer.
      ``SELECT`` needs no such test — it opens a statement wherever it appears.
    """
    for match in _STATEMENT.finditer(masked):
        if match.group(1).upper() == "SELECT" or _CTE.match(masked, match.start()):
            return match.start()
    return None


def _statement_from(candidate: str) -> str | None:
    """Slice one statement out of *candidate*, or ``None`` if it holds no SQL."""
    masked = _mask_comments(candidate)
    start = _statement_start(masked)
    if start is None:
        return None
    statement, masked = candidate[start:], masked[start:]
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
    # Terminal, not provisional: Spider publishes no number this task could be
    # aligned against. The docstring's closing paragraph has why.
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
            "answer CORRECT against any gold returning no rows. "
            "TEXT-FACTORY DELTA MEASURED (2026-09-03, "
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
            "flags -- one full dev pass per model quoted above (1,034 + 1,031 "
            "graded rollouts; the Qwen pass is an earlier one than the row "
            "quoted here) plus upstream's own shipped "
            "evaluation_examples/predict.txt (322 pairs, 4 db_ids). "
            "Zero upstream crashes and zero gold failures, the latter asserted "
            "by construction since this port raises on a gold it cannot run. "
            "The shape of both comparisons is pinned hermetically in the tests. "
            "NOTHING FURTHER IS OWED, and experimental is the TERMINAL status "
            "rather than a placeholder: there is no published number to anchor "
            "against, which is a property of Spider and not a gap in this work. "
            "Its leaderboard is almost all fine-tuned systems, and the one "
            "published figure using this prompt (Rajkumar's code-davinci-002) "
            "is a retired completion model, which is the divergence this task "
            "documents -- so only a completion-faithful spider_0shot_base_gen "
            "sibling could carry a Target:/Measured: block. "
            "Bounds measured on both paths and no bound binds: 1,034 dev "
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
        # Beyond `float`: `score_key` names a column rather than measuring one,
        # `score_ci95` is a bound pair, and `ci95_units` maps metric to
        # population.
        dict[str, float | str | list[float] | dict[str, str]],
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
                # Only a TIMEOUT is swallowed: the prediction is a shape the
                # grader cannot bound, which is the model's problem. Every other
                # exception propagates to `fails` as `exception::<class>`, which
                # `grade_one` depends on rather than merely tolerates -- it
                # *raises* on a gold it cannot run, our bug, so swallowing would
                # record that as the model answering wrongly. SQL that will not
                # run never reaches here; `grade_one` scores it `False` and names
                # the reason in `error`.
                #
                # This is the one place the test-suite fan-out is a risk rather
                # than a cost. Every statement is individually bounded, but a
                # sample runs ~78 of them, so their sum can exceed the per-sample
                # budget where two never could -- and a prediction slow enough to
                # do that is scored wrong even if it is right. The slowest sample
                # of the gold pass takes 1.08 s against a 30 s budget, so the
                # exposure is real but far from the measured range, and it cannot
                # be reduced without changing the metric: a shorter per-statement
                # deadline would bind on a real gold.
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
                    # `None`, not `False`: the parser may never have run here,
                    # and `False` would charge a timeout to `n_parser_rejected`
                    # -- the one number that makes the two parse-gated columns
                    # readable. A counter that quietly absorbs unrelated
                    # failures is worse than one that omits them.
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
                        # Not a metric: says whether the two reference metrics
                        # scored this prediction's answer or only its syntax.
                        # Passed through rather than coerced -- `bool(None)` is
                        # exactly the misreport the timeout branch avoids.
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
        #: Each sample's contribution to the headline, for the interval below.
        per_sample: list[float] = []
        for final in finals:
            passed_here = 0
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                metrics = rollout.get("metrics") or {}
                extra = rollout.get("extra") or {}
                passed = bool(metrics.get("test_suite"))
                n_suite += passed
                passed_here += passed
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
            per_sample.append(float(passed_here))

        # Denominator spans the full requested set: a pipeline failure produced
        # no gradeable answer and counts as wrong, matching upstream (whose
        # total is every dev example) and the *_gen family.
        total = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / total, 2)) if total else (lambda c: 0.0)
        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": rate(n_suite),
            "test_suite_accuracy": rate(n_suite),
            # Reference columns, both parse-gated -- read them next to
            # `n_parser_rejected`, never on their own. See the module docstring.
            "execution_accuracy": rate(n_exec),
            "exact_match": rate(n_exact),
            "n": float(total),
            "fails": float(len(fails)),
            # Predictions that would not run at all (syntax error, deadline,
            # row cap), on either path. Separates "wrong answer" from "no
            # answer", which the headline cannot.
            "n_execution_errors": float(n_execution_errors),
            # The size of the parse gate: these score 0 on both reference
            # columns whatever SQLite returns for them, which is what makes
            # those two rates readable. Not an error metric, and it does not
            # touch `score`. Named for the actor -- this task has two graders
            # and only one of them parses.
            "n_parser_rejected": float(n_parser_rejected),
            SCORE_KEY_FIELD: "test_suite_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Per-hardness rates are over rollouts actually GRADED in each bucket,
        # not the requested set -- a failed sample never reveals which bucket its
        # gold belongs to -- so each carries its own count. Split on the HEADLINE
        # only, matching upstream, which prints the breakdown for whichever
        # `--etype` it ran: repeating it for two parse-gated columns would mostly
        # report where the parser loses.
        for level, (correct, seen) in by_hardness.items():
            metrics[f"test_suite_accuracy_{level}"] = (
                round(100 * correct / seen, 2) if seen else 0.0
            )
            metrics[f"n_{level}"] = float(seen)
        grouping = self.problem_groups(finals)
        # Clustered on problems, over the REQUESTED denominator, so a fail is
        # charged as wrong here exactly as it is in `score`. Only the headline
        # gets one: the two reference columns are gated by a parser whose
        # rejections are not sampling noise, so an interval on them would
        # describe the wrong source of variation, and the per-hardness rates are
        # a breakdown rather than a headline.
        return (
            metrics
            | health_metrics(finals)
            | interval_metrics(
                per_sample,
                denominator=total,
                group_keys=None if grouping is None else grouping.keys,
                n_problems=None if grouping is None else grouping.n_problems,
                # `test_suite_accuracy` is `score` under its own name, so it
                # carries the same interval rather than a second estimate.
                aliases=("test_suite_accuracy",),
            )
        )
