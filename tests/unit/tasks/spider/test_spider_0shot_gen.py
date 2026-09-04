"""Unit tests for spider_0shot_gen.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.spider import SpiderDataset
from sieval.tasks.spider.spider_0shot_gen import SpiderZeroShotGenTask, extract_sql
from tests.conftest import HandlerTransport


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording calls."""

    def __init__(self, reply: str, model: str = "mock"):
        self._reply = reply
        self.calls: list[str] = []
        super().__init__(model=model, api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.calls.append(str(req.input))
        return Response(
            texts=(self._reply,) * req.sampling.n,
            finish_reasons=("stop",) * req.sampling.n,
        )


# --- SQL extraction ---------------------------------------------------------


def test_extract_prefers_a_fenced_block():
    assert extract_sql("blah\n```sql\nSELECT 1\n```\ntrailing") == "SELECT 1"


def test_extract_accepts_an_unlabelled_fence():
    assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"


def test_extract_falls_back_to_a_bare_statement():
    assert extract_sql("Here you go: SELECT 3 FROM t") == "SELECT 3 FROM t"


def test_extract_takes_the_last_fence_when_several_are_present():
    """Models routinely show working, then give the answer last."""
    reply = "```sql\nSELECT 1\n```\nactually, better:\n```sql\nSELECT 2\n```"
    assert extract_sql(reply) == "SELECT 2"


def test_extract_handles_a_with_clause():
    extracted = extract_sql("```sql\nWITH c AS (SELECT 1) SELECT * FROM c\n```")
    assert extracted is not None
    assert extracted.startswith("WITH")


def test_extract_strips_a_trailing_semicolon():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


@pytest.mark.parametrize("label", ["sql", "SQL", "sqlite", "SQLite", "mysql", "pgsql"])
def test_extract_accepts_any_fence_label(label):
    """The prompt says "valid SQLite", so ```sqlite is a label models reach for.

    Matching only ```sql fell through to the raw reply, whose slice runs to the
    end of the string and carries the closing backticks into the statement —
    SQLite then fails it on `unrecognized token: "```"`, scoring a correct
    answer wrong.
    """
    assert extract_sql(f"```{label}\nSELECT 1\n```") == "SELECT 1"


def test_extract_never_returns_a_fence_marker():
    """Whatever the label, no extraction may carry backticks into the SQL."""
    for label in ("sql", "sqlite", "python", "text", ""):
        extracted = extract_sql(f"Answer:\n```{label}\nSELECT 1 FROM t\n```")
        assert extracted == "SELECT 1 FROM t"


def test_extract_skips_a_trailing_block_that_holds_no_sql():
    """A ```json note after the answer must not beat the SQL block.

    Guards the interaction with the permissive label above: once every label
    matches, the LAST block is no longer necessarily the answer.
    """
    reply = '```sql\nSELECT 1\n```\n```json\n{"note": "done"}\n```'
    assert extract_sql(reply) == "SELECT 1"


def test_extract_takes_the_last_unlabelled_dialect_fence():
    """The label must be matched, not merely survived.

    Truncating at a stray ``` rescues a single ```sqlite block, so it hides a
    fence pattern that recognises only ```sql. It cannot rescue this: with the
    blocks unmatched the reply is treated as prose and the FIRST statement
    wins, handing back the working instead of the answer.
    """
    reply = "```sqlite\nSELECT 1\n```\nactually, better:\n```sqlite\nSELECT 2\n```"
    assert extract_sql(reply) == "SELECT 2"


def test_extract_prefers_a_dialect_fence_over_prose_saying_select():
    """`select` as an English verb must not outrank a real fenced answer."""
    reply = "First I will select the right table.\n```sqlite\nSELECT 2 FROM t\n```"
    assert extract_sql(reply) == "SELECT 2 FROM t"


def test_extract_drops_an_unpaired_fence_marker_after_the_statement():
    """A lone ``` never reaches the SQL, even with no fence pair to match.

    A reply cut off at `max_tokens` mid-fence leaves an opener the fence
    pattern cannot consume, so the prose fallback runs with backticks still in
    the slice — the shape SQLite rejects as `unrecognized token`.
    """
    assert extract_sql("SELECT 1 FROM t\n```js") == "SELECT 1 FROM t"


def test_extract_cuts_prose_after_the_statement_terminator():
    reply = "SELECT count(*) FROM singer; This counts the singers."
    assert extract_sql(reply) == "SELECT count(*) FROM singer"


def test_extract_keeps_a_semicolon_inside_a_string_literal():
    """A `;` in a literal terminates nothing — splitting on it would truncate."""
    reply = "```sql\nSELECT * FROM t WHERE name = 'a;b'\n```"
    assert extract_sql(reply) == "SELECT * FROM t WHERE name = 'a;b'"


def test_extract_handles_a_doubled_quote_before_a_terminator():
    """SQLite escapes a quote by doubling it; the toggle must stay balanced."""
    reply = "```sql\nSELECT * FROM t WHERE name = 'it''s'; -- done\n```"
    assert extract_sql(reply) == "SELECT * FROM t WHERE name = 'it''s'"


def test_extract_returns_none_when_nothing_looks_like_sql():
    assert extract_sql("I cannot answer that.") is None


def test_extract_returns_none_for_an_empty_reply():
    assert extract_sql("") is None


@pytest.mark.parametrize(
    "comment",
    [
        "-- Find all cars with more than 4 cylinders",
        "-- Count singers with age over 20",
        "-- Step 1: select distinct countries",
        "/* select the youngest singer */",
        "/* join singers with concerts */",
    ],
)
def test_extract_ignores_a_keyword_inside_a_leading_comment(comment):
    """A comment holding "with"/"select" must not become the statement start.

    Both are ordinary English words, so a model that opens with a comment would
    otherwise have the slice begin mid-prose and reach SQLite as
    `with more than 4 cylinders SELECT ...` — one syntax error, scored a wrong
    answer and counted against `n_execution_errors`, so it reads as the model's
    fault rather than the extractor's. Parametrised over both comment syntaxes
    because they are masked by separate branches.
    """
    extracted = extract_sql(f"```sql\n{comment}\nSELECT a FROM t\n```")
    assert extracted is not None
    assert extracted.lower().startswith("select a from t")


def test_extract_keeps_a_comment_that_follows_the_statement():
    """Masking hides comments from the SEARCH, it does not delete them.

    SQLite accepts trailing commentary, and stripping it would be a second,
    unasked-for change to what reaches the grader.
    """
    extracted = extract_sql("```sql\nSELECT a FROM t -- the answer\n```")
    assert extracted == "SELECT a FROM t -- the answer"


def test_extract_does_not_treat_a_dash_dash_inside_a_literal_as_a_comment():
    """`'a--b'` is a value, so the rest of the line is still SQL."""
    reply = "```sql\nSELECT a FROM t WHERE note = 'a--b' AND x = 1\n```"
    assert extract_sql(reply) == "SELECT a FROM t WHERE note = 'a--b' AND x = 1"


@pytest.mark.parametrize(
    "prose",
    [
        "Here is a query with a join:",
        "You can do this with a subquery:",
        "I built it with care, and with two joins.",
        "Sure — starting with the singer table:",
    ],
)
def test_extract_ignores_the_english_word_with_before_bare_sql(prose):
    """Masking cannot reach this: on the fallback the prose is not a comment.

    The comment branch above handles `-- ... with ...`; unfenced prose has no
    marker to mask, so `WITH` is instead accepted only where a CTE can follow.
    Without that the slice reaches SQLite as `with a join: SELECT ...` — one
    syntax error, scored a wrong answer and charged to `n_execution_errors`, so
    it reads as the model's fault rather than the extractor's.
    """
    assert extract_sql(f"{prose} SELECT a FROM t") == "SELECT a FROM t"


def test_extract_returns_none_when_prose_says_with_and_holds_no_sql():
    """No statement at all beats a slice of prose that cannot run."""
    assert extract_sql("This has nothing to do with databases.") is None


@pytest.mark.parametrize(
    "cte",
    [
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "WITH x AS(SELECT 1) SELECT * FROM x",
        "WITH RECURSIVE t(n) AS (SELECT 1) SELECT * FROM t",
        "WITH x(a, b) AS (SELECT 1, 2) SELECT * FROM x",
        'WITH "my cte" AS (SELECT 1) SELECT * FROM "my cte"',
        "WITH [my cte] AS (SELECT 1) SELECT 1",
        "WITH `my cte` AS (SELECT 1) SELECT 1",
        "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a",
        "WITH x\n  AS (\n SELECT 1\n)\nSELECT * FROM x",
        "with x as (select 1) select * from x",
        "WITH x AS MATERIALIZED (SELECT 1) SELECT 1",
    ],
)
def test_extract_still_accepts_every_real_cte_form(cte):
    """The guard above may not cost a real CTE.

    Quoted names, a column list, `RECURSIVE`, `MATERIALIZED`, several CTEs and
    newlines between the parts are all SQLite-legal openings, and a `WITH`
    prediction is exactly the shape the pre-2020 parser already rejects — so
    losing one here would be invisible in `n_parser_rejected` and visible only
    as a lower headline.
    """
    assert extract_sql(cte) == cte


def test_extract_ignores_a_semicolon_inside_a_comment():
    """A `;` in a comment terminates nothing, so the statement is not cut short."""
    reply = "```sql\nSELECT a FROM t\n-- watch out; this is prose\nWHERE x = 1\n```"
    extracted = extract_sql(reply)
    assert extracted is not None
    assert extracted.endswith("WHERE x = 1")


def test_extract_returns_none_for_a_comment_only_reply():
    """A block that is nothing but commentary holds no statement to run."""
    assert extract_sql("```sql\n-- I need to select something, but cannot\n```") is None


# --- report -----------------------------------------------------------------


def _empty_dataset() -> SpiderDataset:
    return SpiderDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([])}))


def _task(**kwargs) -> SpiderZeroShotGenTask:
    return SpiderZeroShotGenTask(
        _empty_dataset(), _ScriptedChatModel(reply="```sql\nSELECT 1\n```"), **kwargs
    )


def _final(
    sample_id: int,
    *,
    test_suite: bool,
    execution: bool,
    exact: bool,
    hardness: str,
    error=None,
    test_suite_error=None,
    parsed: bool | None = True,
) -> TaskContext:
    """One graded rollout, with all three verdicts spelled out separately.

    `test_suite` has no default on purpose: it is the headline, so a fixture
    that forgot it would report zero rather than fail, and every test below
    would then agree with a report that had lost its own score.
    """
    return TaskContext(
        sample_id=sample_id,
        feedback_result=build_judgement_record(
            "SELECT 1",
            [
                build_rollout_judgement(
                    0,
                    # Production derives `correct` from the headline, not from
                    # the execution column; a fixture that derived it from the
                    # other one would hide a swap.
                    test_suite,
                    metrics={
                        "test_suite": test_suite,
                        "execution": execution,
                        "exact_match": exact,
                    },
                    extra={
                        "hardness": hardness,
                        "error": error,
                        "test_suite_error": test_suite_error,
                        "parsed": parsed,
                    },
                )
            ],
        ),
        postprocess_result=build_prediction_record(["SELECT 1"]),
    )


@pytest.mark.anyio
async def test_report_declares_score_key_and_denominator():
    metrics = await _task().report([], [])
    assert metrics["score_key"] == "test_suite_accuracy"
    assert metrics["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_empty_run_still_declares_both_fields():
    """The empty-run guard is a return path too."""
    metrics = await _task().report([], [])
    assert metrics["test_suite_accuracy"] == 0.0
    assert metrics["score"] == 0.0


@pytest.mark.anyio
async def test_score_is_copied_from_the_key_it_names():
    finals = [
        _final(0, test_suite=True, execution=True, exact=True, hardness="easy"),
        _final(1, test_suite=False, execution=False, exact=False, hardness="hard"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["score"] == metrics["test_suite_accuracy"] == 50.0


@pytest.mark.anyio
async def test_the_headline_is_the_test_suite_column_not_the_execution_one():
    """The point of the metric: it must be able to *disagree* with the others.

    Both reference columns are parse-gated, so on real data they read lower
    than the headline for predictions the pre-2020 parser refuses — and one
    reading the wrong key would still look plausible, because on a prediction
    the parser accepts all three usually agree. Only a rollout where they
    differ can tell, so this builds one: right by the test suite, wrong by
    both parse-gated columns.
    """
    finals = [_final(0, test_suite=True, execution=False, exact=False, hardness="easy")]
    metrics = await _task().report(finals, [])
    assert metrics["score"] == 100.0
    assert metrics["test_suite_accuracy"] == 100.0
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["exact_match"] == 0.0
    # And the reverse: agreeing on the one shipped database is exactly what the
    # metric exists to stop counting as correct.
    finals = [_final(0, test_suite=False, execution=True, exact=True, hardness="easy")]
    metrics = await _task().report(finals, [])
    assert metrics["score"] == 0.0
    assert metrics["execution_accuracy"] == 100.0


@pytest.mark.anyio
async def test_exact_match_is_reported_separately_from_execution():
    """The reference columns are distinct from each other, not just from score."""
    finals = [
        _final(0, test_suite=True, execution=True, exact=False, hardness="medium")
    ]
    metrics = await _task().report(finals, [])
    assert metrics["execution_accuracy"] == 100.0
    assert metrics["exact_match"] == 0.0


@pytest.mark.anyio
async def test_a_pipeline_failure_counts_as_wrong():
    """DENOMINATOR_REQUESTED: fails sit in the denominator, not outside it."""
    finals = [_final(0, test_suite=True, execution=True, exact=True, hardness="easy")]
    metrics = await _task().report(finals, [TaskContext(sample_id=1)])
    assert metrics["test_suite_accuracy"] == 50.0
    assert metrics["fails"] == 1.0
    assert metrics["n"] == 2.0


@pytest.mark.anyio
async def test_per_hardness_rates_carry_their_own_counts():
    finals = [
        _final(0, test_suite=True, execution=True, exact=True, hardness="easy"),
        _final(1, test_suite=False, execution=False, exact=False, hardness="easy"),
        _final(2, test_suite=True, execution=True, exact=True, hardness="extra"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["test_suite_accuracy_easy"] == 50.0
    assert metrics["n_easy"] == 2.0
    assert metrics["test_suite_accuracy_extra"] == 100.0
    assert metrics["n_extra"] == 1.0
    # A bucket nothing landed in reports zero over zero, not a crash.
    assert metrics["test_suite_accuracy_hard"] == 0.0
    assert metrics["n_hard"] == 0.0


@pytest.mark.anyio
async def test_the_hardness_split_follows_the_headline():
    """One split, on `score`'s own column — not on a parse-gated one.

    Same trap as the headline itself, and harder to see: a breakdown keyed on
    `execution` would still be labelled `test_suite_accuracy_*`, so it would
    read as the headline's own bucket while reporting where the parser loses.
    """
    finals = [_final(0, test_suite=True, execution=False, exact=False, hardness="easy")]
    metrics = await _task().report(finals, [])
    assert metrics["test_suite_accuracy_easy"] == 100.0


@pytest.mark.anyio
async def test_unexecutable_predictions_are_counted():
    """Separates 'wrong answer' from 'no answer', which the headline cannot."""
    finals = [
        _final(
            0,
            test_suite=False,
            execution=False,
            exact=False,
            hardness="easy",
            error="syntax error",
        ),
        _final(1, test_suite=False, execution=False, exact=False, hardness="easy"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["n_execution_errors"] == 1.0


@pytest.mark.anyio
async def test_a_failure_on_either_path_is_counted_once():
    """Two graders, one count — and a prediction only the new path could not run.

    The metrics now come from two executions, so a prediction can fail on the
    test-suite path while the shipped database happened to answer it. Counting
    only `error` would report that run as having had no execution failures at
    all.
    """
    finals = [
        _final(
            0,
            test_suite=False,
            execution=True,
            exact=True,
            hardness="easy",
            test_suite_error="result exceeded 500000 rows",
        ),
        # Both paths failing is still one unexecutable prediction, not two.
        _final(
            1,
            test_suite=False,
            execution=False,
            exact=False,
            hardness="easy",
            error="syntax error",
            test_suite_error="syntax error",
        ),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["n_execution_errors"] == 2.0


@pytest.mark.anyio
async def test_the_parser_gate_is_sized_without_touching_the_score():
    """`n_parser_rejected` is the size of the gate on the reference columns.

    It is what makes those two rates readable, so it must be published even on
    a run where the headline is perfect — and it must not move `score`, since
    the headline never consults the parser.
    """
    finals = [
        _final(
            0,
            test_suite=True,
            execution=False,
            exact=False,
            hardness="easy",
            parsed=False,
        ),
        _final(1, test_suite=True, execution=True, exact=True, hardness="easy"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["n_parser_rejected"] == 1.0
    assert metrics["score"] == 100.0
    assert metrics["execution_accuracy"] == 50.0


@pytest.mark.anyio
async def test_a_grading_timeout_is_not_counted_as_a_parser_rejection():
    """A timeout means the parser never ran, which is not a rejection.

    `n_parser_rejected` is the size of the gate on the two parse-gated columns
    — the number that explains a 43pp spread between the headline and
    `execution_accuracy` on a real dev pass — so a counter that absorbs
    unrelated failures stops being readable. The rollout is already counted by
    `n_execution_errors`, which is what makes double-counting it wrong rather
    than merely redundant.
    """
    finals = [
        _final(
            0,
            test_suite=False,
            execution=False,
            exact=False,
            hardness="easy",
            error="TimeoutError: grading exceeded 30.0s",
            test_suite_error="TimeoutError: grading exceeded 30.0s",
            parsed=None,
        ),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["n_parser_rejected"] == 0.0
    assert metrics["n_execution_errors"] == 1.0


@pytest.mark.anyio
async def test_an_absent_parsed_flag_is_not_counted_as_a_rejection():
    """Serialization drops a `None`, so a resumed record has no `parsed` key.

    The same rollout must be counted the same way whether it was just graded or
    hydrated from a shard — a resume that started charging timeouts to the
    parser gate would move a published diagnostic with no code change.
    """
    final = TaskContext(
        sample_id=0,
        feedback_result=build_judgement_record(
            "SELECT 1",
            [
                build_rollout_judgement(
                    0,
                    False,
                    metrics={
                        "test_suite": False,
                        "execution": False,
                        "exact_match": False,
                    },
                    # No `parsed` key at all, which is what a shard round-trip
                    # leaves behind for the timeout branch.
                    extra={"hardness": "easy", "error": None, "test_suite_error": None},
                )
            ],
        ),
        postprocess_result=build_prediction_record(["SELECT 1"]),
    )
    metrics = await _task().report([final], [])
    assert metrics["n_parser_rejected"] == 0.0


# --- the headline interval --------------------------------------------------


def _mixed_finals(n: int = 30) -> list[TaskContext]:
    """*n* graded samples that are neither all right nor all wrong.

    `wilson_interval` needs `0 < p < 1`, so a uniform fixture would publish no
    interval at all and every assertion below would pass vacuously.
    """
    return [
        _final(i, test_suite=i % 3 != 0, execution=False, exact=False, hardness="easy")
        for i in range(n)
    ]


@pytest.mark.anyio
async def test_the_headline_publishes_an_interval_with_its_population():
    metrics = await _task().report(_mixed_finals(), [])
    interval = metrics["score_ci95"]
    score = metrics["score"]
    assert isinstance(interval, list)
    assert isinstance(score, float)
    low, high = interval
    assert low < score < high
    assert metrics["n_problems"] == 30.0


@pytest.mark.anyio
async def test_every_published_interval_declares_its_unit():
    """An interval whose unit is undeclared cannot be told from another axis."""
    metrics = await _task().report(_mixed_finals(), [])
    assert metrics["ci95_units"] == {
        "score": "n_problems",
        "test_suite_accuracy": "n_problems",
    }


@pytest.mark.anyio
async def test_the_alias_carries_the_headline_interval_not_a_second_estimate():
    """`test_suite_accuracy` is `score` under its own name, so it is the same
    bound — a consumer keyed on the column name must not get a different one."""
    metrics = await _task().report(_mixed_finals(), [])
    assert metrics["test_suite_accuracy_ci95"] == metrics["score_ci95"]


@pytest.mark.anyio
async def test_the_interval_population_spans_the_requested_set():
    """`DENOMINATOR_REQUESTED`, so a fail is charged as wrong here too.

    The estimate is over the units that came back while scaled to the requested
    denominator, which is what keeps the interval and `score` on one population.
    """
    fails = [TaskContext(sample_id=900), TaskContext(sample_id=901)]
    metrics = await _task().report(_mixed_finals(), fails)
    assert metrics["n_problems"] == 32.0


@pytest.mark.anyio
async def test_a_uniform_run_still_gets_a_one_sided_bound():
    """p == 1 is not "no dispersion to report" — it is a bound touching 100.

    Five for five is weak evidence and the interval says so; dropping it would
    leave the strongest-looking column as the only one with no width.
    """
    finals = [
        _final(i, test_suite=True, execution=True, exact=True, hardness="easy")
        for i in range(5)
    ]
    metrics = await _task().report(finals, [])
    interval = metrics["score_ci95"]
    assert isinstance(interval, list)
    low, high = interval
    assert low < 100.0
    assert high == 100.0


@pytest.mark.anyio
async def test_an_empty_run_publishes_no_interval_and_no_population():
    """Emitted whole or not at all: a population with no bound beside it is a
    count nothing asked for, and there is nothing here to estimate over."""
    metrics = await _task().report([], [])
    assert "score_ci95" not in metrics
    assert "n_problems" not in metrics
    assert "ci95_units" not in metrics


# --- grading failure: only a TIMEOUT is a wrong answer -----------------------


class _Raiser:
    """An async `run_cpu_bound` stand-in that always raises *exc*.

    Counts its calls, so a test cannot pass because the patch was never reached —
    which is what a rename of the grading call site would otherwise look like. A
    class rather than a closure with a function attribute, which `ty` rejects.
    """

    def __init__(self, exc: type[BaseException]):
        self._exc = exc
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise self._exc("grader stub")


def _gradeable(tmp_path) -> tuple[SpiderZeroShotGenTask, dict]:
    """A task with its paths staged, for tests that stub the grader out.

    All three paths must exist — the staged-path properties check that before
    handing any of them to the grader — but none needs contents, since these
    tests never let `grade_one` run. Staging is not optional even here: a
    missing path raises `ValueError` from `_staged`, which is *also* one of the
    classes the propagation test parametrises, so a fixture that skipped one
    would let that param pass without the grader ever being called.
    """
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    suite_dir = tmp_path / "test_suite" / "database"
    suite_dir.mkdir(parents=True)
    tables = tmp_path / "tables.json"
    tables.write_text("[]")
    task = _task(
        db_dir=str(db_dir),
        tables_json_path=str(tables),
        test_suite_db_dir=str(suite_dir),
    )
    return task, {"db_id": "concert_singer", "query": "SELECT count(*) FROM singer"}


@pytest.mark.anyio
async def test_a_grader_timeout_is_a_wrong_answer(tmp_path, monkeypatch):
    """The half that stays swallowed — a prediction the grader cannot bound.

    `report` charges fails to the denominator either way, so propagating this
    one would only trade a truthful number for a scarier-looking one. It is
    still recorded in `error`, so `n_execution_errors` separates it from a
    prediction that merely ran and disagreed.
    """
    task, raw = _gradeable(tmp_path)
    stub = _Raiser(TimeoutError)
    monkeypatch.setattr("sieval.tasks.spider.spider_0shot_gen.run_cpu_bound", stub)
    post = build_prediction_record(["SELECT count(*) FROM singer"])
    _, judgement = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
    )
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["metrics"]["test_suite"] is False
    assert rollout["metrics"]["execution"] is False
    assert rollout["extra"]["error"].startswith("TimeoutError")
    # Recorded on both paths: one grading call computes both, so a timeout
    # leaves neither measured, and `n_execution_errors` reads either key.
    assert rollout["extra"]["test_suite_error"].startswith("TimeoutError")
    assert stub.calls > 0, "the grading call site moved; this test intercepted nothing"


@pytest.mark.parametrize("exc", [ValueError, AttributeError, ImportError, OSError])
@pytest.mark.anyio
async def test_a_broken_grader_propagates_instead_of_scoring_zero(
    exc, tmp_path, monkeypatch
):
    """A grader that is BROKEN rather than slow must not read as a wrong answer.

    `grade_one` *relies* on this rather than merely tolerating it: it raises
    `ValueError` on a gold it cannot parse — our bug, not a model failure — and
    a missing database file surfaces as `OSError` from the same call. Swallowed,
    both scored the sample 0 and left `fails` at 0, so a staging mistake or a
    defect in the vendored parser produced a low score on a run that looked
    clean. Propagated, the runner writes `exception::<class>` on the sample and
    `fails` becomes the signal.
    """
    task, raw = _gradeable(tmp_path)
    stub = _Raiser(exc)
    monkeypatch.setattr("sieval.tasks.spider.spider_0shot_gen.run_cpu_bound", stub)
    post = build_prediction_record(["SELECT count(*) FROM singer"])
    with pytest.raises(exc):
        await task.feedback(
            post, TaskContext(sample_id=0, raw_sample=raw, postprocess_result=post)
        )
    assert stub.calls > 0, "the grading call site moved; this test intercepted nothing"


# --- staged-path resolution -------------------------------------------------


@pytest.mark.parametrize(
    "attribute", ["db_dir", "tables_json_path", "test_suite_db_dir"]
)
def test_a_missing_staged_path_is_a_loud_stop(attribute):
    """A run that cannot find its data must not grade zeros silently.

    Three paths, three separate stops, and the headline's own archive is the
    one that would be easiest to lose: it is a 1.3 GB download the other two do
    not need, so a machine with the dataset staged and the suite absent is the
    ordinary way this goes wrong. Matched on the *quoted* attribute name rather
    than a bare substring — `db_dir` occurs inside `test_suite_db_dir`, so the
    loose pattern passes whichever path is actually missing.
    """
    with pytest.raises(ValueError, match=f"'{attribute}'"):
        _ = getattr(_task(), attribute)


GOLD = "SELECT count(*) FROM singer"

_TABLES_JSON = (
    '[{"db_id": "concert_singer", "table_names_original": ["singer"], '
    '"table_names": ["singer"], '
    '"column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"]], '
    '"column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"]], '
    '"column_types": ["text", "number", "text", "number"], '
    '"foreign_keys": [], "primary_keys": [1]}]'
)


def _write_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id int, name text, age int)")
    conn.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _pipeline(tmp_path, reply: str) -> tuple[SpiderZeroShotGenTask, dict]:
    """A task staged with real data for both graders, and its one dev row.

    Miniature version of the archive's own construction: the shipped database
    and the first distilled one hold singers who are all over 30, the second
    holds one who is not. A prediction that filters on age therefore agrees
    with the gold everywhere except `_1` — which is the only place the two
    metrics can be told apart, so it is what makes an end-to-end assertion on
    the headline mean anything.
    """
    db_dir = tmp_path / "database"
    _write_db(
        db_dir / "concert_singer" / "concert_singer.sqlite",
        [(1, "Joe", 41), (2, "Ann", 52)],
    )
    suite_dir = tmp_path / "test_suite" / "database"
    _write_db(
        suite_dir / "concert_singer" / "concert_singer.sqlite",
        [(1, "Joe", 41), (2, "Ann", 52)],
    )
    _write_db(
        suite_dir / "concert_singer" / "concert_singer_1.sqlite",
        [(1, "Joe", 41), (2, "Ann", 22)],
    )
    tables = tmp_path / "tables.json"
    tables.write_text(_TABLES_JSON)

    row = {
        "db_id": "concert_singer",
        "query": GOLD,
        "query_toks": [],
        "query_toks_no_value": [],
        "question": "How many singers do we have?",
        "question_toks": [],
    }
    dataset = SpiderDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])})
    )
    task = SpiderZeroShotGenTask(
        dataset,
        _ScriptedChatModel(reply=reply),
        db_dir=str(db_dir),
        tables_json_path=str(tables),
        test_suite_db_dir=str(suite_dir),
    )
    return task, row


async def _run(task, row) -> dict:
    """preprocess -> infer -> postprocess -> feedback -> report, no stubs."""
    ctx = TaskContext(sample_id=0, raw_sample=row)
    pre = await task.preprocess(row, ctx)
    assert "CREATE TABLE singer" in pre["prompt"][0]["content"]
    assert pre["reference"] == GOLD
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    finalize, judgement = await task.feedback(post, ctx)
    assert finalize is True
    scored = TaskContext(
        sample_id=0,
        raw_sample=row,
        feedback_result=judgement,
        postprocess_result=post,
    )
    return {
        "prediction": post["rollouts"][0].get("prediction"),
        "rollout": judgement["rollouts"][0],
        "metrics": await task.report([scored], []),
    }


@pytest.mark.anyio
async def test_full_pipeline_scores_a_correct_answer(tmp_path):
    """Runs the real stages against real (tiny) databases rather than stubbing
    the grader, so a break anywhere in the wiring shows up here."""
    task, row = _pipeline(tmp_path, reply=f"```sql\n{GOLD}\n```")
    result = await _run(task, row)

    assert result["prediction"] == GOLD
    rollout = result["rollout"]
    assert rollout["correct"] is True
    assert rollout["metrics"]["test_suite"] is True
    assert rollout["metrics"]["exact_match"] is True
    assert rollout["extra"]["hardness"] == "easy"

    metrics = result["metrics"]
    assert metrics["score"] == 100.0
    assert metrics["test_suite_accuracy"] == 100.0
    assert metrics["execution_accuracy"] == 100.0
    assert metrics["exact_match"] == 100.0
    assert metrics["n_easy"] == 1.0


@pytest.mark.anyio
async def test_full_pipeline_fails_an_answer_that_only_agrees_coincidentally(
    tmp_path,
):
    """The end-to-end proof that the headline is the *new* metric.

    `WHERE age > 30` is not the question asked; it returns the gold's answer on
    the shipped database only because every singer there happens to be over 30.
    The pre-2020 reading has one database to ask and calls that correct — so it
    is reported, unchanged, as the reference column. The headline asks the
    distilled ones too, finds the disagreement, and scores it wrong. If the
    fan-out were not wired, or the report crowned the wrong column, this test
    is the one that says so rather than agreeing with both readings at once.
    """
    prediction = "SELECT count(*) FROM singer WHERE age > 30"
    task, row = _pipeline(tmp_path, reply=f"```sql\n{prediction}\n```")
    result = await _run(task, row)

    assert result["prediction"] == prediction
    rollout = result["rollout"]
    assert rollout["correct"] is False
    assert rollout["metrics"]["test_suite"] is False
    assert rollout["metrics"]["execution"] is True
    # It ran fine on both paths; disagreeing is not an execution error.
    assert not rollout["extra"]["error"]
    assert not rollout["extra"]["test_suite_error"]

    metrics = result["metrics"]
    assert metrics["score"] == 0.0
    assert metrics["test_suite_accuracy"] == 0.0
    assert metrics["execution_accuracy"] == 100.0
    assert metrics["n_execution_errors"] == 0.0


def test_db_path_is_built_from_the_staged_dir(tmp_path):
    db_dir = tmp_path / "database"
    (db_dir / "concert_singer").mkdir(parents=True)
    target = db_dir / "concert_singer" / "concert_singer.sqlite"
    sqlite3.connect(target).close()
    tables = tmp_path / "tables.json"
    tables.write_text("[]")

    task = _task(db_dir=str(db_dir), tables_json_path=str(tables))
    assert task._db_path("concert_singer") == str(target)
