"""Spider 1.0 — 0-shot generative text-to-SQL, execution- and match-scored.

Spider (Yu et al., EMNLP 2018) is the reference cross-domain text-to-SQL
benchmark. This task evaluates its **dev** split — 1,034 questions over 20
databases, the split the literature reports, because the real test set was held
out for years. Two metrics, both upstream's:

* **Execution accuracy** (headline) — run the predicted SQL and upstream's gold
  against the same database and compare results with ``eval_exec_match``'s
  column-keyed projection.
* **Exact set match** — upstream's clause-by-clause set comparison, at its
  ``DISABLE_VALUE = True`` default, so literal values are not compared. This is
  the metric upstream deprecated in 2020 in favour of test-suite accuracy;
  it is reported because papers still quote it, not because it is the better
  number.

**Test-suite accuracy is not implemented.** It is upstream's official metric
since 2020 but lives in a separate repository with its own distilled databases.
Its absence is a scope decision, not an oversight.

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

**Execution safety.** Upstream's ``eval_exec_match`` opens a *read-write*
connection and runs model-generated SQL with no timeout behind a bare
``except:``. Grading is synchronous on one shared event loop, so an unbounded
query stalls the session rather than one sample. This task therefore carries the
hardened reading — read-only immutable connection, ``ATTACH``/``DETACH`` denied,
and a progress-handler deadline that aborts inside SQLite. Per
``sieval/tasks/CLAUDE.md`` this is the one divergence that does **not** earn a
``_fixed`` variant: a variant exists so two readings can be compared, and the
unsafe reading is not one we will run. Details and the measured bounds live in
``sieval.tasks._spider_exec``.

Upstream is preserved everywhere safety does not object, including where it is
wrong: the comparison stays ``res_map`` equality rather than a plain result-set
compare, an unparseable prediction is still scored against upstream's empty
parse rather than skipped, and exact match runs *after* execution because
``eval_exact_match`` mutates the parse trees in place.

**Safety delta, measured (2026-08-22).** All three obligations the hardening
owes are discharged, none of them needing a model:

* **No bound binds.** Over all 1,034 dev golds the largest result is 20,662 rows
  and the slowest query 0.486 s, against a 100,000-row cap and a 5 s deadline.
  A gold-vs-gold pass scores 1,034/1,034 on both metrics with zero errors, and
  its hardness split (248 easy / 446 medium / 174 hard / 166 extra) reproduces
  Spider's published dev distribution.
* **Quantified score impact: 99.903% verdict parity.** Over 1,033 comparable
  pairs — each dev gold graded against a sibling row's gold from the same
  database, so the mix is realistic and executable without an API — the hardened
  executor and upstream's own ``eval_exec_match`` (called with its own module
  globals, not a reimplementation) agree 1,032 times and differ once. Upstream
  additionally crashed outright on one further pair.
* **Safety, not repair.** Both of those two cases are ``wta_1`` and both trace to
  the same cause, the UTF-8 text factory: upstream's decode error surfaces as
  ``False`` when it hits the prediction (caught by its bare ``except:``) and as a
  crash when it hits the gold. **The read-only connection, the ATTACH denial,
  the deadline and the row cap produced zero verdict differences.** Worst-case
  headline impact is 2 of 1,034 samples, 0.19 pp, and only on models that get
  those two questions right.

Target: published Spider dev execution accuracy for the model under test.

Measured against a published anchor: **not yet**, which is the only reason this
ships ``experimental`` rather than ``stable``. That run needs model access; the
safety work above does not, and is already done.

References:

* Paper: <https://arxiv.org/abs/1809.08887>
* Harness: <https://github.com/taoyds/spider>
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

from ._spider_exec import grade_one
from ._spider_schema import build_prompt

#: Upstream's four difficulty buckets, from `Evaluator.eval_hardness`.
HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")

_FENCE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_STATEMENT = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    """Pull one SQL statement out of a chat reply.

    Prefers the **last** fenced block: models routinely show working and give
    the answer last. Falls back to the text from the first ``SELECT``/``WITH``
    keyword onward. Returns ``None`` when nothing looks like SQL, which is what
    marks the rollout unextracted.
    """
    blocks = _FENCE.findall(text)
    candidate = blocks[-1] if blocks else text
    match = _STATEMENT.search(candidate)
    if match is None:
        return None
    statement = candidate[match.start() :].strip().rstrip(";").strip()
    return statement or None


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
    description="Cross-domain text-to-SQL; exact set match and execution accuracy.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "text-to-sql", "code-exec"),
    model_type="chat",
    # Flipped to "stable" once the safety delta and an alignment run exist.
    status="experimental",
    deps_group="spider",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="taoyds/spider",
        url="https://github.com/taoyds/spider/tree/b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c/",
        notes=(
            "Grader vendored byte-identical in sieval.community.spider except "
            "evaluation.py:29, a flat import that cannot resolve inside a "
            "package. Exact set match is upstream's DISABLE_VALUE=True default. "
            "Execution accuracy diverges for SAFETY ONLY: read-only immutable "
            "connection, ATTACH/DETACH denied, and a progress-handler deadline, "
            "because upstream runs model SQL on a read-write connection with no "
            "timeout behind a bare except. Comparison stays upstream's "
            "column-keyed res_map, and exact match runs after execution because "
            "eval_exact_match mutates the parse trees in place. Two further "
            "verdict-preserving divergences: get_schema is reproduced read-only "
            "(upstream opens read-write; dict equality asserted in tests), and a "
            "surrogateescape text factory, because wta_1.players.last_name is "
            "not valid UTF-8 and upstream fetches gold outside its except, so it "
            "dies on two dev examples rather than scoring them. SAFETY DELTA "
            "MEASURED: 99.903% verdict parity against upstream's own "
            "eval_exec_match over 1,033 comparable pairs (1,032 agree, 1 "
            "differs, plus 1 upstream crash); all three cases are wta_1 and all "
            "trace to the text factory — the read-only connection, ATTACH "
            "denial, deadline and row cap produced zero verdict differences. "
            "Worst-case headline impact 2/1,034 = 0.19pp. Bounds measured "
            "over all 1,034 dev golds: largest result 20,662 rows, slowest "
            "0.486s, against a 100,000-row cap and a 5s deadline. Prompt follows "
            "Rajkumar et al. 2022 (arXiv:2204.00498) CREATE TABLE + 3 example "
            "rows; upstream's trailing bare SELECT becomes a fenced-block "
            "instruction because a chat turn cannot end mid-token. Upstream runs "
            "one deterministic pass per question; n=1 is the protocol, not just "
            "this task's default."
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
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._db_dir = db_dir
        self._tables_json_path = tables_json_path

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
                logger.warning(
                    "Grading sample {} exceeded {}s and was scored wrong; both "
                    "queries are individually bounded, so the cost is likely in "
                    "parsing the prediction.",
                    ctx.sample_id,
                    GRADE_TIMEOUT,
                )
                graded = {
                    "exact_match": False,
                    "execution": False,
                    "hardness": None,
                    "error": f"TimeoutError: grading exceeded {GRADE_TIMEOUT}s",
                }
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    bool(graded["execution"]),
                    metrics={
                        "execution": bool(graded["execution"]),
                        "exact_match": bool(graded["exact_match"]),
                    },
                    extra={
                        "hardness": graded["hardness"],
                        "error": graded["error"],
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        n_exec = 0
        n_exact = 0
        n_execution_errors = 0
        by_hardness: dict[str, list[int]] = {level: [0, 0] for level in HARDNESS_LEVELS}
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                metrics = rollout.get("metrics") or {}
                extra = rollout.get("extra") or {}
                executed = bool(metrics.get("execution"))
                n_exec += executed
                n_exact += bool(metrics.get("exact_match"))
                if extra.get("error"):
                    n_execution_errors += 1
                bucket = by_hardness.get(extra.get("hardness") or "")
                if bucket is not None:
                    bucket[0] += executed
                    bucket[1] += 1

        # Denominator spans the full requested set: a pipeline failure produced
        # no gradeable answer and counts as wrong, matching upstream (whose
        # total is every dev example) and the *_gen family.
        total = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / total, 2)) if total else (lambda c: 0.0)
        metrics: dict[str, float | str] = {
            "score": rate(n_exec),
            "execution_accuracy": rate(n_exec),
            "exact_match": rate(n_exact),
            "n": float(total),
            "fails": float(len(fails)),
            # Predictions that would not run at all (syntax error, deadline,
            # row cap). They score 0 either way; the count separates "wrong
            # answer" from "no answer", which the headline cannot.
            "n_execution_errors": float(n_execution_errors),
            SCORE_KEY_FIELD: "execution_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Per-hardness rates are over rollouts actually GRADED in each bucket,
        # not the requested set: a failed sample never reveals which bucket its
        # gold belongs to. The paired count makes each denominator visible
        # rather than leaving four rates to be read as if they shared one.
        for level, (correct, seen) in by_hardness.items():
            metrics[f"execution_accuracy_{level}"] = (
                round(100 * correct / seen, 2) if seen else 0.0
            )
            metrics[f"n_{level}"] = float(seen)
        return metrics | health_metrics(finals)
