"""LiveOIBench — informatics-olympiad problems, graded in C++ with subtask credit.

Grading runs in the evaluator service (``source="liveoibench"``), which compiles
the submission with ``g++`` and runs every official test case under the
problem's own time and memory limits. Two things make this task unlike the other
code benchmarks here:

* **Partial credit.** A submission is not pass/fail — a subtask pays its points
  only when every test in it passes, and a problem's score is the sum. The
  evaluator therefore returns the whole verdict vector rather than a count.
* **A human baseline.** Every contest ships its real contestants' scores, so the
  report ranks the model inside the contest it was actually sitting.

Test data is read by the evaluator from the materialized corpus by path, not
shipped inline: a problem averages ~140 MB of test cases, which would otherwise
cross the wire once per rollout. Pass ``inline_tests=True`` when the evaluator
cannot see that volume.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import json
import os
import time
from collections import defaultdict
from typing import override

import httpx
from loguru import logger

from sieval.community.liveoibench.code_extractor import CodeExtractor
from sieval.community.liveoibench.payloads import load_code_bundle, load_subtasks
from sieval.community.liveoibench.prompts import build_prompt
from sieval.community.liveoibench.rankings import score_contest
from sieval.community.liveoibench.scoring import interprete_task_result, total_points
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    NonRetriableSampleError,
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
    budget_metrics,
    health_metrics,
)
from sieval.datasets import LiveOIBenchDatasetSample

_UPSTREAM = "https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12"


@sieval_task(
    name="liveoibench_0shot_gen",
    display_name="LiveOIBench (0-shot)",
    description=(
        "LiveOIBench — olympiad C++ problems, subtask-scored and ranked against "
        "human contestants."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "cpp", "code-exec"),
    model_type="chat",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="liveoibench",
        url=f"{_UPSTREAM}/src/judges/base_judge.py",
        notes=(
            "Prompt (judges/problem.py get_prompt), extraction (code_extractor.py), "
            "subtask scoring (base_judge.py interprete_task_result) and ranking "
            "(generate_rankings.py) are ported under sieval/community/liveoibench/. "
            "Execution matches BatchJudge: g++ -std=gnu++17 -Wall -O2 -pipe "
            "-static -g, "
            "RLIMIT_CPU/RLIMIT_AS at the problem's limits with upstream's 20% buffer, "
            "and its output comparison. No checker path: the published dataset ships "
            "no checkers/ directory, so upstream's own judge compares outputs directly "
            "and the 36 min-score subtasks collapse onto the all-or-nothing rule. "
            "Upstream samples n=8 per problem and reports the best; this task defaults "
            "to n=1, so match it with n=8. Scope: the 380 'batch' problems -- the 23 "
            "interactive ones are filtered out by the dataset (no interactor process "
            "yet), so numbers are over 380 problems and are NOT comparable to the "
            "paper's 403-problem table. Codeforces Elo is not computed."
        ),
    ),
)
class LiveOIBenchZeroShotGenTask(
    Task[
        LiveOIBenchDatasetSample,
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
        max_concurrency: int = 4,
        inline_tests: bool = False,
    ):
        """*n* is the sampling budget; upstream draws 8 and reports the best of
        them, which this task reproduces for whatever *n* it is given.

        *inline_tests* ships each problem's test cases in the request body
        instead of naming the directory. Only for an evaluator that cannot read
        the materialized corpus — it moves ~140 MB per rollout on average.
        """
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._inline_tests = inline_tests
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        prompt = build_prompt(
            task=raw["task_name"],
            statement=raw["problem_statement"],
            time_limit=raw["time_limit"],
            memory_limit=raw["memory_limit"],
            starter_codes=load_code_bundle(raw["starter_code"]),
        )
        # No system message: upstream's default run sends the prompt alone
        # (`--system_prompt` is opt-in and supplies a per-model string).
        return build_prompt_record(
            [{"role": "user", "content": prompt}],
            # No `reference`: the ground truth is a test suite and a subtask
            # rubric, described in the judgement's extra instead.
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        task_name = ctx.raw_sample["task_name"]
        # `extract_code` returns ("", 'empty') for an empty fence and
        # (None, 'not_found') for no fence; both normalize to None so
        # `extracted` reports a miss rather than an empty submission.
        return build_prediction_record(
            [
                CodeExtractor.extract_code(choice, task_name)[0] or None
                for choice in inf.texts
            ]
        )

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        subtasks = load_subtasks(raw["subtasks"])
        if not subtasks:
            # Without a rubric there is no score to compute, and re-inferring
            # cannot produce one.
            raise NonRetriableSampleError(
                f"{raw['problem_id']}: no subtasks to score against"
            )
        maximum = total_points(subtasks)
        payload_tests = self._test_payload(raw)

        rollouts: list[RolloutJudgement] = []
        best_index, best_key = None, None
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            verdicts, case_names, msg, resources = await self._grade(
                raw, rollout.get("prediction") or "", payload_tests, idx
            )

            results = [
                {"test_case": name, "correct": correct}
                for name, correct in zip(case_names, verdicts, strict=True)
            ]
            try:
                scored = interprete_task_result(results, subtasks)
            except KeyError as e:
                # A subtask naming a test the corpus does not contain: the two
                # parquets disagree, and no rollout of this sample can be scored.
                raise NonRetriableSampleError(
                    f"{raw['problem_id']}: subtask references unknown test case {e}"
                ) from e

            relative_score = (scored["score"] / maximum * 100) if maximum else 0.0
            rollouts.append(
                build_rollout_judgement(
                    idx,
                    # `correct` is the cross-task axis, so it is the olympiad's
                    # own all-or-nothing reading: every test passed.
                    scored["ace"],
                    score=float(scored["score"]),
                    metrics={
                        "relative_score": relative_score,
                        "tests_passed_pct": scored["tests_passed"] * 100,
                        "ace": scored["ace"],
                    },
                    extra={
                        "msg": msg,
                        "subtasks": scored["subtasks"],
                        "n_cases": len(case_names),
                        "n_passed": sum(verdicts),
                        "resources": resources,
                    },
                )
            )
            # Upstream ranks candidates by score, then relative_score, then
            # tests_passed_pct. Within one problem the first two are the same
            # ordering, so this is (score, tests_passed). Upstream's final
            # tiebreak is execution time, which is not recorded here; the lowest
            # rollout index wins instead, so the pick is deterministic.
            key = (scored["score"], scored["tests_passed"])
            if best_key is None or key > best_key:
                best_index, best_key = idx, key

        best = next(r for r in rollouts if r["index"] == best_index)
        return True, build_judgement_record(
            None,  # the reference is a test suite plus a rubric, not a value
            rollouts,
            score=best["score"],
            metrics={
                "relative_score": best["metrics"]["relative_score"],
                "tests_passed_pct": best["metrics"]["tests_passed_pct"],
                "ace": best["metrics"]["ace"],
            },
            extra={
                "problem_id": raw["problem_id"],
                "contest_id": raw["contest_id"],
                "task_name": raw["task_name"],
                "total_points": maximum,
                "n_subtasks": len(subtasks),
                "best_rollout": best_index,
                "task_type": raw["task_type"],
            },
        )

    def _test_payload(self, raw) -> dict:
        """The request's test half: a directory to read, or the cases inline."""
        tests_dir = raw["tests_dir"]
        if not self._inline_tests:
            return {"test_dir": tests_dir}

        inputs, outputs, names = [], [], []
        for filename in sorted(f for f in os.listdir(tests_dir) if f.endswith(".in")):
            stem = filename[: -len(".in")]
            output_path = os.path.join(tests_dir, f"{stem}.out")
            if not os.path.exists(output_path):
                raise NonRetriableSampleError(
                    f"{raw['problem_id']}: test {stem!r} has no .out in {tests_dir}"
                )
            with open(
                os.path.join(tests_dir, filename), encoding="utf-8", errors="replace"
            ) as f:
                inputs.append(f.read())
            with open(output_path, encoding="utf-8", errors="replace") as f:
                outputs.append(f.read())
            names.append(stem)
        return {"test": {"inputs": inputs, "outputs": outputs, "names": names}}

    async def _grade(
        self, raw, code: str, payload_tests: dict, idx: int
    ) -> tuple[list[bool], list[str], str, dict]:
        """One rollout through the evaluator; returns verdicts and their names."""
        task_name = raw["task_name"]
        # Compiled with the submission when the problem ships one; the header has
        # to be on disk for the submission's own `#include` to resolve.
        files = load_code_bundle(raw["grader_code"])

        try:
            resp = await self._http_client.post(
                self._code_eval_api,
                json={
                    "uuid": f"{raw['problem_id']}-{idx}-{time.perf_counter_ns()}",
                    "source": "liveoibench",
                    "lang": "cpp",
                    # An unextractable answer is "" on the wire, so the evaluator
                    # still runs it and reports a compile error -- a real verdict
                    # rather than a skipped rollout.
                    "code": code,
                    "entry_filename": f"{task_name}.cpp",
                    "files": files,
                    # The problem's own limit, per case; the evaluator applies
                    # upstream's 20% buffer on top.
                    "timeout_per_case": float(raw["time_limit"]),
                    "memory_limit": int(raw["memory_limit"]),
                    **payload_tests,
                },
                # Every case is bounded, so this only has to outlast the whole
                # suite: the per-case wall the evaluator enforces is 120s, and a
                # problem can carry a few hundred cases.
                timeout=self._request_timeout(raw),
            )
            resp.raise_for_status()
            res = resp.json()
            msg = res["msg"]
            data = res["data"] or {}
        except Exception as e:
            logger.warning(
                "Evaluation error for {} rollout {}: [{}] {}",
                raw["problem_id"],
                idx,
                type(e).__name__,
                e,
            )
            raise

        verdicts = data.get("case_verdicts")
        case_names = data.get("case_names")
        if verdicts is None or case_names is None:
            # Reached an evaluator without the C++ path (or an older one): there
            # is no verdict vector to score subtasks from, and `status` alone
            # would silently score every partial solution as zero.
            raise RuntimeError(
                f"evaluator returned no per-case verdicts for {raw['problem_id']!r} "
                f"(msg: {msg}); it needs the liveoibench C++ source "
                "(vendor/code-evaluator/VENDORED.md)."
            )

        resources = {
            key: value
            for key, value in data.items()
            if key not in ("n_cases", "n_passed", "case_verdicts", "case_names")
        }
        return verdicts, case_names, msg, resources

    def _request_timeout(self, raw) -> float:
        """HTTP deadline, outside every budget the evaluator can spend.

        Compilation is bounded at the evaluator's 60s default and each case at
        its own 120s wall, which only a process burning no CPU can reach. The
        suite runs at most 4 cases at a time there, so the bound is derived from
        the case count rather than assumed.
        """
        try:
            n_cases = sum(1 for f in os.listdir(raw["tests_dir"]) if f.endswith(".in"))
        except OSError:
            n_cases = 100
        per_case = min(120.0, float(raw["time_limit"]) * 1.2 + 5.0)
        return 60.0 + per_case * n_cases + 30.0

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if not finals:
            return {
                "score": 0.0,
                "fails": float(len(fails)),
                "relative_score": 0.0,
                SCORE_KEY_FIELD: "relative_score",
                DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            }

        # A failed sample scores 0 rather than dropping out of the mean, so the
        # denominator is every sample requested.
        relative = (
            sum(f.feedback_result["metrics"]["relative_score"] for f in finals) / total
        )
        pass_rate = (
            sum(f.feedback_result["metrics"]["tests_passed_pct"] for f in finals)
            / total
        )
        ace_rate = (
            sum(1 for f in finals if f.feedback_result["metrics"]["ace"]) / total * 100
        )

        metrics: dict[str, float | str] = {
            "score": relative,
            "fails": float(len(fails)),
            "relative_score": relative,
            "pass_rate": pass_rate,
            "ace_rate": ace_rate,
            "n_problems": float(len(finals)),
            SCORE_KEY_FIELD: "relative_score",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        metrics |= self._human_metrics(finals)
        metrics |= budget_metrics(
            [f.feedback_result["n_rollouts"] for f in finals], n=self._n, k=self._n
        )
        return metrics | health_metrics(finals)

    def _human_metrics(self, finals) -> dict[str, float]:
        """Percentile and medals, per contest, against that contest's field.

        Upstream re-totals the humans over exactly the tasks the model was scored
        on, so a contest whose interactive problems were filtered out still
        compares like with like. Contests are averaged unweighted, as upstream
        averages its per-contest percentiles.
        """
        contestants = self._load_contestants()
        if not contestants:
            return {}

        by_contest: dict[str, dict[str, float]] = defaultdict(dict)
        for final in finals:
            extra = final.feedback_result["extra"]
            by_contest[extra["contest_id"]][extra["task_name"]] = final.feedback_result[
                "score"
            ]

        percentiles: list[float] = []
        medals: list[str] = []
        unmatched = 0
        for contest_id, model_scores in by_contest.items():
            contest = contestants.get(contest_id)
            if contest is None:
                unmatched += 1
                continue
            result = score_contest(
                contest["rankings"],
                model_scores,
                contest_id=contest_id,
                gold_cutoff=contest["gold_cutoff"],
                silver_cutoff=contest["silver_cutoff"],
                bronze_cutoff=contest["bronze_cutoff"],
            )
            if result["human_percentile"] is not None:
                percentiles.append(result["human_percentile"])
            if result["medal"] is not None:
                medals.append(result["medal"])

        if unmatched:
            logger.warning(
                "{} contest(s) had no contestant record; excluded from the "
                "human metrics.",
                unmatched,
            )

        metrics: dict[str, float] = {"n_contests": float(len(by_contest))}
        if percentiles:
            metrics["human_percentile"] = sum(percentiles) / len(percentiles)
            metrics["n_contests_ranked"] = float(len(percentiles))
        if medals:
            metrics["medal_rate"] = (
                sum(1 for m in medals if m in ("Gold", "Silver", "Bronze"))
                / len(medals)
                * 100
            )
            metrics["gold_rate"] = (
                sum(1 for m in medals if m == "Gold") / len(medals) * 100
            )
        return metrics

    def _load_contestants(self) -> dict[str, dict]:
        """The contestant table, keyed by contest, or ``{}`` when unavailable."""
        path = getattr(self.dataset, "contestants_path", None)
        if not path or not os.path.exists(path):
            logger.warning(
                "No contestant results at {}; reporting scores without human ranking.",
                path,
            )
            return {}

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        contestants: dict[str, dict] = {}
        for row in table.to_pylist():
            try:
                rankings = json.loads(row["contestants_ranking"] or "[]")
            except json.JSONDecodeError:
                continue
            contestants[row["contest_id"]] = {
                "rankings": rankings,
                "gold_cutoff": row["gold_cutoff"],
                "silver_cutoff": row["silver_cutoff"],
                "bronze_cutoff": row["bronze_cutoff"],
            }
        return contestants

    @override
    async def shutdown(self):
        await self._http_client.aclose()
