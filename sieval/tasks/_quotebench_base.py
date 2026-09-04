"""Shared half of the QuoteBench contract tasks.

One benchmark, two generation contracts. A contract is not a prompt variant: it
declares what happens to the reply between the model and the shell, and the
paper's finding is that two matched command paths can score alike while ranking
models differently. So each contract is its own registered task, and everything
except the system prompt and the transport name lives here.

sieval executes nothing. The reply is POSTed to the vendored code-evaluator's
`quotebench` source, which builds the task's filesystem fixture, runs one
`bash -c` payload inside it, and returns the exact-final-state verdict plus a
failure class. Both sides vendor upstream's task definitions; the response
carries the digest the evaluator graded with, and this module refuses to score a
verdict whose digest does not match the one the prompts were built from.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, override

import httpx
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    interval_metrics,
)
from sieval.datasets import QuoteBenchDatasetSample

#: The modules that define what is asked and what is accepted. Must match
#: `exec_quotebench._DIGEST_MODULES` in the evaluator, including the order.
_DIGEST_MODULES = ("core.py", "scenarios.py", "shellesc.py")

#: Upstream's failure taxonomy (`harness.classify`), in the order it tests them.
FAILURE_CLASSES = (
    "pass",
    "environment-invalid",
    "timeout",
    "shell-syntax",
    "tool-error",
    "runtime-error",
    "silent-wrong",
)


class DigestMismatch(RuntimeError):
    """The evaluator graded against different task definitions than we prompted from.

    Not a wrong answer and not a transient fault: every number a run produced
    after this point would be a mix of two fixture sets, and would still look
    plausible. Raising is the only honest outcome.
    """


@lru_cache(maxsize=1)
def local_digest() -> str:
    """sha256 over sieval's own copy of the task definitions."""
    import sieval.community.quotebench as pkg

    assert pkg.__file__ is not None
    root = Path(pkg.__file__).parent
    digest = hashlib.sha256()
    for name in _DIGEST_MODULES:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def assert_digest(*, local: str, remote: str | None) -> None:
    """Refuse to score unless both sides hold the same task definitions."""
    if remote is None:
        raise DigestMismatch(
            "evaluator returned no scenarios_digest: it predates the quotebench "
            "source, or is a build without it"
        )
    if remote != local:
        raise DigestMismatch(
            f"evaluator graded against scenarios {remote[:12]}, we prompted from "
            f"{local[:12]} -- one of the two vendored copies has moved"
        )


class QuoteBenchTask(
    Task[
        QuoteBenchDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one; `list[float]` carries an interval, and
        # `dict[str, str]` the `ci95_units` map naming each interval's unit.
        dict[str, float | str | list[float] | dict[str, str]],
    ]
):
    """Abstract: a leaf sets SYSTEM_PROMPT and CONTRACT and nothing else."""

    #: Upstream's system prompt for this contract, used verbatim.
    SYSTEM_PROMPT: ClassVar[str]
    #: Transport name on the wire. Upstream's released rollouts spell these
    #: `raw` and `nested`; its own CLI spells the second `nested-shell` and
    #: rejects `nested`, so the released spelling is the one that travels.
    CONTRACT: ClassVar[str]

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        n: int = 1,
        max_concurrency: int = 4,
        grade_timeout: float = 60.0,
    ):
        """*grade_timeout* is the HTTP deadline for one grading call.

        Upstream bounds the command itself at 15 s (`harness.TIMEOUT_S`), and a
        timed-out command returns a verdict rather than hanging, so this only has
        to be comfortably outside that. It is not the command's budget and
        changing it does not change what upstream measures.
        """
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grade_timeout = grade_timeout
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": raw["instruction"]},
            ],
            # No `reference`: the ground truth is the task's own check over the
            # final filesystem state, not a value. Described at judgement time.
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # No extraction, deliberately. Upstream passes the reply verbatim to
        # `bash -c` and scores a fenced or chatty answer as the shell failure it
        # becomes -- that is the measurement, so stripping fences here would
        # repair the thing being measured. Empty normalizes to None so
        # `extracted` reports it as a miss.
        return build_prediction_record([choice or None for choice in inf.texts])

    @override
    async def feedback(self, post, ctx):
        task_id = ctx.raw_sample["task_id"]
        expected_digest = local_digest()

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "quotebench",
                        # An empty reply is None here but "" on the wire, so the
                        # evaluator still runs it and returns a real verdict
                        # rather than the sample being skipped.
                        "code": rollout.get("prediction") or "",
                        "kwargs": {"task_id": task_id, "contract": self.CONTRACT},
                    },
                    timeout=self._grade_timeout,
                )
                resp.raise_for_status()
                res = resp.json()
                correct, reason = res["status"], res["msg"]
                data = res["data"]
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {} ({}): [{}] {}",
                    idx,
                    task_id,
                    type(e).__name__,
                    e,
                )
                raise e

            if data is None:
                # The evaluator answers `data=None` only when nothing ran: an
                # unknown task id, an unknown contract, malformed kwargs. That is
                # our bug or a version skew, never a wrong command, so it must
                # not be recorded as a failed rollout.
                raise RuntimeError(
                    f"quotebench evaluator rejected sample {task_id!r} on "
                    f"contract {self.CONTRACT!r}: {reason}"
                )
            assert_digest(local=expected_digest, remote=data.get("scenarios_digest"))

            rollouts.append(
                build_rollout_judgement(
                    idx,
                    correct,
                    extra={
                        # The failure taxonomy comes from the evaluator rather
                        # than from a client-side classifier over `reason`: it is
                        # upstream's own `harness.classify`, computed where the
                        # exit code and stderr are.
                        "error_class": data.get("error_class"),
                        "exit_code": data.get("exit_code"),
                        "timed_out": data.get("timed_out"),
                        "reason": reason,
                    },
                )
            )

        return True, build_judgement_record(
            None,  # the reference is the check procedure below, not a value
            rollouts,
            extra={
                "scenario": ctx.raw_sample["scenario"],
                "tier": ctx.raw_sample["tier"],
                "hazards": list(ctx.raw_sample["hazards"]),
                "contract": self.CONTRACT,
            },
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        grouping = self.problem_groups(finals)

        # First rollout per sample, in percentage points -- the units
        # `pass_rate_pct` is published in, so the interval brackets the number
        # printed beside it.
        first_pct = [
            100.0
            if ((f.feedback_result or {}).get("rollouts") or [{}])[0].get("correct")
            else 0.0
            for f in finals
        ]
        pass_rate_pct = (sum(first_pct) / total) if total else 0.0

        classes: Counter[str] = Counter()
        by_tier: dict[int, list[float]] = {}
        by_scenario: dict[str, list[float]] = {}
        for final in finals:
            judgement = final.feedback_result or {}
            rollouts = judgement.get("rollouts") or [{}]
            head = rollouts[0]
            hit = 100.0 if head.get("correct") else 0.0
            classes[str((head.get("extra") or {}).get("error_class") or "unknown")] += 1
            extra = judgement.get("extra") or {}
            scenario = str(extra.get("scenario", "unknown"))
            by_tier.setdefault(int(extra.get("tier", -1)), []).append(hit)
            by_scenario.setdefault(scenario, []).append(hit)

        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": pass_rate_pct,
            "pass_rate_pct": pass_rate_pct,
            SCORE_KEY_FIELD: "pass_rate_pct",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            "fails": len(fails),
        }
        # One count per failure class, always all seven, so a class that did not
        # occur reads as zero rather than as a column this run forgot.
        for name in FAILURE_CLASSES:
            metrics[f"n_{name.replace('-', '_')}"] = classes.get(name, 0)
        unknown = sum(v for k, v in classes.items() if k not in FAILURE_CLASSES)
        metrics["n_unknown_class"] = unknown

        # Per-tier and per-scenario rates carry no interval: each is a different
        # population from the headline's, and publishing an interval would owe a
        # per-axis population count that means something else in every row.
        for tier, hits in sorted(by_tier.items()):
            metrics[f"pass_rate_pct_tier{tier}"] = sum(hits) / len(hits)
        for scenario, hits in sorted(by_scenario.items()):
            key = scenario.replace("-", "_")
            metrics[f"pass_rate_pct_{key}"] = sum(hits) / len(hits)

        # `interval_metrics` names the headline `score`; `pass_rate_pct` is the
        # SAME number under upstream's own spelling, so it rides along as an
        # alias on this one call rather than getting a second interval computed
        # the same way.
        return (
            metrics
            | interval_metrics(
                first_pct,
                denominator=total,
                group_keys=None if grouping is None else grouping.keys,
                n_problems=None if grouping is None else grouping.n_problems,
                aliases=("pass_rate_pct",),
            )
            | health_metrics(finals)
        )

    @override
    async def shutdown(self):
        await self._http_client.aclose()
