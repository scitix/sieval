import base64
import json
import os
import pickle
import time
import zlib
from typing import override

import httpx
from loguru import logger

from sieval.community.livecodebench.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)
from sieval.community.livecodebench.utils.extraction_utils import extract_code
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
    sampling_report,
)
from sieval.datasets import LiveCodeBenchDatasetSample


@sieval_task(
    name="livecodebench_code_generation_0shot_gen",
    display_name="LiveCodeBench Code Generation (0-shot)",
    description="LiveCodeBench — contamination-free code benchmark, generation subset.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="livecodebench",
        url="https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py",
        notes=(
            "Prompt templates and extract_code vendored from "
            "lcb_runner/{prompts,utils}. Grading budget matches upstream: 6s per "
            "test case (codegen_metrics(..., timeout=6)), via `timeout_per_case`. "
            "Runs from before that rule landed used one whole-suite wall and are "
            "NOT comparable -- re-grading 90 recorded rollouts cost 2.22 pp."
        ),
    ),
)
class LiveCodeBenchCodeGenerationZeroShotGenTask(
    Task[
        LiveCodeBenchDatasetSample,
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
        cot: bool = False,
        k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        timeout_per_case: float = 6.0,
    ):
        """*timeout_per_case* is the budget **each test case** gets, which is how
        official LiveCodeBench grades: ``lcb_runner`` re-arms ``signal.alarm(timeout)``
        inside the case loop of ``grade_call_based`` / ``grade_stdio``, and
        ``codegen_metrics(..., timeout=6)`` supplies the ``6.0`` default kept here.
        The whole-suite wall is derived from it and not configurable (see
        :meth:`feedback`).

        It replaces an earlier ``timeout``, the base of a whole-suite wall and a
        different rule; passing that now raises ``TypeError`` rather than silently
        grading by it.
        """
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._cot = cot
        self._k = k
        self._n = n
        self._timeout_per_case = timeout_per_case
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        question = {
            "question_content": raw["question_content"],
            "starter_code": raw["starter_code"],
        }
        prompt = get_generic_question_template_answer(question, self._cot)
        return build_prompt_record(
            [
                {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                {"role": "user", "content": prompt},
            ],
            # No `reference`: the ground truth is a test suite, not a value. It is
            # described at judgement time instead (see feedback's sample-level extra).
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Empty extraction normalizes to None so `extracted` reports it as a miss.
        return build_prediction_record(
            [extract_code(choice) or None for choice in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        public_test_cases = json.loads(ctx.raw_sample["public_test_cases"])
        private_test_cases = ctx.raw_sample["private_test_cases"]
        try:
            private_test_cases = json.loads(ctx.raw_sample["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(private_test_cases.encode("utf-8"))
                    )
                )
            )
        metadata = json.loads(ctx.raw_sample["metadata"])

        cases = public_test_cases + private_test_cases
        inputs = [t["input"] for t in cases]
        outputs = [t["output"] for t in cases]
        fn_name = metadata.get("func_name", None)

        # Cases are budgeted individually, so this wall is only a backstop, for a
        # worker wedged where a per-case signal cannot reach it. Upstream's own shape:
        # `check_correctness` joins its worker at `(timeout + 1) * n + 5`. Sent rather
        # than left to the evaluator's identical derivation, so the HTTP deadline below
        # is provably outside it. Mirrored in the sibling k-shot base task.
        suite_timeout = (self._timeout_per_case + 1.0) * len(inputs) + 5.0

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "livecodebench",
                        # An unextractable answer is None here but "" on the wire,
                        # so the evaluator still runs it and reports a compile
                        # error -- the pre-protocol behaviour, and a real verdict
                        # rather than a skipped rollout.
                        "code": rollout.get("prediction") or "",
                        "test": {
                            "inputs": inputs,
                            "outputs": outputs,
                            "fn_name": fn_name,
                        },
                        "timeout": suite_timeout,
                        "timeout_per_case": self._timeout_per_case,
                    },
                    # allow more time than the suite wall, for network latency
                    timeout=suite_timeout + 2,
                )
                resp.raise_for_status()
                res = resp.json()
                # should raise error if no `status` & `msg` field
                correct, msg = res["status"], res["msg"]
                data = res["data"] or {}
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {}: [{}] {}",
                    idx,
                    type(e).__name__,
                    e,
                )
                raise e

            # n_cases / n_passed need an evaluator that reports test-case progress
            # (vendor/code-evaluator/VENDORED.md); against an older one they are
            # simply absent, which reads as unknown rather than as zero.
            #
            # `msg` is stored raw and deliberately not bucketed into a failure
            # taxonomy: it is free text from a separately deployed service whose
            # wording has already drifted once, so a client-side classifier
            # decays silently. A category belongs on the response next to
            # n_cases/n_passed, where the evaluator already knows the answer.
            rollouts.append(
                build_rollout_judgement(
                    idx,
                    correct,
                    extra={
                        "msg": msg,
                        "n_cases": data.get("n_cases"),
                        "n_passed": data.get("n_passed"),
                        "resources": {
                            key: value
                            for key, value in data.items()
                            if key not in ("n_cases", "n_passed")
                        },
                    },
                )
            )

        return True, build_judgement_record(
            None,  # the reference is the test suite below, not a value
            rollouts,
            extra={
                "n_test_cases": len(cases),
                "n_public_cases": len(public_test_cases),
                "n_private_cases": len(private_test_cases),
                "io_mode": "fn_call" if fn_name else "stdio",
                "func_name": fn_name,
            },
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        # Kept as the original substring check rather than switching to the
        # `failure` category, so the counter stays byte-identical across the
        # protocol migration.
        timeouts = sum(
            1
            for f in finals
            for r in f.feedback_result["rollouts"]
            # A null msg from the evaluator is absent on disk -- default it.
            if "timeout" in (r["extra"].get("msg") or "").lower()
        )
        # `votes=False`: two correct programs are not one answer, so there is
        # nothing well-defined to take a majority over (RFC #74).
        rolled = sampling_report(
            finals, n=self._n, k=self._k, denominator=total, votes=False
        )
        # Read back out of the shared block, so `score` cannot drift from it.
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str] = {
            "score": pass_at_1,
            "fails": len(fails),
            "timeouts": timeouts,
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        # Outside the gate: extraction health is a fact about the parser, not
        # about the draw, and n=1 is where a stopped extractor hides longest.
        return metrics | health_metrics(finals)

    @override
    async def shutdown(self):
        await self._http_client.aclose()
