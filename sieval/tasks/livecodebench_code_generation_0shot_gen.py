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
            "lcb_runner/{prompts,utils}."
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
        dict[str, float],
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
        timeout: float = 6.0,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._cot = cot
        self._k = k
        self._n = n
        self._timeout = timeout
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
                        "code": rollout["prediction"] or "",
                        "test": {
                            "inputs": inputs,
                            "outputs": outputs,
                            "fn_name": fn_name,
                        },
                        # All N cases share one sequential budget, so scale by N.
                        # Approximates official per-case 6s within a single run.
                        "timeout": self._timeout + len(inputs) * 2.0,
                    },
                    # allow more time for more test cases
                    # with extra buffer for network latency
                    timeout=self._timeout + len(inputs) * 2 + 2,
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
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        timeouts = 0
        for f in finals:
            judgement = f.feedback_result
            n_samples = judgement["n_rollouts"]
            correct_num = judgement["n_correct"]
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)
            # Kept as the original substring check rather than switching to the
            # `failure` category, so the counter stays byte-identical across the
            # protocol migration.
            timeouts += sum(
                1
                for r in judgement["rollouts"]
                # A null msg from the evaluator is absent on disk -- default it.
                if "timeout" in (r["extra"].get("msg") or "").lower()
            )

        pass_at_1 = pass_at_1_total * 100 / total
        metrics = {
            "score": pass_at_1,
            "fails": len(fails),
            "timeouts": timeouts,
            "pass@1": pass_at_1,
        }
        if self._k > 1:
            metrics[f"pass@{self._k}"] = pass_at_k_total * 100 / total
        return metrics

    @override
    async def shutdown(self):
        await self._http_client.aclose()

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n < k:
            return 0.0
        if c == 0:
            return 0.0
        # Formula: 1 - product_{i=0}^{k-1} (n - c - i) / (n - i)
        # This calculates the probability that all k samples are wrong
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
