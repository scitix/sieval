import os
import re
import time
from typing import override

import httpx
from loguru import logger

from sieval.community.simple_evals.humaneval_eval import QUERY_TEMPLATE
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
from sieval.datasets import HumanEvalDatasetSample


@sieval_task(
    name="human_eval_0shot_gen",
    display_name="HumanEval (0-shot, generative)",
    description="OpenAI HumanEval — 164 Python functions evaluated with pass@k.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/humaneval_eval.py",
        notes="Prompt style follows simple-evals HumanEval; QUERY_TEMPLATE is sieval-local.",  # noqa: E501
    ),
)
class HumanEvalZeroShotGenTask(
    Task[
        HumanEvalDatasetSample,
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
        k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        timeout: float = 3.0,  # official HumanEval per-exec budget (flat, single run)
    ):
        super().__init__(dataset=dataset, model=model, name=name)
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
        return build_prompt_record(
            [{"role": "user", "content": QUERY_TEMPLATE.format(prompt=raw["prompt"])}],
            # No `reference`: the ground truth is a test suite, not a value. It is
            # described at judgement time instead (see feedback's sample-level extra).
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        predictions: list[str | None] = []
        for choice in inf.texts:
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(choice)
            extracted_answer = matches[0] if len(matches) >= 1 else choice
            extracted_answer = extracted_answer[
                extracted_answer.find(":\n    ") + 2 :
            ]  # remove signature
            # Empty extraction normalizes to None so `extracted` reports the miss.
            predictions.append(extracted_answer or None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            # An unextractable answer is None here but "" on the wire, so the
            # evaluator still runs it and reports a compile error -- the
            # pre-protocol behaviour, and a real verdict rather than a skip.
            check_program = (
                ctx.raw_sample["prompt"]
                + (rollout["prediction"] or "")
                + "\n"
                + ctx.raw_sample["test"]
                + "\n"
                + f"check({ctx.raw_sample['entry_point']})"
            )
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "human-eval",
                        "code": check_program,
                        "timeout": self._timeout,
                    },
                    timeout=self._timeout + 2,  # extra buffer for network latency
                )
                resp.raise_for_status()
                res = resp.json()
                data = res["data"] or {}
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"],
                        res["status"],
                        extra={
                            "msg": res["msg"],
                            # Absent against an evaluator that predates test-case
                            # progress reporting -- which reads as unknown, not zero.
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
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {}: [{}] {}",
                    idx,
                    type(e).__name__,
                    e,
                )
                raise e

        return True, build_judgement_record(
            None,  # the reference is the test suite below, not a value
            rollouts,
            extra={"entry_point": ctx.raw_sample["entry_point"]},
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
