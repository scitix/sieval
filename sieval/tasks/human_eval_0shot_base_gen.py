"""HumanEval zero-shot base-model generative task.

AI-Generated Code - GPT-5.5-Codex (OpenAI)
"""

import os
import time
from typing import TypedDict, override

import httpx
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import HumanEvalDatasetSample


class ResourceMetrics(TypedDict):
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float


class Feedback(TypedDict):
    correct: bool
    msg: str
    metrics: ResourceMetrics | None


STOP_SEQUENCES = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


@sieval_task(
    name="human_eval_0shot_base_gen",
    display_name="HumanEval (0-shot, base generative)",
    description="OpenAI HumanEval for base completion models evaluated with pass@k.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/humaneval/humaneval.yaml"
        ),
        notes=(
            "Aligned with lm-evaluation-harness humaneval.yaml prompt, stop "
            "sequences, max_gen_toks, zero-shot setting, repeats=1, and raw "
            "completion filtering; code execution is handled by the SiEval "
            "code-eval API."
        ),
    ),
)
class HumanEvalZeroShotBaseGenTask(
    Task[
        HumanEvalDatasetSample,
        str,
        ModelOutput,
        list[str],
        list[Feedback],
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
        max_tokens: int = 1024,
        max_concurrency: int = 4,
        timeout: float = 5.0,
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n
        self._max_tokens = max_tokens
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._stop = stop
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        return raw["prompt"]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(
            pre,
            n=self._n,
            max_tokens=self._max_tokens,
            stop=list(self._stop),
        )

    @override
    async def postprocess(self, inf, ctx):
        return list(inf.texts)

    @override
    async def feedback(self, post, ctx):
        feedbacks = [
            {"correct": False, "msg": "Not evaluated", "metrics": None}
            for _ in range(len(post))
        ]

        for idx, pred in enumerate(post):
            check_program = (
                ctx.raw_sample["prompt"]
                + pred
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
                    },
                    timeout=self._timeout + 2,
                )
                resp.raise_for_status()
                res = resp.json()
                feedbacks[idx] = {
                    "correct": res["status"],
                    "msg": res["msg"],
                    "metrics": res["data"],
                }
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {}: [{}] {}",
                    idx,
                    type(e).__name__,
                    e,
                )
                raise e

        return True, feedbacks

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        timeouts = 0
        for f in finals:
            feedbacks = f.feedback_result
            n_samples = len(feedbacks)
            correct_num = sum(1 for f in feedbacks if f["correct"])
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)
            timeouts += sum(1 for fb in feedbacks if "timeout" in fb["msg"].lower())

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
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
