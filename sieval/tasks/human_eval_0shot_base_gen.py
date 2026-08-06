"""HumanEval zero-shot base-model generative task.

Reproduces the lm-evaluation-harness ``humaneval.yaml`` for base completion
models, scored with pass@k via the SiEval code-eval API.

Decoding follows the harness defaults and is configured on the model, not
injected by this task: greedy sampling (``temperature=0``, ``top_p=1``) and
``max_gen_toks=1024``. Set these through the model ``args`` or per-task
``infer_args`` in the run config. The task owns only the prompt-coupled
``stop`` sequences and ``n`` (the pass@k sampling count).

HumanEval has no single canonical Qwen2.5-72B-Base score: references span
roughly six points — DeepSeek-V3 Table 3 reports 53.0, while the Qwen2.5
technical report self-reports 59.1 (with no published eval config). Because
this task reproduces lm-eval-harness rather than Qwen's own setup, its score
is expected to land between those references rather than match the Qwen
self-report.

AI-Generated Code - GPT-5.5-Codex (OpenAI)
"""

import os
import time
from typing import override

import httpx
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
from sieval.datasets import HumanEvalDatasetSample

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
            "code-eval API. No single canonical Qwen2.5-72B-Base target "
            "exists: DeepSeek-V3 Table 3 reports 53.0 and the Qwen2.5 "
            "technical report self-reports 59.1 (no published eval config). "
            "This task reproduces lm-eval-harness, so its score is expected "
            "to land between those references rather than match Qwen's "
            "self-report."
        ),
    ),
)
class HumanEvalZeroShotBaseGenTask(
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
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n
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
        return build_prompt_record(
            raw["prompt"],
            # No `reference`: the ground truth is a test suite, not a value --
            # described at judgement time instead.
        )

    @override
    async def infer(self, pre, ctx):
        # Decoding params (temperature, top_p, max_tokens) come from the
        # model's configured args / per-task infer_args, not this task. Only
        # the prompt-coupled stop sequences and the pass@k sample count live
        # here.
        return await self.model.agenerate(
            pre["prompt"],
            n=self._n,
            stop=list(self._stop),
        )

    @override
    async def postprocess(self, inf, ctx):
        # A blank completion normalizes to None so `extracted` stays a real signal.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        rollouts: list[RolloutJudgement] = []

        for rollout in post["rollouts"]:
            idx = rollout["index"]
            # An unextractable completion is None here but "" on the wire, so the
            # evaluator still runs it and reports a compile error -- the
            # pre-protocol behaviour, and a real verdict rather than a skip.
            check_program = (
                ctx.raw_sample["prompt"]
                + (rollout.get("prediction") or "")
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
                    timeout=self._timeout + 2,
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
                            # progress reporting -- unknown, not zero.
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
            None,  # the reference is the test suite, not a value
            rollouts,
            extra={"entry_point": ctx.raw_sample["entry_point"]},
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails), "timeouts": 0, "pass@1": 0.0}

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
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
