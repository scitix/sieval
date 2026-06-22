"""
MBPP few-shot base-model generative task.

Reproduces the lm-evaluation-harness MBPP 3-shot setup: the
``You are an expert Python programmer...`` prompt with ``[BEGIN]``/``[DONE]``
delimiters, the ``[DONE]`` stop token, and the fixed task_id 2/3/4 few-shot
examples. The few-shot set and the 1-10/11-510/511-600/601-974 split ranges
follow the original google-research MBPP README.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, override

import httpx
from loguru import logger

from sieval.community.mbpp import list_fewshot_samples
from sieval.core.models import ModelOutput
from sieval.core.tasks import EvalMode, ReferenceImpl, Task, sieval_task
from sieval.datasets import MBPPDatasetSample

DEFAULT_NUM_SHOTS = 3
STOP_SEQUENCES = ("[DONE]",)


class ResourceMetrics(TypedDict):
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float


class Feedback(TypedDict):
    correct: bool
    msg: str
    metrics: ResourceMetrics | None


@sieval_task(
    name="mbpp_kshot_base_gen",
    display_name="MBPP (few-shot, base generative)",
    description="MBPP few-shot code generation with pass@k execution scoring.",
    eval_mode=EvalMode.GEN,
    n_shot=DEFAULT_NUM_SHOTS,
    tags=("english", "python", "code-exec", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url="https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/mbpp/mbpp.yaml",
        notes=(
            "Prompt, [DONE] stop token, and default task_id 2/3/4 few-shot "
            "samples mirror lm-eval MBPP; num_shots is configurable via YAML "
            "task args. Greedy generation (temperature=0, top_p=1, "
            "max_tokens=1024). Published Qwen2.5-72B-Base MBPP 3-shot Pass@1 "
            "is 76.0 (Qwen3 report, Table 3) and 72.6 (DeepSeek-V3 report, "
            "Table 3)."
        ),
    ),
)
class MBPPFewShotBaseGenTask(
    Task[
        MBPPDatasetSample,
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
        *,
        num_shots: int | None = None,
        k: int | None = None,
        pass_k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        timeout: float = 5.0,
        stop: Sequence[str] | None = STOP_SEQUENCES,
    ):
        if num_shots is not None and k is not None and num_shots != k:
            raise ValueError(
                f"num_shots and k must match when both are set; got {num_shots} and {k}"
            )
        if num_shots is not None:
            resolved_num_shots = num_shots
        elif k is not None:
            resolved_num_shots = k
        else:
            resolved_num_shots = DEFAULT_NUM_SHOTS
        if resolved_num_shots < 0:
            raise ValueError(f"num_shots must be >= 0, got {resolved_num_shots}")
        if pass_k < 1:
            raise ValueError(f"pass_k must be >= 1, got {pass_k}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if pass_k > n:
            raise ValueError(
                f"pass_k must be <= n; got pass_k={pass_k} and n={n}. "
                "pass@k needs at least k samples per problem."
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        available_shots = len(list_fewshot_samples())
        if resolved_num_shots > available_shots:
            raise ValueError(
                "MBPP lm-eval few-shot prompt provides at most "
                f"{available_shots} examples; got num_shots={resolved_num_shots}."
            )

        super().__init__(dataset=dataset, model=model, name=name)
        self._num_shots = resolved_num_shots
        self._pass_k = pass_k
        self._n = n
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._stop = tuple(stop) if stop is not None else ()
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )
        self._few_shot_prefix: str | None = None

    def _format_tests(self, sample: Mapping[str, Any]) -> str:
        tests = [str(test) for test in sample.get("test_list", [])[:3]]
        return "\n".join(tests).strip()

    def _doc_to_text(self, sample: Mapping[str, Any]) -> str:
        return (
            "You are an expert Python programmer, and here is your task: "
            f"{sample['text']} "
            "Your code should pass these tests:\n\n"
            f"{self._format_tests(sample)}\n"
            "[BEGIN]\n"
        )

    def _build_few_shot_str(self) -> str:
        parts: list[str] = []
        for example in list_fewshot_samples()[: self._num_shots]:
            parts.append(self._doc_to_text(example))
            parts.append(f"{example['code']}\n[DONE]\n\n")
        return "".join(parts)

    def _get_few_shot_prefix(self) -> str:
        # The prefix only depends on self._num_shots, so build it once and
        # reuse it for every sample rather than rebuilding per preprocess call.
        if self._few_shot_prefix is None:
            self._few_shot_prefix = self._build_few_shot_str()
        return self._few_shot_prefix

    @override
    async def setup(self):
        self._few_shot_prefix = self._build_few_shot_str()

    @override
    async def preprocess(self, raw, ctx):
        return f"{self._get_few_shot_prefix()}{self._doc_to_text(raw)}"

    @override
    async def infer(self, pre, ctx):
        # Forward the pass@k count and the stop token; decoding params come
        # from the model config.
        kwargs: dict[str, object] = {"n": self._n}
        if self._stop:
            kwargs["stop"] = list(self._stop)
        return await self.model.agenerate(pre, **kwargs)

    @override
    async def postprocess(self, inf, ctx):
        return [text.split("[DONE]", maxsplit=1)[0] for text in inf.texts]

    @override
    async def feedback(self, post, ctx):
        feedbacks = [
            {"correct": False, "msg": "Not evaluated", "metrics": None}
            for _ in range(len(post))
        ]

        # Score against the same three tests shown in the prompt, as lm-eval
        # does (candidate + test_list[0..2]).
        tests = self._format_tests(ctx.raw_sample)

        for idx, pred in enumerate(post):
            try:
                check_program = "\n".join(p for p in (pred, tests) if p).strip()
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "mbpp",
                        "code": check_program,
                        "timeout": self._timeout,
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
    async def report(self, finals, fails) -> dict[str, float]:
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        timeouts = 0
        for f in finals:
            feedbacks = f.feedback_result
            n_samples = len(feedbacks)
            correct_num = sum(1 for fb in feedbacks if fb["correct"])
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._pass_k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._pass_k)
            timeouts += sum(1 for fb in feedbacks if "timeout" in fb["msg"].lower())

        pass_at_1 = pass_at_1_total * 100 / total
        metrics = {
            "score": pass_at_1,
            "fails": len(fails),
            "timeouts": timeouts,
            "pass@1": pass_at_1,
        }
        if self._pass_k > 1:
            metrics[f"pass@{self._pass_k}"] = pass_at_k_total * 100 / total
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
