"""
MBPP few-shot base-model generative task.

Reproduces the lm-evaluation-harness MBPP 3-shot setup: the
``You are an expert Python programmer...`` prompt with ``[BEGIN]``/``[DONE]``
delimiters, the ``[DONE]`` stop token, and the fixed task_id 2/3/4 few-shot
examples, evaluated on the ``test`` split. The few-shot set follows the
original google-research MBPP README.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import os
import time
from collections.abc import Mapping
from typing import Any, override

import httpx
from loguru import logger

from sieval.community.mbpp import list_fewshot_samples
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
from sieval.datasets import MBPPDatasetSample

DEFAULT_NUM_SHOTS = 3
STOP_SEQUENCES = ("[DONE]",)


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
            "samples mirror lm-eval MBPP; n_shot (few-shot count) is "
            "configurable "
            "via YAML task args. Greedy generation (temperature=0, top_p=1, "
            "max_tokens=1024). Published Qwen2.5-72B-Base MBPP 3-shot Pass@1 "
            "is 76.0 (Qwen3 report, Table 3) and 72.6 (DeepSeek-V3 report, "
            "Table 3); DeepSeek-V3 leaves its MBPP protocol unspecified, so "
            "the gap to the Qwen-aligned number is a protocol difference, not "
            "an implementation error."
        ),
    ),
)
class MBPPFewShotBaseGenTask(
    Task[
        MBPPDatasetSample,
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
        *,
        n_shot: int = DEFAULT_NUM_SHOTS,
        k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        # lm-eval scores MBPP via HF code_eval, whose default timeout is 3.0s
        # (lm-eval does not override it); match upstream.
        timeout: float = 3.0,
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k > n:
            raise ValueError(
                f"k must be <= n; got k={k} and n={n}. "
                "pass@k needs at least k samples per problem."
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        available_shots = len(list_fewshot_samples())
        if n_shot > available_shots:
            raise ValueError(
                "MBPP lm-eval few-shot prompt provides at most "
                f"{available_shots} examples; got n_shot={n_shot}."
            )

        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._k = k
        self._n = n
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._stop = stop
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )
        self._few_shot_prefix: str | None = None

    def _format_tests(self, sample: Mapping[str, Any]) -> str:
        # lm-eval joins the first three tests verbatim (no strip); some samples
        # carry a trailing space, kept here for byte-exact prompt fidelity.
        tests = [str(test) for test in sample.get("test_list", [])[:3]]
        return "\n".join(tests)

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
        for example in list_fewshot_samples()[: self.n_shot]:
            parts.append(self._doc_to_text(example))
            parts.append(f"{example['code']}\n[DONE]\n\n")
        return "".join(parts)

    def _get_few_shot_prefix(self) -> str:
        # The prefix only depends on self.n_shot, so build it once and
        # reuse it for every sample rather than rebuilding per preprocess call.
        if self._few_shot_prefix is None:
            self._few_shot_prefix = self._build_few_shot_str()
        return self._few_shot_prefix

    @override
    async def setup(self):
        self._few_shot_prefix = self._build_few_shot_str()

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            f"{self._get_few_shot_prefix()}{self._doc_to_text(raw)}",
            # No `reference`: the ground truth is the assert-statement suite, not
            # a value -- described at judgement time instead.
        )

    @override
    async def infer(self, pre, ctx):
        # Forward the sample count and the [DONE] stop token; decoding params
        # come from the model config.
        return await self.model.agenerate(
            pre["prompt"], n=self._n, stop=list(self._stop)
        )

    @override
    async def postprocess(self, inf, ctx):
        # A blank completion normalizes to None so `extracted` stays a real signal.
        return build_prediction_record(
            [text.split("[DONE]", maxsplit=1)[0] or None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        rollouts: list[RolloutJudgement] = []

        # Score against the same three tests shown in the prompt, as lm-eval
        # does (candidate + test_list[0..2]).
        tests = self._format_tests(ctx.raw_sample)

        for rollout in post["rollouts"]:
            idx = rollout["index"]
            try:
                # An unextractable completion is None here but "" on the wire, so
                # the evaluator still runs the tests alone and reports a real
                # verdict -- the pre-protocol behaviour, not a skipped rollout.
                pred = rollout.get("prediction") or ""
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
            # The reference is a procedure, so `extra` describes it: these are the
            # same three asserts the prompt showed, which is what was run.
            extra={"tests": tests},
        )

    @override
    async def report(self, finals, fails) -> dict[str, float | str]:
        total = len(finals) + len(fails)
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
