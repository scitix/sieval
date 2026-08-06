"""
LiveCodeBench code-generation few-shot base-model generative task.

A few-shot continuation prompt for base models (no system message / chat
template), scored by executing each generated program against the public +
private test cases.

Comparison target: DeepSeek-V3 technical report Table 3, "LiveCodeBench-Base
(Pass@1), 3-shot", over the 2024-08-01..2024-11-01 (`0801-1101`) problem window;
the Qwen2.5-72B-Base anchor there is Pass@1 = 12.9. The window and dataset version
are set via dataset YAML `args` (`version_tag`/`start_date`/`end_date`), not here.

Metric: Pass@1 (and Pass@k when `k > 1`) via the unbiased estimator over `n`
samples, matching upstream `codegen_metrics`. Extraction convention: upstream's
`extract_code(lmstyle="GenericBase")` — the raw completion stripped, with no
```-fence parsing (base models continue raw code after `### Answer`).

Repro decoding (greedy Pass@1): `temperature=0`, `top_p=1`, `max_gen_toks=2000`.
These are NOT set by the task (forwarding them into `agenerate` would silently
override the model config via `{**self._kwargs, **kwargs}`); supply them through
the model config or per-task `infer_args`. The task forwards only `n` (pass@k)
and `stop` — `("###",)`, prompt-coupled to the `### Question`/`### Answer`
template and identical to the upstream runner default (`--stop "###"`).

Deviations from the upstream LCB runner (complete list):
  1. Prompt builder generalized from upstream's hardcoded single example to
     `n_shot` (default 3 vs upstream's 1), and takes a `dict` rather than a
     `CodeGenerationProblem`; for `n_shot=1` the rendered prompt is byte-identical
     to upstream.
  2. Few-shot pools: upstream ships 2 examples per pool (`func`/`stdin`); a 3rd
     sieval-authored example is appended to each so the default `n_shot=3`
     (DeepSeek's setting) is reachable. At `n_shot=3` that 3rd example is NOT
     upstream. DeepSeek did not publish its exact few-shot examples, so scores
     approximate rather than exactly reproduce the reference.
  3. Eval sandbox: code is executed by an external service at
     `$SIEVAL_CODE_EVAL_API` (POST code + test cases), not upstream's in-process
     `check_correctness`. The execution budget is client-owned, sent in the request
     body, but enforced per test case as upstream does (`timeout_per_case`, default
     6s) rather than as one wall for the whole suite. Runs from before that rule
     landed used the whole-suite wall and are NOT comparable.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

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
    get_base_model_fewshot_prefix,
    get_base_model_target_block,
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

N_SHOT = 3
STOP_SEQUENCES = ("###",)


@sieval_task(
    name="livecodebench_code_generation_kshot_base_gen",
    display_name="LiveCodeBench Code Generation (few-shot, base generative)",
    description=(
        "LiveCodeBench — contamination-free code benchmark, base-model few-shot "
        "generation subset."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=N_SHOT,
    tags=("english", "python", "code-exec", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="livecodebench",
        url="https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py",
        notes=(
            "Base-model few-shot template vendored from "
            "lcb_runner/prompts.get_base_model_question_template_answer and "
            "extract_code(lmstyle='GenericBase') from lcb_runner/utils. Default "
            "n_shot=3 and stop=('###',) follow DeepSeek-V3 Table 3 "
            "'LiveCodeBench-Base (Pass@1), 3-shot' and upstream's runner default. "
            "Upstream ships only 2 few-shot examples per pool, so the 3rd example "
            "is sieval-authored. Recommended problem window 2024-08-01..2024-11-01 "
            "is configured via dataset YAML args (version_tag/start_date/end_date). "
            "Grading budget matches upstream: each test case gets its own 6s, "
            "tunable via `timeout_per_case`. Runs recorded before that rule landed "
            "used one whole-suite wall instead and are NOT comparable."
        ),
    ),
)
class LiveCodeBenchCodeGenerationFewShotBaseGenTask(
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
        *,
        n_shot: int = N_SHOT,
        k: int = 1,
        n: int = 1,
        stop: tuple[str, ...] = STOP_SEQUENCES,
        max_concurrency: int = 4,
        timeout_per_case: float = 6.0,
    ):
        """*timeout_per_case* is the budget **each test case** gets -- upstream's rule
        and its ``6.0`` default (``codegen_metrics(..., timeout=6)``, deviation 3
        above). The whole-suite wall is derived from it and not configurable.

        It replaces an earlier ``timeout``, the base of a whole-suite wall and a
        different rule; passing that now raises ``TypeError``.
        """
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._k = k
        self._n = n
        self._stop = stop
        self._timeout_per_case = timeout_per_case
        # Fixed few-shot prefixes, keyed by whether the target has starter code
        # (func vs stdin pool). Computed once in setup(); never per sample.
        self._fewshot_prefix: dict[bool, str] = {}
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def setup(self) -> None:
        self._fewshot_prefix = {
            has_starter: get_base_model_fewshot_prefix(has_starter, self.n_shot)
            for has_starter in (False, True)
        }

    @override
    async def preprocess(self, raw, ctx):
        # Prefixes are pre-filled by setup() (both keys), so index directly —
        # no per-sample rebuild and no mutation of shared state under concurrency.
        prefix = self._fewshot_prefix[bool(raw["starter_code"])]
        return build_prompt_record(
            prefix
            + get_base_model_target_block(raw["question_content"], raw["starter_code"]),
            # No `reference`: the ground truth is a test suite, not a value.
        )

    @override
    async def infer(self, pre, ctx):
        # Decoding params (temperature/max_tokens/...) come from model config or
        # per-task infer_args, NOT this task — avoid silently overriding the
        # model's configured args. Forward only `n` (pass@k) and the
        # prompt-coupled `stop`. The branch keeps `stop` out of the kwargs when
        # unset so it can't clobber the model's configured stop via the merge.
        if self._stop:
            return await self.model.agenerate(
                pre["prompt"], n=self._n, stop=list(self._stop)
            )
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Empty extraction normalizes to None so `extracted` reports the miss.
        return build_prediction_record(
            [extract_code(choice, "GenericBase") or None for choice in inf.texts]
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

        rollouts: list[RolloutJudgement] = []

        cases = public_test_cases + private_test_cases
        inputs = [t["input"] for t in cases]
        outputs = [t["output"] for t in cases]
        fn_name = metadata.get("func_name", None)

        # Cases are budgeted individually, so this wall is only a backstop, for a
        # worker wedged where a per-case signal cannot reach it. Upstream's own shape:
        # `check_correctness` joins its worker at `(timeout + 1) * n + 5`. Sent rather
        # than left to the evaluator's identical derivation, so the HTTP deadline below
        # is provably outside it. Mirrored in the sibling 0-shot task.
        suite_timeout = (self._timeout_per_case + 1.0) * len(inputs) + 5.0

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
                        # error -- a real verdict rather than a skipped rollout.
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
