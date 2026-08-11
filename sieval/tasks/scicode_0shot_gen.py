"""SciCode 0-shot task for instruct/chat models.

Each sample is one main problem decomposed into dependent sub-steps. Generation
is sequential *within* a problem: the prompt for step *i* embeds the model's own
code from steps ``1..i-1`` (upstream's default self-dependency setting, not gold
context). Different problems still run concurrently as separate samples.

Evaluation mirrors upstream ``test_generated_code.py`` — per step, concatenate
``required_dependencies`` + prior-step functions + current-step function + the
step's test cases, then execute — but the numeric targets that upstream reads
from ``test_data.h5`` inside the sandbox are read here on the eval side and
inlined into the program, keeping sieval's code-eval sandbox stateless. Three
scientist-authored steps (13.6, 62.1, 76.3) are not generated or tested; their
gold code is used only as context for later steps.

Metrics: sub-problem accuracy (passing steps / tested steps) and main-problem
accuracy (problems whose every tested step passes) — the headline resolve rate.
Both treat a problem that failed the pipeline as unsolved, so neither denominator
shrinks when a sample errors out.

Stage-output protocol: **one rollout per problem**. A rollout means an independent
attempt at the same thing, and a problem's sub-steps are neither independent (each
builds on the model's own prior code) nor attempts at the same thing, so mapping
steps onto ``rollouts[]`` would make ``n_correct``/``n_rollouts`` read as a pass
rate over repeated tries. The whole dependent sequence is the attempt: ``correct``
is "every tested step passed" (main-problem accuracy's numerator), ``score`` and
``metrics`` carry the problem's own step pass rate, and the per-step verdicts sit
in ``extra``. Report-time sub-problem accuracy pools the raw step counts from
``extra`` rather than averaging those per-problem rates — problems have different
step counts, so the two are different numbers.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import os
import time
from typing import Literal, TypedDict, override

import httpx
from anyio.to_thread import run_sync
from loguru import logger

from sieval.community.scicode import (
    build_test_program,
    encode_targets,
    extract_function_name,
    extract_python_script,
    generate_prompt_with_steps,
    get_function_from_code,
    is_special_step,
    process_hdf5_to_tuple,
    special_step_code,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    TaskStageOutput,
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
)
from sieval.core.types import JSONValue
from sieval.core.utils.meta import build_stage_meta
from sieval.datasets import SciCodeDatasetSample


class StepCode(TypedDict):
    step_number: str
    tested: bool
    # dependencies + prior-step funcs + current-step func; None for special steps
    code_content: str | None
    # Just this step's extracted code, without the accumulated context above. This
    # is the model's answer for the step, so it becomes the prediction; kept
    # separately rather than re-running the extractor on raw_response, and cheap
    # next to code_content (which repeats every prior step). "" for special steps.
    extracted_code: str
    # Raw model response for this step, kept for provenance/debugging. Empty for
    # special (gold) steps that are not generated.
    raw_response: str
    # True when extract_python_script found no code in raw_response — the step
    # (and any later step depending on it) will fail; surfaced in the report.
    empty_extraction: bool


class StepProgram(TypedDict):
    step_number: str
    program: str
    empty_extraction: bool


class StepFeedback(TypedDict):
    step_number: str
    correct: bool
    msg: str
    empty_extraction: bool


@sieval_task(
    name="scicode_0shot_gen",
    display_name="SciCode (0-shot, generative)",
    description="SciCode — research coding benchmark with dependent sub-steps; sub-problem and main-problem accuracy.",  # noqa: E501
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    deps_group="scicode",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="scicode",
        url="https://github.com/scicode-bench/SciCode/tree/69a8cfc829fe8788a426ce8b5de6292366dce7ef/eval/scripts",
        notes=(
            "Vendored from upstream eval/scripts into community/scicode: prompt "
            "templates, code/h5 parsers, comparison helpers, and the 3 "
            "non-generated gold steps (13.6/62.1/76.3, inlined in "
            "_gold_steps.py). special_step_mode selects gold injection and is "
            "SCORE-RELEVANT: 'verbatim' (default) injects the whole gold block; "
            "'extract' byte-matches upstream gencode_json.py, whose "
            "get_function_from_code matches the header's def before class and so "
            "silently drops the class wrapper for 13.6/62.1 (open upstream bugs "
            "#59/#49), leaving every dependent step to fail with NameError and "
            "making problems 13/62 structurally unsolvable. Use 'extract' when "
            "comparing against public leaderboard numbers (produced with that "
            "behavior). Calibration (Qwen2-72B, committed code, T=0 greedy, full "
            "test split, self-dependency): official expected values apply only "
            "to 'extract'. Observed extract main accuracy matches the official "
            "figures (1.5 without background, paper Table 2; 4.6 with background, "
            "Table 3). Observed main/sub pairs are extract no-background "
            "1.5/13.9, verbatim no-background 1.5/13.9, extract with-background "
            "4.6/24.0, and verbatim with-background 4.6/25.0; verbatim has no "
            "official expected value. Without background, the two modes have "
            "identical pass/fail outcomes because Qwen's affected downstream code "
            "is wrong "
            "even after the missing classes are restored. With background, "
            "verbatim rescues 13.14/62.2/62.4 (+3), as it does on a stronger "
            "model (gpt-5.6), increasing the sub-step count from 69/288 to 72/288; "
            "main remains 3/65 because neither affected problem is fully solved. "
            "In an earlier calibration, an unrelated byte-identical program for "
            "stochastic step 14.2 flipped verdict between paired runs. Its unseeded "
            "Monte Carlo test provides evidence of an observed ±1-problem (about "
            "1.5 point) execution-side main-accuracy jitter. In this committed "
            "rerun, 14.2 passes in both modes and does not contribute to their "
            "difference. "
            "Compatibility/parity adaptations and other deviations from upstream: "
            "(1) problems 2/28 import scipy.integrate.simps, removed in SciPy 1.14 "
            "(open upstream issue #2); a conditional legacy-compatible wrapper "
            "restores that API for any program referencing simps. That is wider "
            "than problems 2/28: problems 12/57 declare a scipy.integrate module "
            "handle and the model calls integrate.simps(...) through it, which "
            "raises AttributeError rather than ImportError, so import_errors=0 "
            "does not prove those steps ran. Both routes are covered. Replaying "
            "every newly covered step of the four-way calibration turned 13-14 "
            "environment failures per run into the model's own errors and "
            "produced no additional pass, so the wrapper leaves all four Qwen "
            "scores unchanged; a stronger model would otherwise lose up to "
            "13/288 sub-steps and all of problem 12, which is 0/14 without it. "
            "simps is the only removed API any problem declares (checked across "
            "both splits), but the sandbox otherwise runs its own pinned SciPy "
            "rather than the pre-1.14 one behind the official numbers, so "
            "model-generated code reaching for another retired API fails here "
            "though it would pass upstream; the prompt constraining models to the "
            "declared dependencies bounds that exposure. Steps the wrapper "
            "un-blocks execute for real instead of failing instantly, so problem "
            "12 adds substantial wall clock (12.14 alone exceeds the 1800 s "
            "step timeout); the timeouts counter surfaces this; "
            "(2) numeric h5 "
            "targets are read eval-side and inlined into the "
            "sandbox program (upstream reads test_data.h5 in-subprocess); "
            "(3) execution runs on a remote code-eval service over HTTP, not an "
            "in-process subprocess — the service caps sandbox memory (1 GB "
            "default) while upstream subprocesses are uncapped, so memory-heavy "
            "numeric steps could OOM here yet pass upstream; (4) pipeline "
            "failures count as unsolved problems in BOTH accuracies -- a failed "
            "problem's tested steps are recovered from its raw sample and stay in "
            "the sub-problem denominator scoring zero, so a full test split keeps "
            "the fixed 288-step denominator the official sub-problem figures are "
            "computed over rather than shrinking to whichever steps happened to "
            "run; the unevaluated_steps counter reports how many never executed. "
            "Reproducing official numbers requires greedy decoding "
            "(temperature=0) in the model config; with_background defaults to "
            "False (the official headline mode). The code-eval service image "
            "must provide the sandbox scientific stack including sympy, which "
            "the vendored comparison shim injects — see the evaluator's "
            "requirements/scicode.txt."
        ),
    ),
)
class SciCodeZeroShotGenTask(
    Task[
        SciCodeDatasetSample,
        PromptRecord,
        # infer is not a record stage, and its value is not a single ModelOutput:
        # a problem is one model call per generated step. The box carries the
        # per-step code as the stage value while `meta` reports every call's token
        # usage to the profiler (the mmmlu_kshot_clp precedent).
        TaskStageOutput[list[StepCode]],
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
        with_background: bool = False,
        h5_path: str | None = None,
        max_concurrency: int = 4,
        timeout: float = 1800.0,  # matches upstream test_generated_code.py
        special_step_mode: Literal["extract", "verbatim"] = "verbatim",
    ):
        if special_step_mode not in ("extract", "verbatim"):
            raise ValueError(
                "special_step_mode must be 'extract' or 'verbatim', "
                f"got {special_step_mode!r}"
            )
        super().__init__(dataset=dataset, model=model, name=name)
        self._with_background = with_background
        self._special_step_mode = special_step_mode
        self._h5_path = h5_path
        self._timeout = timeout
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def setup(self):
        if self._h5_path is None:
            self._h5_path = getattr(self.dataset, "h5_path", None)
        if not self._h5_path or not os.path.exists(self._h5_path):
            raise FileNotFoundError(
                "SciCode numeric test data not found. Run "
                "`sieval dataset download scicode` to stage raw_ground.h5, or pass "
                f"h5_path=. Resolved path: {self._h5_path!r}"
            )

    @override
    async def preprocess(self, raw, ctx):
        sub_steps = raw["sub_steps"]
        problem_id = str(raw["problem_id"])
        generated = [
            i for i in range(len(sub_steps)) if not is_special_step(problem_id, i)
        ]
        # A problem is a *sequence* of prompts, not one: step i's prompt embeds the
        # model's own code from steps 1..i-1, so only the first generated step's
        # prompt exists before inference. The rest are assembled in infer as that
        # code arrives, and are deliberately not persisted -- each restates every
        # prior step, so keeping all of them costs quadratic text for content
        # already on disk as `code_content`.
        previous_llm_code: list[str | None] = [None] * len(sub_steps)
        first = generated[0] if generated else None
        if first is None:
            # Nothing is generated for this problem. No such problem exists in
            # either split, but the record still has to be well-formed; infer's
            # loop makes no model call and the problem scores zero over zero steps.
            prompt, previous_code = "", ""
        else:
            # Every step before the first generated one is a gold step, by
            # definition of "first", and gold code is static -- which is exactly
            # what makes this prompt knowable before any inference.
            for i in range(first):
                previous_llm_code[i] = self._gold_code(sub_steps[i])
            prompt, previous_code = generate_prompt_with_steps(
                sub_steps,
                raw["required_dependencies"],
                first + 1,
                previous_llm_code,
                self._with_background,
            )
        return build_prompt_record(
            [{"role": "user", "content": prompt}] if first is not None else [],
            # No `reference`: the ground truth is a per-step test suite, a procedure
            # rather than a value. It is described at judgement time instead.
            extra={
                "problem_id": problem_id,
                # Say which step `prompt` belongs to and how many prompts the
                # sample really takes, so it cannot be misread as the whole input.
                "first_generated_step": (
                    sub_steps[first]["step_number"] if first is not None else None
                ),
                "n_generated_steps": len(generated),
                "with_background": self._with_background,
                # Handed to infer rather than recomputed there, so the code context
                # recorded for the first step cannot drift from the prompt above.
                "previous_code": previous_code,
            },
        )

    @override
    async def infer(self, pre, ctx):
        raw = ctx.raw_sample
        sub_steps = raw["sub_steps"]
        problem_id = str(raw["problem_id"])
        deps = raw["required_dependencies"]
        tot = len(sub_steps)

        previous_llm_code: list[str | None] = [None] * tot
        steps_out: list[StepCode] = []
        outputs: list[ModelOutput] = []
        # preprocess already assembled the first generated step's prompt -- it is
        # the record's `prompt`. Reuse it rather than rebuilding it here, so the
        # model provably receives what was recorded.
        seed_prompt = pre["prompt"]
        seed_previous_code = pre["extra"]["previous_code"]
        seeded = False

        for i in range(tot):
            step = sub_steps[i]
            step_number = step["step_number"]

            if is_special_step(problem_id, i):
                # Scientist-authored gold code: context only, never generated/tested.
                previous_llm_code[i] = self._gold_code(step)
                steps_out.append(
                    {
                        "step_number": step_number,
                        "tested": False,
                        "code_content": None,
                        "extracted_code": "",
                        "raw_response": "",
                        "empty_extraction": False,
                    }
                )
                continue

            if not seeded:
                messages, previous_code = seed_prompt, seed_previous_code
                seeded = True
            else:
                prompt, previous_code = generate_prompt_with_steps(
                    sub_steps, deps, i + 1, previous_llm_code, self._with_background
                )
                messages = [{"role": "user", "content": prompt}]
            output = await self.model.agenerate(messages, n=1)
            outputs.append(output)
            # Guard against an empty choices list (aborted/filtered response):
            # treat it like an empty extraction below instead of raising.
            raw_response = output.texts[0] if output.texts else ""
            extracted = extract_python_script(raw_response)
            # An extraction with no def/class means the model returned prose, was
            # truncated, or emitted an unfenced answer; the step will fail and so
            # will any later step calling its function. Log it and flag for the
            # report instead of silently scoring 0.
            empty = "def " not in extracted and "class " not in extracted
            if empty:
                logger.warning(
                    "SciCode empty code extraction: problem {} step {} "
                    "(finish={}, raw_len={}); dependent steps may cascade-fail.",
                    problem_id,
                    step_number,
                    (output.finish_reasons or ["?"])[0],
                    len(raw_response),
                )
            previous_llm_code[i] = extracted
            # Matches upstream save_response_with_steps: `{previous_code}\n{code}`.
            steps_out.append(
                {
                    "step_number": step_number,
                    "tested": True,
                    "code_content": f"{previous_code}\n{extracted}",
                    "extracted_code": extracted,
                    "raw_response": raw_response,
                    "empty_extraction": empty,
                }
            )

        # Box the structured per-step code as the stage value while recording
        # token usage from every model call via the stage meta.
        return TaskStageOutput(value=steps_out, meta=build_stage_meta(*outputs))

    @override
    async def postprocess(self, inf, ctx):
        steps_out: list[StepCode] = inf.value
        sub_steps = ctx.raw_sample["sub_steps"]
        by_number = {s["step_number"]: s for s in sub_steps}

        pending: list[tuple[StepCode, str, list[str]]] = []
        for sc in steps_out:
            code = sc["code_content"]
            if not sc["tested"] or code is None:
                continue
            pending.append((sc, code, by_number[sc["step_number"]]["test_cases"]))

        def read_targets() -> dict[str, str]:
            # Numeric targets come from a large h5 file: blocking I/O that would
            # stall the event loop for every other in-flight sample. One worker-
            # thread hop per problem reads (and pickles) them all; h5py serializes
            # HDF5 access internally, so concurrent per-sample threads are safe.
            return {
                sc["step_number"]: encode_targets(
                    process_hdf5_to_tuple(sc["step_number"], len(cases), self._h5_path)
                )
                for sc, _code, cases in pending
            }

        # `anyio.to_thread`, not `asyncio.to_thread`: the latter uses the loop's
        # own executor and so escapes anyio's CapacityLimiter, putting these
        # reads outside the session's thread budget. Shares the default limiter
        # with the loader and the deployer (grading has its own — see
        # `core/utils/offload.py`).
        targets_by_step = await run_sync(read_targets)

        programs: list[StepProgram] = []
        for sc, code, cases in pending:
            step_number = sc["step_number"]
            programs.append(
                {
                    "step_number": step_number,
                    "program": build_test_program(
                        code, targets_by_step[step_number], cases
                    ),
                    "empty_extraction": sc["empty_extraction"],
                }
            )

        # One rollout per problem, not one per step -- see the module docstring. The
        # prediction is the per-step code the model wrote; it is `None` (so
        # `extracted` reports the miss) only when no tested step yielded any
        # extractable code at all. A partial miss is still a real answer that scores
        # badly, and its per-step `empty_extraction` flags ride with the programs.
        step_code: list[JSONValue] = [
            {"step_number": sc["step_number"], "code": sc["extracted_code"]}
            for sc, _code, _cases in pending
        ]
        any_extracted = any(not sc["empty_extraction"] for sc, _code, _cases in pending)
        return build_prediction_record(
            [step_code if any_extracted else None],
            # The executable programs are the test harness, not the answer, so they
            # are mechanism detail -- and feedback reads them from here.
            extra={"programs": programs},
        )

    @override
    async def feedback(self, post, ctx):
        # Steps are evaluated sequentially ON PURPOSE. Task-runner sample
        # concurrency determines how many problems are in flight; max_concurrency
        # only caps simultaneous HTTP connections to the code-eval service.
        # Fanning out per-step would multiply that load. Worst-case wall clock is
        # steps x timeout for a problem whose steps all run long — more likely
        # under "verbatim", where dependent steps genuinely execute (under
        # "extract" they die instantly on NameError).
        feedbacks: list[StepFeedback] = []
        for step in post["extra"]["programs"]:
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{step['step_number']}-{time.perf_counter_ns()}",
                        "source": "scicode",
                        "code": step["program"],
                        "timeout": self._timeout,
                    },
                    # Must comfortably exceed the eval's OWN subprocess timeout
                    # (self._timeout) plus server-side profiling/serialization
                    # overhead. Too tight and a step that legitimately runs to
                    # the eval timeout trips a client ReadTimeout, failing the
                    # WHOLE problem instead of recording just that step as
                    # failed — the service already returns status=False (msg
                    # "subprocess timeout") on its own timeout.
                    # Do not time out while waiting for a pooled connection: the
                    # task runner may have more samples in flight than this
                    # client's max_connections limit. Connect/read/write retain
                    # the subprocess timeout plus transport-overhead buffer.
                    timeout=httpx.Timeout(self._timeout + 120, pool=None),
                )
                resp.raise_for_status()
                res = resp.json()
                feedbacks.append(
                    {
                        "step_number": step["step_number"],
                        "correct": res["status"],
                        "msg": res["msg"],
                        "empty_extraction": step["empty_extraction"],
                    }
                )
            except Exception as e:
                logger.warning(
                    "SciCode eval error for step {}: [{}] {}",
                    step["step_number"],
                    type(e).__name__,
                    e,
                )
                raise e

        n_tested = len(feedbacks)
        n_correct = sum(1 for fb in feedbacks if fb["correct"])
        # `correct` is main-problem accuracy's numerator: every tested step passed.
        # That is the axis comparable to other tasks -- "solved the problem" -- so
        # the step pass rate becomes partial credit rather than the headline.
        solved = bool(feedbacks) and n_correct == n_tested
        rate = n_correct / n_tested if n_tested else 0.0
        metrics: dict[str, bool | float] = {
            "main_problem_solved": solved,
            "sub_problem_pass_rate": rate,
        }
        return True, build_judgement_record(
            None,  # the reference is a procedure: this problem's per-step test suites
            [
                build_rollout_judgement(
                    0, solved, score=rate, metrics=metrics, extra={"steps": feedbacks}
                )
            ],
            score=rate,
            metrics=metrics,
            extra={
                # Raw counts, not just the rate above: report()'s sub-problem
                # accuracy pools these across problems, and per-problem rates cannot
                # reconstruct a pooled one when problems have different step counts.
                "correct_steps": n_correct,
                "total_steps": n_tested,
            },
        )

    @override
    async def report(self, finals, fails):
        total_problems = len(finals) + len(fails)
        if total_problems == 0:
            # Declared on this path too: which population the headline would have
            # been averaged over is a property of the task, not of the run.
            return {
                "score": 0.0,
                "fails": len(fails),
                SCORE_KEY_FIELD: "main_problem_accuracy",
                DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            }

        correct_steps = 0
        total_steps = 0
        correct_problems = 0
        empty_extractions = 0
        timeouts = 0
        memory_errors = 0
        import_errors = 0
        for f in finals:
            judgement = f.feedback_result
            # Pooled from the per-problem raw counts, not averaged from the
            # per-problem rates in `metrics` -- problems have different step counts,
            # so those are two different numbers.
            correct_steps += judgement["extra"]["correct_steps"]
            total_steps += judgement["extra"]["total_steps"]
            # Reads the headline verdict (one rollout, so n_correct is 1 for a solved
            # problem) instead of recomputing it, so the two cannot disagree.
            correct_problems += judgement["n_correct"]
            feedbacks = judgement["rollouts"][0]["extra"]["steps"]
            empty_extractions += sum(
                1 for fb in feedbacks if fb.get("empty_extraction")
            )
            messages = [str(fb.get("msg", "")).lower() for fb in feedbacks]
            timeouts += sum("timeout" in msg for msg in messages)
            memory_errors += sum("memoryerror" in msg for msg in messages)
            # ModuleNotFoundError is an ImportError subclass, but the evaluator
            # reports the concrete class name, which does not contain
            # "importerror". It is the signature of a package missing from the
            # code-eval image, so it must not read as import_errors=0.
            import_errors += sum(
                "importerror" in msg or "modulenotfounderror" in msg for msg in messages
            )

        # A pipeline failure is an unsolved problem in BOTH accuracies. Its tested
        # steps are recoverable from the raw sample, so they stay in the
        # sub-problem denominator scoring zero instead of vanishing from it.
        # Dropping them would push the two metrics in opposite directions -- main
        # diluted by the failure, sub silently inflated by its removal -- and
        # shrink a full test split below the fixed 288-step denominator the
        # official sub-problem numbers are computed over.
        unevaluated_steps = 0
        for f in fails:
            raw = f.raw_sample
            if raw is None:
                # Failed before the sample was loaded, so there is nothing to
                # count; such a problem still dilutes main-problem accuracy.
                continue
            problem_id = str(raw["problem_id"])
            unevaluated_steps += sum(
                1
                for idx in range(len(raw["sub_steps"]))
                if not is_special_step(problem_id, idx)
            )
        total_steps += unevaluated_steps

        main_accuracy = correct_problems * 100 / total_problems
        sub_accuracy = correct_steps * 100 / total_steps if total_steps else 0.0
        return {
            "score": main_accuracy,
            "main_problem_accuracy": main_accuracy,
            "sub_problem_accuracy": sub_accuracy,
            "correct_problems": correct_problems,
            "total_problems": total_problems,
            "correct_steps": correct_steps,
            "total_steps": total_steps,
            # How much of `total_steps` never ran because its problem failed the
            # pipeline. Those steps score zero, so a non-zero value here means
            # sub-problem accuracy is bounded by pipeline health rather than by
            # model capability alone -- read it alongside `fails`.
            "unevaluated_steps": unevaluated_steps,
            # Steps where the model produced no extractable code (truncation /
            # unfenced / prose). A non-zero count means some failures are
            # generation-side, not solution-correctness — investigate raw_response.
            "empty_extractions": empty_extractions,
            # Step-level execution failures remain incorrect answers, but surface
            # their causes so report.json does not hide systemic environment or
            # resource failures behind an otherwise healthy pipeline fails=0.
            "timeouts": timeouts,
            "memory_errors": memory_errors,
            "import_errors": import_errors,
            "fails": len(fails),
            # Main- and sub-problem accuracy are co-equal published rates over
            # different units (problems vs steps); `score_key` says which one the
            # headline is. `requested`: a problem that failed the pipeline counts
            # against `total_problems` rather than being excluded.
            SCORE_KEY_FIELD: "main_problem_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }

    @override
    async def shutdown(self):
        await self._http_client.aclose()

    def _gold_code(self, step) -> str:
        """Scientist-authored code for a special step, per ``special_step_mode``.

        Shared by preprocess and infer because both assemble a prompt whose context
        contains it: preprocess for the first generated step, infer for the rest.
        Injecting it two different ways would make the recorded prompt disagree with
        the one the model saw.
        """
        gold = special_step_code(step["step_number"])
        if self._special_step_mode == "extract":
            # Upstream behavior (gencode_json.py): re-extract the node named by the
            # header. For 13.6/62.1 the header's `def __init__` is matched before
            # `class`, so this yields a bare method and drops the class wrapper — a
            # known upstream bug (scicode-bench/SciCode #59, #49) that makes
            # dependent steps fail with NameError. Kept as an opt-in for parity with
            # the public leaderboard numbers.
            return get_function_from_code(
                gold, extract_function_name(step["function_header"])
            )
        # "verbatim" (default): inject the whole gold block (keeps the class) — the
        # fix proposed in #59/#49. Deviates from the public (buggy) pipeline by
        # design; see reference_impl.notes.
        return gold
