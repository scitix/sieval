"""SciTaRC — 0-shot generative table QA, LLM-grader + exact-match scored.

Generative port of SciTaRC (JHU-CLSP; Wang et al., COLM 2026). The model is
handed the LaTeX source of the table(s) a question needs and answers in free
text under an ``Answer:`` instruction; a separate **LLM grader** rates the
extracted answer against the gold on upstream's ternary scale, which is
binarised so only a full 1.0 counts. Headline metric is that accuracy, reported
next to upstream's second column: a strict, case-sensitive exact match.

Upstream's protocol is a grid — ``plan_mode`` (``none`` / ``auto`` / ``oracle``)
x ``exec_mode`` (``language`` / ``code``). This task is the ``none`` x
``language`` cell, which is the one behind upstream's zero-shot leaderboard
(Table 2, N=371); the other five are deliberately not ported:

* **The two ``code`` cells are an execution-safety stop, not a scope call.**
  Upstream's ``extract_answer_code`` ``exec``s model-authored Python in-process
  with the real ``__builtins__``, no sandbox and no timeout, behind a bare
  ``except:`` that turns any failure into an empty answer. sieval grades
  synchronously on one shared event loop, so an unbounded ``exec`` stalls the
  session rather than the sample. Reproducing it is out of the question; a
  sandboxed reading is a separate task, not a hardening of this one, so this
  file does not claim to cover the PoT rows of Table 2.
* **The ``auto`` / ``oracle`` plan cells are upstream's Table 4 ablation**, run
  over four models rather than the leaderboard. They are cheap to add later as
  sibling tasks over the same dataset — the ``plan`` column and the pseudo-code
  spec are both already in place — and are left out because nothing needs them
  yet.

**Upstream's published script cannot have produced most of Table 2, so this is
a chat port on purpose.** ``generate.py`` is vLLM-only and calls
``LLM.generate()``, which applies no chat template (``LLM.chat()`` is the entry
point that does). Yet nine of Table 2's twenty-five rows are proprietary
(GPT-5, Grok-4.1, Claude-Opus-4.5, Gemini-2.5-Flash, Kimi-K2/K2-Thinking,
GPT-4o) and two of those vendors expose no completion endpoint at all; the repo
holds nine files on one branch with no API path anywhere, and the paper reports
no serving details (its only mention of an API is Appendix G's "Proprietary
models are omitted because of API constraints on long prompts"). Several
open-weight rows are equally unreachable that way — GPT-OSS-120B at 62.5 under
harmony formatting, DeepSeek-V3.2-thinking at 73.6, the ``-Think`` checkpoints —
since a reasoning model prompted without its template never opens its reasoning
channel. So the shipped script is a partial artifact, and the chat reading is
the one that can produce the published column. The prompt string itself is
upstream's, byte for byte, sent as a **single user turn**: upstream bakes the
"You are a helpful science assistant" persona into the same string as the table
block, so splitting a system message out of it would change the rendered text.

Decoding params are model-layer, set via ``models:`` / ``infer_args`` — never by
this task. Upstream's ``generate.py`` samples at ``temperature=0.1``,
``top_p=0.95``, ``max_tokens=2048``, ``repetition_penalty=1.05``, one rollout
per question (no repeats — ``n=1`` is the protocol, not just this task's
default).

Grader is a REAL LLM supplied via the ``grader`` task arg on its own
``api_base``/``api_key``; upstream's is ``Llama-3.3-70B-Instruct`` at
``temperature=0.0``, ``top_p=1.0``, ``max_tokens=512``,
``stop=["[Evaluation End]"]``, ``max_model_len=4096``. Correctness depends on
the grader endpoint's model version (not pinnable like a Hub revision) — pin
it, and expect a stronger grader to move the column. Two upstream quirks the
port inherits and records rather than repairs: the ``[Evaluation End]`` stop
sequence cannot be assumed on an arbitrary chat endpoint (the parser truncates
on the marker itself instead), and the rendered grader prompt contains a JSON
example that parses as a perfect score, so a grader that quotes the template
back reads high — see ``sieval.community.scitarc``. Each rollout persists the
grader's ternary ``score``, its ``grader_parsed`` flag and the grader's whole
``ModelOutput`` (``extra.grader_output``), which is the only durable evidence of
a verdict a re-grade need not reproduce.

An answer that extracts to nothing is scored 0.0 **without** a grader call,
matching upstream's ``if item['prediction'].strip()`` filter; those rollouts
carry ``grader_skipped`` and are counted by ``n_unextracted``, not by
``n_grader_unparsed``.

Target: upstream Table 2 (LLM-Judge % / EM %, N=371) — e.g. GPT-5 76.8 / 22.1,
DeepSeek-V3.2 non-thinking 69.3 / 11.6, Qwen2.5-72B-Instruct 42.0 / 4.9,
Llama-3.3-70B-Instruct 34.5 / 5.1. The EM column is grader-independent and is
therefore the cleaner alignment anchor of the two.

References:

* Paper: <https://arxiv.org/abs/2603.08910>
* Harness: <https://github.com/JHU-CLSP/SciTaRC>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.scitarc import (
    CORRECT_SCORE,
    EVAL_PROMPT,
    create_language_prompt,
    exact_match,
    extract_answer_language,
    parse_response,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RequirementContext,
    RolloutJudgement,
    Task,
    TaskModelRequirement,
    TaskRequirements,
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
)
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import SciTaRCDataset, SciTaRCDatasetSample

#: Upstream's ternary scale awards this for a partially correct answer. It is
#: not credited by `accuracy` (only `CORRECT_SCORE` is) but is reported as its
#: own rate, since a run that shifts from 0.0 to 0.5 has moved without moving
#: the headline.
PARTIAL_SCORE = 0.5


@sieval_task(
    name="scitarc_0shot_gen",
    display_name="SciTaRC (0-shot, generative)",
    description=(
        "Composite multi-step QA over raw LaTeX tables, "
        "LLM-grader + exact-match scored."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "reasoning", "tabular"),
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="SciTaRC",
        # The whole harness at the pinned commit: upstream keeps generate.py,
        # evaluate.py, eval_prompt.txt and exact_match.py at the repo root, so
        # the root tree is the directory this task mirrors.
        url="https://github.com/JHU-CLSP/SciTaRC/tree/d96f4e7b0d312cf1bfc2cc16345cd150ba0fa78d/",
        notes=(
            "Generative port of SciTaRC (JHU-CLSP), upstream's plan_mode=none x "
            "exec_mode=language cell — the protocol behind its zero-shot "
            "leaderboard (Table 2, N=371). The prompt, the `Answer:` extractor, "
            "the grader prompt, the grader-reply parser and the exact-match "
            "normaliser are vendored in sieval.community.scitarc and were "
            "checked against upstream by executing upstream's own functions "
            "(lifted by AST, since generate.py imports vllm at module scope) on "
            "shared inputs: EVAL_PROMPT byte-identical to eval_prompt.txt as "
            "loaded (.strip()ed); create_language_prompt/get_table_text "
            "identical over 2,000 randomised table sets; "
            "extract_answer_language over 4,000 replies; parse_response "
            "(score AND reasoning) over 29 hand-built replies plus 3,000 fuzzed "
            "ones; normalize_text over 2,000. Exactly one divergence, "
            'deliberate: on a reply parsing to {"score": null} upstream raises '
            "TypeError out of float(None) and kills the batch, where this port "
            "records the reply as unparsed and scores it 0.0. "
            "CHAT vs COMPLETION: upstream's generate.py is vLLM-only and calls "
            "LLM.generate(), which applies no chat template — but it cannot be "
            "the artifact behind Table 2, whose nine proprietary rows have no "
            "completion endpoint to call (the repo has no API path; the paper "
            "reports no serving details) and whose harmony-format and -Think "
            "open-weight rows could not score as published without their "
            "templates. Ported as chat, one user turn carrying upstream's "
            "prompt string unchanged. "
            "NOT PORTED: the two exec_mode=code cells (upstream execs "
            "model-authored Python in-process with real __builtins__, no "
            "sandbox, no timeout, behind a bare except — an execution-safety "
            "stop, so the PoT rows of Table 2 are out of this task's reach) and "
            "the plan_mode=auto/oracle cells (Table 4's four-model ablation). "
            "GRADING: upstream's ternary 1.0/0.5/0.0 binarised so only 1.0 is "
            "correct, matching its summary; an answer extracting to nothing is "
            "scored 0.0 with no grader call, as upstream's non-empty filter "
            "does. Exact match is strict and CASE-SENSITIVE (upstream leaves "
            "the .lower() commented out), which is why its published EM column "
            "runs far below the grader column — and being grader-independent it "
            "is the cleaner alignment anchor. "
            "REPRODUCIBILITY: grader is a REAL LLM (upstream: "
            "Llama-3.3-70B-Instruct, temperature=0.0, top_p=1.0, "
            'max_tokens=512, stop=["[Evaluation End]"], max_model_len=4096) '
            "supplied via the `grader` task arg; scores depend on the grader "
            "endpoint's model version, so pin it. The rendered grader prompt "
            "self-parses as 1.0 (its JSON example matches the parser's own "
            "pattern, ahead of the [Evaluation End] truncation point) — "
            "unreachable upstream, whose completion endpoint returns only the "
            "continuation, but live for a chat grader that quotes the template "
            "back; kept as-is and pinned by a test, with the grader's full "
            "ModelOutput persisted so the case is auditable. "
            "SAMPLING: upstream runs one rollout per question and publishes no "
            "repeat protocol; generation at temperature=0.1, top_p=0.95, "
            "max_tokens=2048, repetition_penalty=1.05 (model-layer, set via "
            "models:/infer_args). "
            "status=experimental: faithful to the reachable upstream artifact "
            "and parity-checked against it, but not yet measured against a "
            "published row."
        ),
    ),
)
class SciTaRCZeroShotGenTask(
    Task[
        SciTaRCDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    @classmethod
    @override
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        candidate = super().model_requirements_for(context)
        grader = cls._bind_role_requirement(
            context,
            "grader",
            TaskRequirements(input=InputKind.CHAT),
        )
        return candidate + grader

    def __init__(
        self,
        dataset: SciTaRCDataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
        models_by_role: Mapping[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._resolve_role_model(
            "grader",
            grader,
            models_by_role,
            build=self._build_grader,
        )

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config
        mapping (the YAML path, e.g. ``{model: ..., api_base: ...}``). Grading
        is mandatory — exact match alone is not this task's headline — so
        ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "SciTaRC requires an LLM grader. Pass `grader:` in the task args — "
            "a model-config dict such as {model: llama-3.3-70b-instruct, "
            "api_base: ..., api_key: ..., temperature: 0}. Upstream grades with "
            "Llama-3.3-70B-Instruct; pin the grader to compare against its "
            "published column."
        )

    @override
    async def preprocess(self, raw, ctx):
        # One user turn: upstream's prompt already carries the persona, the
        # table block and the `Answer:` instruction in a single string, so
        # lifting a system message out of it would change the rendered text.
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": create_language_prompt(
                        raw["question"], raw["relevant_tables"]
                    ),
                }
            ],
            reference=raw["answer"],
            extra={"paper": raw["paper"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # `extract_answer_language` falls back to the whole reply when no
        # `Answer:` marker is present, so `None` here means the *extracted*
        # answer was blank — which is exactly the condition upstream's grader
        # filter skips on, and what `n_unextracted` counts.
        predictions: list[str | None] = []
        for text in inf.texts:
            extracted = extract_answer_language(text)
            predictions.append(extracted if extracted.strip() else None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout, recording the grader's full output.

        ``extra["grader_output"]`` is the grader's whole ``ModelOutput``
        flattened to a plain dict — reply text, reasoning, usage, finish
        reasons, model id. Nothing is hand-picked, so a future ``ModelOutput``
        field is captured for free, and a reasoning grader that spends its
        entire budget thinking is distinguishable (via ``finish_reasons``) from
        an empty API response.

        ``grader_parsed`` is the companion to upstream's undifferentiated 0.0:
        upstream defaults an unreadable reply to the same score a genuinely
        wrong answer earns, so without the flag ``n_grader_unparsed`` could not
        be counted at all. ``grader_skipped`` marks the rollouts whose answer
        extracted to nothing — upstream never asks the grader about those, so
        they are a third case again, and folding them into either count would
        misname them.

        The ternary score goes in ``score`` and the exact match in ``metrics``:
        both are measured per rollout, and EM is co-equal rather than
        incidental — upstream publishes it as its own column.
        """
        gold = ctx.raw_sample["answer"]
        question = ctx.raw_sample["question"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction")
            metrics = {"exact_match": exact_match(predicted, gold)}
            if predicted is None:
                # Upstream's `if item['prediction'].strip()` filter: no grader
                # call, score 0.0, no reasoning.
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"],
                        False,
                        score=0.0,
                        metrics=metrics,
                        extra={"grader_skipped": True, "grader_parsed": False},
                    )
                )
                continue

            prompt = EVAL_PROMPT.format(
                question=question,
                ground_truth=gold,
                prediction=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            score, reasoning, parsed = parse_response(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    score == CORRECT_SCORE,
                    score=score,
                    metrics=metrics,
                    extra={
                        "grader_skipped": False,
                        "grader_parsed": parsed,
                        "grader_reasoning": reasoning,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        n_correct = 0
        n_partial = 0
        n_exact = 0
        n_graded = 0
        n_grader_unparsed = 0
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                extra = rollout.get("extra") or {}
                if rollout["correct"]:
                    n_correct += 1
                elif rollout.get("score") == PARTIAL_SCORE:
                    n_partial += 1
                if (rollout.get("metrics") or {}).get("exact_match"):
                    n_exact += 1
                if extra.get("grader_skipped"):
                    continue
                n_graded += 1
                if not extra.get("grader_parsed"):
                    n_grader_unparsed += 1

        # Denominator spans the full requested set: a pipeline failure produced
        # no gradeable answer and counts as wrong, matching upstream (whose
        # total is every generated row) and the *_gen family.
        n = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / n, 2)) if n else (lambda c: 0.0)
        return {
            "score": rate(n_correct),
            "accuracy": rate(n_correct),
            "exact_match": rate(n_exact),
            # Upstream's middle grade. It is not in its summary, but it is on
            # its scale, and a run that moved 0.0 -> 0.5 has changed without
            # changing the headline.
            "partial": rate(n_partial),
            "n": float(n),
            "n_graded": float(n_graded),
            "fails": float(len(fails)),
            "n_grader_unparsed": float(n_grader_unparsed),
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            # `n_grader_unparsed` counts the GRADER failing to answer;
            # `n_unextracted` counts the candidate producing nothing to grade,
            # which is also the count of rollouts the grader was never asked
            # about. Both score 0.0, and without the pair they are
            # indistinguishable in the report. Deliberately only
            # `health_metrics` and not the rest of the sampling block: RFC #74
            # defers pass@k / maj@k for the LLM-graded family, while this one
            # measures extraction rather than the draw and is outside that gate.
        } | health_metrics(finals)
