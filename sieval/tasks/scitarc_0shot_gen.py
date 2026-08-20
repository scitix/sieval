"""SciTaRC — 0-shot generative table QA, LLM-grader + exact-match scored.

Generative port of SciTaRC (JHU-CLSP; Wang et al., COLM 2026). The model gets
the LaTeX source of the table(s) a question needs and answers in free text under
an ``Answer:`` instruction; an **LLM grader** rates the extracted answer against
the gold on upstream's ternary scale, binarised so only a full 1.0 counts. That
accuracy is the headline, reported next to upstream's second column: a strict,
case-sensitive exact match.

Upstream's protocol is a grid — ``plan_mode`` (``none``/``auto``/``oracle``) x
``exec_mode`` (``language``/``code``). This is the ``none`` x ``language`` cell,
the one behind upstream's zero-shot leaderboard (Table 2, N=371). The other five
are not ported:

* **The two ``code`` cells are an execution-safety stop, not a scope call.**
  ``extract_answer_code`` ``exec``s model-authored Python in-process with the
  real ``__builtins__``, no sandbox, no timeout, behind a bare ``except:``.
  Grading is synchronous on one shared event loop, so an unbounded ``exec``
  stalls the session, not the sample. A sandboxed reading is a separate task, so
  Table 2's PoT rows are out of this file's reach.
* **The ``auto``/``oracle`` plan cells are upstream's Table 4 ablation**, over
  four models rather than the leaderboard. Cheap to add later as siblings — the
  ``plan`` column and the pseudo-code spec are already in the dataset.

**Ported as chat, not completion, because upstream's script cannot have produced
most of Table 2.** ``generate.py`` is vLLM-only and calls ``LLM.generate()``,
not ``LLM.chat()``, so no chat template is applied. Yet nine of Table 2's 25
rows are proprietary (GPT-5, Grok-4.1, Claude-Opus-4.5, Gemini-2.5-Flash,
Kimi-K2/K2-Thinking, GPT-4o), two of those vendors expose no completion endpoint,
and the repo holds no API path — the paper's only mention of one is Appendix G's
"Proprietary models are omitted because of API constraints on long prompts".
The harmony-format and ``-Think`` open-weight rows (GPT-OSS-120B 62.5,
DeepSeek-V3.2-thinking 73.6) could not score as published without their
templates either. The prompt string is upstream's byte for byte, sent as a
**single user turn** — upstream bakes its persona into the same string as the
table block, so splitting out a system message would change the rendered text.

Decoding params are model-layer, via ``models:``/``infer_args`` — never this
task. Upstream samples at ``temperature=0.1``, ``top_p=0.95``,
``max_tokens=2048``, ``repetition_penalty=1.05``, one rollout per question
(``n=1`` is the protocol, not just this task's default).

Grader is a REAL LLM via the ``grader`` task arg on its own
``api_base``/``api_key``; upstream's is ``Llama-3.3-70B-Instruct`` at
``temperature=0.0``, ``top_p=1.0``, ``max_tokens=512``,
``stop=["[Evaluation End]"]``, ``max_model_len=4096``. Scores depend on the
grader endpoint's model version (not pinnable like a Hub revision) — pin it, and
expect a stronger grader to move the column. Two upstream assumptions a chat
endpoint breaks are handled in the parser, not the prompt: the stop sequence
cannot be assumed, so the parser truncates on the marker itself; and the
prompt's JSON example would parse as a perfect score, so the parser reads only
what follows the last ``[Evaluation Start]`` — see ``sieval.community.scitarc``.
Each rollout persists the ternary ``score``, the ``grader_parsed`` flag and the
grader's whole ``ModelOutput`` (``extra.grader_output``).

An answer extracting to nothing is scored 0.0 **without** a grader call, matching
upstream's ``if item['prediction'].strip()`` filter; those rollouts carry
``grader_skipped`` and are counted by ``n_unextracted``, not
``n_grader_unparsed``.

Target: upstream Table 2 (LLM-Judge % / EM %, N=371) — GPT-5 76.8 / 22.1,
DeepSeek-V3.2 non-thinking 69.3 / 11.6, Qwen2.5-72B-Instruct 42.0 / 4.9,
Llama-3.3-70B-Instruct 34.5 / 5.1.

**Measured** (2026-08-20; Llama-3.3-70B-Instruct as both candidate and grader —
upstream's own — via OpenRouter pinned to Crusoe bf16, upstream's sampling
params, all 371 rows, ``fails=0``, ``n_grader_unparsed=0``): **34.77 accuracy /
19.68 exact match**, against the row's 34.5 / 5.1.

The headline reproduces: +0.27pp against a paired sigma of 3.49pp is 0.08 sigma,
indistinguishable. The paper's independent "collapse[s] on 65.5% of the
benchmark" agrees (100 − 34.77 = 65.23).

**The exact-match column does not, and that is the protocol showing through, not
a porting defect.** +14.58pp at 6.18 sigma is not sampling noise. EM is
grader-independent but *not* protocol-independent: upstream's extractor falls
back to the WHOLE reply when no ``Answer:`` marker is present, so it scores
whether the answer got *formatted* as much as whether it was right — and
formatting is what a chat template governs. This port emitted the marker on
97.0% of rollouts (median extracted answer 8 chars vs a 21.5-char mean gold),
which is what lets EM fire at all; a template-free completion call is what
yields upstream's near-zero column. Table 2's shape agrees: its ~0-EM rows are
open-weight (Llama-3.1-8B 0.0, Qwen-3-8B 0.0, GPT-OSS-120B 0.3, Gemma-3-27B 0.8)
while the rows that score (Grok-4 34.0, DeepSeek-V3.2-thinking 32.1, GPT-5 22.1)
are API models that had to use chat. Compare the headline against Table 2's
LLM-Judge column; read ``exact_match`` as this task's own measurement.

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
    status="stable",
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
            "leaderboard (Table 2, N=371). Prompt, `Answer:` extractor, grader "
            "prompt, grader-reply parser and exact-match normaliser are "
            "vendored in sieval.community.scitarc and checked by executing "
            "upstream's own functions (AST-lifted, since generate.py imports "
            "vllm at module scope): EVAL_PROMPT byte-identical to "
            "eval_prompt.txt as loaded (.strip()ed); "
            "create_language_prompt/get_table_text over 2,000 randomised table "
            "sets; extract_answer_language over 4,000 replies; parse_response "
            "(score AND reasoning) over 29 hand-built plus 3,000 fuzzed; "
            "normalize_text over 2,000. "
            'TWO DELIBERATE DIVERGENCES. (1) On a reply parsing to {"score": '
            "null} upstream raises TypeError out of float(None) and kills the "
            "batch; this port records it unparsed and scores 0.0. (2) The reply "
            "is cut to the text after the last [Evaluation Start] before "
            "parsing. Upstream's completion endpoint returns only the "
            "continuation, so its parser never sees the prompt; a chat grader "
            "can quote the template back, and the prompt's JSON example — ahead "
            "of the [Evaluation End] truncation point, and Method 1 takes the "
            "FIRST brace-run — would then outrank a real verdict following it. "
            "The cut rebuilds that boundary rather than retuning the score "
            "patterns, and is a no-op on every reply upstream could produce. "
            "Score impact, replayed over the stored run below: 0.00pp — the "
            "marker appears in 4 of 369 grader replies, every one with its "
            "verdict after it, 0 verdicts change. Pinned by tests on the bare "
            "echo, echo-then-verdict and first-match ordering; the grader's "
            "full ModelOutput is persisted so such replies stay auditable. "
            "CHAT vs COMPLETION: generate.py is vLLM-only and calls "
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
            "stop, so Table 2's PoT rows are out of reach) and "
            "plan_mode=auto/oracle (Table 4's four-model ablation). "
            "GRADING: upstream's ternary 1.0/0.5/0.0 binarised so only 1.0 is "
            "correct, matching its summary; an answer extracting to nothing "
            "scores 0.0 with no grader call, as upstream's non-empty filter "
            "does. Exact match is strict and CASE-SENSITIVE (upstream leaves "
            "the .lower() commented out). Grader is a REAL LLM via the `grader` "
            "task arg (upstream: Llama-3.3-70B-Instruct, temperature=0.0, "
            'top_p=1.0, max_tokens=512, stop=["[Evaluation End]"], '
            "max_model_len=4096); scores depend on the grader endpoint's model "
            "version, so pin it. "
            "SAMPLING: one rollout per question, no published repeat protocol; "
            "generation at temperature=0.1, top_p=0.95, max_tokens=2048, "
            "repetition_penalty=1.05 (model-layer, via models:/infer_args). "
            "VALIDATION: Llama-3.3-70B-Instruct scored 34.77 accuracy / 19.68 "
            "exact match vs the Table 2 row's 34.5 / 5.1 (2026-08-20, all 371 "
            "rows, fails=0, n_grader_unparsed=0, n_unextracted=2, "
            "n_truncated=5 at upstream's own max_tokens=2048; grader was "
            "upstream's own Llama-3.3-70B-Instruct, OpenRouter pinned to "
            'Crusoe bf16 to match dtype="bfloat16"). The headline is 0.08 sigma '
            "off (+0.27pp against a paired sigma of 3.49pp) — indistinguishable. "
            "EM is +14.58pp at 6.18 sigma, real and expected: EM is "
            "grader-independent but NOT protocol-independent, because "
            "upstream's extractor falls back to the whole reply absent an "
            "`Answer:` marker, so it scores formatting as much as correctness "
            "and formatting is what the chat template governs. This port "
            "emitted the marker on 97.0% of rollouts (median extracted answer "
            "8 chars vs 21.5-char mean gold); a template-free completion call "
            "is what produces upstream's near-zero open-weight EM column, "
            "whose shape corroborates it. Compare the headline against Table "
            "2's LLM-Judge column; read exact_match as this task's own "
            "measurement. status=stable on the headline, with that EM column "
            "as its one stated limit."
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
            # Rollouts the grader was CALLED on: `grader_skipped` excluded,
            # unparsed included — the denominator `n_grader_unparsed` is a rate
            # over. Spelled out because the judged family splits three ways:
            # hle/sysbench count parsed replies only, simpleqa_verified/
            # complex_constraints count every rollout in `finals`.
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
