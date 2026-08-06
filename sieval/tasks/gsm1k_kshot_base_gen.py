"""
GSM1k few-shot generative task — the protocol Scale AI published GSM1k with.

Port of Scale's `gsm1k` task from its lm-evaluation-harness fork (pinned commit
`39294c6f`, `lm_eval/tasks/gsm1k/gsm1k_scale.yaml`):

* Prompt: `n_shot` exemplars of `"Question: {question}\\nAnswer: {answer}\\n\\n"`
  followed by `"Question: {question}\\nAnswer:"` — upstream's `doc_to_text` with
  its `doc_to_target` of `"{{answer}}"`, i.e. the exemplar answer verbatim,
  calculator annotations (`<<9*7=63>>`) and `#### N` line included.
* Answer extraction: upstream declares exactly one filter, `flexible-extract` —
  `regex` with `group_select: -1` over `(-?[$0-9.,]{2,})|(-?[0-9]+)`, then
  `take_first`. That is lm-eval's `RegexFilter`: `findall`, take the **last**
  match, first non-empty group, strip. It is also what the paper describes,
  "extracts the last numeric answer in the response and compares this to the
  correct answer".
* Scoring: `exact_match` with `ignore_case: true` and `regexes_to_ignore`
  `[",", "\\$", "(?s).*#### ", "\\.$"]`, applied to prediction **and** gold in
  that order, then lowercased. `_normalize_exact_match` reproduces the list
  verbatim; the third entry cannot fire after extraction (both sides are bare
  numbers by then) and is kept only so the normalizer reads as upstream's list
  rather than a subset of it.
* Stop sequences: `until: ["Question:", "</s>", "<|im_end|>"]`.

Metric names say which extraction rule produced them, and there is deliberately
**no** bare `exact_match` key. GSM1k's whole use is a paired diff against GSM8K,
and `gsm8k_kshot_base_gen` spells its *strict* (`#### N`) metric `exact_match`
while upstream GSM1k's only metric is the flexible one — so a shared key would
let a reader diff two different extraction rules and see a gap that is pure
extraction. Both rules are therefore reported over the same response:
`flexible_exact_match` (upstream's filter, and the headline `score` / `correct`)
and `strict_exact_match` (the `#### N` rule, which on GSM1k also measures whether
the model followed the 5-shot format at all, since the gold carries no `####`).

Deviations from Scale's harness (documented, not silent):

* **Few-shot exemplars are fixed, not resampled per question.** Upstream draws
  "five random examples from GSM8k to use as n-shot examples, which vary for each
  new question"; `_GSM8K_FEWSHOT_EXAMPLES` is one fixed set of 5, so every sample
  in a run shares one prompt prefix. This is the house pattern (`n_shot`
  exemplars sampled once, as in `gsm8k_kshot_base_gen`), and for a paired
  GSM8K/GSM1k read it is the stronger design: holding the prefix fixed keeps
  exemplar variance out of the diff, which is what upstream's single shared
  prompt is reaching for. It does move absolute scores relative to upstream's
  published column.
* **Exemplar provenance.** The 5 pairs are rows of `openai/gsm8k` train (revision
  `740312add88f781978c0658806c59bc2815b9866`, MIT) — GSM8k train is where
  upstream's exemplars come from too, and GSM1k has no train split and ships no
  worked solutions, so it cannot supply chain-of-thought exemplars of its own.
  They were taken once as `shuffle(seed=1234)[:5]`, which is what
  `gsm8k_kshot_base_gen` draws at `n_shot=5, fewshot_seed=1234` (verified
  byte-identical with `datasets` 4.4.1): running that sibling at those settings
  gives the same prompt prefix, so the pair can be run prompt-controlled.
* **`report()` divides by `len(finals) + len(fails)`** — a pipeline failure scores
  wrong rather than being excluded, the convention across this repo's tasks. Note
  `gsm8k_kshot_base_gen` divides by `len(finals)` instead, so a paired diff must
  be read with both `fails` counts in view; the two rules agree when `fails == 0`.

Comparison targets — the paper's Table 1, GSM8k → GSM1k (5-shot, temperature 0):
Meta-Llama-3-8B-Instruct 0.752 → 0.690 (diff 0.062), Meta-Llama-3-70B-Instruct
0.914 → 0.900 (0.014), Mistral-7B-Instruct-v0.2 0.428 → 0.419 (0.009), phi-2
0.566 → 0.504 (0.063). Instruct checkpoints appear in a base-model task because
upstream applies **no** chat template: its `lm_eval` invocation passes no
`--apply_chat_template`, so every model was prompted with this raw 5-shot
completion. The **diff** is the measurement; treat a single absolute column as
alignment evidence only after checking `fails` and the exemplar deviation above.

Repro decoding (model-layer assets — set via `models:` / `infer_args`, not here):
greedy `temperature=0`, and `max_tokens=1000` — upstream's one deliberate change
to lm-eval's defaults was raising the generation cap "from 256 to 1000" so chains
of thought are not truncated. This task forwards only the stop sequences, which
are coupled to the prompt format.

`status="experimental"`: faithful to upstream's declared protocol by
construction, but the fixed-exemplar deviation changes every prompt and no run
has yet been checked against the published column.

References:

* Paper: <https://arxiv.org/abs/2405.00332>
* Task config: <https://github.com/scaleapi/gsm1k_eval/blob/39294c6f31855aca8255b6174b22fc3a6311be0b/lm_eval/tasks/gsm1k/gsm1k_scale.yaml>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re
from typing import override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.datasets import GSM1KDatasetSample

N_SHOT = 5
STOP_SEQUENCES = ("Question:", "</s>", "<|im_end|>")

_STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_ANSWER_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

# `openai/gsm8k` train rows at revision 740312add88f781978c0658806c59bc2815b9866,
# taken as shuffle(seed=1234)[:5] — see the module docstring for why GSM1k borrows
# GSM8k's exemplars and why they are fixed. Verbatim, including the `<<...>>`
# calculator annotations and the `#### N` line, which is what upstream's
# `doc_to_target: "{{answer}}"` feeds the model. Upstream's en dash and right
# single quote are written as `\u2013` / `\u2019` so nobody can silently "fix"
# them to ASCII and break byte-fidelity.
_GSM8K_FEWSHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Rodney has 35 dollars more than Ian. Ian has half as much money as Jessica "
        "has. If Jessica has 100 dollars, how much more money does Jessica have than "
        "Rodney?",
        "Ian has 100/2 = <<100/2=50>>50 dollars.\n"
        "Rodney has 50+35 = <<50+35=85>>85 dollars.\n"
        "Jessica has 100-85 = <<100-85=15>>15 more dollars than Rodney.\n"
        "#### 15",
    ),
    (
        "Lynne bought 7 books about cats and 2 books about the solar system. She also "
        "bought 3 magazines. Each book cost 7$ and each magazine cost $4. How much did "
        "Lynne spend in all?",
        "Lynne bought a total of 7 + 2 = <<7+2=9>>9 books\n"
        "The books cost Lynne 9 x 7 = $<<9*7=63>>63\n"
        "For 3 magazines, Lynne spent 3 x 4 = $<<3*4=12>>12\n"
        "In total, Lynne spent 63 + 12 = <<63+12=75>>75$\n"
        "#### 75",
    ),
    (
        "Traci and Harris are baking cakes together. Traci has brought flour from her "
        "own house and Harris has 400g of flour in his house. Each cake needs 100g of "
        "flour and Traci and Harris have created 9 cakes each. How much flour, in "
        "grams, did Traci bring from her own house?",
        "To make the cakes, Traci and Harris used a total of 9 cakes * 100g of flour "
        "per cake = <<9*100=900>>900g of flour.\n"
        "Traci therefore brought 900g of needed flour \u2013 400g of flour from "
        "Harris\u2019 flour = 500g of flour from her own house.\n"
        "#### 500",
    ),
    (
        "Carly recently graduated and is looking for work in a field she studied for. "
        "She sent 200 job applications to companies in her state, and twice that "
        "number to companies in other states. Calculate the total number of job "
        "applications she has sent so far.",
        "If she sent 200 job applications to her state, she sent 200*2 = "
        "<<200*2=400>>400 job applications to other states.\n"
        "The total number of job applications she has sent is 400+200 = "
        "<<400+200=600>>600\n"
        "#### 600",
    ),
    (
        "Bert was able to sell 8 toy phones for $18 each, while Tory was able to sell "
        "7 toy guns for $20 each. How much more did Bert earn than Tory?",
        "Bert was able to earn 8 x $18 = $<<8*18=144>>144 for the toy phones.\n"
        "While Tory was able to earn 7 x $20 = $<<7*20=140>>140 for the toy guns.\n"
        "Therefore, Bert was able to earn $144 \u2013 $140 = $<<144-140=4>>4 more than "
        "Tory.\n"
        "#### 4",
    ),
)


def _format_example(question: str, answer: str | None) -> str:
    """Render upstream's `doc_to_text`, with the target appended for an exemplar."""
    prompt = f"Question: {question}\nAnswer:"
    if answer is not None:
        prompt += f" {answer}\n\n"
    return prompt


def _normalize_exact_match(text: str) -> str:
    # lm-eval `exact_match`: `regexes_to_ignore` in upstream's order, then
    # `ignore_case`. The `#### ` entry is a no-op on an already-extracted number
    # (both extractors return one) and is kept to mirror upstream's list.
    text = re.sub(r",", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text.strip())
    return text.strip().lower()


def _extract_strict_answer(text: str) -> str:
    """The `#### N` rule. `""` when the response never emits that format."""
    match = _STRICT_ANSWER_RE.search(text)
    return _normalize_exact_match(match.group(1)) if match else ""


def _extract_flexible_answer(text: str) -> str:
    """Upstream's only filter: last regex match, first non-empty group, stripped."""
    matches = _FLEXIBLE_ANSWER_RE.findall(text)
    if not matches:
        return ""
    last = next((part for part in matches[-1] if part), "")
    return _normalize_exact_match(last) if last else ""


@sieval_task(
    name="gsm1k_kshot_base_gen",
    display_name="GSM1k (few-shot, base generative)",
    description="GSM1k few-shot eval on Scale AI's published lm-eval-harness protocol.",
    eval_mode=EvalMode.GEN,
    n_shot=N_SHOT,
    tags=("english", "math-word-problems", "open-ended", "base-model"),
    model_type="gen",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="scaleapi/gsm1k_eval",
        url=(
            "https://github.com/scaleapi/gsm1k_eval/blob/39294c6f31855aca8255b6174b22fc3a6311be0b/lm_eval/tasks/gsm1k/gsm1k_scale.yaml"
        ),
        notes=(
            "Scale's own lm-evaluation-harness fork task `gsm1k`: prompt "
            '"Question: {q}\\nAnswer:" with 5 GSM8k-train exemplars, one '
            "`flexible-extract` filter (last numeric match) and `exact_match` "
            "with regexes_to_ignore [',', '$', '(?s).*#### ', '.$'] + "
            "ignore_case. Upstream resamples the 5 exemplars per question and "
            "raises max generation length from 256 to 1000 tokens; this task "
            "fixes one exemplar set (documented in the module docstring) and "
            "leaves max_tokens to the model layer, where 1000 matches "
            "upstream. Repeats: upstream runs `repeats: 1`, greedy at "
            "temperature 0 — match it with n=1 and temperature=0. GSM1k is a "
            "paired benchmark: read it as a diff against GSM8K on the same "
            "extraction rule, not as a standalone score."
        ),
    ),
)
class GSM1KFewShotBaseGenTask(
    Task[
        GSM1KDatasetSample,
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
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if not 0 <= n_shot <= len(_GSM8K_FEWSHOT_EXAMPLES):
            raise ValueError(
                "n_shot must be between 0 and "
                f"{len(_GSM8K_FEWSHOT_EXAMPLES)} (the number of vendored GSM8k "
                f"exemplars), got {n_shot}"
            )
        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._stop = stop

    @override
    async def preprocess(self, raw, ctx):
        prefix = "".join(
            _format_example(question, answer)
            for question, answer in _GSM8K_FEWSHOT_EXAMPLES[: self.n_shot]
        )
        return build_prompt_record(
            prefix + _format_example(raw["question"], None),
            reference=_normalize_exact_match(raw["answer"]),
        )

    @override
    async def infer(self, pre, ctx):
        # Keep `stop` out of the kwargs when unset so it can't clobber the
        # model's configured stop via the `{**self._kwargs, **kwargs}` merge.
        if self._stop:
            return await self.model.agenerate(pre["prompt"], stop=list(self._stop))
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        text = inf.texts[0] if inf.texts else ""
        # The flexible rule is upstream's only filter, so it is the headline
        # prediction; the strict `#### N` rule is a second extraction RULE over
        # the same response, not a second rollout, so it rides in `extra`. `""`
        # there means "this rule found nothing" (the sibling GSM8K task's
        # convention) — a `None` would be dropped by serialization and read as
        # "never measured" on resume.
        return build_prediction_record(
            [_extract_flexible_answer(text) or None],
            extra={"strict_prediction": _extract_strict_answer(text)},
        )

    @override
    async def feedback(self, post, ctx):
        gold = _normalize_exact_match(ctx.raw_sample["answer"])
        # Both extraction rules are recorded as co-equal metrics over one
        # response; `correct` is DERIVED from the flexible one (upstream's single
        # filter) so the headline and the metric cannot drift.
        prediction = post["rollouts"][0].get("prediction") or ""
        metrics: dict[str, bool | float] = {
            "flexible_exact_match": prediction == gold,
            "strict_exact_match": post["extra"]["strict_prediction"] == gold,
        }
        return True, build_judgement_record(
            gold,
            [
                build_rollout_judgement(
                    0, bool(metrics["flexible_exact_match"]), metrics=metrics
                )
            ],
            metrics=metrics,
        )

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set: a pipeline failure counts as
        # wrong, not as an excluded sample.
        total = len(finals) + len(fails)
        if total == 0:
            return {
                "score": 0.0,
                "fails": len(fails),
                "flexible_exact_match": 0.0,
                "strict_exact_match": 0.0,
            }
        flexible = (
            100
            * sum(
                1
                for ctx in finals
                if ctx.feedback_result["metrics"]["flexible_exact_match"]
            )
            / total
        )
        strict = (
            100
            * sum(
                1
                for ctx in finals
                if ctx.feedback_result["metrics"]["strict_exact_match"]
            )
            / total
        )
        return {
            "score": flexible,
            "fails": len(fails),
            "flexible_exact_match": flexible,
            "strict_exact_match": strict,
        }
