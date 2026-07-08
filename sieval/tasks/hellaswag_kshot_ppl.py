"""
HellaSwag k-shot log-likelihood (PPL) task — commonsense sentence completion.

Aligned with the EleutherAI lm-evaluation-harness ``hellaswag`` task
(``output_type: multiple_choice``): for each of the 4 endings the conditional
log-probability of the ending given the context is scored, then ``acc`` is the
argmax over the raw log-prob sums and ``acc_norm`` is the argmax over the
character-length-normalized sums (the headline HellaSwag metric, reported as
``score``).

Few-shot (``k`` > 0) replicates lm-eval's ``ConfigurableTask`` few-shot assembly
for multiple-choice: ``k`` labeled examples sampled from the ``train`` split,
each rendered as ``query + " " + gold_ending`` (``target_delimiter`` = " "; the
few-shot target is the gold *choice text*, not the label index), joined by
``"\\n\\n"`` (``fewshot_delimiter``) with a trailing ``"\\n\\n"``, then the target
query, with no leading description. ``k=0`` reproduces the 0-shot form. This
assembly is byte-verified against the reference-commit source: hellaswag
overrides neither delimiter (defaults ``" "`` / ``"\n\n"``); ``doc_to_target``
maps the digit ``label`` to an int (``ast.literal_eval`` when ``isdigit()`` and
``doc_to_choice`` is set), so the rendered answer is ``choices[label]`` (the gold
ending text), not the index; ``labeled_examples`` carries the trailing delimiter.
The exact assembled context is byte-pinned by the ``k=1``/``k=2`` tests.

Validated: Qwen2.5-72B-Base, 10-shot, ``acc_norm`` 87.4 over the full
10042-sample validation split, ``fails=0``, deterministic (two byte-identical
runs). Comparison target (cross-check only): the DeepSeek-V3 report (Table 6)
lists Qwen2.5-72B-Base HellaSwag = 84.8 — from DeepSeek's internal harness with
an underspecified protocol (prompt / acc-vs-``acc_norm`` / shot count), so the
~2.6-pt gap is a different-harness delta, not a reproduction failure. The
reproduced method is lm-eval ``hellaswag`` (``acc_norm``).

Query/choice construction is the lm-eval ``process_docs`` logic, factored into
``sieval.community.hellaswag`` (``preprocess`` + ``process_doc``), applied to both
the exemplars and the target doc. Per choice the continuation log-prob is read
from an echoed ``alogprobs`` call on ``context + " " + ending``.

Deviations from lm-eval-harness:
- lm-eval splits context/continuation at the token-id level
  (``len(tok_encode(context))``); this layer has no tokenizer, so the continuation
  tokens are isolated by reconstructing character offsets over the echoed token
  texts (``_continuation_logprob``). Equivalent under faithful detokenization
  with the space delimiter (clean token boundary).
- ``alogprobs`` scores with ``max_tokens=1`` (lm-eval uses 0, pure echo), so the
  echoed response carries one trailing generated token; it is dropped by bounding
  the continuation char span at ``len(prompt)``.
- Exemplar *selection* differs: we draw one fixed set via
  ``Dataset.retrieve_samples`` (seed 1234), computed once in ``setup()`` and
  reused for every eval doc (repo convention); lm-eval draws via its own
  ``ContextSampler`` RNG. Both are ``k`` random train exemplars and the assembly
  *format* is identical (verified above), so the aggregate ``acc_norm`` is robust
  to the draw — only the specific exemplars chosen differ.

Repro decoding: deterministic log-prob scoring — ``alogprobs`` uses
``temperature=0`` and ``echo=True``; no sampling params apply.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from sieval.community.hellaswag import process_doc
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.core.utils.ppl import total_logprob
from sieval.datasets import HellaSwagDatasetSample

N_SHOT = 10
DEFAULT_FEWSHOT_SEED = 1234
FEWSHOT_SPLIT = "train"
FEWSHOT_DELIMITER = "\n\n"


class Preprocessed(TypedDict):
    context: str  # few-shot prefix + target query
    choices: list[str]
    gold: int


class Prediction(TypedDict):
    pred_acc: int
    pred_acc_norm: int
    logprobs: list[float]
    norm_logprobs: list[float]


class Feedback(TypedDict):
    acc: bool
    acc_norm: bool
    gold: int
    pred_acc: int
    pred_acc_norm: int


def _continuation_logprob(
    tokens: list[str] | None,
    token_logprobs: list[float | None] | None,
    *,
    prompt: str,
    context_char_len: int,
) -> float:
    """Sum the echoed log-probs of the continuation tokens of *prompt*.

    The continuation is ``prompt[context_char_len:]`` (the ``" " + ending``).
    Tokens are located by reconstructing cumulative character offsets over the
    echoed token texts, in prompt-relative coordinates: any BOS prefix or
    trailing generated token is handled by anchoring on where *prompt* appears
    in the detokenized stream and bounding the span at ``len(prompt)``.

    Raises ``RuntimeError`` if the continuation cannot be isolated (empty
    logprobs, prompt not found in the detokenized stream, or no continuation
    token in range) — echo scoring must fail loud rather than score a
    mislocated/partial span silently.
    """
    if not tokens or not token_logprobs:
        raise RuntimeError("sglang returned empty echoed logprobs; cannot score.")
    size = min(len(tokens), len(token_logprobs))
    tokens = tokens[:size]
    token_logprobs = token_logprobs[:size]

    base = "".join(tokens).find(prompt)
    if base < 0:
        raise RuntimeError(
            "cannot locate the prompt in the echoed token stream "
            "(detokenization mismatch); continuation cannot be isolated."
        )

    cont_tokens: list[str] = []
    cont_logprobs: list[float | None] = []
    offset = 0
    for tok, logprob in zip(tokens, token_logprobs, strict=True):
        start = offset - base
        end = start + len(tok)
        offset += len(tok)
        if start >= context_char_len and end <= len(prompt):
            cont_tokens.append(tok)
            cont_logprobs.append(logprob)

    total, count = total_logprob(cont_tokens, cont_logprobs, skip_first=False)
    if count == 0:
        raise RuntimeError("no continuation tokens isolated; cannot score.")
    return total


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


@sieval_task(
    name="hellaswag_kshot_ppl",
    display_name="HellaSwag (k-shot, log-likelihood)",
    description="Commonsense sentence completion via per-ending log-likelihood.",
    eval_mode=EvalMode.PPL,
    n_shot=N_SHOT,
    tags=("english", "multiple-choice", "commonsense"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/hellaswag/hellaswag.yaml"
        ),
        notes=(
            "multiple_choice log-likelihood; acc + acc_norm (character-length "
            "normalized, reported as score). Few-shot exemplars (k, default 10) "
            "sampled from train, rendered 'query gold_ending' and joined by "
            "blank lines per lm-eval; k=0 is 0-shot. Continuation log-probs read "
            "via echoed alogprobs; context/continuation split by char offset "
            "(no tokenizer at this layer). Infra requirement: echo scoring reads "
            "the prompt's input logprobs, so the serving backend's prefix caching "
            "must be disabled (cached positions are not recomputed -> truncated "
            "logprobs); the backend layer owns the specific flag."
        ),
    ),
)
class HellaSwagFewShotPPLTask(
    Task[
        HellaSwagDatasetSample,
        Preprocessed,
        list[ModelOutput],
        Prediction,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = N_SHOT,
        fewshot_split: str = FEWSHOT_SPLIT,
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._fewshot_split = fewshot_split
        self._fewshot_seed = fewshot_seed
        # Built once in setup() (framework contract: runs before any sample);
        # "" is the k=0 prefix and a typed placeholder never observed post-setup.
        self._fewshot_prefix: str = ""

    @override
    async def setup(self) -> None:
        self._fewshot_prefix = self._build_fewshot_prefix()

    @override
    async def preprocess(self, raw, ctx):
        doc = process_doc(raw)
        context = self._fewshot_prefix + doc["query"]
        return {"context": context, "choices": doc["choices"], "gold": doc["gold"]}

    @override
    async def infer(self, pre, ctx):
        context = pre["context"]
        # echo=True is structural to PPL scoring: it returns the conditional
        # log-prob of every echoed continuation token (one call per ending).
        # logprobs=0: we only read each token's own log-prob, never the top-k
        # alternatives, so skip that extraction/payload over the whole prompt
        # (sglang: top_logprobs_num=0; input_token_logprobs are still returned).
        return [
            await self.model.alogprobs(f"{context} {choice}", echo=True, logprobs=0)
            for choice in pre["choices"]
        ]

    @override
    async def postprocess(self, inf, ctx):
        pre = ctx.preprocess_result
        context = pre["context"]

        logprobs: list[float] = []
        norm_logprobs: list[float] = []
        for choice, out in zip(pre["choices"], inf, strict=True):
            ll = _continuation_logprob(
                out.logprobs_tokens,
                out.logprobs,
                prompt=f"{context} {choice}",
                context_char_len=len(context),
            )
            logprobs.append(ll)
            # lm-eval acc_norm divides by the choice character length (no delimiter)
            norm_logprobs.append(ll / len(choice) if choice else ll)

        return {
            "pred_acc": _argmax(logprobs),
            "pred_acc_norm": _argmax(norm_logprobs),
            "logprobs": logprobs,
            "norm_logprobs": norm_logprobs,
        }

    @override
    async def feedback(self, post, ctx):
        gold = ctx.preprocess_result["gold"]
        return True, {
            "acc": post["pred_acc"] == gold,
            "acc_norm": post["pred_acc_norm"] == gold,
            "gold": gold,
            "pred_acc": post["pred_acc"],
            "pred_acc_norm": post["pred_acc_norm"],
        }

    @override
    async def report(self, finals, fails):
        # Denominator counts pipeline failures as wrong (full-set accuracy),
        # matching the `mbpp` sibling. (The MCQ siblings openbookqa/cmmlu instead
        # exclude fails, and lm-eval has no pipeline-failure concept — this is a
        # deliberate choice, moot here since fails=0.) A regression test locks it.
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "acc": 0.0, "acc_norm": 0.0, "fails": 0}
        acc_num = sum(1 for ctx in finals if ctx.feedback_result["acc"])
        acc_norm_num = sum(1 for ctx in finals if ctx.feedback_result["acc_norm"])
        acc = 100 * acc_num / total
        acc_norm = 100 * acc_norm_num / total
        return {
            "score": acc_norm,
            "acc": acc,
            "acc_norm": acc_norm,
            "fails": len(fails),
        }

    def _build_fewshot_prefix(self) -> str:
        if self._k == 0:
            return ""
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "HellaSwag k-shot PPL task requires a "
                f"{self._fewshot_split!r} split for few-shot exemplars."
            )
        if len(split) < self._k:
            raise ValueError(
                "HellaSwag k-shot PPL task requires at least "
                f"{self._k} examples in split {self._fewshot_split!r}; "
                f"found {len(split)}."
            )
        examples = self.dataset.retrieve_samples(
            self._k,
            split=self._fewshot_split,
            mode="random",
            seed=self._fewshot_seed,
        )
        rendered = []
        for example in examples:
            doc = process_doc(example)
            gold_ending = doc["choices"][doc["gold"]]
            rendered.append(f"{doc['query']} {gold_ending}")
        return FEWSHOT_DELIMITER.join(rendered) + FEWSHOT_DELIMITER
