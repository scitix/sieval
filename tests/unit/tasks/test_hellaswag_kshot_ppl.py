"""Unit tests for the HellaSwag k-shot log-likelihood (PPL) task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.hellaswag import HellaSwagDataset
from sieval.tasks.hellaswag_kshot_ppl import (
    HellaSwagFewShotPPLTask,
    _argmax,
    _continuation_logprob,
)

QUERY = "Activity label: a person"
# Chosen so raw-LL argmax (acc) and length-normalized argmax (acc_norm) DIVERGE.
CHOICES = ["ab", "cdcde", "x" * 30, "y" * 10]  # char lengths 2, 5, 30, 10
LLS = [-1.0, -5.0, -3.0, -4.0]  # norms: -0.50, -1.00, -0.10, -0.40
# acc -> argmax(LL) = 0 ; acc_norm -> argmax(LL / len) = 2
GOLD = 2


def _doc(activity: str, ctx_a: str, ctx_b: str, endings: list[str], label: str) -> dict:
    return {
        "ind": 0,
        "activity_label": activity,
        "ctx_a": ctx_a,
        "ctx_b": ctx_b,
        "ctx": f"{ctx_a} {ctx_b}",
        "endings": endings,
        "source_id": "s",
        "split": "train",
        "split_type": "t",
        "label": label,
    }


TRAIN_A = _doc(
    "Cooking", "She cracks an egg.", "the woman", ["pours it in.", "x", "y", "z"], "0"
)
TRAIN_B = _doc(
    "Driving", "He starts the car.", "the man", ["drives away.", "x", "y", "z"], "0"
)
RENDERED_A = "Cooking: She cracks an egg. The woman pours it in."
RENDERED_B = "Driving: He starts the car. The man drives away."


def _crafted_output(
    model: GenModel, prompt: str, choice: str, ll: float
) -> ModelOutput:
    """Echoed logprobs for *prompt* (= context + ' ' + choice) summing to *ll*.

    Layout stresses the helper: a non-empty leading BOS token, the (arbitrary,
    possibly few-shot-prefixed) context as one token, the continuation split
    across two tokens, and a trailing generated token (``max_tokens=1``).
    """
    context = prompt[: len(prompt) - len(choice) - 1]  # strip trailing " " + choice
    cont = " " + choice
    mid = max(1, len(cont) // 2)
    part1, part2 = cont[:mid], cont[mid:]
    tokens = ["<s>", context, part1, part2, "<gen>"]
    logprobs = [None, -42.0, ll / 2, ll / 2, -99.0]
    return ModelOutput(
        model=model.meta(),
        texts=["<gen>"],
        logprobs_tokens=tokens,
        logprobs=logprobs,
    )


class _PPLMockModel(GenModel):
    def __init__(self, choices: list[str], lls: list[float]):
        super().__init__(model="mock-gen", api_key="fake")
        self._ll_by_choice: dict[str, float] = dict(zip(choices, lls, strict=True))
        self.calls: list[str] = []
        self.echos: list[bool] = []
        self.logprobs_args: list[int] = []

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])

    async def _alogprobs_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (max_tokens, temperature, kwargs)
        self.calls.append(prompt)
        self.echos.append(echo)
        self.logprobs_args.append(logprobs)
        # CHOICES are mutually non-suffix, so exactly one endswith-matches.
        choice = next(c for c in self._ll_by_choice if prompt.endswith(c))
        return _crafted_output(self, prompt, choice, self._ll_by_choice[choice])


def _make_task(
    k: int = 0, train_docs: list[dict] | None = None
) -> tuple[HellaSwagFewShotPPLTask, _PPLMockModel]:
    splits = {
        "test": HFDataset.from_list([_doc("T", "A.", "he", ["a", "b", "c", "d"], "0")])
    }
    if train_docs is not None:
        splits["train"] = HFDataset.from_list(train_docs)
    dataset = HellaSwagDataset(_hf_dict=HFDatasetDict(splits))
    model = _PPLMockModel(CHOICES, LLS)
    return HellaSwagFewShotPPLTask(dataset, model, k=k), model


def _pre(context: str = QUERY) -> dict:
    return {"context": context, "choices": CHOICES, "gold": GOLD}


# ---------------------------------------------------------------- helper units


def test_continuation_logprob_isolates_continuation_with_bos_and_gen_tail():
    prompt = f"{QUERY} {CHOICES[0]}"
    tokens = ["<s>", QUERY, " ", CHOICES[0], "<gen>"]
    logprobs = [None, -42.0, -0.4, -0.6, -99.0]
    ll = _continuation_logprob(
        tokens, logprobs, prompt=prompt, context_char_len=len(QUERY)
    )
    assert ll == pytest.approx(-1.0)


def test_continuation_logprob_clean_stream_no_bos():
    prompt = "abc de"
    tokens = ["abc", " de", "Z"]
    logprobs = [-1.0, -2.0, -7.0]
    ll = _continuation_logprob(tokens, logprobs, prompt=prompt, context_char_len=3)
    assert ll == pytest.approx(-2.0)


def test_continuation_logprob_empty_raises():
    # empty echoed logprobs → loud fail, never a silent -inf that mis-scores
    with pytest.raises(RuntimeError):
        _continuation_logprob(None, None, prompt="x", context_char_len=0)


def test_continuation_logprob_raises_when_prompt_not_located():
    # tokens don't reconstruct the prompt → cannot isolate continuation → raise
    with pytest.raises(RuntimeError):
        _continuation_logprob(
            ["foo", "bar"], [-1.0, -2.0], prompt="unrelated text", context_char_len=3
        )


def test_argmax_returns_first_index_on_ties():
    assert _argmax([-1.0, -1.0, -2.0]) == 0
    assert _argmax([-3.0, -0.5, -0.5]) == 1


# ----------------------------------------------------------- few-shot prefix


def test_k_negative_rejected():
    dataset = HellaSwagDataset(
        _hf_dict=HFDatasetDict(
            {
                "test": HFDataset.from_list(
                    [_doc("T", "A.", "h", ["a", "b", "c", "d"], "0")]
                )
            }
        )
    )
    with pytest.raises(ValueError, match="k must be >= 0"):
        HellaSwagFewShotPPLTask(dataset, _PPLMockModel(CHOICES, LLS), k=-1)


@pytest.mark.anyio
async def test_k0_context_is_just_the_query():
    task, _ = _make_task(k=0)
    await task.setup()
    raw = _doc("X", "A.", "he", ["one", "two", "three", "four"], "2")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert pre["context"] == "X: A. He"  # no prefix
    assert pre["gold"] == 2


@pytest.mark.anyio
async def test_k1_fewshot_prefix_rendered_query_space_gold_ending():
    task, _ = _make_task(k=1, train_docs=[TRAIN_A])
    await task.setup()
    assert task._fewshot_prefix == f"{RENDERED_A}\n\n"

    raw = _doc("X", "A.", "he", ["one", "two", "three", "four"], "2")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    # byte-identical to lm-eval's few-shot assembly: `query + " " + gold_ending`,
    # "\n\n"-joined with trailing "\n\n", then the target query (no description).
    assert pre["context"] == f"{RENDERED_A}\n\nX: A. He"


@pytest.mark.anyio
async def test_k2_prefix_joins_two_exemplars_with_blank_lines():
    task, _ = _make_task(k=2, train_docs=[TRAIN_A, TRAIN_B])
    await task.setup()
    prefix = task._fewshot_prefix
    parts = prefix.split("\n\n")
    assert prefix.endswith("\n\n")
    assert parts[-1] == ""  # trailing delimiter
    assert set(parts[:-1]) == {RENDERED_A, RENDERED_B}  # order-independent


def test_build_fewshot_prefix_raises_when_train_too_small():
    task, _ = _make_task(k=3, train_docs=[TRAIN_A])  # only 1 < 3
    with pytest.raises(ValueError, match="at least 3 examples"):
        task._build_fewshot_prefix()


# ----------------------------------------------------------------- pipeline


@pytest.mark.anyio
async def test_infer_prepends_prefix_and_issues_one_echo_call_per_choice():
    task, model = _make_task(k=1, train_docs=[TRAIN_A])
    await task.setup()
    prefix = task._fewshot_prefix
    pre = _pre(context=f"{prefix}{QUERY}")
    out = await task.infer(pre, TaskContext(sample_id=0, preprocess_result=pre))
    assert len(out) == len(CHOICES)
    assert all(call.startswith(prefix) for call in model.calls)
    assert model.calls == [f"{prefix}{QUERY} {c}" for c in CHOICES]
    assert model.echos == [True] * len(CHOICES)
    # top-k logprobs are never read → not requested (cheaper over a long prompt)
    assert model.logprobs_args == [0] * len(CHOICES)


@pytest.mark.anyio
async def test_acc_and_acc_norm_diverge_and_score_is_acc_norm():
    task, _ = _make_task(k=0)
    pre = _pre()
    inf = await task.infer(pre, TaskContext(sample_id=0, preprocess_result=pre))
    ctx = TaskContext(sample_id=0, preprocess_result=pre, infer_result=inf)

    post = await task.postprocess(inf, ctx)
    assert post["pred_acc"] == 0  # argmax raw log-likelihood
    assert post["pred_acc_norm"] == 2  # argmax length-normalized log-likelihood

    finalize, fb = await task.feedback(post, ctx)
    assert finalize is True
    assert fb["acc"] is False  # 0 != gold(2)
    assert fb["acc_norm"] is True  # 2 == gold(2)

    report = await task.report(
        [TaskContext(sample_id=0, preprocess_result=pre, feedback_result=fb)], []
    )
    assert report["acc"] == 0.0
    assert report["acc_norm"] == 100.0
    assert report["score"] == report["acc_norm"]  # headline metric is acc_norm
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_fewshot_scoring_unaffected_by_prefix():
    # With a k=1 prefix, continuation isolation still scores only the ending,
    # so the acc/acc_norm predictions match the k=0 case.
    task, _ = _make_task(k=1, train_docs=[TRAIN_A])
    await task.setup()
    prefix = task._fewshot_prefix
    pre = _pre(context=f"{prefix}{QUERY}")
    inf = await task.infer(pre, TaskContext(sample_id=0, preprocess_result=pre))
    post = await task.postprocess(
        inf, TaskContext(sample_id=0, preprocess_result=pre, infer_result=inf)
    )
    assert post["pred_acc"] == 0
    assert post["pred_acc_norm"] == 2


@pytest.mark.anyio
async def test_report_handles_empty_finals():
    task, _ = _make_task(k=0)
    report = await task.report([], [TaskContext(sample_id=0)])
    assert report == {"score": 0.0, "acc": 0.0, "acc_norm": 0.0, "fails": 1}


@pytest.mark.anyio
async def test_report_counts_fails_in_denominator():
    # 1 correct final + 1 pipeline fail → 50% (fails in denominator), not 100%.
    # Locks against reverting to the old len(finals)-only denominator.
    task, _ = _make_task(k=0)
    fb = {"acc": True, "acc_norm": True, "gold": 0, "pred_acc": 0, "pred_acc_norm": 0}
    finals = [TaskContext(sample_id=0, feedback_result=fb)]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)
    assert report["acc"] == 50.0
    assert report["acc_norm"] == 50.0
    assert report["score"] == 50.0
    assert report["fails"] == 1
