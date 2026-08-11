"""Unit tests for the GSM1k k-shot base generative task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.gsm1k import GSM1KDataset, GSM1KDatasetSample
from sieval.tasks.gsm1k_kshot_base_gen import (
    _GSM8K_FEWSHOT_EXAMPLES,
    N_SHOT,
    STOP_SEQUENCES,
    GSM1KFewShotBaseGenTask,
    _extract_flexible_answer,
    _extract_strict_answer,
)


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[" Work shown.\n#### 42"])

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
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(answer: str = "42") -> GSM1KDatasetSample:
    return {"question": "What is 40 + 2?", "answer": answer}


def _task(n_shot: int = 0) -> tuple[GSM1KFewShotBaseGenTask, _CapturingGenModel]:
    # GSM1k ships a `test` split only — a task needing a `train` split could not
    # run on it at all, which is why the exemplars are vendored.
    dataset = GSM1KDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingGenModel()
    return GSM1KFewShotBaseGenTask(dataset, model, n_shot=n_shot), model


def test_vendored_exemplars_carry_gsm8k_cot_shape():
    # A GSM8k-train exemplar is what teaches the `#### N` final-answer format the
    # strict rule reads; losing it would silently zero `strict_exact_match`.
    assert len(_GSM8K_FEWSHOT_EXAMPLES) == N_SHOT == 5
    for question, answer in _GSM8K_FEWSHOT_EXAMPLES:
        assert question.strip() == question
        # A worked solution, then `#### N` on its own final line.
        assert "\n#### " in answer
        assert answer.split("\n")[-1].removeprefix("#### ").isdigit()
        assert _extract_strict_answer(answer) == answer.rsplit("#### ", 1)[1]


def test_extractors_implement_upstreams_two_rules():
    # flexible-extract = last regex match, so a trailing restatement wins.
    assert _extract_flexible_answer("First 7, then finally 42.") == "42"
    # regexes_to_ignore: commas, dollar signs and a trailing period all dropped.
    assert _extract_flexible_answer("It costs $1,234.") == "1234"
    # strict-match needs the `#### ` delimiter and finds nothing without it.
    assert _extract_strict_answer("Therefore the answer is 42.") == ""
    assert _extract_strict_answer("Final.\n#### 1,234.") == "1234"
    # Neither rule can extract from a response carrying no number.
    assert _extract_flexible_answer("I cannot solve this.") == ""
    assert _extract_strict_answer("I cannot solve this.") == ""


@pytest.mark.anyio
async def test_zero_shot_prompt_is_upstreams_doc_to_text():
    task, _ = _task(n_shot=0)

    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    assert pre["prompt"] == "Question: What is 40 + 2?\nAnswer:"
    # GSM1k's gold is already the bare final answer: no `####` split, unlike
    # openai/gsm8k.
    assert pre["reference"] == "42"


@pytest.mark.anyio
async def test_fewshot_prompt_blocks_match_upstream_format():
    task, _ = _task(n_shot=2)
    first_q, first_a = _GSM8K_FEWSHOT_EXAMPLES[0]

    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    assert pre["prompt"].startswith(f"Question: {first_q}\nAnswer: {first_a}\n\n")
    assert pre["prompt"].endswith("Question: What is 40 + 2?\nAnswer:")
    # n_shot exemplar blocks plus the unanswered question.
    assert pre["prompt"].count("Question: ") == 3


@pytest.mark.anyio
async def test_prompt_prefix_is_fixed_across_samples():
    # The documented deviation from upstream: exemplars do NOT vary per question.
    task, _ = _task(n_shot=5)

    first = await task.preprocess(_sample(), TaskContext(sample_id=0))
    second = await task.preprocess(
        {"question": "What is 1 + 1?", "answer": "2"}, TaskContext(sample_id=1)
    )

    tail = "Question: What is 40 + 2?\nAnswer:"
    assert first["prompt"].removesuffix(tail) == second["prompt"].removesuffix(
        "Question: What is 1 + 1?\nAnswer:"
    )


def test_n_shot_is_bounded_by_the_vendored_exemplar_count():
    dataset = GSM1KDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingGenModel()

    with pytest.raises(ValueError, match="n_shot must be between 0 and 5"):
        GSM1KFewShotBaseGenTask(dataset, model, n_shot=6)
    with pytest.raises(ValueError, match="n_shot must be between 0 and 5"):
        GSM1KFewShotBaseGenTask(dataset, model, n_shot=-1)


@pytest.mark.anyio
async def test_infer_only_forwards_prompt_coupled_stop():
    task, model = _task()

    await task.infer(
        {"prompt": "prompt"}, TaskContext(sample_id=0, raw_sample=_sample())
    )

    assert model.last_kwargs == {"stop": list(STOP_SEQUENCES)}


@pytest.mark.anyio
async def test_headline_follows_the_flexible_rule_not_the_strict_one():
    task, model = _task()
    raw = _sample()
    inferred = ModelOutput(
        model=model.meta(),
        texts=["No delimiter here, but the final sentence says 42."],
    )
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)

    post = await task.postprocess(inferred, ctx)
    finalize, feedback = await task.feedback(post, ctx)
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)], []
    )

    assert finalize is True
    assert post["rollouts"][0]["prediction"] == "42"
    # Per-rollout, not sample-level: the strict rule is a fact about one response.
    assert post["rollouts"][0]["extra"]["strict_prediction"] == ""
    assert "extra" not in post
    # Upstream's only filter is the flexible one, so it drives `correct`.
    assert feedback["metrics"]["flexible_exact_match"] is True
    assert feedback["metrics"]["strict_exact_match"] is False
    assert feedback["rollouts"][0]["correct"] is True
    assert report["score"] == report["flexible_exact_match"] == 100.0
    assert report["strict_exact_match"] == 0.0
    # No bare `exact_match` key: it means the STRICT rule in gsm8k_kshot_base_gen,
    # so sharing the name would let a paired diff compare two different rules.
    assert "exact_match" not in report


@pytest.mark.anyio
async def test_unextractable_response_is_none_and_scores_wrong():
    task, model = _task()
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["I cannot solve this."])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)

    post = await task.postprocess(inferred, ctx)
    _, feedback = await task.feedback(post, ctx)

    assert post["rollouts"][0]["prediction"] is None
    assert post["rollouts"][0]["extracted"] is False
    assert feedback["metrics"]["flexible_exact_match"] is False
    assert feedback["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_report_counts_pipeline_failures_as_wrong():
    task, model = _task()
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["The answer is 42."])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)
    post = await task.postprocess(inferred, ctx)
    _, feedback = await task.feedback(post, ctx)

    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)],
        [TaskContext(sample_id=1, raw_sample=raw)],
    )

    assert report["fails"] == 1
    # 1 correct out of (1 final + 1 fail), not out of 1 final.
    assert report["score"] == 50.0


@pytest.mark.anyio
async def test_report_on_empty_set_reports_zero_for_both_rules():
    task, _ = _task()

    report = await task.report([], [])

    # The declarations ride on the empty report too: a zero-sample run still has
    # to say which key the headline was copied from and how it counted failures,
    # or on disk it is indistinguishable from a report that declared neither.
    assert report == {
        "score": 0.0,
        "fails": 0,
        "flexible_exact_match": 0.0,
        "strict_exact_match": 0.0,
        "score_key": "flexible_exact_match",
        "denominator_policy": "requested",
        # Extraction health rides on both halves of the pair, so a diff can never
        # be confused with a difference in how often extraction failed.
        "n_unextracted": 0.0,
    }
