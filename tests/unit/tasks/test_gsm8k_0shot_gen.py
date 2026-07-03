"""Unit tests for the DeepSeek-Math-aligned GSM8K 0-shot task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.gsm8k import GSM8KDataset, GSM8KDatasetSample
from sieval.tasks.gsm8k_0shot_gen import (
    COT_INSTRUCTION,
    GSM8KZeroShotGenTask,
    _gold_answer,
)


class _CapturingChatModel(ChatModel):
    def __init__(self, text: str):
        super().__init__(model="mock-chat", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self._text = text

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[self._text])

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(answer: str = "Solution.\n#### 42") -> GSM8KDatasetSample:
    return {"question": "What is 40 + 2?", "answer": answer}


def _task(text: str):
    dataset = GSM8KDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingChatModel(text=text)
    return GSM8KZeroShotGenTask(dataset, model), model


# --- Pinning: prompt instruction is byte-for-byte DeepSeek markup_question(en, cot) ---


def test_cot_instruction_pinned():
    assert COT_INSTRUCTION == (
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


@pytest.mark.anyio
async def test_preprocess_appends_instruction_single_user_turn():
    task, _ = _task("x")
    messages = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is 40 + 2?" + COT_INSTRUCTION


# --- Gold derivation matches process_gsm8k_test (####-split, commas removed) ---


def test_gold_answer_strip_and_decomma():
    assert _gold_answer("reasoning ...\n#### 1,000") == "1000"
    assert _gold_answer("#### 42") == "42"


# --- postprocess uses DeepSeek extract_answer(exhaust=False): boxed wins ---


@pytest.mark.anyio
async def test_postprocess_prefers_boxed():
    task, model = _task("x")
    inf = ModelOutput(
        model=model.meta(), texts=["Work.\nSo the answer is $\\boxed{42}$."]
    )
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post == "42"


@pytest.mark.anyio
async def test_postprocess_last_number_fallback():
    task, model = _task("x")
    inf = ModelOutput(model=model.meta(), texts=["first 12 then finally 30"])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))
    assert post == "30"


# --- scoring via vendored is_correct/math_equal (numeric isclose) ---


@pytest.mark.anyio
async def test_feedback_numeric_equal_via_math_equal():
    task, model = _task("x")
    raw = _sample(answer="Work.\n#### 1,000")
    inf = ModelOutput(
        model=model.meta(), texts=["...so the answer is $\\boxed{1000.0}$."]
    )
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    finalize, fb = await task.feedback(post, ctx)
    assert finalize is True
    assert fb["answer"] == "1000"
    assert fb["correct"] is True


@pytest.mark.anyio
async def test_feedback_wrong_answer():
    task, model = _task("x")
    raw = _sample(answer="Work.\n#### 42")
    inf = ModelOutput(model=model.meta(), texts=["The answer is $\\boxed{7}$."])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["correct"] is False


# --- report accuracy + infer injects no decode params ---


@pytest.mark.anyio
async def test_report_accuracy():
    task, _ = _task("x")
    raw = _sample()
    finals = [
        TaskContext(sample_id=0, raw_sample=raw, feedback_result={"correct": True}),
        TaskContext(sample_id=1, raw_sample=raw, feedback_result={"correct": False}),
    ]
    report = await task.report(finals, [])
    assert report == {"score": 50.0, "fails": 0, "accuracy": 50.0}


@pytest.mark.anyio
async def test_report_empty_finals():
    task, _ = _task("x")
    report = await task.report([], [])
    assert report == {"score": 0.0, "fails": 0, "accuracy": 0.0}


@pytest.mark.anyio
async def test_report_counts_fails_in_denominator():
    # Denominator is len(finals) + len(fails), matching the math-0shot-gen
    # family: a pipeline failure counts as wrong, not as an excluded sample.
    task, _ = _task("x")
    raw = _sample()
    finals = [
        TaskContext(sample_id=0, raw_sample=raw, feedback_result={"correct": True}),
    ]
    fails = [TaskContext(sample_id=1, raw_sample=raw)]
    report = await task.report(finals, fails)
    assert report == {"score": 50.0, "fails": 1, "accuracy": 50.0}


@pytest.mark.anyio
async def test_infer_injects_no_decode_params():
    task, model = _task("x")
    pre = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    await task.infer(pre, TaskContext(sample_id=0, raw_sample=_sample()))
    for forbidden in ("temperature", "top_p", "max_tokens", "n", "stop"):
        assert forbidden not in model.last_kwargs
