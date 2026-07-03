"""Unit tests for the OpenBookQA k-shot generative task.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.openbookqa import OBQA_PROMPT_TEMPLATE
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.openbookqa import OpenBookQADataset, OpenBookQADatasetSample
from sieval.tasks.openbookqa_kshot_gen import (
    STOP_SEQUENCES,
    OpenBookQAFewShotGenTask,
)


class _CapturingChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=["The answer is A."])


def _sample(stem: str, answer_key: str = "A") -> OpenBookQADatasetSample:
    return {
        "id": f"id-{stem}",
        "question_stem": stem,
        "choices": {"text": [f"{stem}-a", f"{stem}-b", f"{stem}-c", f"{stem}-d"]},
        "answerKey": answer_key,
    }


def _dataset(train: list[OpenBookQADatasetSample]) -> OpenBookQADataset:
    return OpenBookQADataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(s) for s in train]),
                "test": HFDataset.from_list([dict(_sample("q-test"))]),
            }
        )
    )


def _expected_question(sample: OpenBookQADatasetSample) -> str:
    texts = sample["choices"]["text"]
    return OBQA_PROMPT_TEMPLATE.format(
        question_stem=sample["question_stem"],
        A=texts[0],
        B=texts[1],
        C=texts[2],
        D=texts[3],
    )


@pytest.mark.anyio
async def test_zero_shot_prompt_has_no_fewshot_prefix():
    dataset = _dataset([_sample("q-train", "B")])
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), k=0)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre == [{"role": "user", "content": _expected_question(raw)}]


@pytest.mark.anyio
async def test_kshot_prefix_uses_fixed_first_k_train_rows_with_answer():
    train = [_sample("q0", "A"), _sample("q1", "C"), _sample("q2", "D")]
    dataset = _dataset(train)
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), k=2)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    content = pre[0]["content"]

    # Fixed first 2 train rows, each with its answerKey appended, then the question.
    expected_prefix = (
        f"{_expected_question(train[0])} A\n\n{_expected_question(train[1])} C\n\n"
    )
    assert content == expected_prefix + _expected_question(raw)
    # Third train row must not leak into a k=2 prompt.
    assert "q2" not in content


@pytest.mark.anyio
async def test_multiturn_renders_alternating_user_assistant_turns():
    train = [_sample("q0", "A"), _sample("q1", "C")]
    dataset = _dataset(train)
    task = OpenBookQAFewShotGenTask(
        dataset, _CapturingChatModel(), k=2, fewshot_as_multiturn=True
    )
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    # Each shot becomes a user(question) + assistant(answerKey) pair, then the
    # final query as a trailing user turn — no single-turn packing.
    assert pre == [
        {"role": "user", "content": _expected_question(train[0])},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": _expected_question(train[1])},
        {"role": "assistant", "content": "C"},
        {"role": "user", "content": _expected_question(raw)},
    ]


@pytest.mark.anyio
async def test_infer_does_not_forward_decoding_params():
    dataset = _dataset([_sample("q0")])
    model = _CapturingChatModel()
    task = OpenBookQAFewShotGenTask(dataset, model, k=0)

    raw = _sample("q-test")
    await task.infer(
        [{"role": "user", "content": "x"}],
        TaskContext(sample_id=0, raw_sample=raw),
    )

    for forbidden in ("temperature", "top_p", "max_tokens", "n", "stop"):
        assert forbidden not in model.last_kwargs


def test_stop_sequences_pinned():
    # Coupled to the few-shot block layout (examples begin with "Question:").
    assert STOP_SEQUENCES == ("\nQuestion:",)


@pytest.mark.anyio
async def test_infer_bounds_generation_at_kshot_but_not_zero_shot():
    dataset = _dataset([_sample("q0", "A"), _sample("q1", "C")])

    # k>0: bound the run-on that would let a trailing match override the answer.
    model_k = _CapturingChatModel()
    task_k = OpenBookQAFewShotGenTask(dataset, model_k, k=2)
    await task_k.infer(
        [{"role": "user", "content": "x"}],
        TaskContext(sample_id=0, raw_sample=_sample("q-test")),
    )
    assert model_k.last_kwargs.get("stop") == list(STOP_SEQUENCES)

    # k=0: no stop — preserves upstream 0-shot parity.
    model_0 = _CapturingChatModel()
    task_0 = OpenBookQAFewShotGenTask(dataset, model_0, k=0)
    await task_0.infer(
        [{"role": "user", "content": "x"}],
        TaskContext(sample_id=0, raw_sample=_sample("q-test")),
    )
    assert "stop" not in model_0.last_kwargs


@pytest.mark.anyio
async def test_feedback_and_report_accuracy_and_field_types():
    dataset = _dataset([_sample("q0")])
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), k=0)

    correct_raw = _sample("q-test", "A")
    wrong_raw = _sample("q-test", "B")
    # postprocess extracts "A" from the mock "The answer is A." response.
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["The answer is A."]),
        TaskContext(sample_id=0, raw_sample=correct_raw),
    )
    assert post == "A"

    _, fb_correct = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=correct_raw)
    )
    _, fb_wrong = await task.feedback(
        post, TaskContext(sample_id=1, raw_sample=wrong_raw)
    )
    assert fb_correct["correct"] is True
    assert fb_wrong["correct"] is False

    finals = [
        TaskContext(sample_id=0, raw_sample=correct_raw, feedback_result=fb_correct),
        TaskContext(sample_id=1, raw_sample=wrong_raw, feedback_result=fb_wrong),
    ]
    report = await task.report(
        finals,
        [TaskContext(sample_id=2, raw_sample=_sample("q-fail"))],
    )

    # 1 correct out of 2 finalized samples; fails counted separately as int.
    assert report["score"] == 50.0
    # `accuracy` names the metric behind `score`; they must agree.
    assert report["accuracy"] == 50.0
    assert report["fails"] == 1
    assert isinstance(report["fails"], int)
    # MCQ tasks report accuracy only — no pass@1 (sibling consistency).
    assert "pass@1" not in report


def test_negative_k_rejected():
    dataset = _dataset([_sample("q0")])
    with pytest.raises(ValueError, match="k must be >= 0"):
        OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), k=-1)
