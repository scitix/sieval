"""Unit tests for the OpenBookQA k-shot generative task.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.openbookqa import OBQA_PROMPT_TEMPLATE
from sieval.core.models import ModelOutput, Request, Response, SamplingParams
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.datasets.openbookqa import OpenBookQADataset, OpenBookQADatasetSample
from sieval.tasks.openbookqa_kshot_gen import (
    STOP_SEQUENCES,
    OpenBookQAFewShotGenTask,
)
from tests.conftest import HandlerTransport


class _CapturingChatModel(ChatModel):
    def __init__(self):
        self.last_req: Request | None = None
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=("The answer is A.",))


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
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), n_shot=0)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre["prompt"] == [{"role": "user", "content": _expected_question(raw)}]
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == raw["answerKey"]


@pytest.mark.anyio
async def test_kshot_prefix_uses_fixed_first_k_train_rows_with_answer():
    train = [_sample("q0", "A"), _sample("q1", "C"), _sample("q2", "D")]
    dataset = _dataset(train)
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), n_shot=2)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    content = pre["prompt"][0]["content"]

    # Fixed first 2 train rows, each with its answerKey appended, then the question.
    expected_prefix = (
        f"{_expected_question(train[0])} A\n\n{_expected_question(train[1])} C\n\n"
    )
    assert content == expected_prefix + _expected_question(raw)
    # Third train row must not leak into a n_shot=2 prompt.
    assert "q2" not in content


@pytest.mark.anyio
async def test_multiturn_renders_alternating_user_assistant_turns():
    train = [_sample("q0", "A"), _sample("q1", "C")]
    dataset = _dataset(train)
    task = OpenBookQAFewShotGenTask(
        dataset, _CapturingChatModel(), n_shot=2, fewshot_as_multiturn=True
    )
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    # Each shot becomes a user(question) + assistant(answerKey) pair, then the
    # final query as a trailing user turn — no single-turn packing.
    assert pre["prompt"] == [
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
    task = OpenBookQAFewShotGenTask(dataset, model, n_shot=0)

    raw = _sample("q-test")
    await task.infer(
        {"prompt": [{"role": "user", "content": "x"}]},
        TaskContext(sample_id=0, raw_sample=raw),
    )

    req = model.last_req
    assert req is not None
    # No decoding params forwarded: default sampling (n=1, everything else unset).
    assert req.sampling == SamplingParams()
    assert req.dialect_options is None


def test_stop_sequences_pinned():
    # Coupled to the few-shot block layout (examples begin with "Question:").
    assert STOP_SEQUENCES == ("\nQuestion:",)


@pytest.mark.anyio
async def test_infer_bounds_generation_at_kshot_but_not_zero_shot():
    dataset = _dataset([_sample("q0", "A"), _sample("q1", "C")])

    # n_shot>0: bound the run-on that would let a trailing match override the answer.
    model_k = _CapturingChatModel()
    task_k = OpenBookQAFewShotGenTask(dataset, model_k, n_shot=2)
    await task_k.infer(
        {"prompt": [{"role": "user", "content": "x"}]},
        TaskContext(sample_id=0, raw_sample=_sample("q-test")),
    )
    assert model_k.last_req is not None
    assert model_k.last_req.sampling == SamplingParams(stop=STOP_SEQUENCES)

    # n_shot=0: no stop — preserves upstream 0-shot parity.
    model_0 = _CapturingChatModel()
    task_0 = OpenBookQAFewShotGenTask(dataset, model_0, n_shot=0)
    await task_0.infer(
        {"prompt": [{"role": "user", "content": "x"}]},
        TaskContext(sample_id=0, raw_sample=_sample("q-test")),
    )
    assert model_0.last_req is not None
    assert model_0.last_req.sampling == SamplingParams()


@pytest.mark.anyio
async def test_feedback_and_report_accuracy_and_field_types():
    dataset = _dataset([_sample("q0")])
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), n_shot=0)

    correct_raw = _sample("q-test", "A")
    wrong_raw = _sample("q-test", "B")
    # postprocess extracts "A" from the mock "The answer is A." response.
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["The answer is A."]),
        TaskContext(sample_id=0, raw_sample=correct_raw),
    )
    assert post["rollouts"][0]["prediction"] == "A"

    _, fb_correct = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=correct_raw)
    )
    _, fb_wrong = await task.feedback(
        post, TaskContext(sample_id=1, raw_sample=wrong_raw)
    )
    assert fb_correct["rollouts"][0]["correct"] is True
    assert fb_wrong["rollouts"][0]["correct"] is False

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
    # JUDGED: the fail is outside the population, so 2 -- not 3.
    assert report["n_problems"] == 2
    lo, hi = report["score_ci95"]
    assert lo < report["score"] < hi
    # `accuracy` is `score` under its own name, so it carries the same bounds --
    # a reader keyed on the column `score_key` names finds its companion.
    assert report["accuracy_ci95"] == [lo, hi]
    assert report["ci95_units"] == {"score": "n_problems", "accuracy": "n_problems"}
    # The task tests call report() directly, so the runner's finalizer never sees
    # this dict -- run the validator here or a missing declaration ships.
    assert interval_declaration_problems(report) == []


def test_negative_k_rejected():
    dataset = _dataset([_sample("q0")])
    with pytest.raises(ValueError, match="n_shot must be >= 0"):
        OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), n_shot=-1)


# --- the few-shot pool must supply every shot meta.json records -------------


@pytest.mark.anyio
async def test_setup_aborts_when_train_split_is_shorter_than_n_shot():
    """A short pool used to truncate silently while meta.json still said n_shot.

    retrieve_samples clips out-of-range indices, so n_shot=5 against two rows
    rendered two shots and recorded five. setup() now aborts before any spend.
    """
    dataset = _dataset([_sample("q0"), _sample("q1")])
    task = OpenBookQAFewShotGenTask(dataset, _CapturingChatModel(), n_shot=5)
    assert task.n_shot == 5
    with pytest.raises(ValueError, match="requires at least 5 examples"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_aborts_when_fewshot_split_is_absent():
    dataset = _dataset([_sample("q0")])
    task = OpenBookQAFewShotGenTask(
        dataset, _CapturingChatModel(), n_shot=1, fewshot_split="nope"
    )
    with pytest.raises(ValueError, match="requires a 'nope' split"):
        await task.setup()


@pytest.mark.anyio
async def test_exactly_enough_examples_is_accepted():
    train = [_sample("q0", "A"), _sample("q1", "C")]
    task = OpenBookQAFewShotGenTask(_dataset(train), _CapturingChatModel(), n_shot=2)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    content = pre["prompt"][0]["content"]
    # Both exemplars reached the prompt — the count is not merely declared.
    assert "q0" in content
    assert "q1" in content


@pytest.mark.anyio
async def test_zero_shot_needs_no_fewshot_split():
    """n_shot=0 renders no block, so a missing pool is not an error."""
    dataset = _dataset([_sample("q0")])
    task = OpenBookQAFewShotGenTask(
        dataset, _CapturingChatModel(), n_shot=0, fewshot_split="nope"
    )
    await task.setup()
    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert pre["prompt"][0]["content"] == _expected_question(raw)
