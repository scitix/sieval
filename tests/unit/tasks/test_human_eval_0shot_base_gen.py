"""Unit tests for the HumanEval zero-shot base generative task.

AI-Generated Code - GPT-5.5-Codex (OpenAI)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
)
from sieval.datasets.human_eval import HumanEvalDataset, HumanEvalDatasetSample
from sieval.tasks.human_eval_0shot_base_gen import (
    STOP_SEQUENCES,
    HumanEvalZeroShotBaseGenTask,
)


def _judgement(*rollouts: tuple[bool, str]):
    """A JudgementRecord from (correct, msg) pairs -- the shape report() reads."""
    return build_judgement_record(
        None,
        [
            build_rollout_judgement(i, correct, extra={"msg": msg})
            for i, (correct, msg) in enumerate(rollouts)
        ],
    )


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_prompt = ""
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=["    return x + 1"])

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


def _sample() -> HumanEvalDatasetSample:
    return {
        "prompt": "def add_one(x):\n",
        "canonical_solution": "    return x + 1\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2",
        "entry_point": "add_one",
    }


def _task(
    **kwargs,
) -> tuple[HumanEvalZeroShotBaseGenTask, _CapturingGenModel, HumanEvalDataset]:
    dataset = HumanEvalDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingGenModel()
    return HumanEvalZeroShotBaseGenTask(dataset, model, **kwargs), model, dataset


def test_default_stop_matches_lm_eval_harness():
    assert STOP_SEQUENCES == ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


@pytest.mark.anyio
async def test_preprocess_and_infer_use_base_completion_prompt():
    task, model, _ = _task(n=2, stop=("\nclass",))
    try:
        raw = _sample()
        pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

        await task.infer(pre, TaskContext(sample_id=0, raw_sample=raw))

        assert pre["prompt"] == raw["prompt"]
        assert model.last_prompt == raw["prompt"]
        assert model.last_kwargs["n"] == 2
        assert model.last_kwargs["stop"] == ["\nclass"]
        # Decoding params (max_tokens, temperature, top_p) are owned by the
        # model config / infer_args, never injected by the task layer.
        assert "max_tokens" not in model.last_kwargs
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_postprocess_keeps_raw_completions_like_lm_eval_harness():
    task, model, _ = _task()
    try:
        inferred = ModelOutput(
            model=model.meta(),
            texts=[
                "```python\ndef add_one(x):\n    return x + 1\n```",
                "'    return x + 1'",
                "    return x + 1\n\ndef helper():\n    return 0",
            ],
        )

        post = await task.postprocess(
            inferred,
            TaskContext(sample_id=0, raw_sample=_sample(), infer_result=inferred),
        )

        assert [r["prediction"] for r in post["rollouts"]] == list(inferred.texts)
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_report_counts_finals_and_fails_like_chat_human_eval_task():
    task, _, _ = _task(k=2, n=2)
    try:
        report = await task.report(
            [
                TaskContext(
                    sample_id=0,
                    raw_sample=_sample(),
                    feedback_result=_judgement((True, "passed"), (False, "timeout")),
                ),
                TaskContext(
                    sample_id=1,
                    raw_sample=_sample(),
                    feedback_result=_judgement((False, "failed"), (False, "failed")),
                ),
            ],
            [TaskContext(sample_id=2, raw_sample=_sample())],
        )

        assert report["fails"] == 1
        assert report["timeouts"] == 1
        assert report["score"] == pytest.approx(100 / 6)
        assert report["pass@1"] == pytest.approx(100 / 6)
        assert report["pass@k"] == pytest.approx(100 / 3)
        assert "pass@2" not in report  # the key carries a literal `k`
    finally:
        await task.shutdown()
