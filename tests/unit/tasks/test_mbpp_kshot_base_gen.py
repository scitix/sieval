import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.mbpp import MBPPDataset, MBPPDatasetSample
from sieval.tasks.mbpp_kshot_base_gen import MBPPFewShotBaseGenTask


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=["def f():\n    pass\n[DONE]"])

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


def _sample() -> MBPPDatasetSample:
    return {
        "task_id": 11,
        "text": "Write a function to return 1.",
        "code": "def one():\n    return 1",
        "test_list": ["assert one() == 1"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }


def _dataset() -> MBPPDataset:
    sample = _sample()
    return MBPPDataset(
        _hf_dict=HFDatasetDict(
            {
                "prompt": HFDataset.from_list([dict(sample)]),
                "test": HFDataset.from_list([dict(sample)]),
            }
        )
    )


@pytest.mark.anyio
async def test_preprocess_uses_yaml_configured_k():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=2)

    prompt = await task.preprocess(_sample(), TaskContext(0, _sample()))
    await task.shutdown()

    assert prompt.count("[DONE]") == 2
    assert "similar_elements" in prompt
    assert "is_not_prime" in prompt
    assert "heap_queue_largest" not in prompt


@pytest.mark.anyio
async def test_k_zero_is_allowed():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=0)

    prompt = await task.preprocess(_sample(), TaskContext(0, _sample()))
    await task.shutdown()

    assert "[DONE]" not in prompt
    assert prompt.count("[BEGIN]") == 1


def test_k_above_lm_eval_examples_raises():
    with pytest.raises(ValueError, match="at most 3 examples"):
        MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=4)


@pytest.mark.anyio
async def test_infer_forwards_n_and_stop_but_not_decoding_params():
    model = _CapturingGenModel()
    task = MBPPFewShotBaseGenTask(
        _dataset(),
        model,
        k=0,
        n=3,
    )

    result = await task.infer("prompt", TaskContext(0, _sample()))
    await task.shutdown()

    assert result.texts == ["def f():\n    pass\n[DONE]"]
    assert model.last_kwargs["n"] == 3
    assert model.last_kwargs["stop"] == ["[DONE]"]
    # Decoding params stay in the model layer; the task must not inject them.
    assert "max_tokens" not in model.last_kwargs


def test_pass_k_above_n_raises():
    with pytest.raises(ValueError, match="pass_k must be <= n"):
        MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), pass_k=2, n=1)


def _final(feedbacks: list[dict]) -> TaskContext:
    return TaskContext(sample_id=0, raw_sample=_sample(), feedback_result=feedbacks)


@pytest.mark.anyio
async def test_report_pass_at_1_counts_fails_in_denominator():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=0)
    finals = [
        _final([{"correct": True, "msg": "ok", "metrics": None}]),
        _final([{"correct": False, "msg": "assertion failed", "metrics": None}]),
    ]
    # One failed sample (e.g. eval-server error) must lower the score, not be
    # dropped from the denominator.
    fails = [_final([])]

    report = await task.report(finals, fails)
    await task.shutdown()

    assert report["fails"] == 1
    # 1 correct out of 3 total (2 finals + 1 fail) = 33.33...
    assert report["score"] == pytest.approx(100 / 3)
    assert report["pass@1"] == pytest.approx(100 / 3)
    assert "pass@2" not in report


@pytest.mark.anyio
async def test_report_pass_at_k_and_timeouts():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=0, pass_k=2, n=2)
    finals = [
        # 1 of 2 samples correct → pass@1 = 0.5, pass@2 = 1.0
        _final(
            [
                {"correct": True, "msg": "ok", "metrics": None},
                {"correct": False, "msg": "Timeout exceeded", "metrics": None},
            ]
        ),
    ]

    report = await task.report(finals, [])
    await task.shutdown()

    assert report["score"] == pytest.approx(50.0)
    assert report["pass@1"] == pytest.approx(50.0)
    assert report["pass@2"] == pytest.approx(100.0)
    assert report["timeouts"] == 1


@pytest.mark.anyio
async def test_report_empty_returns_zero():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=0)
    report = await task.report([], [])
    await task.shutdown()

    assert report == {"score": 0.0, "fails": 0}


@pytest.mark.anyio
async def test_postprocess_strips_done_token():
    task = MBPPFewShotBaseGenTask(_dataset(), _CapturingGenModel(), k=0)
    output = ModelOutput(
        model=task.model.meta(),
        texts=["def one():\n    return 1\n[DONE]"],
    )

    post = await task.postprocess(output, TaskContext(0, _sample()))
    await task.shutdown()

    assert post == ["def one():\n    return 1\n"]
