"""Unit tests for the LiveCodeBench code-generation few-shot base-model task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.livecodebench.prompts.code_generation import (
    get_base_model_fewshot_prefix,
    get_base_model_question_template_answer,
    get_base_model_target_block,
)
from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.livecodebench_code_generation import LiveCodeBenchDataset
from sieval.tasks.livecodebench_code_generation_kshot_base_gen import (
    N_SHOT,
    STOP_SEQUENCES,
    LiveCodeBenchCodeGenerationFewShotBaseGenTask,
)

_STARTER = "class Solution:\n    def solve(self) -> int:\n        "


class _CapturingGenModel(GenModel):
    def __init__(self, texts: list[str] | None = None):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self._texts = texts if texts is not None else ["print(1)"]

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=list(self._texts))

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


def _raw(starter_code: str = "") -> dict:
    return {"question_content": "TARGET_QUESTION", "starter_code": starter_code}


def _task(
    *, texts: list[str] | None = None, **kwargs
) -> tuple[LiveCodeBenchCodeGenerationFewShotBaseGenTask, _CapturingGenModel]:
    dataset = LiveCodeBenchDataset(
        _hf_dict=HFDatasetDict(
            {"test": HFDataset.from_list([_raw()])},
        )
    )
    model = _CapturingGenModel(texts=texts)
    task = LiveCodeBenchCodeGenerationFewShotBaseGenTask(dataset, model, **kwargs)
    return task, model


# --------------------------------------------------------------------------- #
# Prompt builder (community function)
# --------------------------------------------------------------------------- #
def test_stdin_pool_used_without_starter_code_and_no_starter_block():
    prompt = get_base_model_question_template_answer(_raw(starter_code=""), 1)
    # one example + the target question, no starter-code section for stdin problems
    assert prompt.count("### Question") == 2
    assert "### Starter Code" not in prompt
    # target question is appended last with an empty answer to be completed
    assert prompt.rstrip().endswith("### Answer")
    assert "TARGET_QUESTION" in prompt


def test_func_pool_used_with_starter_code_includes_starter_blocks():
    prompt = get_base_model_question_template_answer(_raw(starter_code=_STARTER), 3)
    # three examples + the target question
    assert prompt.count("### Question") == 4
    assert prompt.count("### Starter Code") == 4
    assert _STARTER in prompt


def test_n_shot_count_controls_number_of_examples():
    for n_shot in (0, 1, 2):
        prompt = get_base_model_question_template_answer(_raw(""), n_shot)
        assert prompt.count("### Question") == n_shot + 1


def test_n_shot_out_of_range_raises():
    with pytest.raises(ValueError):
        get_base_model_question_template_answer(_raw(""), 99)
    with pytest.raises(ValueError):
        get_base_model_question_template_answer(_raw(""), -1)


def test_stop_and_n_shot_pinned_to_upstream():
    # Upstream LCB runner default is `--stop "###"` (split on ',' -> ["###"]);
    # default 3-shot matches DeepSeek-V3 Table 3. Pin both so a drift is loud.
    assert STOP_SEQUENCES == ("###",)
    assert N_SHOT == 3


def test_prefix_plus_target_equals_faithful_builder():
    # The decomposed prefix + target block must reproduce the faithful upstream
    # entry point byte-for-byte, for both pools.
    for starter in ("", _STARTER):
        q = _raw(starter_code=starter)
        rebuilt = get_base_model_fewshot_prefix(bool(starter), 2) + (
            get_base_model_target_block(q["question_content"], q["starter_code"])
        )
        assert rebuilt == get_base_model_question_template_answer(q, 2)


# --------------------------------------------------------------------------- #
# Task stages
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_preprocess_returns_base_prompt_string():
    task, _ = _task(n_shot=2)
    try:
        await task.setup()  # framework contract: setup() runs before preprocess()
        prompt = await task.preprocess(
            _raw(starter_code=_STARTER),
            TaskContext(sample_id=0, raw_sample=_raw(starter_code=_STARTER)),
        )
        assert isinstance(prompt, str)
        assert prompt.count("### Question") == 3  # 2 shots + target
        assert "### Starter Code" in prompt
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_setup_caches_fewshot_prefix_and_preprocess_reuses_it():
    task, _ = _task(n_shot=2)
    try:
        await task.setup()
        # both pools (stdin / func) precomputed once, not per sample
        assert set(task._fewshot_prefix) == {False, True}
        cached_true = task._fewshot_prefix[True]

        raw = _raw(starter_code=_STARTER)
        out = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
        # preprocess output is the cached prefix + the per-sample target block
        assert out.startswith(cached_true)
        assert out == get_base_model_question_template_answer(raw, 2)
        # cache object identity unchanged → no rebuild happened in preprocess
        assert task._fewshot_prefix[True] is cached_true
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_infer_forwards_only_stop_and_n_not_decoding_params():
    # Decoding params (temperature/max_tokens) must come from model config /
    # infer_args, never from the task layer (would silently override model args).
    task, model = _task(n=4)
    try:
        await task.infer("prompt", TaskContext(sample_id=0, raw_sample=_raw()))
    finally:
        await task.shutdown()

    assert model.last_kwargs["stop"] == ["###"]
    assert model.last_kwargs["n"] == 4
    assert "max_tokens" not in model.last_kwargs
    assert "temperature" not in model.last_kwargs


@pytest.mark.anyio
async def test_postprocess_strips_each_choice_generic_base():
    # GenericBase extraction returns the raw completion, stripped (no ``` fences).
    texts = ["  print(1)\n", "\nclass Solution:\n    pass\n  "]
    task, _ = _task(texts=texts)
    inferred = ModelOutput(model=task.model.meta(), texts=texts)
    try:
        post = await task.postprocess(
            inferred,
            TaskContext(sample_id=0, raw_sample=_raw(), infer_result=inferred),
        )
    finally:
        await task.shutdown()

    assert post == ["print(1)", "class Solution:\n    pass"]


@pytest.mark.anyio
async def test_invalid_init_args_raise():
    with pytest.raises(ValueError):
        _task(n_shot=-1)


@pytest.mark.anyio
async def test_report_pass_at_1_and_pass_at_k():
    task, _ = _task(k=2)
    try:
        # one sample, two generations, one correct -> pass@1 = 0.5, pass@2 = 1.0
        feedback = [
            {"correct": True, "msg": "ok", "metrics": None},
            {"correct": False, "msg": "wrong answer", "metrics": None},
        ]
        report = await task.report(
            [TaskContext(sample_id=0, raw_sample=_raw(), feedback_result=feedback)],
            [],
        )
    finally:
        await task.shutdown()

    assert report["pass@1"] == 50.0
    assert report["pass@2"] == 100.0
    assert report["score"] == report["pass@1"]
    assert report["fails"] == 0
