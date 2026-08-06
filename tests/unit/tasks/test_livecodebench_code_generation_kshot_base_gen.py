"""Unit tests for the LiveCodeBench code-generation few-shot base-model task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import json

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
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.livecodebench_code_generation import LiveCodeBenchDataset
from sieval.tasks.livecodebench_code_generation_kshot_base_gen import (
    N_SHOT,
    STOP_SEQUENCES,
    LiveCodeBenchCodeGenerationFewShotBaseGenTask,
)

_STARTER = "class Solution:\n    def solve(self) -> int:\n        "


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
        pre = await task.preprocess(
            _raw(starter_code=_STARTER),
            TaskContext(sample_id=0, raw_sample=_raw(starter_code=_STARTER)),
        )
        prompt = pre["prompt"]
        assert isinstance(prompt, str)
        # The ground truth is a test suite, so there is no `reference` value.
        assert "reference" not in pre
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
        pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
        out = pre["prompt"]
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
        await task.infer(
            {"prompt": "prompt"}, TaskContext(sample_id=0, raw_sample=_raw())
        )
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

    assert [r["prediction"] for r in post["rollouts"]] == [
        "print(1)",
        "class Solution:\n    pass",
    ]


@pytest.mark.anyio
async def test_invalid_init_args_raise():
    with pytest.raises(ValueError):
        _task(n_shot=-1)


@pytest.mark.anyio
async def test_report_pass_at_1_and_pass_at_k():
    task, _ = _task(k=2)
    try:
        # one sample, two generations, one correct -> pass@1 = 0.5, pass@2 = 1.0
        feedback = _judgement((True, "ok"), (False, "wrong answer"))
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


# --------------------------------------------------------------------------- #
# Execution budget on the wire (deviation 4 -- see the module docstring)
# --------------------------------------------------------------------------- #
_PUBLIC = [{"input": "1\n", "output": "2", "testtype": "stdin"}]
_PRIVATE = [
    {"input": "2\n", "output": "4", "testtype": "stdin"},
    {"input": "3\n", "output": "6", "testtype": "stdin"},
]
_N_CASES = len(_PUBLIC) + len(_PRIVATE)


class _Response:
    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {
            "status": True,
            "msg": "",
            "data": {"n_cases": _N_CASES, "n_passed": _N_CASES},
        }


class _CapturingEvaluator:
    """Stands in for the code-eval service, recording what was asked of it."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.deadlines: list[float] = []

    async def post(self, url, *, json, timeout):
        _ = url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        return _Response()

    async def aclose(self) -> None:
        return None


def _raw_with_cases() -> dict:
    return {
        "question_content": "Double the input.",
        "starter_code": "",
        "public_test_cases": json.dumps(_PUBLIC),
        "private_test_cases": json.dumps(_PRIVATE),
        "metadata": json.dumps({}),
    }


async def _post_one(**kwargs) -> _CapturingEvaluator:
    """Run `feedback` for a single rollout and hand back what the evaluator saw."""
    task, _ = _task(**kwargs)
    await task._http_client.aclose()  # the real client is never used
    evaluator = _CapturingEvaluator()
    task._http_client = evaluator
    raw = _raw_with_cases()
    try:
        await task.feedback(
            build_prediction_record(["print(int(input()) * 2)"]),
            TaskContext(sample_id=0, raw_sample=raw),
        )
    finally:
        await task.shutdown()
    return evaluator


@pytest.mark.anyio
async def test_the_default_is_upstreams_six_seconds_per_case():
    # codegen_metrics(..., timeout=6). Pinned so a drift from upstream is loud.
    evaluator = await _post_one()

    (body,) = evaluator.bodies
    assert body["timeout_per_case"] == 6.0


@pytest.mark.anyio
async def test_the_suite_wall_is_derived_as_upstreams_backstop():
    # Must stay in step with the sibling 0-shot task, which computes the same wall.
    evaluator = await _post_one(timeout_per_case=6.0)

    (body,) = evaluator.bodies
    # check_correctness joins its worker at (timeout + 1) * n + 5.
    assert body["timeout"] == (6.0 + 1.0) * _N_CASES + 5.0 == 26.0
    assert evaluator.deadlines == [26.0 + 2]


@pytest.mark.anyio
async def test_the_old_whole_suite_knob_is_gone_not_silently_reinterpreted():
    # `timeout` used to be the base of a whole-suite wall; a stale config setting it
    # must fail at construction rather than be regraded under the new rule.
    with pytest.raises(TypeError):
        await _post_one(timeout=30.0)
