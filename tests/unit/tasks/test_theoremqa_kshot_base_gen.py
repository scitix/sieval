"""
Unit tests for TheoremQA k-shot base generative task.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import importlib
import subprocess
import sys

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks.context import TaskContext

_TASK_MODULE = "sieval.tasks.theoremqa_kshot_base_gen"
_DATASET_MODULE = "sieval.datasets.theoremqa"
_TASK_EXPORTS = ("TheoremQAKShotBaseGenTask",)
_DATASET_EXPORTS = ("TheoremQADataset", "TheoremQADatasetSample")


def _drop_theoremqa_modules() -> None:
    sys.modules.pop(_TASK_MODULE, None)
    sys.modules.pop(_DATASET_MODULE, None)
    for package_name, exports in (
        ("sieval.tasks", _TASK_EXPORTS),
        ("sieval.datasets", _DATASET_EXPORTS),
    ):
        package = sys.modules.get(package_name)
        if package is None:
            continue
        for export in exports:
            package.__dict__.pop(export, None)


@pytest.fixture(autouse=True)
def _preserve_registries():
    from sieval.core.datasets.meta import DATASET_REGISTRY, SAMPLE_TO_DATASET
    from sieval.core.tasks.meta import _TASK_CLASSES, TASK_REGISTRY

    task_snapshot = dict(TASK_REGISTRY)
    task_classes_snapshot = dict(_TASK_CLASSES)
    dataset_snapshot = dict(DATASET_REGISTRY)
    sample_map_snapshot = dict(SAMPLE_TO_DATASET)

    TASK_REGISTRY.clear()
    _TASK_CLASSES.clear()
    DATASET_REGISTRY.clear()
    SAMPLE_TO_DATASET.clear()
    _drop_theoremqa_modules()
    try:
        yield
    finally:
        _drop_theoremqa_modules()
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(task_snapshot)
        _TASK_CLASSES.clear()
        _TASK_CLASSES.update(task_classes_snapshot)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(dataset_snapshot)
        SAMPLE_TO_DATASET.clear()
        SAMPLE_TO_DATASET.update(sample_map_snapshot)


class _MockGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=["The answer is 4"])

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
        raise NotImplementedError


def _task_module():
    return importlib.import_module(_TASK_MODULE)


def _theoremqa_examples():
    return _task_module()._THEOREMQA_EXAMPLES


def _dataset():
    sample = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    dataset_module = importlib.import_module(_DATASET_MODULE)
    return dataset_module.TheoremQADataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([sample])})
    )


def _task(k: int | None = None):
    task_module = _task_module()
    return task_module.TheoremQAKShotBaseGenTask(_dataset(), _MockGenModel(), k=k)


def test_import_does_not_pull_latex2sympy():
    code = (
        "import sys\n"
        "import sieval.tasks.theoremqa_kshot_base_gen\n"
        "assert 'latex2sympy2' not in sys.modules, "
        "'latex2sympy2 must be lazy-imported'\n"
        "assert 'latex2sympy2_extended' not in sys.modules, "
        "'latex2sympy2_extended must be lazy-imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.anyio
async def test_report_divides_metrics_by_completed_finals_only():
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(k=0)
    final_ctx = TaskContext(
        sample_id=0,
        raw_sample=raw,
        feedback_result={"correct": True, "pred": "4", "answer": "4"},
    )
    failed_ctx = TaskContext(sample_id=1, raw_sample=raw)

    report = await task.report([final_ctx], [failed_ctx])

    assert report["score"] == 100.0
    assert report["accuracy"] == 100.0
    assert report["fails"] == 1.0
    assert report["empty"] == 0.0


@pytest.mark.anyio
async def test_infer_only_forwards_prompt_coupled_stop():
    task_module = _task_module()
    model = _MockGenModel()
    task = task_module.TheoremQAKShotBaseGenTask(_dataset(), model, k=0)

    await task.infer(
        "prompt",
        TaskContext(sample_id=0, raw_sample={"Question": "What is 2+2?"}),
    )

    assert model.last_kwargs == {"stop": task_module._STOP_TOKENS}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("k", "expected_examples"),
    [(None, None), (0, 0), (2, 2)],
)
async def test_preprocess_uses_configured_k(k, expected_examples):
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(k=k)
    examples = _theoremqa_examples()
    if expected_examples is None:
        expected_examples = len(examples)

    prompt = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert prompt.count("Problem:\n") == expected_examples + 1
    assert prompt.endswith("Problem:\nWhat is 2+2?\nSolution:\n")
    if expected_examples == 0:
        assert examples[0][0] not in prompt
    if expected_examples == 2:
        assert examples[1][0] in prompt
        assert examples[2][0] not in prompt


@pytest.mark.anyio
async def test_preprocess_preserves_official_runtime_prompt_artifacts():
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(k=3)

    prompt = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert "\u2248 833.33 frames" in prompt
    assert "Bytes/frame is approximately 833.33 frames" not in prompt
    upstream_control_line = (
        "Let's calculate the numerical value of "
        "$\\left[\x0crac{10}{3}, \x0crac{4}{3}\x0dight]_C$ "
        "as [3.33, 1.33]."
    )
    assert upstream_control_line in prompt


def test_constructor_rejects_negative_k():
    with pytest.raises(ValueError, match="k must"):
        _task(k=-1)


def test_constructor_rejects_too_many_examples():
    k = len(_theoremqa_examples()) + 1
    with pytest.raises(ValueError, match="k must"):
        _task(k=k)


@pytest.mark.parametrize("k", [True, 1.5, "2"])
def test_constructor_rejects_non_integer_k(k):
    with pytest.raises(TypeError, match="k must be an int"):
        _task(k=k)
