"""Import-discipline and few-shot-pool tests for the DROP k-shot task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import subprocess
import sys

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.drop import DROPDataset, DROPDatasetSample
from sieval.tasks.drop_kshot_gen import DROPFewShotGenTask


def test_import_does_not_pull_drop_eval_backend():
    # drop_eval pulls scipy; importing the task for registration must not.
    code = (
        "import sys\n"
        "import sieval.tasks.drop_kshot_gen\n"
        "assert 'sieval.community.simple_evals.drop_eval' not in sys.modules, "
        "'drop_eval backend must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules
    # don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        return ModelOutput(model=self.meta(), texts=["Answer: 1"])


def _sample(tag: str) -> DROPDatasetSample:
    return {
        "context": f"ctx-{tag}",
        "completion": f"ans-{tag}",
        "ref_text": f"ans-{tag}",
    }


def _dataset(train: list[DROPDatasetSample] | None) -> DROPDataset:
    splits = {"test": HFDataset.from_list([dict(_sample("eval"))])}
    if train is not None:
        splits["train"] = HFDataset.from_list([dict(s) for s in train])
    return DROPDataset(_hf_dict=HFDatasetDict(splits))


# --- the few-shot pool must supply every shot meta.json records ------------


@pytest.mark.anyio
async def test_setup_aborts_when_train_split_is_shorter_than_n_shot():
    """A short pool used to truncate silently while meta.json still said n_shot.

    The abort belongs in setup() so it lands before any inference spend.
    """
    task = DROPFewShotGenTask(_dataset([_sample("t0")]), _StubChatModel(), n_shot=3)
    assert task.n_shot == 3
    with pytest.raises(ValueError, match="requires at least 3 examples"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_aborts_when_fewshot_split_is_absent():
    task = DROPFewShotGenTask(_dataset(None), _StubChatModel(), n_shot=3)
    with pytest.raises(ValueError, match="requires a 'train' split"):
        await task.setup()


@pytest.mark.anyio
async def test_exactly_enough_examples_is_accepted():
    train = [_sample("t0"), _sample("t1"), _sample("t2")]
    task = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await task.setup()
    raw = _sample("eval")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    # All three exemplars reached the prompt — the count is not merely declared.
    content = pre["prompt"][0]["content"]
    for tag in ("t0", "t1", "t2"):
        assert f"ctx-{tag}" in content


@pytest.mark.anyio
async def test_zero_shot_needs_no_train_split():
    """n_shot=0 renders an empty block, so a missing pool is not an error."""
    task = DROPFewShotGenTask(_dataset(None), _StubChatModel(), n_shot=0)
    await task.setup()
    raw = _sample("eval")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert "ctx-t0" not in pre["prompt"][0]["content"]


def test_negative_n_shot_rejected():
    with pytest.raises(ValueError, match="n_shot must be >= 0"):
        DROPFewShotGenTask(_dataset([_sample("t0")]), _StubChatModel(), n_shot=-1)


# --- hoisting the draw into setup() must not move the prompt ---------------


@pytest.mark.anyio
async def test_prompt_is_identical_with_and_without_setup():
    """The exemplar draw is seeded, so caching it in setup() is a pure hoist.

    preprocess() keeps a lazy fallback for callers that skip setup(); both paths
    must render the same bytes, or the hoist changed scores.
    """
    train = [_sample(f"t{i}") for i in range(5)]
    raw = _sample("eval")
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    cached = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await cached.setup()
    from_setup = await cached.preprocess(raw, ctx)

    lazy = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    from_fallback = await lazy.preprocess(raw, ctx)

    assert from_setup == from_fallback


@pytest.mark.anyio
async def test_repeated_preprocess_is_stable():
    train = [_sample(f"t{i}") for i in range(5)]
    task = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await task.setup()
    raw = _sample("eval")
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    first = await task.preprocess(raw, ctx)
    second = await task.preprocess(raw, ctx)
    assert first == second
