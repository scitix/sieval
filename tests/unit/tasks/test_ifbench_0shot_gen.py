"""Unit tests for the IFBench zero-shot generative task.

AI-Generated Code - GPT-5 (OpenAI)
"""

import sys
import types
from dataclasses import dataclass

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext, build_prediction_record
from sieval.datasets.ifbench import IFBenchDataset
from sieval.tasks import ifbench_0shot_gen as module
from sieval.tasks.ifbench_0shot_gen import IFBenchZeroShotGenTask


@dataclass
class _FakeInputExample:
    key: str
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, object]]


@dataclass
class _FakeOutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def _sample(key: str, prompt: str) -> dict[str, object]:
    return {
        "key": key,
        "prompt": prompt,
        "instruction_id_list": ["format:no_whitespace", "format:title_case"],
        "kwargs": [{"unused": None}, {}],
    }


def _task() -> IFBenchZeroShotGenTask:
    sample = _sample("ifbench-1", "final prompt")
    dataset = IFBenchDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([sample]),
                "test": HFDataset.from_list([sample]),
            }
        )
    )
    model = ChatModel(model="mock-chat", api_key="fake")
    return IFBenchZeroShotGenTask(dataset, model)


def _install_fake_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("sieval.community.ifbench.evaluation_lib")
    fake_module.__dict__["InputExample"] = _FakeInputExample

    def strict(
        inp: _FakeInputExample,
        prompt_to_response: dict[str, str],
        *,
        instruction_dict: dict[str, type] | None = None,
    ):
        assert prompt_to_response == {"final prompt": "final response"}
        # `None` is "grade through the vendored registry". Asserting it here is
        # what keeps the unqualified task upstream's: if it ever started passing
        # an overlay, this test -- not a run -- is where that surfaces.
        assert instruction_dict is None
        return _FakeOutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=prompt_to_response[inp.prompt],
            follow_all_instructions=False,
            follow_instruction_list=[True, False],
        )

    def loose(
        inp: _FakeInputExample,
        prompt_to_response: dict[str, str],
        *,
        instruction_dict: dict[str, type] | None = None,
    ):
        assert prompt_to_response == {"final prompt": "final response"}
        assert instruction_dict is None
        return _FakeOutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=prompt_to_response[inp.prompt],
            follow_all_instructions=True,
            follow_instruction_list=[True, True],
        )

    fake_module.__dict__["test_instruction_following_strict"] = strict
    fake_module.__dict__["test_instruction_following_loose"] = loose
    monkeypatch.setitem(
        sys.modules,
        "sieval.community.ifbench.evaluation_lib",
        fake_module,
    )


@pytest.fixture(autouse=True)
def fresh_nltk_check():
    """Keep `_ensure_nltk_resources`' cache from leaking between tests.

    Autouse because the cache is process-global: a test that lets a success
    through decides the outcome of every later one, whichever order they run in.
    """
    module._ensure_nltk_resources.cache_clear()
    yield
    module._ensure_nltk_resources.cache_clear()


@pytest.fixture
def staged_nltk(monkeypatch: pytest.MonkeyPatch):
    """Report every corpus as present, so grading tests do not need real data.

    `feedback()` verifies the corpora for real, which is the point of the check
    -- but a test driving it through a faked evaluator should still pass on a box
    that has never staged them.
    """
    import nltk

    monkeypatch.setattr(nltk.data, "find", lambda path: path)


@pytest.mark.usefixtures("staged_nltk")
@pytest.mark.anyio
async def test_report_scores_finals_and_counts_fails(monkeypatch: pytest.MonkeyPatch):
    _install_fake_evaluator(monkeypatch)
    task = _task()
    # Grading moved from report() into feedback() (the relocation #60 made for
    # IFEval), so the verdict is produced here and report() only aggregates. The
    # expected numbers below are unchanged from the pre-migration shape -- that
    # equality IS the parity check.
    raw = _sample("ifbench-1", "final prompt")
    grading_ctx = TaskContext(sample_id=0, raw_sample=raw)
    _, judgement = await task.feedback(
        build_prediction_record(["final response"]), grading_ctx
    )
    final_ctx = TaskContext(
        sample_id=0,
        raw_sample=raw,
        feedback_result=judgement,
    ).to_final()
    failed_ctx = TaskContext(
        sample_id=1,
        raw_sample=_sample("ifbench-2", "failed prompt"),
    ).to_failed(None, "error", "boom")

    report = await task.report([final_ctx], [failed_ctx])

    assert report == {
        "fails": 1,
        "strict_prompt_level_accuracy": 0.0,
        "strict_instruction_level_accuracy": 50.0,
        "loose_prompt_level_accuracy": 100.0,
        "loose_instruction_level_accuracy": 100.0,
        "score": 100.0,
        # IFBench's headline is the LOOSE prompt-level rate where IFEval's is
        # strict — the four rates are co-equal, so only this key says which.
        # `judged`: the failed context is counted in `fails`, not scored as a
        # prompt that followed nothing (which is why `score` is 100.0, not 50.0).
        "score_key": "loose_prompt_level_accuracy",
        "denominator_policy": "judged",
    }


def test_ensure_nltk_resources_passes_when_every_resource_is_staged(
    monkeypatch: pytest.MonkeyPatch,
):
    import nltk

    checked: list[str] = []

    def fake_find(path):
        checked.append(path)
        return path

    monkeypatch.setattr(nltk.data, "find", fake_find)
    module._ensure_nltk_resources()
    assert set(checked) == set(module._NLTK_RESOURCES)


def test_ensure_nltk_resources_names_every_missing_resource(
    monkeypatch: pytest.MonkeyPatch,
):
    import nltk

    absent = {"corpora/stopwords", "taggers/averaged_perceptron_tagger_eng"}

    def fake_find(path):
        if path in absent:
            raise LookupError(path)
        return path

    monkeypatch.setattr(nltk.data, "find", fake_find)
    with pytest.raises(LookupError) as excinfo:
        module._ensure_nltk_resources()

    message = str(excinfo.value)
    # All of them, not just the first one to fail: stopping at the first would
    # make an offline box stage its corpora one run at a time.
    for resource in absent:
        assert resource in message
    # And nothing that *is* staged. `tokenizers/punkt` is a prefix of
    # `tokenizers/punkt_tab`, so this one assertion rules out naming either.
    assert "tokenizers/punkt" not in message


def test_a_passing_check_runs_once_per_process(monkeypatch: pytest.MonkeyPatch):
    import nltk

    checked: list[str] = []

    def fake_find(path):
        checked.append(path)
        return path

    monkeypatch.setattr(nltk.data, "find", fake_find)
    module._ensure_nltk_resources()
    module._ensure_nltk_resources()
    # `nltk.data.find` walks every entry on `nltk.data.path`, and `feedback()`
    # calls this per sample -- 4 walks x set size is worth caching away.
    assert len(checked) == len(module._NLTK_RESOURCES)


def test_a_failing_check_is_not_cached(monkeypatch: pytest.MonkeyPatch):
    import nltk

    checked: list[str] = []

    def fake_find(path):
        checked.append(path)
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", fake_find)
    for _ in range(2):
        with pytest.raises(LookupError):
            module._ensure_nltk_resources()

    # `functools.cache` stores return values only, never raises -- which is what
    # keeps a broken box failing every sample instead of being marked done by the
    # first. A hand-rolled "checked already" flag would lose that.
    assert len(checked) == 2 * len(module._NLTK_RESOURCES)


def test_the_verified_resources_match_upstreams_download_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    import nltk

    # Reached through `instructions`, not imported directly: that module points
    # NLTK at the sieval cache dir *before* loading `instructions_util`, and
    # loading `instructions_util` is what runs the staging attempt.
    from sieval.community.ifbench import instructions

    looked_up: list[str] = []

    def fake_find(path):
        looked_up.append(path)
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk, "download", lambda name, **_kwargs: True)

    instructions.instructions_util.download_nltk_resources()

    # `_NLTK_RESOURCES` is a second copy of upstream's list. Driving upstream's own
    # helper with every lookup failing records what it asks for, so a corpus added
    # there but not here fails here rather than in a run.
    assert set(looked_up) == set(module._NLTK_RESOURCES)
    assert len(looked_up) == len(module._NLTK_RESOURCES)


@pytest.mark.anyio
async def test_feedback_stops_when_the_corpora_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    # Teeth for the call site, not the helper: unwire the check from `feedback()`
    # and this is what goes red. The faked evaluator grades happily with no NLTK
    # data, which is exactly how the real one hid the problem.
    _install_fake_evaluator(monkeypatch)

    import nltk

    def fake_find(path):
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", fake_find)

    task = _task()
    with pytest.raises(LookupError) as excinfo:
        await task.feedback(
            build_prediction_record(["final response"]),
            TaskContext(sample_id=0, raw_sample=_sample("ifbench-1", "final prompt")),
        )
    assert "SIEVAL_IFBENCH_NLTK_DATA" in str(excinfo.value)
