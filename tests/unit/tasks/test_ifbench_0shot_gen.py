"""Unit tests for the IFBench zero-shot generative task.

AI-Generated Code - GPT-5 (OpenAI)
"""

import subprocess
import sys
import types
from dataclasses import dataclass

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.ifbench import IFBenchDataset
from sieval.tasks.ifbench_0shot_gen import IFBenchZeroShotGenTask


def test_import_does_not_pull_evaluation_lib():
    # evaluation_lib pulls optional IFBench scorers; registration must not import it.
    code = (
        "import sys\n"
        "import sieval.tasks.ifbench_0shot_gen\n"
        "assert 'sieval.community.ifbench.evaluation_lib' not in sys.modules, "
        "'evaluation_lib must be lazy-imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


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

    def strict(inp: _FakeInputExample, prompt_to_response: dict[str, str]):
        assert prompt_to_response == {"final prompt": "final response"}
        return _FakeOutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=prompt_to_response[inp.prompt],
            follow_all_instructions=False,
            follow_instruction_list=[True, False],
        )

    def loose(inp: _FakeInputExample, prompt_to_response: dict[str, str]):
        assert prompt_to_response == {"final prompt": "final response"}
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


@pytest.mark.anyio
async def test_report_scores_finals_and_counts_fails(monkeypatch: pytest.MonkeyPatch):
    _install_fake_evaluator(monkeypatch)
    task = _task()
    final_ctx = TaskContext(
        sample_id=0,
        raw_sample=_sample("ifbench-1", "final prompt"),
        feedback_result="final response",
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
    }
