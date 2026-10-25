"""The refusal contract: exit codes and the ``SIEVAL_REFUSAL`` line are pinned.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json

import pytest

from sieval.cli.refusal import (
    EXIT_CODES,
    MESSAGE_LIMIT,
    REFUSAL_LINE_VERSION,
    classify_refusals,
    report_refusals,
)
from sieval.core.runners import StageResumeError
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.core.tasks.python_requires import (
    MissingPythonRequirementsError,
    UnmetPythonRequirement,
)
from sieval.infer.introspect import LaunchGpuError


def _payload(line: str) -> dict:
    prefix, _, body = line.partition(" ")
    assert prefix == "SIEVAL_REFUSAL"
    return json.loads(body)


class TestTheContractIsPinned:
    """Changing any of these breaks every scheduler that matches on them."""

    def test_exit_codes(self):
        assert EXIT_CODES == {"resume": 3, "hardware": 4, "environment": 5}

    def test_line_version(self):
        assert REFUSAL_LINE_VERSION == 2

    @pytest.mark.parametrize(
        ("exc", "kind", "reason", "task"),
        [
            (StageResumeError("inference", "m", task="t"), "resume", "inference", "t"),
            (StageResumeError("data", "m", task="t"), "resume", "data", "t"),
            (ResumeVersionError("m", task="t"), "resume", "schema", "t"),
            (LaunchGpuError("no_gpu", "m"), "hardware", "no_gpu", None),
            (LaunchGpuError("mismatch", "m"), "hardware", "mismatch", None),
        ],
    )
    def test_kind_and_reason(self, exc, kind, reason, task):
        (refusal,), code = classify_refusals(exc)
        assert (refusal.kind, refusal.reason, refusal.task) == (kind, reason, task)
        assert code == EXIT_CODES[kind]


def _missing(*tasks: str) -> MissingPythonRequirementsError:
    return MissingPythonRequirementsError(
        {
            task: [UnmetPythonRequirement("rdkit", "rdkit>=2023.9", "not installed")]
            for task in tasks
        }
    )


class TestEnvironmentRefusal:
    def test_one_refusal_per_task_lacking_a_package(self):
        refusals, code = classify_refusals(_missing("a", "b"))
        assert code == 5
        assert [(r.kind, r.reason, r.task) for r in refusals] == [
            ("environment", "python_requires", "a"),
            ("environment", "python_requires", "b"),
        ]

    def test_the_line_names_the_package(self, capsys):
        assert report_refusals(_missing("a")) == 5
        assert _payload(capsys.readouterr().err.strip()) == {
            "v": 2,
            "kind": "environment",
            "reason": "python_requires",
            "task": "a",
            "message": "Missing Python packages: rdkit>=2023.9 "
            "[import rdkit]: not installed",
        }

    def test_beside_a_resume_refusal_there_is_no_single_code(self):
        group = ExceptionGroup(
            "x", [_missing("a"), StageResumeError("data", "m", task="b")]
        )
        refusals, code = classify_refusals(group)
        assert len(refusals) == 2
        assert code is None


class TestClassify:
    def test_a_crash_is_no_refusal(self):
        assert classify_refusals(RuntimeError("boom")) == ([], None)

    def test_a_task_batch_of_refusals_takes_their_code(self):
        group = ExceptionGroup(
            "2 of 2 task(s) failed: a, b",
            [
                StageResumeError("inference", "m", task="a"),
                ResumeVersionError("m", task="b"),
            ],
        )
        refusals, code = classify_refusals(group)
        assert [r.task for r in refusals] == ["a", "b"]
        assert code == 3

    def test_a_refusal_beside_a_crash_exits_like_a_crash(self):
        """A retry may succeed for the crashed task, so it is not a refusal."""
        group = ExceptionGroup(
            "2 of 2 task(s) failed: a, b",
            [StageResumeError("inference", "m", task="a"), ValueError("b")],
        )
        refusals, code = classify_refusals(group)
        assert [r.task for r in refusals] == ["a"]
        assert code is None

    def test_two_kinds_have_no_single_code(self):
        group = ExceptionGroup(
            "x", [StageResumeError("data", "m"), LaunchGpuError("no_gpu", "m")]
        )
        assert classify_refusals(group)[1] is None

    def test_nested_groups_are_walked(self):
        inner = ExceptionGroup("inner", [StageResumeError("data", "m", task="a")])
        refusals, code = classify_refusals(ExceptionGroup("outer", [inner]))
        assert [r.reason for r in refusals] == ["data"]
        assert code == 3


class TestLine:
    def test_carries_the_first_line_of_the_message(self, capsys):
        exc = StageResumeError(
            "inference", "Resume refused: stage differs.\n  recorded: /x", task="t"
        )
        assert report_refusals(exc) == 3

        out = capsys.readouterr()
        assert out.out == ""
        (line,) = out.err.splitlines()
        assert _payload(line) == {
            "v": 2,
            "kind": "resume",
            "reason": "inference",
            "task": "t",
            "message": "Resume refused: stage differs.",
        }

    def test_omits_task_when_unknown(self, capsys):
        report_refusals(LaunchGpuError("mismatch", "other GPUs"))
        assert "task" not in _payload(capsys.readouterr().err.strip())

    def test_a_long_message_is_capped(self, capsys):
        report_refusals(LaunchGpuError("no_gpu", "x" * 1000))
        message = _payload(capsys.readouterr().err.strip())["message"]
        assert len(message) == MESSAGE_LIMIT
        assert message.endswith("…")

    def test_one_line_per_refusal_even_when_the_code_is_1(self, capsys):
        group = ExceptionGroup(
            "x", [StageResumeError("data", "m", task="a"), ValueError("b")]
        )
        assert report_refusals(group) is None
        assert len(capsys.readouterr().err.splitlines()) == 1

    def test_a_crash_prints_nothing(self, capsys):
        assert report_refusals(ValueError("boom")) is None
        assert capsys.readouterr().err == ""
