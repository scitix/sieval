"""
``resume``: one option for an in-place resume and a copy, and its deprecated
aliases ``auto_resume`` and ``resume_from``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path

import pytest

from sieval.core.runners.resume_option import (
    as_resume,
    fold_resume_aliases,
    resume_source,
)
from sieval.core.runners.runner import TaskRunnerConfig


def _fold(resume=None, **aliases):
    return fold_resume_aliases(
        resume, new="resume", auto="auto_resume", source="resume_from", **aliases
    )


class TestAsResume:
    @pytest.mark.parametrize("value", [None, True, False])
    def test_bools_and_none_stay(self, value):
        assert as_resume(value) is value

    def test_a_string_is_a_directory(self):
        assert as_resume("~/runs/a") == Path("~/runs/a").expanduser()

    def test_a_path_is_kept(self, tmp_path):
        assert as_resume(tmp_path) == tmp_path

    def test_an_empty_string_is_refused(self):
        with pytest.raises(ValueError, match="true, false or a directory"):
            as_resume("")

    def test_another_type_is_refused_naming_where(self):
        with pytest.raises(TypeError, match="runner_config.resume"):
            as_resume(3, "runner_config.resume")


class TestResumeSource:
    def test_true_resumes_in_place(self, tmp_path):
        assert resume_source(True, tmp_path) is None

    def test_no_resume_copies_nothing(self, tmp_path):
        assert resume_source(None, tmp_path) is None
        assert resume_source(False, tmp_path) is None

    def test_another_directory_is_copied(self, tmp_path):
        assert resume_source(tmp_path / "old", tmp_path / "new") == tmp_path / "old"

    def test_the_own_directory_resumes_in_place(self, tmp_path):
        assert resume_source(tmp_path / "a" / ".." / "own", tmp_path / "own") is None


class TestAliases:
    def test_none_given_leaves_resume(self, warnings):
        assert _fold(True) is True
        assert _fold(None) is None
        assert warnings == []

    @pytest.mark.parametrize("value", [True, False])
    def test_auto_resume_maps_to_resume_and_warns(self, warnings, value):
        assert _fold(auto_resume=value) is value
        assert warnings == [
            f"auto_resume is deprecated; use resume: {str(value).lower()}"
        ]

    def test_resume_from_maps_to_a_directory_and_warns(self, warnings, tmp_path):
        assert _fold(resume_from=str(tmp_path)) == tmp_path
        assert warnings == ["resume_from is deprecated; use resume <dir>"]

    def test_each_alias_warns_once_per_process(self, warnings, tmp_path):
        _fold(resume_from=str(tmp_path))
        _fold(resume_from=str(tmp_path))
        assert len(warnings) == 1

    def test_resume_from_wins_over_auto_resume(self, warnings, tmp_path):
        assert _fold(auto_resume=True, resume_from=str(tmp_path)) == tmp_path
        assert len(warnings) == 2

    def test_an_empty_resume_from_is_unset(self, warnings):
        assert _fold(resume_from="") is None
        assert warnings == []

    @pytest.mark.parametrize(
        ("resume", "aliases", "expected"),
        [
            (True, {"auto_resume": True}, True),
            (False, {"auto_resume": False}, False),
            (Path("/d"), {"auto_resume": True}, Path("/d")),
            (True, {"resume_from": "/d"}, Path("/d")),
            (Path("/d"), {"resume_from": "/d/"}, Path("/d")),
        ],
    )
    def test_agreeing_values_are_one(self, warnings, resume, aliases, expected):
        assert _fold(resume, **aliases) == expected

    @pytest.mark.parametrize(
        ("resume", "aliases"),
        [
            (True, {"auto_resume": False}),
            (False, {"auto_resume": True}),
            (Path("/d"), {"auto_resume": False}),
            (False, {"resume_from": "/d"}),
            (Path("/d"), {"resume_from": "/e"}),
        ],
    )
    def test_disagreeing_values_are_refused(self, warnings, resume, aliases):
        with pytest.raises(ValueError, match="disagree; set only resume"):
            _fold(resume, **aliases)

    def test_a_non_bool_auto_resume_is_refused(self):
        with pytest.raises(TypeError, match="auto_resume must be true or false"):
            _fold(auto_resume="yes")

    def test_a_bool_resume_from_is_refused(self):
        with pytest.raises(TypeError, match="resume_from must be a directory"):
            _fold(resume_from=True)


class TestTaskRunnerConfig:
    def test_resume_defaults_to_none(self):
        assert TaskRunnerConfig().resume is None

    def test_a_string_resume_is_a_path(self, tmp_path):
        """YAML hands the runner a string; the field is typed for Python."""
        cfg = TaskRunnerConfig(resume=str(tmp_path))  # ty: ignore[invalid-argument-type]
        assert cfg.resume == tmp_path

    def test_auto_resume_folds_into_resume(self, warnings):
        cfg = TaskRunnerConfig(auto_resume=True)
        assert cfg.resume is True
        assert cfg.auto_resume is None
        assert warnings == ["auto_resume is deprecated; use resume: true"]

    def test_resume_from_folds_into_resume(self, warnings, tmp_path):
        cfg = TaskRunnerConfig(resume_from=str(tmp_path))
        assert cfg.resume == tmp_path
        assert cfg.resume_from is None
        assert warnings == ["resume_from is deprecated; use resume <dir>"]

    def test_an_alias_disagreeing_with_resume_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="disagree"):
            TaskRunnerConfig(resume=tmp_path / "a", resume_from=str(tmp_path / "b"))
