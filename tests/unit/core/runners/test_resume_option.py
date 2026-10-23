"""
``resume``: one option for an in-place resume and a copy.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path

import pytest

from sieval.core.runners.resume_option import as_resume, resume_source
from sieval.core.runners.runner import TaskRunnerConfig


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


class TestTaskRunnerConfig:
    def test_resume_defaults_to_none(self):
        assert TaskRunnerConfig().resume is None

    def test_a_string_resume_is_a_path(self, tmp_path):
        """YAML hands the runner a string; the field is typed for Python."""
        cfg = TaskRunnerConfig(resume=str(tmp_path))  # ty: ignore[invalid-argument-type]
        assert cfg.resume == tmp_path
