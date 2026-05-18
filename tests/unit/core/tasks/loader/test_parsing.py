"""
Tests for low-level TaskLoader parsing utilities: _try_task_action,
_parse_idx_line, _parse_idx_file, _should_replace_offset.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from pathlib import Path

from sieval.core.tasks.context import TaskAction, TaskStage
from sieval.core.tasks.loader import (
    _parse_idx_file,
    _parse_idx_line,
    _should_replace_offset,
    _try_task_action,
)


class TestTryTaskAction:
    def test_valid_actions(self):
        assert _try_task_action("preprocess") == TaskAction.PREPROCESS
        assert _try_task_action("infer") == TaskAction.INFER
        assert _try_task_action("postprocess") == TaskAction.POSTPROCESS
        assert _try_task_action("feedback") == TaskAction.FEEDBACK

    def test_none_returns_none(self):
        assert _try_task_action(None) is None

    def test_empty_returns_none(self):
        assert _try_task_action("") is None

    def test_invalid_returns_none(self):
        assert _try_task_action("bogus") is None


class TestShouldReplaceOffset:
    def test_replace_when_current_missing(self):
        candidate = (0, -1, Path("0/final/0.jsonl"), 10, 20)
        assert _should_replace_offset(None, candidate)

    def test_stage_prefers_higher_iteration(self):
        current = (0, -1, Path("0/final/0.jsonl"), 100, 20)
        candidate = (1, -1, Path("0/final/0.jsonl"), 0, 20)
        assert _should_replace_offset(current, candidate)

    def test_stage_same_iteration_same_shard_prefers_later_offset(self):
        current = (1, -1, Path("1/final/0.jsonl"), 100, 20)
        candidate = (1, -1, Path("1/final/0.jsonl"), 120, 20)
        assert _should_replace_offset(current, candidate)

    def test_best_same_iteration_prefers_higher_rank(self):
        current = (1, 1, Path("1/preprocessed/0.jsonl"), 100, 20)
        candidate = (1, 2, Path("1/inferred/0.jsonl"), 0, 20)
        assert _should_replace_offset(current, candidate)

    def test_cross_shard_tiebreak_is_deterministic(self):
        current = (0, -1, Path("0/preprocessed/10.jsonl"), 0, 20)
        candidate = (0, -1, Path("0/preprocessed/20.jsonl"), 0, 20)
        assert _should_replace_offset(current, candidate)


class TestParseIdxLine:
    def test_basic_line(self):
        parts = ["0", "0", "final", "0", "128", "", "", "0"]
        rec = _parse_idx_line(parts)
        assert rec is not None
        assert rec["sample_id"] == 0
        assert rec["iteration"] == 0
        assert rec["stage"] == TaskStage.FINAL
        assert rec["offset"] == 0
        assert rec["length"] == 128

    def test_with_error_fields(self):
        parts = ["5", "1", "failed", "256", "64", "infer", "exception::Timeout", "2"]
        rec = _parse_idx_line(parts)
        assert rec is not None
        assert rec["error_action"] == "infer"
        assert rec["error_reason"] == "exception::Timeout"
        assert rec["retry_count"] == 2

    def test_string_sample_id(self):
        parts = ["sample_abc", "0", "inferred", "100", "50"]
        rec = _parse_idx_line(parts)
        assert rec is not None
        assert rec["sample_id"] == "sample_abc"

    def test_too_few_parts(self):
        parts = ["0", "0", "final", "0"]
        assert _parse_idx_line(parts) is None

    def test_invalid_stage(self):
        parts = ["0", "0", "bogus_stage", "0", "128"]
        assert _parse_idx_line(parts) is None

    def test_non_numeric_offset(self):
        parts = ["0", "0", "final", "abc", "128"]
        assert _parse_idx_line(parts) is None


class TestParseIdxFile:
    def test_valid_file(self, tmp_path):
        idx_file = tmp_path / "test.idx"
        idx_file.write_text(
            "0\t0\tfinal\t0\t100\t\t\t0\n1\t0\tfinal\t100\t120\t\t\t0\n"
        )
        records = _parse_idx_file(idx_file)
        assert len(records) == 2
        assert records[0]["sample_id"] == 0
        assert records[1]["sample_id"] == 1

    def test_empty_file(self, tmp_path):
        idx_file = tmp_path / "empty.idx"
        idx_file.write_text("")
        assert _parse_idx_file(idx_file) == []

    def test_missing_file(self, tmp_path):
        assert _parse_idx_file(tmp_path / "nonexistent.idx") == []

    def test_skips_blank_lines(self, tmp_path):
        idx_file = tmp_path / "test.idx"
        idx_file.write_text(
            "0\t0\tfinal\t0\t100\t\t\t0\n\n1\t0\tfinal\t100\t120\t\t\t0\n"
        )
        records = _parse_idx_file(idx_file)
        assert len(records) == 2

    def test_skips_malformed_lines(self, tmp_path):
        idx_file = tmp_path / "test.idx"
        idx_file.write_text("0\t0\tfinal\t0\t100\nbad_line\n1\t0\tfinal\t100\t120\n")
        records = _parse_idx_file(idx_file)
        assert len(records) == 2
