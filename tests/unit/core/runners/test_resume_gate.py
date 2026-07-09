"""Tests for sieval.core.runners.resume_gate — pure version-verdict ladder."""

from pathlib import Path

import orjson
import pytest

from sieval.core.runners.resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_reject_message,
    resume_version_verdict,
)
from sieval.core.runners.runner import gate_resume_version


def _write_meta(root: Path, version: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_bytes(
        orjson.dumps({"version": version, "deterministic": False})
    )


class TestResumeVersionVerdict:
    def test_exact_match(self):
        assert resume_version_verdict("0.6.0", "0.6.0").action is ResumeAction.EXACT

    def test_exact_match_dev_build(self):
        v = "0.5.1.dev24+gabc"
        assert resume_version_verdict(v, v).action is ResumeAction.EXACT

    def test_compatible_same_minor_under_1_0(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.3").action is ResumeAction.COMPATIBLE
        )

    def test_reject_minor_break_under_1_0(self):
        assert resume_version_verdict("0.6.0", "0.7.0").action is ResumeAction.REJECT

    def test_reject_1_0_boundary(self):
        assert resume_version_verdict("0.9.5", "1.0.0").action is ResumeAction.REJECT

    def test_compatible_same_major_post_1_0(self):
        assert (
            resume_version_verdict("1.2.0", "1.5.9").action is ResumeAction.COMPATIBLE
        )

    def test_reject_major_break_post_1_0(self):
        assert resume_version_verdict("1.9.0", "2.0.0").action is ResumeAction.REJECT

    def test_reject_dev_mismatch(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1.dev3+gxyz").action
            is ResumeAction.REJECT
        )

    def test_reject_local_mismatch(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1+local").action is ResumeAction.REJECT
        )

    def test_reject_unparseable(self):
        assert (
            resume_version_verdict("not-a-version", "0.6.0").action
            is ResumeAction.REJECT
        )

    def test_reject_zero_version_mismatch(self):
        assert resume_version_verdict("0.0.0", "0.6.0").action is ResumeAction.REJECT

    def test_zero_vs_zero_is_exact(self):
        # EM precedence (rule 1) beats the 0.0.0 reject (rule 2).
        assert resume_version_verdict("0.0.0", "0.0.0").action is ResumeAction.EXACT

    def test_reject_dev_mismatch_run_side(self):
        # dev marker on the FIRST arg (v_run) must also reject
        assert (
            resume_version_verdict("0.6.1.dev3+gxyz", "0.6.0").action
            is ResumeAction.REJECT
        )

    def test_reject_zero_version_cur_side(self):
        # 0.0.0 on the SECOND arg (v_cur) must also reject
        assert resume_version_verdict("0.6.0", "0.0.0").action is ResumeAction.REJECT

    def test_reject_reason_is_populated(self):
        assert resume_version_verdict("0.6.0", "0.7.0").reason != ""


class TestFormatRejectMessage:
    def test_contains_versions_reason_and_recovery(self):
        msg = format_reject_message("0.6.0", "0.7.0", "incompatible version series")
        assert "0.6.0" in msg
        assert "0.7.0" in msg
        assert "incompatible version series" in msg
        assert "start fresh" in msg


def test_resume_version_error_is_runtimeerror():
    assert issubclass(ResumeVersionError, RuntimeError)


class TestGateResumeVersion:
    def test_exact_passes(self, tmp_path):
        _write_meta(tmp_path, "0.6.0")
        gate_resume_version(tmp_path, "0.6.0")  # no raise

    def test_compatible_passes(self, tmp_path):
        _write_meta(tmp_path, "0.6.0")
        gate_resume_version(tmp_path, "0.6.3")  # no raise

    def test_incompatible_raises(self, tmp_path):
        _write_meta(tmp_path, "0.6.0")
        with pytest.raises(ResumeVersionError, match="incompatible version series"):
            gate_resume_version(tmp_path, "0.7.0")

    def test_missing_meta_raises(self, tmp_path):
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.6.0")

    def test_unreadable_meta_raises(self, tmp_path):
        (tmp_path / "meta.json").write_bytes(b"not json{")
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.6.0")

    def test_meta_without_version_key_raises(self, tmp_path):
        (tmp_path / "meta.json").write_bytes(orjson.dumps({"deterministic": True}))
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.6.0")
