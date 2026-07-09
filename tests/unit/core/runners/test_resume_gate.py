"""Tests for sieval.core.runners.resume_gate — pure version-verdict ladder."""

from sieval.core.runners.resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_reject_message,
    resume_version_verdict,
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
