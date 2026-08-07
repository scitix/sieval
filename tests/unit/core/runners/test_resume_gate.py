"""Tests for sieval.core.runners.resume_gate — pure version-verdict ladder."""

from pathlib import Path

import orjson
import pytest

from sieval.core.runners.resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_identity_reject_message,
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


class TestRejectReasonIdentifiesTheRule:
    """Each rejection names *which* rule fired, not merely that one did.

    The four reject reasons lead an operator to four different fixes — fix the
    version string, reinstall a released build, pin a non-dev build, or match the
    series. Asserting only that `reason` is non-empty lets any of them be swapped
    for another without a test noticing, including a swap that sends the operator
    somewhere useless.
    """

    def test_unparseable_says_so(self):
        assert (
            resume_version_verdict("not-a-version", "0.6.0").reason
            == "version string is unparseable"
        )

    def test_unknown_version_says_so(self):
        assert (
            resume_version_verdict("0.0.0", "0.6.0").reason
            == "version is unknown (0.0.0)"
        )

    def test_dev_build_says_so(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1.dev3+gxyz").reason
            == "development/local build cannot be matched non-exactly"
        )

    def test_local_build_shares_the_dev_reason(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1+local").reason
            == "development/local build cannot be matched non-exactly"
        )

    def test_series_break_says_so(self):
        assert (
            resume_version_verdict("0.6.0", "0.7.0").reason
            == "incompatible version series"
        )

    def test_a_permitted_resume_carries_no_reason(self):
        # `reason` is documented as populated only for REJECT; a stray reason on
        # an accepted resume would surface as an operator-facing message about a
        # run that was never blocked.
        assert resume_version_verdict("0.6.0", "0.6.0").reason == ""
        assert resume_version_verdict("0.6.0", "0.6.3").reason == ""


class TestLadderPrecedence:
    """The ladder's *order* is the contract, not just its membership."""

    def test_exact_match_outranks_every_reject_rule(self):
        # Resuming your own build is always allowed, even when the string would
        # fail every later rule.
        for v in ("0.0.0", "not-a-version", "0.5.1.dev24+gabc", "0.6.0+local"):
            assert resume_version_verdict(v, v).action is ResumeAction.EXACT

    def test_unparseable_outranks_the_unknown_check(self):
        # An unparseable string cannot be compared to 0.0.0 at all, so it must be
        # reported as unparseable rather than as "unknown".
        assert (
            resume_version_verdict("not-a-version", "0.0.0").reason
            == "version string is unparseable"
        )

    def test_unknown_outranks_the_dev_check(self):
        assert (
            resume_version_verdict("0.0.0", "0.6.1.dev3+gxyz").reason
            == "version is unknown (0.0.0)"
        )

    def test_dev_outranks_the_series_check(self):
        # Different series *and* a dev build: the dev build is the actionable
        # one, since matching the series would still not make it resumable.
        assert (
            resume_version_verdict("0.6.0", "1.2.0.dev1").reason
            == "development/local build cannot be matched non-exactly"
        )


class TestBreakAxis:
    """Under 1.0 the break axis is minor; from 1.0 on it is major."""

    def test_patch_bumps_stay_compatible_under_1_0(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.99").action is ResumeAction.COMPATIBLE
        )

    def test_minor_is_the_break_axis_under_1_0(self):
        assert resume_version_verdict("0.6.9", "0.7.0").action is ResumeAction.REJECT

    def test_minor_bumps_stay_compatible_from_1_0(self):
        assert (
            resume_version_verdict("1.0.0", "1.99.0").action is ResumeAction.COMPATIBLE
        )

    def test_major_is_the_break_axis_from_1_0(self):
        assert resume_version_verdict("1.99.0", "2.0.0").action is ResumeAction.REJECT

    def test_crossing_into_1_0_breaks(self):
        # The axis itself changes here, so the pair is incompatible in both
        # directions regardless of how close the numbers look.
        assert resume_version_verdict("0.9.9", "1.0.0").action is ResumeAction.REJECT
        assert resume_version_verdict("1.0.0", "0.9.9").action is ResumeAction.REJECT

    def test_pinnable_prereleases_are_not_treated_as_dev_builds(self):
        # A pre-release is a fixed artifact, unlike dev/local; it falls through
        # to the series check rather than being rejected as unpinnable.
        assert (
            resume_version_verdict("0.6.0", "0.6.1rc1").action
            is ResumeAction.COMPATIBLE
        )
        assert (
            resume_version_verdict("0.6.0", "0.7.0rc1").reason
            == "incompatible version series"
        )


class TestFormatIdentityRejectMessage:
    """Previously untested entirely — 18 mutants with no test to see them.

    This is the message shown when a result directory was produced by a
    *different* task. Its own text states the stake: a finished run is matched by
    path alone, so resuming here would hand back the persisted task's report as
    this task's result without running a sample. The operator can only act on it
    if it names both tasks and both ways out.
    """

    def _msg(self) -> str:
        return format_identity_reject_message("gsm8k_0shot_gen", "math_500_0shot_gen")

    def test_names_both_tasks(self):
        msg = self._msg()
        assert "gsm8k_0shot_gen" in msg
        assert "math_500_0shot_gen" in msg

    def test_distinguishes_persisted_from_current(self):
        # Both names appear; without labels the operator cannot tell which is
        # which, and the two fixes are not symmetric.
        msg = self._msg()
        persisted = msg.index("gsm8k_0shot_gen")
        current = msg.index("math_500_0shot_gen")
        assert msg.index("persisted") < persisted < current
        assert "meta.json" in msg, "the operator has to know where to look"

    def test_states_the_consequence(self):
        msg = self._msg()
        assert "without running a single sample" in msg

    def test_offers_both_recovery_paths(self):
        msg = self._msg()
        assert "Remove the result_dir and start fresh" in msg
        assert "Give this task its own result_dir" in msg

    def test_leads_with_the_abort(self):
        assert self._msg().startswith("Resume aborted:")


class TestFormatRejectMessageStructure:
    """The message body, not just that four substrings appear somewhere."""

    def _msg(self) -> str:
        return format_reject_message("0.6.0", "0.7.0", "incompatible version series")

    def test_leads_with_the_abort_and_names_the_subject(self):
        msg = self._msg()
        assert msg.startswith("Resume aborted:")
        assert "sieval version is incompatible" in msg

    def test_labels_which_version_is_which(self):
        msg = self._msg()
        assert msg.index("persisted") < msg.index("0.6.0")
        assert msg.index("current") < msg.index("0.7.0")
        assert "meta.json" in msg

    def test_carries_the_reason_under_its_label(self):
        assert "reason: incompatible version series" in self._msg()

    def test_offers_both_recovery_paths(self):
        msg = self._msg()
        assert "Remove the result_dir and start fresh" in msg
        assert "Reinstall sieval matching the persisted version series" in msg

    def test_the_two_builders_do_not_share_a_recovery_path(self):
        # They abort for different reasons and the second option differs: a
        # version mismatch is fixed by reinstalling, an identity mismatch by
        # giving the task its own directory. Swapping them would send the
        # operator down a route that cannot work.
        version_msg = self._msg()
        identity_msg = format_identity_reject_message("a_task", "b_task")
        assert "Reinstall sieval" in version_msg
        assert "Reinstall sieval" not in identity_msg
        assert "its own result_dir" in identity_msg
        assert "its own result_dir" not in version_msg


class TestUnpinnableBuildsRejectOnEitherSide:
    """`local` and `dev` are independent markers; either alone must reject.

    The earlier run-side test used "0.6.1.dev3+gxyz", which carries *both*, so it
    could not tell an `or` in this guard from an `and` — a build tagged only
    local would have fallen through to the series check and resumed.
    """

    def test_local_only_on_the_run_side(self):
        assert (
            resume_version_verdict("0.6.1+local", "0.6.0").action is ResumeAction.REJECT
        )

    def test_dev_only_on_the_run_side(self):
        assert (
            resume_version_verdict("0.6.1.dev3", "0.6.0").action is ResumeAction.REJECT
        )

    def test_local_only_on_the_current_side(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1+local").action is ResumeAction.REJECT
        )

    def test_dev_only_on_the_current_side(self):
        assert (
            resume_version_verdict("0.6.0", "0.6.1.dev3").action is ResumeAction.REJECT
        )


class TestRejectMessagesExplainThemselves:
    """The explanation is the load-bearing half of an operator message.

    Both builders spell out *why* the abort happened. Without it the operator
    sees two versions or two task names and a pair of options, with nothing to
    choose between them.
    """

    def test_identity_message_explains_the_path_match(self):
        msg = format_identity_reject_message("a_task", "b_task")
        assert "a finished run is matched by path alone" in msg
        assert "would hand back the persisted task's report" in msg

    def test_version_message_names_the_incompatibility(self):
        msg = format_reject_message("0.6.0", "0.7.0", "incompatible version series")
        assert "sieval version is incompatible with the persisted run" in msg

    def test_both_messages_end_on_an_actionable_choice(self):
        for msg in (
            format_identity_reject_message("a_task", "b_task"),
            format_reject_message("0.6.0", "0.7.0", "series"),
        ):
            assert "Either:" in msg
            assert "  1. " in msg and "  2. " in msg
