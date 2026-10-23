"""Tests for sieval.core.runners.resume_gate: the result schema version gate."""

from pathlib import Path

import orjson
import pytest

from sieval.core.runners.resume_gate import (
    ResumeAction,
    ResumeVersionError,
    format_reject_message,
    resume_schema_verdict,
    schema_from_version,
)
from sieval.core.runners.runner import gate_resume_version
from sieval.core.tasks.run_meta import RESULT_SCHEMA_VERSION

_UNSET = object()


def _write_meta(root: Path, version: str, schema: object = _UNSET) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta: dict = {"version": version, "deterministic": False}
    if schema is not _UNSET:
        meta["result_schema_version"] = schema
    (root / "meta.json").write_bytes(orjson.dumps(meta))


def test_result_schema_version_is_1():
    assert RESULT_SCHEMA_VERSION == 1


class TestRecordedSchemaDecides:
    def test_same_schema_same_version_is_exact(self):
        assert resume_schema_verdict(1, "0.11.0", "0.11.0").action is ResumeAction.EXACT

    def test_same_schema_across_minor_releases_is_compatible(self):
        verdict = resume_schema_verdict(1, "0.11.0", "0.12.0")
        assert verdict.action is ResumeAction.COMPATIBLE
        assert verdict.reason == ""

    def test_same_schema_across_major_releases_is_compatible(self):
        assert (
            resume_schema_verdict(1, "1.0.0", "2.3.0").action is ResumeAction.COMPATIBLE
        )

    def test_a_local_build_resumes_another_local_build(self):
        verdict = resume_schema_verdict(1, "0.10.2+gc18745b", "0.11.0+g1234567")
        assert verdict.action is ResumeAction.COMPATIBLE

    def test_a_dev_build_resumes_a_release(self):
        assert (
            resume_schema_verdict(1, "0.11.0", "0.11.1.dev3+gxyz").action
            is ResumeAction.COMPATIBLE
        )

    def test_a_recorded_schema_ignores_the_version_string(self):
        # The version is provenance once the schema is recorded.
        for v_run in ("not-a-version", "0.0.0", "0.6.0"):
            assert (
                resume_schema_verdict(1, v_run, "0.11.0").action
                is ResumeAction.COMPATIBLE
            )

    def test_a_schema_mismatch_is_rejected(self):
        verdict = resume_schema_verdict(1, "0.11.0", "0.11.0", current_schema=2)
        assert verdict.action is ResumeAction.REJECT
        assert verdict.reason == "result schema version 1 differs from 2"

    def test_a_newer_recorded_schema_is_rejected(self):
        verdict = resume_schema_verdict(2, "0.12.0", "0.11.0")
        assert verdict.action is ResumeAction.REJECT
        assert verdict.reason == "result schema version 2 differs from 1"

    @pytest.mark.parametrize("bad", ["1", 1.0, True, [1], {}])
    def test_a_non_integer_schema_is_rejected(self, bad):
        verdict = resume_schema_verdict(bad, "0.11.0", "0.11.0")
        assert verdict.action is ResumeAction.REJECT
        assert "is not an integer" in verdict.reason


class TestMissingSchemaMapsFromVersion:
    @pytest.mark.parametrize(
        "version",
        ["0.10.0", "0.10.1", "0.10.2", "0.10.2+gc18745b", "0.10.3.dev4+ga9adfbe"],
    )
    def test_0_10_and_later_is_schema_1(self, version):
        assert schema_from_version(version) == 1
        assert (
            resume_schema_verdict(None, version, "0.11.0").action
            is ResumeAction.COMPATIBLE
        )

    def test_0_10_resumes_under_a_local_0_11_build(self):
        assert (
            resume_schema_verdict(None, "0.10.2+gc18745b", "0.11.0+gabc").action
            is ResumeAction.COMPATIBLE
        )

    def test_same_version_without_schema_is_exact(self):
        assert resume_schema_verdict(None, "0.10.2", "0.10.2").action is (
            ResumeAction.EXACT
        )

    @pytest.mark.parametrize(
        "version", ["0.9.0", "0.9.9", "0.10.0rc1", "0.10.0.dev1", "0.6.0"]
    )
    def test_before_0_10_is_rejected(self, version):
        assert schema_from_version(version) is None
        verdict = resume_schema_verdict(None, version, "0.11.0")
        assert verdict.action is ResumeAction.REJECT
        assert version in verdict.reason
        assert "no result_schema_version" in verdict.reason

    @pytest.mark.parametrize("version", ["not-a-version", "0.0.0"])
    def test_unparseable_or_unknown_is_rejected(self, version):
        assert schema_from_version(version) is None
        assert (
            resume_schema_verdict(None, version, "0.11.0").action is ResumeAction.REJECT
        )

    def test_even_an_identical_pre_0_10_version_is_rejected(self):
        # The schema decides first: 0.9 directories are not schema 1.
        assert (
            resume_schema_verdict(None, "0.9.0", "0.9.0").action is ResumeAction.REJECT
        )

    def test_a_mapped_schema_still_has_to_match(self):
        verdict = resume_schema_verdict(None, "0.10.2", "0.12.0", current_schema=2)
        assert verdict.action is ResumeAction.REJECT
        assert verdict.reason == "result schema version 1 differs from 2"


class TestFormatRejectMessage:
    def _msg(self) -> str:
        return format_reject_message(
            2, "0.12.0", "0.11.0", "result schema version 2 differs from 1"
        )

    def test_leads_with_the_abort(self):
        assert self._msg().startswith("Resume aborted:")

    def test_names_schema_versions_and_sieval_versions_per_side(self):
        persisted, current = self._msg().splitlines()[1:3]
        assert "meta.json" in persisted
        assert "result_schema_version 2" in persisted
        assert "sieval 0.12.0" in persisted
        assert current.lstrip().startswith("current:")
        assert "result_schema_version 1" in current
        assert "sieval 0.11.0" in current

    def test_an_absent_schema_says_so(self):
        msg = format_reject_message(None, "0.9.0", "0.11.0", "reason")
        assert "result_schema_version absent, sieval 0.9.0" in msg

    def test_carries_the_reason_and_both_recovery_paths(self):
        msg = self._msg()
        assert "reason: result schema version 2 differs from 1" in msg
        assert "Either:" in msg
        assert "Remove the result_dir and start fresh" in msg
        assert "result schema version" in msg.splitlines()[-1]


def test_resume_version_error_is_runtimeerror():
    assert issubclass(ResumeVersionError, RuntimeError)


class TestGateResumeVersion:
    def test_exact_passes(self, tmp_path):
        _write_meta(tmp_path, "0.11.0", 1)
        gate_resume_version(tmp_path, "0.11.0")  # no raise

    def test_a_0_10_directory_passes_under_0_11(self, tmp_path):
        _write_meta(tmp_path, "0.10.2")
        gate_resume_version(tmp_path, "0.11.0")  # no raise

    def test_a_local_build_passes_across_builds(self, tmp_path):
        _write_meta(tmp_path, "0.10.2+gc18745b")
        gate_resume_version(tmp_path, "0.11.0+g1234567")  # no raise

    def test_a_0_9_directory_raises(self, tmp_path):
        _write_meta(tmp_path, "0.9.0")
        with pytest.raises(ResumeVersionError, match="0.9.0"):
            gate_resume_version(tmp_path, "0.11.0")

    def test_a_schema_mismatch_raises(self, tmp_path):
        _write_meta(tmp_path, "0.12.0", 2)
        with pytest.raises(
            ResumeVersionError, match="result schema version 2 differs from 1"
        ):
            gate_resume_version(tmp_path, "0.11.0")

    def test_missing_meta_raises(self, tmp_path):
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.11.0")

    def test_unreadable_meta_raises(self, tmp_path):
        (tmp_path / "meta.json").write_bytes(b"not json{")
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.11.0")

    def test_meta_without_version_key_raises(self, tmp_path):
        (tmp_path / "meta.json").write_bytes(orjson.dumps({"result_schema_version": 1}))
        with pytest.raises(ResumeVersionError):
            gate_resume_version(tmp_path, "0.11.0")
