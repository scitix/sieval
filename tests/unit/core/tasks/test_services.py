"""Tests for the service declarations a task makes, and their grader provenance.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json

import pytest

from sieval.core.tasks.run_meta import grader_identity
from sieval.core.tasks.services import (
    CODE_EVAL,
    SERVICE_ENDPOINT_ENV,
    VENDORED_SERVICES_ENV,
    check_service_names,
    service_identities,
    vendored_services,
)

_URL = "http://127.0.0.1:4000/evaluations"


def _started(**entry: object) -> dict[str, str]:
    record = {"commit": "abc1234", "digest": "sha256:d", "endpoint": _URL, **entry}
    return {
        VENDORED_SERVICES_ENV: json.dumps({CODE_EVAL: record}),
        SERVICE_ENDPOINT_ENV[CODE_EVAL]: _URL,
    }


class TestServiceIdentities:
    def test_a_service_sieval_started_is_vendored(self):
        [entry] = service_identities([CODE_EVAL], _started())
        assert entry == {
            "name": CODE_EVAL,
            "source": "vendored",
            "commit": "abc1234",
            "digest": "sha256:d",
        }

    def test_a_preset_endpoint_is_external_and_its_url_is_not_recorded(self):
        env = {SERVICE_ENDPOINT_ENV[CODE_EVAL]: "http://sandbox.internal:11451/x"}
        [entry] = service_identities([CODE_EVAL], env)
        assert entry == {"name": CODE_EVAL, "source": "external"}
        assert "sandbox" not in json.dumps(entry)

    def test_a_record_for_another_endpoint_does_not_claim_this_one(self):
        """A stale record must not label a user-set endpoint as vendored."""
        env = _started()
        env[SERVICE_ENDPOINT_ENV[CODE_EVAL]] = "http://elsewhere/evaluations"
        [entry] = service_identities([CODE_EVAL], env)
        assert entry["source"] == "external"

    def test_the_runtime_is_carried_when_recorded(self):
        runtime = {"python": "3.12.1", "platform": "linux", "packages": {}}
        [entry] = service_identities([CODE_EVAL], _started(runtime=runtime))
        assert entry.get("runtime") == runtime
        [bare] = service_identities([CODE_EVAL], _started())
        assert "runtime" not in bare

    def test_a_missing_commit_is_none(self):
        [entry] = service_identities([CODE_EVAL], _started(commit=None))
        assert entry["source"] == "vendored"
        assert entry["commit"] is None

    @pytest.mark.parametrize("raw", ["", "  ", "not json", "[1]", '{"code-eval": 1}'])
    def test_malformed_records_read_as_nothing_started(self, raw):
        assert vendored_services({VENDORED_SERVICES_ENV: raw}) == {}

    def test_no_services_no_entries(self):
        assert service_identities([], _started()) == []


class TestDeclarations:
    def test_unknown_names_are_refused(self):
        with pytest.raises(ValueError, match=r"unknown service\(s\) \['sandbox'\]"):
            check_service_names({"sandbox", CODE_EVAL}, "Task 't'")

    def test_known_names_pass(self):
        check_service_names({CODE_EVAL}, "Task 't'")


class TestGraderIdentity:
    def test_a_task_without_services_digests_as_before(self):
        before = grader_identity({"m": "sha256:1"}, [])
        assert "services" not in before
        assert before == grader_identity({"m": "sha256:1"}, [], [])

    def test_vendored_and_external_digest_differently(self):
        vendored = service_identities([CODE_EVAL], _started())
        external = service_identities([CODE_EVAL], {})
        a = grader_identity({"m": "sha256:1"}, [], vendored)
        b = grader_identity({"m": "sha256:1"}, [], external)
        c = grader_identity({"m": "sha256:1"}, [])
        assert len({a["digest"], b["digest"], c["digest"]}) == 3
        assert a["services"] == vendored
        assert b["services"] == [{"name": CODE_EVAL, "source": "external"}]

    def test_a_changed_runtime_changes_the_digest(self):
        def digest_for(numpy: str) -> str:
            runtime = {
                "python": "3.12",
                "platform": "linux",
                "packages": {"numpy": numpy},
            }
            ids = service_identities([CODE_EVAL], _started(runtime=runtime))
            return grader_identity({}, [], ids)["digest"]

        assert digest_for("2.2.0") != digest_for("1.26.4")

    def test_a_changed_vendored_source_changes_the_digest(self):
        one = service_identities([CODE_EVAL], _started(digest="sha256:1"))
        two = service_identities([CODE_EVAL], _started(digest="sha256:2"))
        assert (
            grader_identity({}, [], one)["digest"]
            != grader_identity({}, [], two)["digest"]
        )
