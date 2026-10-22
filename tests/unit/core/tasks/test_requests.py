"""Tests for a task's ``requests.jsonl`` sink and its summary.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from pathlib import Path

import pytest

from sieval.core.models import ChatModel
from sieval.core.tasks import requests as requests_module
from sieval.core.tasks.requests import REQUESTS_FILE, TaskRequestLog
from sieval.core.types import JSONValue


def _line(**overrides: JSONValue) -> dict[str, JSONValue]:
    line: dict[str, JSONValue] = {
        "run": "run-1",
        "phase": "infer",
        "model": {"id": "cand"},
        "attempt": 1,
        "attempts": 1,
        "outcome": "2xx",
        "latency_s": 0.5,
        "queue_s": 0.1,
        "connect_s": 0.01,
        "retry_after_s": None,
        "in_flight": {"task": {"borrowed": 2, "total": 4}},
    }
    line.update(overrides)
    return line


def _log(tmp_path: Path, **kwargs: int) -> TaskRequestLog:
    return TaskRequestLog(
        tmp_path,
        subject=None,
        judges={},
        max_bytes=kwargs.get("max_bytes", 1 << 20),
        max_files=kwargs.get("max_files", 2),
    )


class TestRoles:
    def test_subject_and_judges_are_told_apart_by_call_key(
        self, tmp_path: Path
    ) -> None:
        subject = ChatModel(api_key="k", model="cand")
        judge = ChatModel(api_key="k", model="judge")
        log = TaskRequestLog(
            tmp_path,
            subject=subject,
            judges={"grader": judge},
            max_bytes=1,
            max_files=0,
        )
        assert log.role_of(subject.meta()) == "subject"
        assert log.role_of(judge.meta()) == "grader"
        # A judge whose defaults the task derived still resolves by name.
        assert log.role_of(judge.with_args(temperature=0.3).meta()) == "grader"
        assert log.role_of(ChatModel(api_key="k", model="other").meta()) == "unknown"

    def test_a_judge_configured_as_the_subject_counts_as_the_subject(
        self, tmp_path: Path
    ) -> None:
        subject = ChatModel(api_key="k", model="same")
        log = TaskRequestLog(
            tmp_path,
            subject=subject,
            judges={"grader": ChatModel(api_key="k", model="same")},
            max_bytes=1,
            max_files=0,
        )
        assert log.role_of(subject.meta()) == "subject"


class TestSummary:
    def test_counts_outcomes_latency_waits_and_limiter_peaks(
        self, tmp_path: Path
    ) -> None:
        log = _log(tmp_path)
        log.record(
            _line(
                attempts=3,
                outcome="429",
                retry_after_s=1.5,
                in_flight={"task": {"borrowed": 4, "total": 4}},
            )
        )
        log.record(_line(attempt=2, attempts=3, outcome="5xx", retry_after_s=2.0))
        log.record(_line(attempt=3, attempts=3, latency_s=2.5, queue_s=None))
        log.record(_line(phase="judge", model={"id": "j"}, outcome="timeout_read"))

        summary = log.summary()
        assert summary is not None
        assert summary["run"] == "run-1"
        by_phase = summary["by_phase"]
        cand = by_phase["infer"]["cand"]
        assert cand["calls"] == 1
        assert cand["attempts"] == 3
        assert cand["retried_calls"] == 1
        assert cand["max_attempts"] == 3
        assert cand["outcomes"] == {"2xx": 1, "429": 1, "5xx": 1}
        assert cand["retry_wait_s_total"] == 3.5
        assert cand["in_flight"] == {"task": {"peak": 4, "limit": 4}}
        assert cand["latency"]["count"] == 3
        assert cand["latency"]["max_s"] == 2.5
        # Only first attempts carry a queue wait; every sent attempt a connect.
        assert cand["queue"]["count"] == 2
        assert cand["connect"]["count"] == 3
        judge = by_phase["judge"]["j"]
        assert judge["outcomes"] == {"timeout_read": 1}

        counts = log.counts()
        assert counts is not None
        assert counts["infer"]["cand"] == {
            "calls": 1,
            "attempts": 3,
            "outcomes": {"2xx": 1, "429": 1, "5xx": 1},
            "in_flight": {"task": {"peak": 4, "limit": 4}},
        }

    def test_no_request_no_summary(self, tmp_path: Path) -> None:
        log = _log(tmp_path)
        assert log.summary() is None
        assert log.counts() is None


class TestPersistence:
    @pytest.mark.anyio
    async def test_flush_appends_and_a_second_run_appends_again(
        self, tmp_path: Path
    ) -> None:
        first = _log(tmp_path)
        first.record(_line(run="run-1"))
        await first.flush()
        second = _log(tmp_path)
        second.record(_line(run="run-2"))
        second.record(_line(run="run-2", attempt=2))
        await second.flush()
        await second.flush()  # nothing buffered: nothing written

        lines = [
            json.loads(raw)
            for raw in (tmp_path / REQUESTS_FILE).read_text().splitlines()
        ]
        assert [line["run"] for line in lines] == ["run-1", "run-2", "run-2"]

    @pytest.mark.anyio
    async def test_maybe_flush_waits_for_enough_lines_or_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests_module, "FLUSH_LINES", 2)
        clock = [100.0]
        monkeypatch.setattr(requests_module.time, "monotonic", lambda: clock[0])
        log = _log(tmp_path)
        path = tmp_path / REQUESTS_FILE

        log.record(_line())
        await log.maybe_flush()
        assert not path.exists()

        log.record(_line())
        await log.maybe_flush()
        assert len(path.read_text().splitlines()) == 2

        log.record(_line())
        await log.maybe_flush()
        assert len(path.read_text().splitlines()) == 2
        clock[0] += requests_module.FLUSH_INTERVAL_S
        await log.maybe_flush()
        assert len(path.read_text().splitlines()) == 3

    @pytest.mark.anyio
    async def test_rotation_keeps_a_bounded_number_of_files(
        self, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, max_bytes=1, max_files=2)
        for index in range(4):
            log.record(_line(attempt=index + 1))
            await log.flush()

        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["requests.3.jsonl", "requests.4.jsonl"]
        assert json.loads((tmp_path / "requests.4.jsonl").read_text())["attempt"] == 4
        # The summary is folded in as lines arrive: rotation loses none.
        summary = log.summary()
        assert summary is not None
        assert summary["by_phase"]["infer"]["cand"]["attempts"] == 4

    @pytest.mark.anyio
    async def test_no_rotated_files_means_the_full_file_is_dropped(
        self, tmp_path: Path
    ) -> None:
        log = _log(tmp_path, max_bytes=1, max_files=0)
        log.record(_line())
        await log.flush()
        assert list(tmp_path.iterdir()) == []

    def test_bounds_are_validated(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            _log(tmp_path, max_bytes=0)
        with pytest.raises(ValueError, match="max_files"):
            _log(tmp_path, max_files=-1)
