"""
Pattern: every model request attempt recorded beside a task's shards.

A YAML run of a candidate and an LLM judge -- real clients, an in-process
wire -- writes one ``requests.jsonl`` line per attempt, a per-phase summary in
``profile.json`` and its counts in ``report.json``. Recording changes no score
and no identity, carries no credential, and a resume appends.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from sieval.cli.leaderboard.session import arun_session
from sieval.core.tasks.requests import REQUESTS_FILE
from tests.integration.test_api_key_env import (
    CANDIDATE_HOST,
    SAMPLES,
    _Wire,
    _write_config,
)
from tests.integration.test_api_key_env import wire as wire  # noqa: F401 -- fixture


def _rate_limit_once() -> Callable[[httpx.Request], httpx.Response | None]:
    """A 429 for each candidate question's first request, then the host's answer."""

    limited: set[str] = set()

    def before(request: httpx.Request) -> httpx.Response | None:
        if request.url.host != CANDIDATE_HOST:
            return None
        question = json.loads(request.content)["messages"][-1]["content"]
        if question in limited:
            return None
        limited.add(question)
        return httpx.Response(
            429,
            json={"error": {"message": "slow down"}},
            headers={"retry-after-ms": "1", "x-request-id": "req-429"},
        )

    return before


def _lines(task_dir: Path) -> list[dict]:
    return [
        json.loads(raw) for raw in (task_dir / REQUESTS_FILE).read_text().splitlines()
    ]


def _without_requests_option(config: Path, record: bool) -> Path:
    text = config.read_text()
    text = text.replace(
        "      dump_progress: false\n",
        f"      dump_progress: false\n      record_requests: {str(record).lower()}\n",
    )
    config.write_text(text)
    return config


class TestRequestRecords:
    @pytest.mark.anyio
    async def test_attempts_are_recorded_and_retries_are_visible(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,  # noqa: F811
    ) -> None:
        wire.before = _rate_limit_once()
        monkeypatch.setenv("CANDIDATE_KEY", "SENTINEL-CANDIDATE")
        monkeypatch.setenv("JUDGE_KEY", "SENTINEL-JUDGE")
        report = await arun_session(_write_config(tmp_path))
        task_dir = tmp_path / "results" / "judged"

        assert report["judged"]["accuracy"] == 1.0
        lines = _lines(task_dir)
        infer = [line for line in lines if line["phase"] == "infer"]
        judge = [line for line in lines if line["phase"] == "judge"]
        # One 429 then a 200 per sample: two attempts of one call each.
        assert len(infer) == 2 * len(SAMPLES)
        for sample in range(len(SAMPLES)):
            attempts = sorted(
                (line for line in infer if line["sample_id"] == sample),
                key=lambda line: line["attempt"],
            )
            assert [a["status"] for a in attempts] == [429, 200]
            assert [a["attempts"] for a in attempts] == [2, 2]
            assert attempts[0]["request_id"] == "req-429"
            assert attempts[0]["retry_after_s"] > 0
            assert attempts[0]["role"] == "subject"
        assert len(judge) == len(SAMPLES)
        assert {line["role"] for line in judge} == {"grader"}
        assert {line["stage"] for line in judge} == {"feedback"}
        meta = json.loads((task_dir / "meta.json").read_text())
        assert {line["run"] for line in lines} == {meta["runs"][-1]["id"]}

        profile = json.loads((task_dir / "profile.json").read_text())
        candidate = profile["requests"]["by_phase"]["infer"]["org/candidate"]
        assert candidate["calls"] == len(SAMPLES)
        assert candidate["attempts"] == 2 * len(SAMPLES)
        assert candidate["outcomes"] == {"2xx": 2, "429": 2}
        counts = json.loads((task_dir / "report.json").read_text())["sieval"]
        assert counts["requests"]["infer"]["org/candidate"]["attempts"] == 4
        assert counts["requests"]["judge"]["gemini-judge"]["calls"] == 2

        record = (task_dir / REQUESTS_FILE).read_text()
        assert "SENTINEL" not in record
        assert "authorization" not in record.lower()

    @pytest.mark.anyio
    async def test_recording_changes_no_score_and_no_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,  # noqa: F811
    ) -> None:
        monkeypatch.setenv("CANDIDATE_KEY", "k1")
        monkeypatch.setenv("JUDGE_KEY", "k2")
        reports, identities = [], []
        for record in (True, False):
            run_dir = tmp_path / str(record)
            run_dir.mkdir()
            config = _without_requests_option(_write_config(run_dir), record)
            reports.append(await arun_session(config))
            task_dir = run_dir / "results" / "judged"
            identities.append(
                json.loads((task_dir / "meta.json").read_text())["identity"]
            )
            assert (task_dir / REQUESTS_FILE).exists() is record
            saved = json.loads((task_dir / "report.json").read_text())
            assert ("requests" in saved["sieval"]) is record

        on, off = reports
        assert on["judged"]["accuracy"] == off["judged"]["accuracy"]
        assert identities[0] == identities[1]

    @pytest.mark.anyio
    async def test_a_resume_appends_and_a_fast_resume_adds_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,  # noqa: F811
    ) -> None:
        monkeypatch.setenv("CANDIDATE_KEY", "k1")
        monkeypatch.setenv("JUDGE_KEY", "k2")
        config = _write_config(tmp_path)
        await arun_session(config)
        task_dir = tmp_path / "results" / "judged"
        first = (task_dir / REQUESTS_FILE).read_text()

        await arun_session(config, resume=True)

        assert (task_dir / REQUESTS_FILE).read_text() == first
