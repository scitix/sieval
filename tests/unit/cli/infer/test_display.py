"""Tests for infer display_status, parse_phase, and probe_and_sync logic.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sieval.cli.infer.lifecycle import display_status, parse_phase, probe_and_sync
from sieval.infer.config import InferCondition, InferHandle, InferPhase


class TestDisplayStatus:
    def test_pending(self) -> None:
        assert (
            display_status(
                InferPhase.PENDING,
                {"ready": InferCondition(status=False, reason="deploying")},
            )
            == "Pending"
        )

    def test_running_ready(self) -> None:
        assert (
            display_status(
                InferPhase.RUNNING,
                {"ready": InferCondition(status=True)},
            )
            == "Ready"
        )

    def test_running_health_check_failed(self) -> None:
        assert (
            display_status(
                InferPhase.RUNNING,
                {"ready": InferCondition(status=False, reason="health_check_failed")},
            )
            == "NotReady (health_check_failed)"
        )

    def test_running_no_health_url(self) -> None:
        assert (
            display_status(
                InferPhase.RUNNING,
                {"ready": InferCondition(status=False, reason="no_health_url")},
            )
            == "Running (no health check)"
        )

    def test_stopped(self) -> None:
        assert (
            display_status(
                InferPhase.STOPPED,
                {"ready": InferCondition(status=False, reason="process_exited")},
            )
            == "Stopped (process_exited)"
        )

    def test_failed_with_reason(self) -> None:
        assert (
            display_status(
                InferPhase.FAILED,
                {"ready": InferCondition(status=False, reason="deploy_error")},
            )
            == "Failed (deploy_error)"
        )

    def test_stopped_no_reason(self) -> None:
        assert (
            display_status(
                InferPhase.STOPPED,
                {"ready": InferCondition(status=False)},
            )
            == "Stopped"
        )

    def test_running_no_ready_condition(self) -> None:
        """Empty conditions dict → NotReady (unknown)."""
        assert display_status(InferPhase.RUNNING, {}) == "NotReady (unknown)"

    def test_failed_no_ready_condition(self) -> None:
        """Empty conditions dict + terminal phase → bare label."""
        assert display_status(InferPhase.FAILED, {}) == "Failed"

    def test_stopped_no_ready_condition(self) -> None:
        assert display_status(InferPhase.STOPPED, {}) == "Stopped"

    def test_pending_empty_conditions(self) -> None:
        assert display_status(InferPhase.PENDING, {}) == "Pending"

    def test_stopping(self) -> None:
        assert display_status(InferPhase.STOPPING, {}) == "Stopping"

    def test_stopping_with_conditions(self) -> None:
        assert (
            display_status(
                InferPhase.STOPPING,
                {"ready": InferCondition(status=False, reason="shutting_down")},
            )
            == "Stopping"
        )


class TestParsePhase:
    def test_current_format(self) -> None:
        assert parse_phase({"phase": "running"}) == InferPhase.RUNNING

    def test_all_phases(self) -> None:
        for p in InferPhase:
            assert parse_phase({"phase": p.value}) == p

    def test_legacy_starting(self) -> None:
        """Legacy handle with 'status: starting' maps to PENDING."""
        assert parse_phase({"status": "starting"}) == InferPhase.PENDING

    def test_legacy_running(self) -> None:
        """Legacy handle without 'status: starting' maps to RUNNING."""
        assert parse_phase({"status": "running"}) == InferPhase.RUNNING
        assert parse_phase({}) == InferPhase.RUNNING

    def test_unknown_phase_value(self) -> None:
        """Unknown phase string falls back to legacy logic."""
        assert parse_phase({"phase": "bogus"}) == InferPhase.RUNNING

    def test_empty_phase_string(self) -> None:
        assert parse_phase({"phase": ""}) == InferPhase.RUNNING


def _make_handle() -> InferHandle:
    return InferHandle(
        backend="sglang",
        handle_id="12345",
        endpoint="http://localhost:30000/v1",
        metadata={"role": "full", "health_url": "http://localhost:30000/health"},
    )


class TestProbeAndSync:
    """Tests for probe_and_sync write-back logic."""

    @pytest.mark.anyio
    async def test_phase_change_writes_back(self, tmp_path: Path) -> None:
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        handle_path.write_text(
            json.dumps(
                {
                    "phase": "running",
                    "conditions": {
                        "ready": {"status": False, "reason": "health_check_failed"},
                    },
                }
            )
        )

        new_cond = InferCondition(status=False, reason="process_exited")
        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.STOPPED, {"ready": new_cond}),
        ):
            phase, conditions = await probe_and_sync(handle, handle_path)

        assert phase == InferPhase.STOPPED
        saved = json.loads(handle_path.read_text())
        assert saved["phase"] == "stopped"
        assert saved["conditions"]["ready"]["reason"] == "process_exited"

    @pytest.mark.anyio
    async def test_condition_change_without_phase_change_writes_back(
        self, tmp_path: Path
    ) -> None:
        """Condition flips (NotReady → Ready) while phase stays RUNNING."""
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        handle_path.write_text(
            json.dumps(
                {
                    "phase": "running",
                    "conditions": {
                        "ready": {"status": False, "reason": "health_check_failed"},
                    },
                }
            )
        )

        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.RUNNING, {"ready": InferCondition(status=True)}),
        ):
            phase, conditions = await probe_and_sync(handle, handle_path)

        assert phase == InferPhase.RUNNING
        assert conditions["ready"].status is True
        saved = json.loads(handle_path.read_text())
        assert saved["conditions"]["ready"]["status"] is True

    @pytest.mark.anyio
    async def test_no_change_skips_write(self, tmp_path: Path) -> None:
        """When phase and conditions are unchanged, file is not rewritten."""
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        original = {
            "phase": "running",
            "conditions": {"ready": {"status": True, "reason": ""}},
        }
        handle_path.write_text(json.dumps(original))
        mtime_before = handle_path.stat().st_mtime_ns

        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.RUNNING, {"ready": InferCondition(status=True)}),
        ):
            await probe_and_sync(handle, handle_path)

        assert handle_path.stat().st_mtime_ns == mtime_before

    @pytest.mark.anyio
    async def test_stopping_not_downgraded_to_running(self, tmp_path: Path) -> None:
        """STOPPING must not regress to RUNNING even if deployer reports RUNNING."""
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        handle_path.write_text(
            json.dumps(
                {
                    "phase": "stopping",
                    "conditions": {
                        "ready": {"status": True, "reason": ""},
                    },
                }
            )
        )

        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.RUNNING, {"ready": InferCondition(status=True)}),
        ):
            phase, _ = await probe_and_sync(handle, handle_path)

        # Returned phase reflects what deployer reported
        assert phase == InferPhase.RUNNING
        # But the file preserves STOPPING
        saved = json.loads(handle_path.read_text())
        assert saved["phase"] == "stopping"

    @pytest.mark.anyio
    async def test_stopping_allows_terminal_phase(self, tmp_path: Path) -> None:
        """STOPPING should transition to STOPPED when deployer confirms exit."""
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        handle_path.write_text(
            json.dumps(
                {
                    "phase": "stopping",
                    "conditions": {
                        "ready": {"status": True, "reason": ""},
                    },
                }
            )
        )

        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(
                InferPhase.STOPPED,
                {"ready": InferCondition(status=False, reason="process_exited")},
            ),
        ):
            phase, _ = await probe_and_sync(handle, handle_path)

        assert phase == InferPhase.STOPPED
        saved = json.loads(handle_path.read_text())
        assert saved["phase"] == "stopped"

    @pytest.mark.anyio
    async def test_legacy_file_upgraded(self, tmp_path: Path) -> None:
        """Legacy file without 'phase' key gets upgraded on first probe."""
        handle = _make_handle()
        handle_path = tmp_path / "model.json"
        handle_path.write_text(json.dumps({"status": "running", "backend": "sglang"}))

        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.RUNNING, {"ready": InferCondition(status=True)}),
        ):
            await probe_and_sync(handle, handle_path)

        saved = json.loads(handle_path.read_text())
        assert saved["phase"] == "running"
        assert "conditions" in saved
