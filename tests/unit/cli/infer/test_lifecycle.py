"""Tests for launch_model and cleanup_model shared lifecycle.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sieval.cli.infer.lifecycle import cleanup_model, launch_model
from sieval.infer.config import InferHandle


def _make_handle(pid: str = "99999") -> InferHandle:
    return InferHandle(
        backend="sglang",
        handle_id=pid,
        endpoint="http://localhost:30000/v1",
        metadata={"role": "full", "health_url": "http://localhost:30000/health"},
    )


@pytest.fixture(autouse=True)
def _use_tmp_handle_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect HANDLE_DIR to a temp directory for all tests."""
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path)


class TestLaunchModel:
    @pytest.mark.anyio
    async def test_claim_deploy_save(self, tmp_path: Path) -> None:
        """Full lifecycle: claim → deploy → env → save handle."""
        handle = _make_handle()
        mock_deployer = AsyncMock()
        mock_deployer.deploy.return_value = [handle]

        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            handles, env = await launch_model(
                "test-model",
                [],
                backend="sglang",
                deployer=mock_deployer,
            )

        assert len(handles) == 1
        assert handles[0].handle_id == "99999"

        # Handle file should exist with running phase
        handle_path = tmp_path / "test-model.json"
        assert handle_path.exists()
        data = json.loads(handle_path.read_text())
        assert data["phase"] == "running"
        assert data["endpoint"] == "http://localhost:30000/v1"

    @pytest.mark.anyio
    async def test_pending_handle_created_before_deploy(self, tmp_path: Path) -> None:
        """Pending handle file exists before deploy() is called."""
        claimed = False

        async def _slow_deploy(*_args, **_kwargs):  # noqa: ANN002, ANN003
            nonlocal claimed
            # At this point, claim should have already happened
            pending_path = tmp_path / "test-model.json"
            claimed = pending_path.exists()
            return [_make_handle()]

        mock_deployer = AsyncMock()
        mock_deployer.deploy.side_effect = _slow_deploy

        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await launch_model(
                "test-model",
                [],
                backend="sglang",
                deployer=mock_deployer,
            )

        assert claimed, "Handle file should exist before deploy() runs"

    @pytest.mark.anyio
    async def test_duplicate_claim_raises(self, tmp_path: Path) -> None:
        """FileExistsError when model name is already claimed."""
        # Pre-create a handle file
        handle_path = tmp_path / "test-model.json"
        handle_path.write_text(json.dumps({"phase": "pending", "pid": 1}))

        mock_deployer = AsyncMock()
        with pytest.raises(FileExistsError):
            await launch_model(
                "test-model",
                [],
                backend="sglang",
                deployer=mock_deployer,
            )

        # deploy should not have been called
        mock_deployer.deploy.assert_not_called()

    @pytest.mark.anyio
    async def test_already_claimed_skips_claim(self, tmp_path: Path) -> None:
        """already_claimed=True skips claim, goes straight to deploy."""
        # Pre-create pending handle (simulates _try_claim in infer start)
        handle_path = tmp_path / "test-model.json"
        handle_path.write_text(
            json.dumps({"phase": "pending", "pid": 1, "backend": "sglang"})
        )

        handle = _make_handle()
        mock_deployer = AsyncMock()
        mock_deployer.deploy.return_value = [handle]

        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            handles, _ = await launch_model(
                "test-model",
                [],
                backend="sglang",
                deployer=mock_deployer,
                already_claimed=True,
            )

        assert len(handles) == 1
        # File updated to running
        data = json.loads(handle_path.read_text())
        assert data["phase"] == "running"

    @pytest.mark.anyio
    async def test_deploy_failure_cleans_up_handle(self, tmp_path: Path) -> None:
        """Handle file is removed when deploy raises."""
        mock_deployer = AsyncMock()
        mock_deployer.deploy.side_effect = RuntimeError("deploy failed")

        with pytest.raises(RuntimeError, match="deploy failed"):
            await launch_model(
                "test-model",
                [],
                backend="sglang",
                deployer=mock_deployer,
            )

        handle_path = tmp_path / "test-model.json"
        assert not handle_path.exists(), "Handle should be cleaned up on failure"


class TestCleanupModel:
    @pytest.mark.anyio
    async def test_stops_and_removes_handle(self, tmp_path: Path) -> None:
        """Cleanup stops services and removes handle file."""
        handle = _make_handle()
        handle_path = tmp_path / "test-model.json"
        handle_path.write_text(json.dumps({"phase": "running"}))

        mock_deployer = AsyncMock()
        await cleanup_model("test-model", [handle], deployer=mock_deployer)

        mock_deployer.delete.assert_awaited_once_with(handle)
        assert not handle_path.exists()

    @pytest.mark.anyio
    async def test_removes_handle_even_if_stop_fails(self, tmp_path: Path) -> None:
        """Handle file removed even when deployer.delete raises."""
        handle = _make_handle()
        handle_path = tmp_path / "test-model.json"
        handle_path.write_text(json.dumps({"phase": "running"}))

        mock_deployer = AsyncMock()
        mock_deployer.delete.side_effect = RuntimeError("stop failed")

        # Should not raise
        await cleanup_model("test-model", [handle], deployer=mock_deployer)
        assert not handle_path.exists()

    @pytest.mark.anyio
    async def test_no_handle_file_is_fine(self, tmp_path: Path) -> None:
        """Cleanup works even when handle file doesn't exist."""
        handle = _make_handle()
        mock_deployer = AsyncMock()

        # Should not raise
        await cleanup_model("nonexistent", [handle], deployer=mock_deployer)
        mock_deployer.delete.assert_awaited_once()
