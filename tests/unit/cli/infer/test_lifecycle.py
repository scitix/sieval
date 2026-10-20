"""Tests for the handle lifecycle: launch, cleanup, transition, logs.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
import typer

from sieval.cli.infer.lifecycle import (
    KEEP_LOGS_ENV,
    UNCHANGED,
    HandleHeldError,
    claim_handle,
    cleanup_model,
    handle_path_for,
    iter_handle_files,
    launch_model,
    load_env,
    load_handle,
    mark_stopping,
    probe_and_sync,
    prune_engine_logs,
    record_log_files,
    remove_run_handle_dir,
    resolve_keep_logs,
    run_handle_dir,
    sweep_stale_run_dirs,
    transition,
)
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferCondition, InferHandle, InferPhase


def _make_handle(pid: str = "99999") -> InferHandle:
    return InferHandle(
        backend="sglang",
        handle_id=pid,
        endpoint="http://localhost:30000/v1",
        metadata={"role": "full", "health_url": "http://localhost:30000/health"},
    )


@pytest.fixture(autouse=True)
def _use_tmp_handle_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect HANDLE_DIR and the engine log dir to temp directories."""
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path)
    monkeypatch.setattr(
        "sieval.cli.infer.lifecycle.ENGINE_LOG_DIR", tmp_path / "engine-logs"
    )


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


# ---------------------------------------------------------------------------
# Per-owner handles, transition(), log retention
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """PID of a process that has exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _increment(path: Path, rounds: int) -> None:
    """Read-modify-write a counter through transition(), *rounds* times."""

    def _bump(current: dict[str, object] | None) -> dict[str, object]:
        count = int(cast(int, (current or {}).get("count", 0)))
        time.sleep(0.001)  # widen the read-modify-write window
        return {"count": count + 1}

    for _ in range(rounds):
        transition(path, _bump)


def _increment_in_child(handle_dir: str, path: str, rounds: int) -> None:
    import sieval.cli.infer.lifecycle as lc

    lc.HANDLE_DIR = Path(handle_dir)
    _increment(Path(path), rounds)


class TestTransition:
    def test_concurrent_threads_lose_no_update(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        threads = [
            threading.Thread(target=_increment, args=(path, 10)) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert json.loads(path.read_text())["count"] == 80
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

    def test_concurrent_processes_lose_no_update(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_increment_in_child, args=(str(tmp_path), str(path), 10))
            for _ in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0
        assert json.loads(path.read_text())["count"] == 40

    def test_update_error_leaves_file_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running"}))

        def _boom(_current: dict[str, object] | None) -> dict[str, object]:
            raise ValueError("refused")

        with pytest.raises(ValueError, match="refused"):
            transition(path, _boom)
        assert json.loads(path.read_text()) == {"phase": "running"}

    def test_none_deletes_and_unchanged_keeps(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running"}))
        mtime = path.stat().st_mtime_ns
        assert transition(path, lambda _c: UNCHANGED) == {"phase": "running"}
        assert path.stat().st_mtime_ns == mtime
        assert transition(path, lambda _c: None) is None
        assert not path.exists()

    @pytest.mark.anyio
    async def test_concurrent_probe_and_sync_share_no_temp_file(
        self, tmp_path: Path
    ) -> None:
        """Two probes writing the same handle each use their own temp file."""
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running", "conditions": {}}))
        statuses = [
            (InferPhase.RUNNING, {"ready": InferCondition(status=True)}),
            (InferPhase.STOPPED, {"ready": InferCondition(False, "process_exited")}),
        ]
        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            side_effect=statuses * 5,
        ):
            async with anyio.create_task_group() as tg:
                for _ in range(10):
                    tg.start_soon(probe_and_sync, _make_handle(), path)
        saved = json.loads(path.read_text())
        assert saved["phase"] in {"running", "stopped"}
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

    @pytest.mark.anyio
    async def test_probe_does_not_recreate_a_removed_handle(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "m.json"
        with patch(
            "sieval.cli.infer.lifecycle.default_deployer.status",
            new_callable=AsyncMock,
            return_value=(InferPhase.STOPPED, {}),
        ):
            await probe_and_sync(_make_handle(), path)
        assert not path.exists()


class TestClaimHandle:
    def test_live_holder_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running", "engine_pid": os.getpid()}))
        with pytest.raises(HandleHeldError) as info:
            claim_handle(path, replace_stale=True)
        assert info.value.pid == os.getpid()
        assert info.value.phase == InferPhase.RUNNING

    def test_stale_handle_replaced_only_when_asked(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running", "engine_pid": _dead_pid()}))
        with pytest.raises(FileExistsError):
            claim_handle(path)
        claim_handle(path, backend="vllm", replace_stale=True)
        data = json.loads(path.read_text())
        assert data["phase"] == "pending"
        assert data["owner_pid"] == os.getpid()
        assert data["backend"] == "vllm"

    def test_corrupt_handle_replaced(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text("{not json")
        claim_handle(path, replace_stale=True)
        assert json.loads(path.read_text())["phase"] == "pending"


class TestOldHandleFormat:
    """Handles written before the rename carry `pid` and `handle_id`."""

    def test_running_handle_loads(self, tmp_path: Path) -> None:
        (tmp_path / "m.json").write_text(
            json.dumps(
                {
                    "phase": "running",
                    "backend": "sglang",
                    "endpoint": "http://localhost:30000/v1",
                    "handle_id": "4242",
                    "metadata": {"role": "full"},
                    "env": {"framework": "sglang==0.4"},
                }
            )
        )
        handle = load_handle("m")
        assert handle.handle_id == "4242"
        assert handle.endpoint == "http://localhost:30000/v1"
        env = load_env("m")
        assert env is not None
        assert env.framework == "sglang==0.4"

    def test_legacy_status_only_handle_loads(self, tmp_path: Path) -> None:
        (tmp_path / "m.json").write_text(
            json.dumps({"status": "running", "backend": "vllm", "handle_id": "7"})
        )
        assert load_handle("m").handle_id == "7"

    def test_pending_handle_held_by_old_pid_field(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "pending", "pid": os.getpid()}))
        with pytest.raises(HandleHeldError):
            claim_handle(path, replace_stale=True)

    def test_rewrite_uses_new_field_names(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"phase": "running", "handle_id": "9", "pid": 3}))
        mark_stopping(path)
        data = json.loads(path.read_text())
        assert data["phase"] == "stopping"
        assert data["engine_pid"] == "9"
        assert data["owner_pid"] == 3
        assert "handle_id" not in data
        assert "pid" not in data


class TestRunHandleDirs:
    @pytest.mark.anyio
    async def test_run_and_start_with_same_model_name_coexist(
        self, tmp_path: Path
    ) -> None:
        """A live `infer start` handle does not block a `sieval run` launch."""
        start_path = tmp_path / "m.json"
        start_data = {"phase": "running", "engine_pid": os.getpid()}
        start_path.write_text(json.dumps(start_data))

        run_path = run_handle_dir() / "m.json"
        mock_deployer = AsyncMock()
        mock_deployer.deploy.return_value = [_make_handle()]
        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await launch_model(
                "m",
                [],
                backend="sglang",
                deployer=mock_deployer,
                handle_path=run_path,
                log_dir=tmp_path / "results" / "logs" / "m",
            )

        assert run_path.parent == tmp_path / f"run-{os.getpid()}"
        assert json.loads(run_path.read_text())["engine_pid"] == 99999
        assert json.loads(start_path.read_text()) == start_data
        names = [name for name, _ in iter_handle_files()]
        assert names == ["m", f"run-{os.getpid()}/m"]

        await cleanup_model(
            "m", [_make_handle()], deployer=mock_deployer, handle_path=run_path
        )
        remove_run_handle_dir(run_path.parent)
        assert not run_path.parent.exists()
        assert start_path.exists()

    @pytest.mark.anyio
    async def test_leftover_handle_from_a_reused_pid_is_replaced(
        self, tmp_path: Path
    ) -> None:
        run_path = run_handle_dir() / "m.json"
        run_path.parent.mkdir()
        run_path.write_text(json.dumps({"phase": "running", "engine_pid": 1 << 22}))
        mock_deployer = AsyncMock()
        mock_deployer.deploy.return_value = [_make_handle()]
        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await launch_model(
                "m",
                [],
                backend="sglang",
                deployer=mock_deployer,
                handle_path=run_path,
                replace_stale=True,
            )
        assert json.loads(run_path.read_text())["engine_pid"] == 99999

    def test_run_dir_removal_keeps_an_earlier_processes_live_engine(
        self, tmp_path: Path
    ) -> None:
        """A leftover handle whose engine is alive outlives this run's cleanup."""
        run_dir = run_handle_dir()
        run_dir.mkdir()
        orphan = run_dir / "m.json"
        orphan.write_text(json.dumps({"phase": "running", "engine_pid": os.getpid()}))
        with pytest.raises(HandleHeldError):
            claim_handle(orphan, replace_stale=True)

        remove_run_handle_dir(run_dir)
        assert orphan.exists()

    def test_run_dir_removal_clears_temp_files_and_the_empty_dir(
        self, tmp_path: Path
    ) -> None:
        run_dir = run_handle_dir()
        run_dir.mkdir()
        (run_dir / ".m.json.abc.tmp").write_text("{}")
        remove_run_handle_dir(run_dir)
        assert not run_dir.exists()

    def test_sweep_removes_dead_owner_dirs_only(self, tmp_path: Path) -> None:
        dead = tmp_path / f"run-{_dead_pid()}"
        dead.mkdir()
        (dead / "m.json").write_text(
            json.dumps({"phase": "running", "engine_pid": _dead_pid()})
        )
        live = run_handle_dir()
        live.mkdir()
        (live / "m.json").write_text(json.dumps({"phase": "pending"}))

        assert sweep_stale_run_dirs() == [dead]
        assert not dead.exists()
        assert (live / "m.json").exists()

    def test_sweep_keeps_a_handle_whose_engine_survived(self, tmp_path: Path) -> None:
        """An orphaned engine stays visible so `infer stop` can still reach it."""
        dead = tmp_path / f"run-{_dead_pid()}"
        dead.mkdir()
        (dead / "a.json").write_text(
            json.dumps({"phase": "running", "engine_pid": os.getpid()})
        )
        (dead / "b.json").write_text(
            json.dumps({"phase": "running", "engine_pid": _dead_pid()})
        )
        assert sweep_stale_run_dirs() == []
        assert (dead / "a.json").exists()
        assert not (dead / "b.json").exists()

    def test_handle_names_cannot_escape_the_handle_dir(self) -> None:
        with pytest.raises(typer.BadParameter):
            handle_path_for("../outside")
        with pytest.raises(typer.BadParameter):
            handle_path_for("run-1/a/b")


class TestKeepLogs:
    @pytest.mark.parametrize(
        ("flag", "env", "expected"),
        [
            (None, None, 3),
            (None, "5", 5),
            (2, "5", 2),
            (0, "5", 0),
            (None, "-1", -1),
            (None, " ", 3),
        ],
    )
    def test_flag_then_env_then_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: int | None,
        env: str | None,
        expected: int,
    ) -> None:
        if env is None:
            monkeypatch.delenv(KEEP_LOGS_ENV, raising=False)
        else:
            monkeypatch.setenv(KEEP_LOGS_ENV, env)
        assert resolve_keep_logs(flag) == expected

    def test_bad_env_value_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KEEP_LOGS_ENV, "many")
        with pytest.raises(typer.BadParameter, match=KEEP_LOGS_ENV):
            resolve_keep_logs(None)


def _seed_logs(log_dir: Path, role: str, count: int, model: str = "m") -> list[Path]:
    """Create *count* logs for *model*'s *role*, oldest first."""
    model_dir = log_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = model_dir / f"{role}-20261020T0000{i:02d}Z-{i:08x}.log"
        path.write_text(str(i))
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
        paths.append(path)
    return paths


class TestEngineLogRetention:
    @pytest.fixture
    def log_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        log_dir = tmp_path / "logs"
        monkeypatch.setattr("sieval.cli.infer.lifecycle.ENGINE_LOG_DIR", log_dir)
        return log_dir

    async def _launch(self, keep: int, tmp_path: Path, model: str = "m") -> Path:
        mock_deployer = AsyncMock()

        async def _deploy(_cmds, *, log_files, **_kwargs):  # noqa: ANN001, ANN003
            log_file = log_files["full"]
            log_file.write_text("engine output")
            handle = _make_handle()
            handle.metadata["log_file"] = str(log_file)
            return [handle]

        mock_deployer.deploy.side_effect = _deploy
        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            handles, _ = await launch_model(
                model,
                [BackendCommand(cli_args=["x"], role="full")],
                backend="sglang",
                deployer=mock_deployer,
                keep_logs=keep,
            )
        data = json.loads((tmp_path / f"{model}.json").read_text())
        assert data["keep_logs"] == keep
        return Path(str(handles[0].metadata["log_file"]))

    @pytest.mark.anyio
    async def test_keep_three_counts_the_new_launch(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        old = _seed_logs(log_dir, "full", 5)
        other_role = _seed_logs(log_dir, "decode", 4)
        new = await self._launch(3, tmp_path)

        assert new.parent == log_dir / "m"
        assert re.fullmatch(r"full-\d{8}T\d{6}Z-[0-9a-f]{8}\.log", new.name)
        remaining = sorted(p for p in (log_dir / "m").glob("full-*.log"))
        assert remaining == sorted([old[3], old[4], new])
        assert all(p.exists() for p in other_role)

    @pytest.mark.anyio
    async def test_retention_counts_each_model_separately(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        """Launching model b never evicts model a's logs, whatever keep is."""
        a_logs = _seed_logs(log_dir, "full", 3, model="a")
        b_logs = _seed_logs(log_dir, "full", 3, model="b")
        new = await self._launch(1, tmp_path, model="b")

        assert new.parent == log_dir / "b"
        assert all(p.exists() for p in a_logs)
        assert not any(p.exists() for p in b_logs)

    @pytest.mark.anyio
    async def test_keep_zero_deletes_the_log_on_stop(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        old = _seed_logs(log_dir, "full", 2)
        new = await self._launch(0, tmp_path)
        assert not any(p.exists() for p in old)
        assert new.exists()

        await cleanup_model("m", [_make_handle()], deployer=AsyncMock())
        assert not new.exists()

    @pytest.mark.anyio
    async def test_negative_keeps_every_log(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        old = _seed_logs(log_dir, "full", 6)
        new = await self._launch(-1, tmp_path)
        await cleanup_model("m", [_make_handle()], deployer=AsyncMock())
        assert all(p.exists() for p in old)
        assert new.exists()

    def test_prune_skips_a_log_a_handle_still_names(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        old = _seed_logs(log_dir, "full", 3)
        (tmp_path / "live.json").write_text(
            json.dumps({"phase": "running", "metadata": {"log_file": str(old[0])}})
        )
        deleted = prune_engine_logs("m", ["full"], 1)
        assert sorted(deleted) == sorted(old[1:])
        assert old[0].exists()

    def test_prune_skips_the_log_of_a_launch_still_starting(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        """A pending handle names its logs before its engine writes metadata."""
        [starting] = _seed_logs(log_dir, "full", 1)
        claim_handle(tmp_path / "m.json")
        record_log_files(tmp_path / "m.json", [starting])

        assert prune_engine_logs("m", ["full"], 1) == []
        assert starting.exists()

    @pytest.mark.anyio
    async def test_concurrent_launch_of_the_same_model_keeps_a_starting_log(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        """Launch 1 is mid-deploy when launch 2 prunes with keep=1."""
        first_log: list[Path] = []
        second_started = anyio.Event()
        first_may_finish = anyio.Event()

        def _deployer(wait: bool) -> AsyncMock:
            async def _deploy(_cmds, *, log_files, **_kwargs):  # noqa: ANN001, ANN003
                log_file = log_files["full"]
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text("engine output")
                if wait:
                    first_log.append(log_file)
                    second_started.set()
                    await first_may_finish.wait()
                handle = _make_handle()
                handle.metadata["log_file"] = str(log_file)
                return [handle]

            dep = AsyncMock()
            dep.deploy.side_effect = _deploy
            return dep

        async def _launch(name: str, wait: bool) -> None:
            await launch_model(
                "m",
                [BackendCommand(cli_args=["x"], role="full")],
                backend="sglang",
                deployer=_deployer(wait),
                handle_path=tmp_path / f"{name}.json",
                keep_logs=1,
            )

        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_launch, "first", True)
                await second_started.wait()
                await _launch("second", False)
                assert first_log[0].exists()
                first_may_finish.set()

    def test_prune_reaches_old_flat_timestamp_names(self, log_dir: Path) -> None:
        log_dir.mkdir()
        legacy = []
        for i in range(3):
            path = log_dir / f"full-2026-01-01_00-00-0{i}.log"
            path.write_text("x")
            os.utime(path, (1_000_000 + i, 1_000_000 + i))
            legacy.append(path)
        unrelated = log_dir / "full-notes.log"
        unrelated.write_text("x")
        assert sorted(prune_engine_logs("m", ["full"], 1)) == legacy[:2]
        assert legacy[2].exists()
        assert unrelated.exists()

    @pytest.mark.anyio
    async def test_result_dir_logs_skip_pruning(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        old = _seed_logs(log_dir, "full", 4)
        mock_deployer = AsyncMock()
        mock_deployer.deploy.return_value = [_make_handle()]
        with patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await launch_model(
                "m",
                [BackendCommand(cli_args=["x"], role="full")],
                backend="sglang",
                deployer=mock_deployer,
                log_dir=tmp_path / "results" / "logs" / "m",
                keep_logs=0,
            )
        kwargs = mock_deployer.deploy.await_args.kwargs
        assert kwargs["log_files"] == {
            "full": tmp_path / "results" / "logs" / "m" / "full.log"
        }
        assert all(p.exists() for p in old)
        assert "keep_logs" not in json.loads((tmp_path / "m.json").read_text())
