"""
Unit tests for TaskProgress.

Focus on externally observable behavior: logging, progress dumps,
and tqdm interactions (via a fake pbar), rather than private fields.
"""

import io
import threading
from pathlib import Path
from unittest.mock import patch

import anyio
import orjson
import pytest
from loguru import logger

from sieval.core.tasks.progress import TaskProgress


# ===================================================================
# Helpers
# ===================================================================
def _capture_logs(fn) -> str:
    sink = io.StringIO()
    logger_id = logger.add(sink, format="{message}")
    try:
        fn()
    finally:
        logger.remove(logger_id)
    return sink.getvalue()


def _close(prog: TaskProgress) -> None:
    anyio.run(prog.aclose)


def _read_progress(root: Path) -> dict:
    return orjson.loads((root / "progress.json").read_bytes())


def make_progress(
    total: int = 10,
    show_progress: bool = True,
    log_interval: float = 9999.0,  # prevent time-based triggers
    log_pct_interval: float = 100.0,  # prevent pct-based triggers unless explicit
    root_dir: Path | None = None,
    dump_progress: bool = False,
    dump_interval: float = 0.0,  # dump immediately in tests
) -> TaskProgress:
    return TaskProgress(
        total=total,
        desc="test",
        show_progress=show_progress,
        log_interval=log_interval,
        log_pct_interval=log_pct_interval,
        root_dir=root_dir,
        dump_progress=dump_progress,
        dump_interval=dump_interval,
    )


class _FakePbar:
    def __init__(self):
        self.n = 0
        self.postfix: dict = {}
        self.updates: list[int] = []
        self.postfix_calls: list[tuple[dict, bool]] = []
        self.closed = False

    def update(self, n: int) -> None:
        self.n += n
        self.updates.append(n)

    def set_postfix(self, value, refresh: bool = False) -> None:
        val = dict(value)
        self.postfix = val
        self.postfix_calls.append((val, refresh))

    def close(self) -> None:
        self.closed = True


# ===================================================================
# TTY detection
# ===================================================================
class TestTTYDetection:
    def test_tty_and_non_tty_modes(self):
        with patch("sys.stderr.isatty", return_value=False):
            non_tty = make_progress(show_progress=True)

        output = _capture_logs(lambda: non_tty.init_state(completed_ids=[0]))
        assert "test started: 1/10 (10.0%)" in output
        _close(non_tty)

        fake_pbar = _FakePbar()
        with (
            patch("sys.stderr.isatty", return_value=True),
            patch(
                "sieval.core.tasks.progress.tqdm", return_value=fake_pbar
            ) as tqdm_cls,
        ):
            tty = make_progress(show_progress=True, total=5)

        output = _capture_logs(lambda: tty.init_state(completed_ids=[0]))
        assert output.strip() == ""
        tqdm_cls.assert_called_once()
        assert fake_pbar.updates == [1]
        _close(tty)
        assert fake_pbar.closed is True

        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm") as tqdm_cls,
        ):
            no_show = make_progress(show_progress=False)
        tqdm_cls.assert_not_called()
        _close(no_show)

        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm") as tqdm_cls,
        ):
            zero_total = make_progress(total=0)
        tqdm_cls.assert_not_called()
        _close(zero_total)


# ===================================================================
# init_state
# ===================================================================
class TestInitState:
    def test_init_populates_dump_fields(self, tmp_path):
        prog = make_progress(total=10, root_dir=tmp_path, dump_progress=True)
        prog.init_state(
            completed_ids=[0, 1, 2],
            failed_ids=[0],
            anomaly_ids=[0],
            anomaly_details={"missing_field": 1},
        )
        _close(prog)

        data = _read_progress(tmp_path)
        assert data["completed"] == 3
        assert data["failed"] == 1
        assert data["anomalies"] == 1
        assert data["anomaly_details"] == {"missing_field": 1}

    def test_init_logs_start(self):
        with patch("sys.stderr.isatty", return_value=False):
            prog = make_progress(
                total=10, show_progress=True, log_interval=0.0, log_pct_interval=0.0
            )

        output = _capture_logs(lambda: prog.init_state(completed_ids=[3, 4, 5]))
        assert "test started: 3/10 (30.0%)" in output
        _close(prog)


# ===================================================================
# update
# ===================================================================
class TestUpdate:
    def test_update_tracks_completed_failed_and_idempotent(self, tmp_path):
        prog = make_progress(total=10, root_dir=tmp_path, dump_progress=True)
        prog.update(sample_id=3, current_hydrated_count=1, failed=True)
        prog.update(sample_id=3, current_hydrated_count=1, failed=True)
        _close(prog)

        data = _read_progress(tmp_path)
        assert data["completed"] == 1
        assert data["failed"] == 1

    def test_update_tracks_and_accumulates_anomalies(self, tmp_path):
        prog = make_progress(total=10, root_dir=tmp_path, dump_progress=True)
        prog.update(
            sample_id=0, current_hydrated_count=1, anomalies={"truncated", "low_score"}
        )
        prog.update(sample_id=1, current_hydrated_count=2, anomalies={"truncated"})
        _close(prog)

        data = _read_progress(tmp_path)
        assert data["anomalies"] == 2
        assert data["anomaly_details"] == {"truncated": 2, "low_score": 1}


# ===================================================================
# tick (non-TTY logging)
# ===================================================================
class TestTick:
    def test_tick_no_log_when_disabled_or_total_zero(self):
        prog = make_progress(show_progress=False)

        def _run_disabled():
            prog.init_state(completed_ids=[])
            prog.update(sample_id=0, current_hydrated_count=1)
            _close(prog)

        disabled_output = _capture_logs(_run_disabled)
        assert disabled_output.strip() == ""

        with patch("sys.stderr.isatty", return_value=False):
            zero_total = make_progress(total=0, show_progress=True)

        zero_output = _capture_logs(lambda: (zero_total.tick(0), _close(zero_total)))
        assert zero_output.strip() == ""

    def test_tick_logs_on_finish(self):
        with patch("sys.stderr.isatty", return_value=False):
            prog = make_progress(
                total=2,
                show_progress=True,
                log_interval=9999.0,
                log_pct_interval=9999.0,
            )

        def _run():
            prog.init_state(completed_ids=[])
            prog.update(sample_id=0, current_hydrated_count=1)
            prog.update(sample_id=1, current_hydrated_count=2)
            _close(prog)

        output = _capture_logs(_run)
        assert "test finished: 2/2 (100.0%)" in output


# ===================================================================
# State dump
# ===================================================================
class TestStateDump:
    def test_dump_writes_file_and_expected_fields(self, tmp_path):
        prog = make_progress(
            total=5, root_dir=tmp_path, dump_progress=True, dump_interval=0.0
        )
        prog.init_state(completed_ids=[0, 1], failed_ids=[0])
        _close(prog)

        progress_file = tmp_path / "progress.json"
        assert progress_file.exists(), "progress.json was not written"
        data = _read_progress(tmp_path)
        assert data["total"] == 5
        assert data["completed"] == 2
        assert data["failed"] == 1
        assert "percent" in data
        assert "timestamp" in data
        assert "finished" in data

    def test_dump_marks_finished_when_complete(self, tmp_path):
        total = 3
        prog = make_progress(
            total=total, root_dir=tmp_path, dump_progress=True, dump_interval=0.0
        )
        for i in range(total):
            prog.update(sample_id=i, current_hydrated_count=i + 1)
        _close(prog)

        data = _read_progress(tmp_path)
        assert data["finished"] is True

    def test_no_dump_when_disabled_or_root_dir_missing(self, tmp_path):
        no_dump_flag = make_progress(root_dir=tmp_path, dump_progress=False)
        no_dump_flag.update(sample_id=0, current_hydrated_count=1)
        _close(no_dump_flag)
        assert not (tmp_path / "progress.json").exists()

        no_root = make_progress(root_dir=None, dump_progress=True)
        no_root.update(sample_id=0, current_hydrated_count=1)
        _close(no_root)
        assert not (tmp_path / "progress.json").exists()

    def test_dump_respects_interval(self, tmp_path):
        """With a large dump_interval, dump is suppressed between calls."""
        prog = make_progress(
            total=5, root_dir=tmp_path, dump_progress=True, dump_interval=9999.0
        )
        # Control perf_counter so update() stays within dump_interval.
        with patch(
            "sieval.core.tasks.progress.time.perf_counter",
            side_effect=[1.0, 2.0, 3.0, 3.5],
        ):
            # Force first dump via init_state (force=True).
            prog.init_state(completed_ids=[])
            dump_after_init = prog._pending_dump

            # Non-forced update should NOT re-dump within interval.
            prog.update(sample_id=0, current_hydrated_count=1)

        assert dump_after_init is not None
        assert prog._pending_dump is dump_after_init, (
            "progress.json was re-dumped within the dump interval"
        )
        _close(prog)


# ===================================================================
# Writes off the event loop
# ===================================================================
class TestOffLoopWrite:
    """progress.json is written by a task in the runner's group, never inline."""

    def test_a_dump_alone_never_touches_disk(self, tmp_path):
        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        with patch("sieval.core.tasks.progress.aatomic_write") as write:
            prog.init_state(completed_ids=[0])
            prog.update(sample_id=1, current_hydrated_count=2)
            prog.set_status({"infer": 1}, 2)
        write.assert_not_called()
        assert not (tmp_path / "progress.json").exists()

    @pytest.mark.anyio
    async def test_the_write_runs_on_a_worker_thread(self, tmp_path):
        threads: list[int] = []

        def _record(path: Path, data: bytes) -> None:
            threads.append(threading.get_ident())
            path.write_bytes(data)

        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        with patch("sieval.core.utils.atomic.atomic_write", _record):
            async with anyio.create_task_group() as tg:
                prog.write_in(tg)
                prog.set_status({"infer": 2}, 2)
            prog.write_in(None)
            await prog.aclose()

        assert len(threads) == 2  # the dump inside the group, then the final one
        assert threading.get_ident() not in threads
        assert _read_progress(tmp_path)["finished"] is False

    @pytest.mark.anyio
    async def test_a_dump_before_write_in_is_written_once_it_is_given(self, tmp_path):
        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        prog.init_state(completed_ids=[0, 1])
        async with anyio.create_task_group() as tg:
            prog.write_in(tg)
        prog.write_in(None)
        assert _read_progress(tmp_path)["completed"] == 2
        await prog.aclose()

    @pytest.mark.anyio
    async def test_dumps_during_a_write_coalesce_into_the_latest(self, tmp_path):
        written: list[dict] = []
        release = anyio.Event()

        async def _slow_write(_path: Path, data: bytes) -> None:
            written.append(orjson.loads(data))
            await release.wait()

        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        with patch("sieval.core.tasks.progress.aatomic_write", _slow_write):
            async with anyio.create_task_group() as tg:
                prog.write_in(tg)
                prog.set_status({"infer": 1}, 1)
                await anyio.wait_all_tasks_blocked()
                # The first write is in flight; these three must not queue.
                prog.set_status({"infer": 2}, 2)
                prog.set_status({"infer": 3}, 3)
                prog.set_status({"infer": 4}, 4)
                release.set()

        assert [w["active_details"] for w in written] == [{"infer": 1}, {"infer": 4}]

    @pytest.mark.anyio
    async def test_a_dump_after_the_writer_drains_starts_a_new_write(self, tmp_path):
        written: list[dict] = []

        async def _write(_path: Path, data: bytes) -> None:
            written.append(orjson.loads(data))

        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        with patch("sieval.core.tasks.progress.aatomic_write", _write):
            async with anyio.create_task_group() as tg:
                prog.write_in(tg)
                prog.set_status({"infer": 1}, 1)
                await anyio.wait_all_tasks_blocked()
                prog.set_status({"infer": 2}, 2)

        assert [w["active_details"] for w in written] == [{"infer": 1}, {"infer": 2}]

    @pytest.mark.anyio
    async def test_write_failure_is_non_fatal(self, tmp_path):
        """A failed write logs a warning; it never fails the runner's group."""
        prog = make_progress(total=2, root_dir=tmp_path, dump_progress=True)

        async def _run() -> None:
            with patch(
                "sieval.core.tasks.progress.aatomic_write",
                side_effect=OSError("disk full"),
            ):
                async with anyio.create_task_group() as tg:
                    prog.write_in(tg)
                    prog.update(sample_id=0, current_hydrated_count=1)
                prog.write_in(None)
                await prog.aclose()

        sink = io.StringIO()
        logger_id = logger.add(sink, format="{message}")
        try:
            await _run()
        finally:
            logger.remove(logger_id)
        assert sink.getvalue().count("Failed to write progress file: disk full") == 2

    @pytest.mark.anyio
    async def test_a_cancelled_write_keeps_its_snapshot(self, tmp_path):
        written: list[dict] = []

        async def _write(_path: Path, data: bytes) -> None:
            if not written:
                written.append({})
                await anyio.sleep_forever()
            written.append(orjson.loads(data))

        prog = make_progress(total=5, root_dir=tmp_path, dump_progress=True)
        with patch("sieval.core.tasks.progress.aatomic_write", _write):
            async with anyio.create_task_group() as tg:
                prog.write_in(tg)
                prog.set_status({"infer": 1}, 1)
                await anyio.wait_all_tasks_blocked()
                tg.cancel_scope.cancel()
            async with anyio.create_task_group() as tg:
                prog.write_in(tg)
            prog.write_in(None)

        assert [w["active_details"] for w in written[1:]] == [{"infer": 1}]

    @pytest.mark.anyio
    async def test_aclose_refuses_while_a_group_is_attached(self, tmp_path):
        prog = make_progress(total=2, root_dir=tmp_path, dump_progress=True)
        async with anyio.create_task_group() as tg:
            prog.write_in(tg)
            with pytest.raises(RuntimeError, match="still attached"):
                await prog.aclose()
        prog.write_in(None)
        await prog.aclose()

    @pytest.mark.anyio
    async def test_aclose_writes_inline_after_the_group_is_detached(self, tmp_path):
        prog = make_progress(total=2, root_dir=tmp_path, dump_progress=True)
        async with anyio.create_task_group() as tg:
            prog.write_in(tg)
        prog.write_in(None)
        # The group is gone; neither may start a task in it.
        prog.update(sample_id=0, current_hydrated_count=1)
        prog.finish()
        await prog.aclose()
        data = _read_progress(tmp_path)
        assert data["completed"] == 1
        assert data["finished"] is True


# ===================================================================
# set_status
# ===================================================================
class TestSetStatus:
    @pytest.mark.anyio
    async def test_set_status_writes_active_counts_to_dump(self, tmp_path):
        prog = make_progress(total=10, root_dir=tmp_path, dump_progress=True)

        async with anyio.create_task_group() as tg:
            prog.write_in(tg)
            prog.set_status({"preprocess": 3, "infer": 7}, 5)
        prog.write_in(None)
        data = _read_progress(tmp_path)
        assert data["active_details"] == {"preprocess": 3, "infer": 7}

        prog.set_status({"infer": 5, "feedback": 2}, 5)
        await prog.aclose()
        data = _read_progress(tmp_path)
        assert data["active_details"] == {"infer": 5, "feedback": 2}

    def test_set_status_triggers_nontty_tick(self):
        """set_status() should invoke tick(), producing non-TTY log output."""
        with patch("sys.stderr.isatty", return_value=False):
            prog = make_progress(total=10, show_progress=True, log_interval=0.0)

        prog.init_state(completed_ids=[0, 1])
        # init_state logs once. Now set_status should trigger tick() and produce
        # another log line since log_interval=0.0 means time gate always passes.
        output = _capture_logs(lambda: prog.set_status({"infer": 3}, 5))
        assert "test" in output
        assert "2/10" in output
        _close(prog)


# ===================================================================
# close
# ===================================================================
class TestClose:
    @pytest.mark.anyio
    async def test_close_triggers_final_dump(self, tmp_path):
        prog = make_progress(
            total=2, root_dir=tmp_path, dump_progress=True, dump_interval=9999.0
        )
        prog.init_state(completed_ids=[0, 1])
        async with anyio.create_task_group() as tg:
            prog.write_in(tg)
        prog.write_in(None)
        # remove file to test that close writes it
        (tmp_path / "progress.json").unlink()
        await prog.aclose()
        assert (tmp_path / "progress.json").exists()


# ===================================================================
# TTY pbar paths (pbar.update / _update_pbar_postfix)
# ===================================================================
class TestPbarPaths:
    """Exercise code paths that only run when a tqdm pbar is active."""

    def test_init_state_pbar_update(self):
        fake_pbar = _FakePbar()
        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm", return_value=fake_pbar),
        ):
            prog = make_progress(total=5, show_progress=True)

        prog.init_state(completed_ids=[0, 1, 2])
        assert fake_pbar.updates == [3]
        assert fake_pbar.n == 3
        _close(prog)

    def test_update_and_set_status_update_postfix(self):
        fake_pbar = _FakePbar()
        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm", return_value=fake_pbar),
        ):
            prog = make_progress(total=5, show_progress=True)

        prog.init_state(completed_ids=[])
        prog.update(
            sample_id=0,
            current_hydrated_count=1,
            failed=True,
            anomalies={"rule_a"},
        )
        prog.set_status({"infer": 2, "feedback": 0}, 1)

        assert fake_pbar.postfix_calls
        assert fake_pbar.postfix_calls[-1][0] == {
            "failed": 1,
            "anomalies": 1,
            "infer": 2,
        }
        _close(prog)

    def test_set_status_clears_existing_postfix_when_empty(self):
        fake_pbar = _FakePbar()
        fake_pbar.postfix = {"failed": 1}
        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm", return_value=fake_pbar),
        ):
            prog = make_progress(total=5, show_progress=True)

        prog.set_status({}, 0)

        assert fake_pbar.postfix_calls
        assert fake_pbar.postfix_calls[-1] == ({}, False)
        _close(prog)


class TestPbarConstruction:
    """What the bar is *built with*, not just whether it is built.

    The three gating conditions were covered; the arguments were not, and they
    carry real behaviour — most of all `position`, which is how MultiTaskRunner
    keeps one runner's bar from overwriting another's.
    """

    def _kwargs(self, **overrides) -> dict:
        with (
            patch("sys.stderr.isatty", return_value=True),
            patch("sieval.core.tasks.progress.tqdm") as tqdm_cls,
        ):
            _close(
                TaskProgress(
                    total=overrides.pop("total", 42),
                    desc=overrides.pop("desc", "a-task"),
                    **overrides,
                )
            )
        return dict(tqdm_cls.call_args.kwargs)

    def test_total_and_desc_are_forwarded(self):
        kwargs = self._kwargs(total=42, desc="a-task")
        assert kwargs["total"] == 42
        assert kwargs["desc"] == "a-task"

    def test_position_is_forwarded(self):
        # MultiTaskRunner passes progress_position=i; collapsing it to a
        # constant makes concurrent runners draw over each other.
        assert self._kwargs(position=3)["position"] == 3

    def test_position_defaults_to_the_first_row(self):
        assert self._kwargs()["position"] == 0

    def test_the_bar_is_left_on_screen(self):
        # leave=False would erase the final counts the moment a run finishes,
        # which is the one moment they are worth reading.
        assert self._kwargs()["leave"] is True

    def test_the_unit_is_samples(self):
        assert self._kwargs()["unit"] == "sample"


class TestNonTtyLogFallback:
    """`_enable_log` is exactly "show_progress and not a TTY"."""

    def _enabled(self, *, show_progress: bool, tty: bool) -> bool:
        with patch("sys.stderr.isatty", return_value=tty):
            prog = make_progress(show_progress=show_progress)
        enabled = prog._enable_log
        _close(prog)
        return enabled

    def test_enabled_only_off_tty(self):
        assert self._enabled(show_progress=True, tty=False) is True

    def test_disabled_on_a_tty_where_the_bar_takes_over(self):
        assert self._enabled(show_progress=True, tty=True) is False

    def test_disabled_when_progress_is_off_entirely(self):
        # Silencing progress must silence both channels, not swap one for the
        # other — otherwise `show_progress=False` still writes to the log.
        assert self._enabled(show_progress=False, tty=False) is False
        assert self._enabled(show_progress=False, tty=True) is False


class TestDumpGating:
    """The dump file exists only when a directory *and* the flag are given."""

    def _has_file(self, *, root_dir, dump_progress: bool) -> bool:
        prog = make_progress(root_dir=root_dir, dump_progress=dump_progress)
        has = prog._progress_file is not None
        _close(prog)
        return has

    def test_both_required(self, tmp_path):
        assert self._has_file(root_dir=tmp_path, dump_progress=True) is True

    def test_a_directory_alone_does_not_enable_it(self, tmp_path):
        assert self._has_file(root_dir=tmp_path, dump_progress=False) is False

    def test_the_flag_alone_does_not_enable_it(self):
        # No directory to write into; enabling here would raise mid-run rather
        # than at construction.
        assert self._has_file(root_dir=None, dump_progress=True) is False

    def test_the_file_is_named_progress_json(self, tmp_path):
        prog = make_progress(root_dir=tmp_path, dump_progress=True)
        assert prog._progress_file == tmp_path / "progress.json"
        _close(prog)
