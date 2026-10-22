"""
Tests for TaskSaver: sharding, the `.idx` index, flush, and report saving.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path

import anyio
import orjson
import pytest

from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext, TaskRunIdentity
from sieval.core.tasks.saver import TaskSaver
from sieval.core.utils import atomic as saver_module

# What `get_task_run_identity` hands the saver for a decorated task; the saver
# itself only passes it through, so a literal keeps these tests off the registry.
_IDENTITY: TaskRunIdentity = {
    "name": "aime_2026_0shot_gen",
    "display_name": "AIME 2026 (0-shot, generative)",
    "dataset": "aime_2026",
    "eval_mode": "gen",
    "n_shot": 0,
    "tags": ["english", "open-ended"],
    "status": "stable",
    "reference_kind": "value",
}


def _idx_ids(root: Path) -> list:
    """Sample IDs the `.idx` files index, in order: what the loader reads."""
    ids = []
    for idx in sorted(root.rglob("*.idx")):
        for line in idx.read_text().splitlines():
            sid = line.split("\t")[0]
            ids.append(int(sid) if sid.isdigit() else sid)
    return ids


def _temp_files(root):
    return sorted(p.name for p in root.glob("*.tmp"))


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path / "saver_test"


@pytest.fixture
def saver(tmp_root):
    return TaskSaver(
        root_dir=tmp_root,
        shard_samples=4,  # small shards for testing
        shard_write_concurrency=2,
        write_buffer_size=2,
        write_buffer_flush_interval=999.0,  # disable time-based flush
        record_type_metadata=True,
        record_meta=True,
    )


def _make_ctx(sid, stage=TaskStage.FINAL, iteration=0, **kwargs):
    return TaskContext(
        sample_id=sid,
        raw_sample={"q": f"question_{sid}"},
        stage=stage,
        iteration=iteration,
        **kwargs,
    )


# ===================================================================
# _shard_id_for
# ===================================================================
class TestShardIdFor:
    def test_sharding_behaviors(self, saver):
        # shard_samples=4, so id 0-3 -> shard 0, 4-7 -> shard 1
        assert saver._shard_id_for(0) == 0
        assert saver._shard_id_for(3) == 0
        assert saver._shard_id_for(4) == 1
        assert saver._shard_id_for(7) == 1

        # String IDs use xxhash, result should be deterministic
        s1 = saver._shard_id_for("sample_a")
        s2 = saver._shard_id_for("sample_a")
        assert s1 == s2
        # Should be within [0, shard_samples)
        assert 0 <= s1 < saver._shard_samples

        # Verify actual distribution across shards (not just > 1)
        shard_counts: dict[int, int] = {}
        for i in range(100):
            shard = saver._shard_id_for(f"sample_{i}")
            assert 0 <= shard < saver._shard_samples, (
                f"shard_id {shard} out of range [0, {saver._shard_samples})"
            )
            shard_counts[shard] = shard_counts.get(shard, 0) + 1
        # With shard_samples=4 and 100 inputs, all shards should be used
        assert len(shard_counts) == saver._shard_samples, (
            f"Expected all {saver._shard_samples} shards used, "
            f"only got {sorted(shard_counts.keys())}"
        )
        # Each shard should have a reasonable share (at least 10%)
        for shard_id, count in shard_counts.items():
            assert count >= 10, (
                f"Shard {shard_id} has only {count}/100 samples — poor distribution"
            )


# ===================================================================
# flush – async write to disk
# ===================================================================
class TestFlush:
    @pytest.mark.anyio
    async def test_flush_empty_noop(self, saver):
        """Flushing with empty queue should not create any files."""
        await saver.flush()
        assert not saver._root_dir.exists() or not any(saver._root_dir.iterdir())

    @pytest.mark.anyio
    async def test_flush_writes_shard_and_index(self, saver, tmp_root):
        ctx = _make_ctx(0, TaskStage.FINAL)
        saver._stage_queue.append(ctx)

        await saver.flush()

        # No manifest: the shard and its index are the record
        assert not (tmp_root / "manifest.json").exists()
        assert _idx_ids(tmp_root) == [0]

        # Shard file should exist
        shard_path = tmp_root / "0" / "final" / "0.jsonl"
        assert shard_path.exists()
        lines = shard_path.read_bytes().strip().split(b"\n")
        assert len(lines) == 1

        # Index file should exist
        idx_path = tmp_root / "0" / "final" / "0.idx"
        assert idx_path.exists()
        idx_lines = idx_path.read_text().strip().split("\n")
        assert len(idx_lines) == 1
        parts = idx_lines[0].split("\t")
        assert parts[0] == "0"  # sample_id
        assert parts[2] == "final"  # stage

    @pytest.mark.anyio
    async def test_flush_appends_to_existing_shard(self, saver, tmp_root):
        """Second flush should append, not overwrite."""
        ctx1 = _make_ctx(0, TaskStage.FINAL)
        saver._stage_queue.append(ctx1)
        await saver.flush()

        ctx2 = _make_ctx(1, TaskStage.FINAL)
        saver._stage_queue.append(ctx2)
        await saver.flush()

        shard_path = tmp_root / "0" / "final" / "0.jsonl"
        lines = shard_path.read_bytes().strip().split(b"\n")
        assert len(lines) == 2
        ids = {orjson.loads(line)["sample_id"] for line in lines}
        assert ids == {0, 1}

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "sample_ids, expected_shards",
        [
            ([0, 1, 2], {"0.jsonl": 3}),  # 3 samples → same shard
            (
                [0, 4],
                {"0.jsonl": 1, "1.jsonl": 1},
            ),  # shard boundary: 0→shard0, 4→shard1
        ],
        ids=["same-shard", "cross-shard"],
    )
    async def test_flush_shard_routing(
        self, saver, tmp_root, sample_ids, expected_shards
    ):
        """Samples route to correct shards based on shard_samples=4."""
        for sid in sample_ids:
            ctx = _make_ctx(sid, TaskStage.FINAL)
            saver._stage_queue.append(ctx)

        await saver.flush()

        shard_dir = tmp_root / "0" / "final"
        for filename, expected_count in expected_shards.items():
            shard_path = shard_dir / filename
            assert shard_path.exists(), f"{filename} should exist"
            lines = shard_path.read_bytes().strip().split(b"\n")
            assert len(lines) == expected_count
            # Verify sample_id routing correctness for cross-shard case
            if filename == "0.jsonl" and expected_count == 1:
                ids = {orjson.loads(line)["sample_id"] for line in lines}
                assert ids == {0}, f"Shard 0 should contain sample_id=0, got {ids}"
            elif filename == "1.jsonl" and expected_count == 1:
                ids = {orjson.loads(line)["sample_id"] for line in lines}
                assert ids == {4}, f"Shard 1 should contain sample_id=4, got {ids}"

    @pytest.mark.anyio
    async def test_flush_different_stages(self, saver, tmp_root):
        """Different stages go to different directories."""
        ctx_pre = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            preprocess_result="preprocessed",
        )
        ctx_inf = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.INFERRED,
            infer_result="inferred",
        )
        for ctx in [ctx_pre, ctx_inf]:
            saver._stage_queue.append(ctx)

        await saver.flush()

        assert (tmp_root / "0" / "preprocessed" / "0.jsonl").exists()
        assert (tmp_root / "0" / "inferred" / "0.jsonl").exists()

    @pytest.mark.anyio
    async def test_flush_different_iterations(self, saver, tmp_root):
        ctx_it0 = _make_ctx(0, TaskStage.FINAL, iteration=0)
        ctx_it1 = _make_ctx(0, TaskStage.FINAL, iteration=1)
        for ctx in [ctx_it0, ctx_it1]:
            saver._stage_queue.append(ctx)

        await saver.flush()

        assert (tmp_root / "0" / "final" / "0.jsonl").exists()
        assert (tmp_root / "1" / "final" / "0.jsonl").exists()

    @pytest.mark.anyio
    async def test_a_failed_flush_keeps_its_batch_for_the_next(
        self, saver, tmp_root, monkeypatch
    ):
        """A flush that dies mid-write (an error, or the storage worker being
        cancelled) leaves its contexts queued, so a later flush still writes them."""
        real_write_shard = TaskSaver._write_shard
        calls = {"n": 0}

        async def _write_shard(self, *args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("EIO")
            return await real_write_shard(self, *args)

        monkeypatch.setattr(TaskSaver, "_write_shard", _write_shard)
        for sid in (0, 1):
            ctx = _make_ctx(sid, TaskStage.FINAL)
            saver._stage_queue.append(ctx)

        with pytest.raises(ExceptionGroup):
            await saver.flush()
        assert [c.sample_id for c in saver._stage_queue] == [0, 1]

        await saver.flush()
        assert saver._stage_queue == []
        lines = (tmp_root / "0" / "final" / "0.jsonl").read_bytes().splitlines()
        assert {orjson.loads(line)["sample_id"] for line in lines} == {0, 1}

    @pytest.mark.anyio
    async def test_drain_writes_what_the_stream_still_holds(self, saver, tmp_root):
        send, recv = anyio.create_memory_object_stream[TaskContext](8)
        leftover = recv.clone()
        for sid in (0, 1, 2):
            send.send_nowait(_make_ctx(sid, TaskStage.FINAL))
        # The consumer stopped early, with three contexts still buffered.
        recv.close()

        await saver.drain(leftover)

        assert _idx_ids(tmp_root) == [0, 1, 2]
        lines = (tmp_root / "0" / "final" / "0.jsonl").read_bytes().splitlines()
        assert len(lines) == 3
        with pytest.raises(anyio.ClosedResourceError):
            leftover.receive_nowait()
        send.close()

    @pytest.mark.anyio
    async def test_drain_stops_at_the_end_of_a_closed_stream(self, saver, tmp_root):
        send, recv = anyio.create_memory_object_stream[TaskContext](8)
        send.send_nowait(_make_ctx(0, TaskStage.FINAL))
        send.close()

        await saver.drain(recv)

        assert (
            len((tmp_root / "0" / "final" / "0.jsonl").read_bytes().splitlines()) == 1
        )


# ===================================================================
# save_report
# ===================================================================
class TestSaveReport:
    @pytest.mark.anyio
    async def test_save_report_overwrite_and_atomic(self, saver, tmp_root):
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        report = {"accuracy": 0.95, "total": 100}
        await saver.save_report(report)

        report_path = tmp_root / "report.json"
        assert report_path.exists()
        loaded = orjson.loads(report_path.read_bytes())
        assert loaded["accuracy"] == 0.95

        await saver.save_report({"v": 1})
        await saver.save_report({"v": 2})
        loaded = orjson.loads((tmp_root / "report.json").read_bytes())
        assert loaded["v"] == 2

        # Temp file should not remain after successful save.
        assert _temp_files(tmp_root) == []

    @pytest.mark.anyio
    async def test_new_files_get_the_umask_mode_not_mkstemps(
        self, saver, tmp_root, monkeypatch
    ):
        """Readers under another uid on a shared volume can open the result."""
        monkeypatch.setattr(saver_module, "_UMASK", 0o022)
        await saver.begin_run(resumed=False)
        await saver.save_report({"v": 1})
        for name in ("report.json", "meta.json"):
            assert (tmp_root / name).stat().st_mode & 0o777 == 0o644, name

    @pytest.mark.anyio
    async def test_a_replaced_file_keeps_its_mode(self, saver, tmp_root):
        tmp_root.mkdir(parents=True)
        (tmp_root / "report.json").write_bytes(b"{}")
        (tmp_root / "report.json").chmod(0o640)
        await saver.save_report({"v": 2})
        assert (tmp_root / "report.json").stat().st_mode & 0o777 == 0o640

    @pytest.mark.anyio
    async def test_save_report_does_not_write_meta(self, saver, tmp_root):
        """save_report should NOT write meta.json — the runner controls that."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        await saver.save_report({"score": 51.01})

        loaded = orjson.loads((tmp_root / "report.json").read_bytes())
        assert loaded == {"score": 51.01}
        assert not (tmp_root / "meta.json").exists()

    @pytest.mark.anyio
    async def test_save_report_failure_fails_the_task(
        self, saver, tmp_root, monkeypatch
    ):
        """`report.json` is the result: a failed write raises, and the temporary
        file does not linger."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)

        def _fail_replace(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(saver_module.os, "replace", _fail_replace)
        with pytest.raises(OSError, match="disk full"):
            await saver.save_report({"accuracy": 0.5})
        assert not (tmp_root / "report.json").exists()
        assert _temp_files(tmp_root) == []


# ===================================================================
# meta.json — rewritten each run, runs[] appended
# ===================================================================
def _read_meta(root):
    return orjson.loads((root / "meta.json").read_bytes())


class TestRunMeta:
    @pytest.mark.anyio
    async def test_fresh_run_creates_dir_and_file(self, tmp_path):
        from sieval import __version__

        root = tmp_path / "fresh"
        saver = TaskSaver(root_dir=root, deterministic=True, task_identity=_IDENTITY)
        assert not root.exists()

        await saver.begin_run(resumed=False, model="candidate")

        meta = _read_meta(root)
        assert meta["version"] == __version__
        assert meta["deterministic"] is True
        assert meta["task"] == _IDENTITY
        assert meta["model"] == "candidate"
        assert meta["identity"] == {
            "schema": 1,
            "data": None,
            "prompt": None,
            "inference": None,
            "grader": None,
        }
        [run] = meta["runs"]
        assert run["sieval"] == __version__
        assert run["start"] == "fresh"
        assert "reran" not in run  # both written only when the run ends
        assert "ended" not in run
        assert "config" not in run  # absent when there is no diff
        assert _temp_files(root) == []

    @pytest.mark.anyio
    async def test_deterministic_false_is_written(self, saver, tmp_root):
        """Missing means a pre-feature run, so false is spelled out."""
        await saver.begin_run(resumed=False)
        assert _read_meta(tmp_root)["deterministic"] is False

    @pytest.mark.anyio
    async def test_task_key_absent_for_undecorated_class(self, tmp_path):
        root = tmp_path / "anonymous"
        await TaskSaver(root_dir=root).begin_run(resumed=False)
        assert "task" not in _read_meta(root)

    @pytest.mark.anyio
    async def test_resume_keeps_origin_and_appends_a_run(self, tmp_path):
        root = tmp_path / "resumed"
        first = TaskSaver(root_dir=root, deterministic=False, task_identity=_IDENTITY)
        await first.begin_run(resumed=False, model="m")
        await first.set_identity(data={"digest": "sha256:d", "samples": 3})
        await first.end_run(["grader", "data", "prompt", "inference"])
        # The directory was created by an older version of the same series.
        meta = _read_meta(root)
        meta["version"] = "0.9.0"
        (root / "meta.json").write_bytes(orjson.dumps(meta))

        second = TaskSaver(root_dir=root, deterministic=True, task_identity=_IDENTITY)
        await second.begin_run(
            resumed=True,
            model="m",
            config_diff=["concurrency_limit: 8 → 2"],
            inference={"digest": "sha256:i"},
        )

        meta = _read_meta(root)
        # Origin fields describe the run that created the directory.
        assert meta["version"] == "0.9.0"
        assert meta["deterministic"] is False
        # A stage not given is carried; a given one replaces it.
        assert meta["identity"]["data"] == {"digest": "sha256:d", "samples": 3}
        assert meta["identity"]["inference"] == {"digest": "sha256:i"}
        first_run, second_run = meta["runs"]
        assert first_run["reran"] == ["data", "prompt", "inference", "grader"]
        assert second_run["start"] == "resume"
        assert second_run["config"] == ["concurrency_limit: 8 → 2"]
        assert "reran" not in second_run
        assert "ended" not in second_run
        assert first_run["ended"] >= first_run["at"]

    @pytest.mark.anyio
    async def test_resume_does_not_backfill_task_into_legacy_meta(self, tmp_path):
        """A pre-feature `meta.json` reads back without `task`, `identity` or
        `runs`; a resume adds the run history but never a `task` block, which
        would describe the resuming process instead of the samples' producer."""
        root = tmp_path / "legacy"
        root.mkdir()
        (root / "meta.json").write_bytes(
            orjson.dumps({"version": "0.6.0", "deterministic": False})
        )
        saver = TaskSaver(root_dir=root, task_identity=_IDENTITY)

        await saver.begin_run(resumed=True)

        meta = _read_meta(root)
        assert "task" not in meta
        assert meta["version"] == "0.6.0"
        assert meta["identity"]["data"] is None
        assert [r["start"] for r in meta["runs"]] == ["resume"]

    @pytest.mark.anyio
    async def test_a_from_run_starts_the_history_and_names_its_source(self, tmp_path):
        root = tmp_path / "copied"
        source = {"dir": "/mnt/old/task", "identity": {"schema": 1}}
        saver = TaskSaver(root_dir=root, deterministic=True, task_identity=_IDENTITY)
        await saver.begin_run(resumed=True, run_from=source)  # type: ignore[arg-type]
        meta = _read_meta(root)
        [run] = meta["runs"]
        assert run["start"] == "from"
        assert run["from"] == source
        assert meta["deterministic"] is True
        assert meta["task"] == _IDENTITY

    @pytest.mark.anyio
    async def test_resume_does_not_backfill_deterministic(self, tmp_path):
        """A `meta.json` that never recorded `deterministic` keeps not saying it;
        the resuming process's value would describe it, not the samples."""
        root = tmp_path / "legacy"
        root.mkdir()
        (root / "meta.json").write_bytes(orjson.dumps({"version": "0.6.0"}))
        saver = TaskSaver(root_dir=root, deterministic=True)

        await saver.begin_run(resumed=True)

        assert "deterministic" not in _read_meta(root)

    @pytest.mark.anyio
    async def test_concurrent_writes_use_their_own_temp_files(
        self, tmp_root, monkeypatch
    ):
        """Two writers never share one temp file, and each fsyncs before the
        replace makes its data visible."""
        await anyio.Path(tmp_root).mkdir(parents=True)
        temps: list[str] = []
        synced: list[int] = []
        real_mkstemp = saver_module.tempfile.mkstemp
        real_fsync = saver_module.os.fsync

        def _mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            temps.append(name)
            return fd, name

        def _fsync(fd):
            synced.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(saver_module.tempfile, "mkstemp", _mkstemp)
        monkeypatch.setattr(saver_module.os, "fsync", _fsync)
        async with anyio.create_task_group() as tg:
            for i in range(4):
                tg.start_soon(
                    saver_module.aatomic_write, tmp_root / "meta.json", b"%d" % i
                )
        assert len(set(temps)) == 4
        assert all(Path(t).parent == tmp_root for t in temps)
        assert len(synced) == 4
        assert (tmp_root / "meta.json").read_bytes() in {b"0", b"1", b"2", b"3"}
        assert _temp_files(tmp_root) == []

    @pytest.mark.anyio
    async def test_resume_without_readable_meta_writes_a_fresh_origin(self, tmp_path):
        from sieval import __version__

        root = tmp_path / "garbled"
        root.mkdir()
        (root / "meta.json").write_bytes(b"{ truncated")
        await TaskSaver(root_dir=root).begin_run(resumed=True)
        meta = _read_meta(root)
        assert meta["version"] == __version__
        assert meta["runs"][0]["start"] == "resume"

    @pytest.mark.anyio
    async def test_identity_of_another_schema_is_not_carried(self, tmp_path):
        root = tmp_path / "schema"
        root.mkdir()
        (root / "meta.json").write_bytes(
            orjson.dumps(
                {
                    "version": "0.9.0",
                    "deterministic": False,
                    "identity": {"schema": 99, "data": {"digest": "x"}},
                    "runs": ["not a run", {"sieval": "0.9.0", "start": "fresh"}],
                }
            )
        )
        await TaskSaver(root_dir=root).begin_run(resumed=True)
        meta = _read_meta(root)
        assert meta["identity"]["schema"] == 1
        assert meta["identity"]["data"] is None
        # Malformed entries are dropped, well-formed ones kept.
        assert [r["start"] for r in meta["runs"]] == ["fresh", "resume"]

    @pytest.mark.anyio
    async def test_set_identity_keeps_what_it_is_not_given(self, saver, tmp_root):
        """Leaving a stage out keeps it; `None` is a value, and sets it unknown."""
        await saver.begin_run(
            resumed=False,
            inference={"digest": "sha256:i"},
            grader={"digest": "sha256:g", "modules": {}, "judges": []},
        )
        await saver.set_identity(data={"digest": "sha256:d", "samples": 1})
        await saver.set_identity(grader=None)
        identity = _read_meta(tmp_root)["identity"]
        assert identity["data"] == {"digest": "sha256:d", "samples": 1}
        assert identity["inference"] == {"digest": "sha256:i"}
        assert identity["prompt"] is None
        assert identity["grader"] is None
        assert saver.inference_identity == {"digest": "sha256:i"}

    @pytest.mark.anyio
    async def test_reported_marks_a_run_that_wrote_its_report(self, saver, tmp_root):
        await saver.begin_run(resumed=False)
        await saver.end_run([], reported=True)
        [run] = _read_meta(tmp_root)["runs"]
        assert run["reran"] == []
        assert run["reported"] is True

    @pytest.mark.anyio
    async def test_a_fast_resume_reports_nothing(self, saver, tmp_root):
        await saver.begin_run(resumed=False)
        await saver.end_run([])
        [run] = _read_meta(tmp_root)["runs"]
        assert "reported" not in run

    @pytest.mark.anyio
    async def test_updates_before_begin_run_raise(self, saver):
        with pytest.raises(RuntimeError, match="begin_run"):
            await saver.set_identity(data=None)
        with pytest.raises(RuntimeError, match="begin_run"):
            await saver.end_run(())
        assert saver.run_meta is None
        assert saver.inference_identity is None

    @pytest.mark.anyio
    async def test_interrupt_ends_the_run_without_reran(self, saver, tmp_root):
        await saver.interrupt_run()  # before begin_run: nothing to record
        assert not (tmp_root / "meta.json").exists()
        await saver.begin_run(resumed=False)
        await saver.interrupt_run()
        run = _read_meta(tmp_root)["runs"][-1]
        assert run["interrupted"] is True
        assert run["ended"] >= run["at"]
        assert "reran" not in run

    @pytest.mark.anyio
    async def test_each_run_keeps_its_own_request_counts(self, saver, tmp_root):
        counts = {
            "infer": {
                "m": {
                    "calls": 2,
                    "attempts": 3,
                    "outcomes": {"2xx": 2, "429": 1},
                    "in_flight": {"task": {"peak": 2, "limit": 4}},
                }
            }
        }
        await saver.begin_run(resumed=False)
        await saver.interrupt_run(requests=counts)
        await saver.begin_run(resumed=True)
        await saver.end_run(["inference"], requests={})  # none made: omitted
        await saver.begin_run(resumed=True)
        await saver.end_run([], requests=counts)

        runs = _read_meta(tmp_root)["runs"]
        assert runs[0]["requests"] == counts
        assert "requests" not in runs[1]
        assert runs[2]["requests"] == counts

    @pytest.mark.anyio
    async def test_end_run_orders_stages(self, saver, tmp_root):
        await saver.begin_run(resumed=False)
        await saver.end_run(["grader", "prompt"])
        assert _read_meta(tmp_root)["runs"][-1]["reran"] == ["prompt", "grader"]
        assert saver.run_meta is not None

    @pytest.mark.anyio
    async def test_failed_write_raises_and_cleans_up(
        self, saver, tmp_root, monkeypatch
    ):
        """`meta.json` says whether a run started here: a write that did not land
        fails the task rather than leaving the directory to be read as fresh."""

        def _fail_replace(*_args):
            raise OSError("read-only file system")

        monkeypatch.setattr(saver_module.os, "replace", _fail_replace)
        with pytest.raises(OSError, match="read-only"):
            await saver.begin_run(resumed=False)
        assert not (tmp_root / "meta.json").exists()
        assert _temp_files(tmp_root) == []


# ===================================================================
# consume_stream
# ===================================================================
class TestConsumeStream:
    @pytest.mark.anyio
    async def test_consume_stream_flushes_during_and_after_stream(
        self, saver, tmp_root
    ):
        """consume_stream should flush on buffer threshold and on final stream close."""
        send, recv = anyio.create_memory_object_stream(10)

        async with send:
            for i in range(3):
                await send.send(_make_ctx(i, TaskStage.FINAL))

        await saver.consume_stream(recv)

        assert sorted(_idx_ids(tmp_root)) == [0, 1, 2]

        send, recv = anyio.create_memory_object_stream(10)

        async with send:
            await send.send(_make_ctx(0, TaskStage.FINAL))

        await saver.consume_stream(recv)

        assert _idx_ids(tmp_root) == [0, 1, 2, 0]

    @pytest.mark.anyio
    async def test_consume_stream_buffer_size_triggers_mid_stream_flush(self, tmp_path):
        """Buffer-size threshold triggers flush before stream closes.

        With write_buffer_size=2, sending 3 items should trigger at least one
        mid-stream flush (at item 2), then a final flush for the remaining item.
        """
        root = tmp_path / "buffer_trigger"
        saver = TaskSaver(
            root_dir=root,
            shard_samples=1024,
            shard_write_concurrency=2,
            write_buffer_size=2,  # flush every 2 items
            write_buffer_flush_interval=999.0,  # disable time-based
            record_type_metadata=True,
            record_meta=True,
        )

        flush_count = 0
        _original_flush = saver.flush

        async def _counting_flush():
            nonlocal flush_count
            flush_count += 1
            await _original_flush()

        saver.flush = _counting_flush  # type: ignore[assignment]

        send, recv = anyio.create_memory_object_stream(10)
        async with send:
            for i in range(3):
                await send.send(_make_ctx(i, TaskStage.FINAL))

        await saver.consume_stream(recv)

        # At least 2 flushes: one mid-stream (buffer full at 2) + one final
        assert flush_count >= 2
        # All 3 samples persisted
        assert sorted(_idx_ids(root)) == [0, 1, 2]


# ===================================================================
# Flush interval (time-based flush)
# ===================================================================
class TestFlushInterval:
    @pytest.mark.anyio
    async def test_flush_triggered_by_interval(self, tmp_path):
        """consume_stream triggers a flush when write_buffer_flush_interval
        elapses, even if the buffer is not full."""
        root = tmp_path / "interval_test"
        saver = TaskSaver(
            root_dir=root,
            shard_samples=4,
            shard_write_concurrency=2,
            write_buffer_size=999,  # very large — won't trigger by size
            write_buffer_flush_interval=0.05,  # 50 ms
            record_type_metadata=True,
            record_meta=True,
        )

        send, recv = anyio.create_memory_object_stream(10)

        async def _produce():
            async with send:
                # Send first context — starts the interval clock
                await send.send(_make_ctx(0, TaskStage.FINAL))
                # Sleep long enough for the flush interval to elapse
                await anyio.sleep(0.1)
                # Second context triggers the time-due check
                await send.send(_make_ctx(1, TaskStage.FINAL))

        async with anyio.create_task_group() as tg:
            tg.start_soon(_produce)
            await saver.consume_stream(recv)

        # Both samples must be persisted
        assert sorted(_idx_ids(root)) == [0, 1]

        # Shard file must contain exactly 2 entries (both flushed)
        shard_path = root / "0" / "final" / "0.jsonl"
        assert shard_path.exists()
        lines = shard_path.read_bytes().strip().split(b"\n")
        assert len(lines) == 2


# ===================================================================
# Serialization content verification
# ===================================================================
class TestSerializationContent:
    @pytest.mark.anyio
    async def test_shard_contains_serialized_context(self, saver, tmp_root):
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"question": "What is 2+2?"},
            stage=TaskStage.POSTPROCESSED,
            preprocess_result="formatted prompt",
            infer_result="4",
            postprocess_result="4",
        )
        saver._stage_queue.append(ctx)
        await saver.flush()

        shard_path = tmp_root / "0" / "postprocessed" / "0.jsonl"
        line = shard_path.read_bytes().strip()
        obj = orjson.loads(line)

        assert obj["sample_id"] == 0
        # raw_sample is NOT serialized — it's reconstructed from the dataset on load
        assert "raw_sample" not in obj
        assert obj["postprocess_result"] == "4"
        assert obj["preprocess_result"] == "formatted prompt"
        assert obj["infer_result"] == "4"

    @pytest.mark.anyio
    async def test_idx_offsets_are_correct(self, saver, tmp_root):
        """Verify idx offsets can be used to seek and read the correct line."""
        for i in range(3):
            ctx = _make_ctx(i, TaskStage.FINAL)
            saver._stage_queue.append(ctx)

        await saver.flush()

        shard_path = tmp_root / "0" / "final" / "0.jsonl"
        idx_path = tmp_root / "0" / "final" / "0.idx"

        shard_bytes = shard_path.read_bytes()
        idx_lines = idx_path.read_text().strip().split("\n")

        for idx_line in idx_lines:
            parts = idx_line.split("\t")
            sid = int(parts[0])
            offset = int(parts[3])
            length = int(parts[4])

            blob = shard_bytes[offset : offset + length]
            obj = orjson.loads(blob)
            assert obj["sample_id"] == sid
