"""
Tests for TaskSaver: sharding, manifest updates, flush, and report saving.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import anyio
import orjson
import pytest

from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.saver import TaskSaver


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
# _update_manifest_entry
# ===================================================================
class TestUpdateManifestEntry:
    def test_final_and_failed_entries(self, saver):
        ctx = _make_ctx(0, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        entry = saver._manifest[0]
        assert entry["sample_id"] == 0
        assert entry["stage"] == "final"
        assert entry["final"] is True
        assert entry["failed"] is False

        ctx = _make_ctx(
            1,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
        )
        saver._update_manifest_entry(ctx)
        entry = saver._manifest[1]
        assert entry["failed"] is True
        assert entry["error_action"] == "infer"
        assert entry["error_reason"] == "exception::TimeoutError"

    def test_retry_count_and_overwrite(self, saver):
        ctx = _make_ctx(2, TaskStage.FAILED, retry_count=3)
        saver._update_manifest_entry(ctx)
        entry = saver._manifest[2]
        assert entry["retry_count"] == 3

        ctx = _make_ctx(3, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        entry = saver._manifest[3]
        assert "retry_count" not in entry

        ctx1 = _make_ctx(0, TaskStage.INFERRED)
        ctx2 = _make_ctx(0, TaskStage.FINAL)
        saver._update_manifest_entry(ctx1)
        saver._update_manifest_entry(ctx2)
        assert saver._manifest[0]["stage"] == "final"


# ===================================================================
# sync_manifest
# ===================================================================
class TestSyncManifest:
    def test_copies_manifest(self, saver):
        initial = {
            0: {
                "sample_id": 0,
                "stage": "initial",
                "iteration": 0,
                "final": False,
                "failed": False,
            }
        }
        saver.sync_manifest(initial)
        assert saver._manifest == initial
        # Should be a copy
        initial[1] = {"sample_id": 1}
        assert 1 not in saver._manifest


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
    async def test_flush_writes_shard_and_manifest(self, saver, tmp_root):
        ctx = _make_ctx(0, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)

        await saver.flush()

        # Manifest should exist
        manifest_path = tmp_root / "manifest.json"
        assert manifest_path.exists()
        manifest_data = orjson.loads(manifest_path.read_bytes())
        assert len(manifest_data) == 1
        assert manifest_data[0]["sample_id"] == 0

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
        saver._update_manifest_entry(ctx1)
        saver._stage_queue.append(ctx1)
        await saver.flush()

        ctx2 = _make_ctx(1, TaskStage.FINAL)
        saver._update_manifest_entry(ctx2)
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
            saver._update_manifest_entry(ctx)
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
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)

        await saver.flush()

        assert (tmp_root / "0" / "preprocessed" / "0.jsonl").exists()
        assert (tmp_root / "0" / "inferred" / "0.jsonl").exists()

    @pytest.mark.anyio
    async def test_flush_different_iterations(self, saver, tmp_root):
        ctx_it0 = _make_ctx(0, TaskStage.FINAL, iteration=0)
        ctx_it1 = _make_ctx(0, TaskStage.FINAL, iteration=1)
        for ctx in [ctx_it0, ctx_it1]:
            saver._update_manifest_entry(ctx)
            saver._stage_queue.append(ctx)

        await saver.flush()

        assert (tmp_root / "0" / "final" / "0.jsonl").exists()
        assert (tmp_root / "1" / "final" / "0.jsonl").exists()


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
        assert not (tmp_root / "report.tmp").exists()

    @pytest.mark.anyio
    async def test_save_report_does_not_write_meta(self, saver, tmp_root):
        """save_report should NOT write meta.json — the runner controls that."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        await saver.save_report({"score": 51.01})

        loaded = orjson.loads((tmp_root / "report.json").read_bytes())
        assert loaded == {"score": 51.01}
        assert not (tmp_root / "meta.json").exists()

    @pytest.mark.anyio
    async def test_write_run_meta(self, saver, tmp_root):
        """write_run_meta should create meta.json with version info."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        await saver.write_run_meta()

        meta = orjson.loads((tmp_root / "meta.json").read_bytes())
        assert "version" in meta
        assert isinstance(meta["version"], str)
        assert len(meta["version"]) > 0

    @pytest.mark.anyio
    async def test_write_run_meta_failure_is_non_fatal(
        self, saver, tmp_root, monkeypatch
    ):
        """write_run_meta failure does not propagate."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)

        import sieval as _sieval_mod

        monkeypatch.delattr(_sieval_mod, "__version__")

        await saver.write_run_meta()

        assert not (tmp_root / "meta.json").exists()

    @pytest.mark.anyio
    async def test_write_run_meta_with_deterministic(self, tmp_root):
        """write_run_meta includes `deterministic: true` when set at construction."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        det_saver = TaskSaver(root_dir=tmp_root, deterministic=True)
        await det_saver.write_run_meta()

        meta = orjson.loads((tmp_root / "meta.json").read_bytes())
        assert meta["deterministic"] is True

    @pytest.mark.anyio
    async def test_write_run_meta_deterministic_false_by_default(self, saver, tmp_root):
        """Default saver writes `deterministic: false` so the field is always
        present — missing key means pre-feature run, false means post-feature
        non-deterministic run."""
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        await saver.write_run_meta()

        meta = orjson.loads((tmp_root / "meta.json").read_bytes())
        assert meta["deterministic"] is False

    @pytest.mark.anyio
    async def test_save_report_failure_is_non_fatal(self, saver, tmp_root, monkeypatch):
        """
        If writing the report fails, the error is logged but no exception is
        raised.
        """
        await anyio.Path(tmp_root).mkdir(parents=True, exist_ok=True)

        async def _fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(anyio, "open_file", _fail)
        # Should not raise
        await saver.save_report({"accuracy": 0.5})


# ===================================================================
# write_run_meta — create-if-absent + self-mkdir (run-start handshake)
# ===================================================================
class TestWriteRunMeta:
    @pytest.mark.anyio
    async def test_creates_dir_and_file(self, tmp_path):
        from sieval import __version__

        root = tmp_path / "fresh"
        saver = TaskSaver(root_dir=root, deterministic=True)
        assert not root.exists()

        await saver.write_run_meta()

        meta_path = root / "meta.json"
        assert meta_path.exists()
        meta = orjson.loads(meta_path.read_bytes())
        assert meta["version"] == __version__
        assert meta["deterministic"] is True

    @pytest.mark.anyio
    async def test_create_if_absent_does_not_clobber(self, tmp_path):
        root = tmp_path / "existing"
        saver = TaskSaver(root_dir=root, deterministic=False)
        await saver.write_run_meta()
        # Simulate an originating version already recorded under this dir.
        (root / "meta.json").write_bytes(
            orjson.dumps({"version": "0.1.0", "deterministic": False})
        )

        await saver.write_run_meta()  # must NOT overwrite

        meta = orjson.loads((root / "meta.json").read_bytes())
        assert meta["version"] == "0.1.0"

    @pytest.mark.anyio
    async def test_exists_check_failure_is_non_fatal(self, tmp_path, monkeypatch):
        async def _boom(_self):
            raise OSError("simulated stat failure")

        monkeypatch.setattr(anyio.Path, "exists", _boom)
        saver = TaskSaver(root_dir=tmp_path / "x", deterministic=False)
        await saver.write_run_meta()  # must NOT raise


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

        manifest_path = tmp_root / "manifest.json"
        assert manifest_path.exists()
        data = orjson.loads(manifest_path.read_bytes())
        assert len(data) == 3

        send, recv = anyio.create_memory_object_stream(10)

        async with send:
            await send.send(_make_ctx(0, TaskStage.FINAL))

        await saver.consume_stream(recv)

        manifest_path = tmp_root / "manifest.json"
        assert manifest_path.exists()

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
        manifest_data = orjson.loads((root / "manifest.json").read_bytes())
        assert len(manifest_data) == 3
        persisted_ids = {e["sample_id"] for e in manifest_data}
        assert persisted_ids == {0, 1, 2}


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
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        manifest_data = orjson.loads(manifest_path.read_bytes())
        assert len(manifest_data) == 2
        persisted_ids = {entry["sample_id"] for entry in manifest_data}
        assert persisted_ids == {0, 1}

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
        saver._update_manifest_entry(ctx)
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
            saver._update_manifest_entry(ctx)
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
