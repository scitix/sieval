"""
Shard corruption and partial-write recovery tests.

Verifies that the loader gracefully handles:
- Truncated JSONL / idx files
- Offset/length mismatches
- Deleted or empty shard files
- Binary garbage in shard data
- Partial idx writes (fewer entries than shard records)
- Multi-shard scenarios where one shard is corrupt

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.tasks.context import TaskContext, TaskStage
from sieval.core.tasks.loader import TaskLoader, _parse_idx_file

from .conftest import make_ctx, make_mock_task, write_contexts


class TestIdxFileCorruption:
    """_parse_idx_file resilience against malformed idx content."""

    def test_truncated_last_line(self, tmp_path):
        """Incomplete last idx line (no trailing newline)
        still parses if fields are present."""
        idx = tmp_path / "0.idx"
        idx.write_text(
            "0\t0\tfinal\t0\t100\t\t\t0\n"  # valid
            "1\t0\tfinal\t100\t12"  # no trailing \n, but all required fields present
        )
        records = _parse_idx_file(idx)
        # Both lines have >= 5 fields, so both parse successfully
        assert len(records) == 2
        assert records[0]["sample_id"] == 0
        assert records[1]["sample_id"] == 1
        assert records[1]["length"] == 12

    def test_truncated_line_too_few_fields(self, tmp_path):
        """Idx line with fewer than 5 fields is skipped."""
        idx = tmp_path / "0.idx"
        idx.write_text(
            "0\t0\tfinal\t0\t100\t\t\t0\n"  # valid (8 fields)
            "1\t0\tfinal\t100\n"  # only 4 fields → dropped
        )
        records = _parse_idx_file(idx)
        assert len(records) == 1
        assert records[0]["sample_id"] == 0

    def test_all_garbage_lines(self, tmp_path):
        """Every line is garbage → empty list."""
        idx = tmp_path / "0.idx"
        idx.write_text("garbage1\ngarbage2\ngarbage3\n")
        assert _parse_idx_file(idx) == []

    def test_binary_content(self, tmp_path):
        """Binary (non-UTF-8) content in idx file → returns empty list."""
        idx = tmp_path / "0.idx"
        idx.write_bytes(b"\x80\x81\x82\xff\xfe\n\x00\x01")
        # May raise UnicodeDecodeError or produce unparseable lines
        records = _parse_idx_file(idx)
        assert records == []


class TestShardCorruptionRoundtrip:
    """Write valid data via TaskSaver, corrupt on disk, verify loader recovery."""

    @pytest.mark.anyio
    async def test_truncated_jsonl_mid_record(self, tmp_path):
        """Shard file truncated mid-JSON → corrupt sample skipped, others load."""
        root = tmp_path / "truncated"
        contexts = [
            TaskContext(
                sample_id=i,
                raw_sample={"q": f"q{i}"},
                stage=TaskStage.FINAL,
                feedback_result={"i": i},
            )
            for i in range(3)
        ]
        await write_contexts(root, contexts, shard_samples=1024)

        # Truncate the shard file to corrupt the last record
        shard_files = list(root.rglob("*.jsonl"))
        assert shard_files
        shard = shard_files[0]
        original = shard.read_bytes()
        shard.write_bytes(original[: len(original) * 2 // 3])

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        # Some samples hydrated, but not all (truncated ones skipped)
        assert len(hydrated_ids) >= 1
        assert len(hydrated_ids) < 3

    @pytest.mark.anyio
    async def test_idx_offset_beyond_shard_size(self, tmp_path):
        """Idx points past shard EOF → read returns short blob → JSON fails → skip."""
        root = tmp_path / "beyond_eof"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            feedback_result={"ok": True},
        )
        await write_contexts(root, [ctx])

        # Rewrite idx with offset + length far beyond shard file size
        idx_files = list(root.rglob("*.idx"))
        assert idx_files
        idx_files[0].write_text("0\t0\tfinal\t99999\t500\t\t\t0\n", encoding="utf-8")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        # Seek past EOF → empty/short read → JSON parse error → not hydrated
        assert 0 not in hydrated_ids

    @pytest.mark.anyio
    async def test_empty_shard_with_valid_idx(self, tmp_path):
        """Shard is 0 bytes but idx has entries → read returns empty → non-fatal."""
        root = tmp_path / "empty_shard"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            feedback_result={"ok": True},
        )
        await write_contexts(root, [ctx])

        # Empty the shard file
        for shard in root.rglob("*.jsonl"):
            shard.write_bytes(b"")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})
        assert 0 not in hydrated_ids

    @pytest.mark.anyio
    async def test_shard_deleted_after_idx_parsed(self, tmp_path):
        """Shard file deleted → outer exception handler catches, no crash."""
        root = tmp_path / "deleted_shard"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            feedback_result={"ok": True},
        )
        await write_contexts(root, [ctx])

        # Delete shard file but keep idx
        for shard in root.rglob("*.jsonl"):
            shard.unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})
        assert 0 not in hydrated_ids

    @pytest.mark.anyio
    async def test_binary_garbage_in_shard(self, tmp_path):
        """Random bytes in shard → orjson.loads fails → all samples skipped."""
        root = tmp_path / "garbage"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            feedback_result={"ok": True},
        )
        await write_contexts(root, [ctx])

        for shard in root.rglob("*.jsonl"):
            shard.write_bytes(b"\x00\xff\xfe\xab" * 100)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})
        assert 0 not in hydrated_ids

    @pytest.mark.anyio
    async def test_partial_idx_fewer_entries_than_shard(self, tmp_path):
        """Idx has N-1 entries for N shard records → only N-1 load."""
        root = tmp_path / "partial_idx"
        contexts = [
            TaskContext(
                sample_id=i,
                raw_sample={"q": f"q{i}"},
                stage=TaskStage.FINAL,
                feedback_result={"i": i},
            )
            for i in range(3)
        ]
        await write_contexts(root, contexts, shard_samples=1024)

        # Remove the last idx entry
        for idx in root.rglob("*.idx"):
            lines = idx.read_text().splitlines(keepends=True)
            if len(lines) >= 2:
                idx.write_text("".join(lines[:-1]))

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        # Only N-1 samples in manifest (idx-driven)
        assert len(loaded) == 2

    @pytest.mark.anyio
    async def test_multiple_shards_one_corrupt(self, tmp_path):
        """Two shards: one valid, one corrupt → valid shard's samples load."""
        root = tmp_path / "multi_shard"
        # shard_samples=2: ids 0,1 → shard 0; ids 2,3 → shard 1
        contexts = [
            TaskContext(
                sample_id=i,
                raw_sample={"q": f"q{i}"},
                stage=TaskStage.FINAL,
                feedback_result={"i": i},
            )
            for i in range(4)
        ]
        await write_contexts(root, contexts, shard_samples=2)

        # Corrupt exactly one shard's JSONL (the one with higher shard id)
        shard_files = sorted(root.rglob("*.jsonl"))
        assert len(shard_files) >= 2, f"Expected >=2 shards, got {len(shard_files)}"
        shard_files[-1].write_bytes(b"corrupted data\n")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        # Exactly 2 samples from the good shard were hydrated
        assert len(hydrated_ids) == 2
        # But not all (corrupt shard's samples failed)
        assert len(hydrated_ids) < 4

    @pytest.mark.anyio
    async def test_corrupt_manifest_valid_shards_rebuilds(self, tmp_path):
        """Corrupt manifest.json → compensate_scan rebuilds from idx files."""
        root = tmp_path / "manifest_corrupt"
        contexts = [make_ctx(i, TaskStage.FINAL) for i in range(3)]
        await write_contexts(root, contexts)

        # Corrupt the manifest
        (root / "manifest.json").write_text("{{not json", encoding="utf-8")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert len(loaded) == 3
        for sid in (0, 1, 2):
            assert loaded[sid].stage == TaskStage.FINAL
