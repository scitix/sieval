"""
Tests for TaskLoader manifest loading, cached report, and manifest status.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import orjson
import pytest

from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.loader import TaskLoader

from .conftest import make_mock_task


class TestLoaderManifest:
    @pytest.mark.anyio
    async def test_load_empty_dir(self, tmp_path):
        task = make_mock_task()
        root = tmp_path / "empty_run"
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        assert contexts == {}

    @pytest.mark.anyio
    async def test_load_manifest_from_disk(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)

        manifest = [
            {
                "sample_id": 0,
                "stage": "final",
                "iteration": 0,
                "final": True,
                "failed": False,
            },
            {
                "sample_id": 1,
                "stage": "failed",
                "iteration": 0,
                "final": False,
                "failed": True,
                "error_action": "infer",
                "error_reason": "exception::Timeout",
            },
        ]
        (root / "manifest.json").write_bytes(orjson.dumps(manifest))

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        assert len(contexts) == 2
        assert contexts[0].stage == TaskStage.FINAL
        assert contexts[1].stage == TaskStage.FAILED
        assert contexts[1].error_action == TaskAction.INFER

    @pytest.mark.anyio
    async def test_get_manifest_status_all_final(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)

        manifest = [
            {
                "sample_id": 0,
                "stage": "final",
                "iteration": 0,
                "final": True,
                "failed": False,
            },
            {
                "sample_id": 1,
                "stage": "final",
                "iteration": 0,
                "final": True,
                "failed": False,
            },
        ]
        (root / "manifest.json").write_bytes(orjson.dumps(manifest))

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        has_failures, all_final = await loader.get_manifest_status()
        assert has_failures is False
        assert all_final is True

    @pytest.mark.anyio
    async def test_get_manifest_status_with_failures(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)

        manifest = [
            {
                "sample_id": 0,
                "stage": "final",
                "iteration": 0,
                "final": True,
                "failed": False,
            },
            {
                "sample_id": 1,
                "stage": "failed",
                "iteration": 0,
                "final": False,
                "failed": True,
            },
        ]
        (root / "manifest.json").write_bytes(orjson.dumps(manifest))

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        has_failures, all_final = await loader.get_manifest_status()
        assert has_failures is True
        assert all_final is False

    @pytest.mark.anyio
    async def test_get_manifest_status_empty(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        has_failures, all_final = await loader.get_manifest_status()
        assert has_failures is False
        assert all_final is False


class TestLoaderCachedReport:
    @pytest.mark.anyio
    async def test_no_report(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        assert await loader.load_cached_report() is None

    @pytest.mark.anyio
    async def test_valid_report(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)
        report = {"accuracy": 0.95}
        (root / "report.json").write_bytes(orjson.dumps(report))

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_cached_report()
        assert loaded == {"accuracy": 0.95}

    @pytest.mark.anyio
    async def test_corrupt_report(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir(parents=True)
        (root / "report.json").write_text("not json{{{")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        assert await loader.load_cached_report() is None


class TestLoaderCorruptManifest:
    @pytest.mark.anyio
    async def test_corrupt_manifest_falls_back_to_shard_scan(self, tmp_path):
        """A corrupt manifest.json should fall back to shard scan, not crash.

        This exercises the compensation path (_compensate_scan) that rebuilds
        state from idx files when the manifest is unreadable.
        """
        from sieval.core.tasks.saver import TaskSaver

        root = tmp_path / "corrupt_manifest"

        # Write a valid FINAL record via saver so idx files exist on disk
        saver = TaskSaver(
            root_dir=root,
            shard_samples=1024,
            record_type_metadata=True,
            record_meta=True,
        )
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.FINAL,
        )
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)
        await saver.flush()

        # Overwrite manifest with invalid JSON
        (root / "manifest.json").write_text("not json{{{")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)

        # Must not raise — should fall back to shard scan
        loaded = await loader.load_initial_state()

        assert len(loaded) == 1, (
            "Expected 1 sample rebuilt from shard scan after corrupt manifest, "
            f"got {len(loaded)}"
        )
        assert loaded[0].stage == TaskStage.FINAL, (
            f"Expected FINAL stage from shard scan, got {loaded[0].stage}"
        )
