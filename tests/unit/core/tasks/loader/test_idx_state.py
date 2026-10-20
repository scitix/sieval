"""
Tests for TaskLoader's state: each sample's latest record, read through `.idx`
and checked against its shard's length; and the cached report.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import orjson
import pytest

from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.loader import TaskLoader

from .conftest import make_ctx, make_mock_task, write_contexts


def _shard(root, stage=TaskStage.FINAL):
    return root / "0" / stage.value / "0.jsonl"


def _idx(root, stage=TaskStage.FINAL):
    return root / "0" / stage.value / "0.idx"


class TestState:
    @pytest.mark.anyio
    async def test_an_empty_dir_has_no_samples(self, tmp_path):
        loader = TaskLoader(task=make_mock_task(), root_dir=tmp_path / "empty")
        assert await loader.load_initial_state() == {}

    @pytest.mark.anyio
    async def test_state_comes_from_the_idx_columns(self, tmp_path):
        failed = make_ctx(
            1,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::Timeout",
            retry_count=2,
        )
        await write_contexts(tmp_path, [make_ctx(0), failed])
        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert contexts[0].stage is TaskStage.FINAL
        assert contexts[1].stage is TaskStage.FAILED
        assert contexts[1].error_action is TaskAction.INFER
        assert contexts[1].error_reason == "exception::Timeout"
        assert contexts[1].retry_count == 2

    @pytest.mark.anyio
    async def test_a_leftover_manifest_is_ignored(self, tmp_path):
        """A directory written before the manifest was removed may still hold
        one; the shards decide."""
        await write_contexts(tmp_path, [make_ctx(0)])
        (tmp_path / "manifest.json").write_bytes(
            orjson.dumps([{"sample_id": 0, "stage": "failed", "iteration": 0}])
        )
        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert contexts[0].stage is TaskStage.FINAL


class TestIdxAgainstShard:
    @pytest.mark.anyio
    async def test_a_record_without_its_idx_line_is_not_done(self, tmp_path):
        """Killed after the shard append, before the `.idx` append: the
        sample reads as its previous record, so that stage reruns."""
        await write_contexts(tmp_path, [make_ctx(0, TaskStage.PREPROCESSED)])
        await write_contexts(tmp_path, [make_ctx(0, TaskStage.FINAL)])
        lines = _idx(tmp_path).read_text().splitlines(keepends=True)
        _idx(tmp_path).write_text("")  # the FINAL line never landed
        assert len(lines) == 1
        assert _shard(tmp_path).stat().st_size > 0

        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert contexts[0].stage is TaskStage.PREPROCESSED

    @pytest.mark.anyio
    async def test_a_line_past_the_shard_end_is_dropped(self, tmp_path):
        await write_contexts(tmp_path, [make_ctx(0), make_ctx(1)])
        shard = _shard(tmp_path)
        first = shard.read_bytes().splitlines(keepends=True)[0]
        shard.write_bytes(first)  # the second record was lost

        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert sorted(contexts) == [0]

    @pytest.mark.anyio
    async def test_a_missing_shard_drops_every_line(self, tmp_path):
        await write_contexts(tmp_path, [make_ctx(0)])
        _shard(tmp_path).unlink()
        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert contexts == {}

    @pytest.mark.anyio
    async def test_an_earlier_record_survives_a_dropped_later_one(self, tmp_path):
        """Two lines for one sample; the later one points past the end."""
        await write_contexts(tmp_path, [make_ctx(0, TaskStage.FINAL)])
        await write_contexts(tmp_path, [make_ctx(0, TaskStage.FINAL, iteration=1)])
        later = tmp_path / "1" / "final" / "0.jsonl"
        later.write_bytes(b"")  # the later record never reached its shard

        contexts = await TaskLoader(
            task=make_mock_task(), root_dir=tmp_path
        ).load_initial_state()
        assert contexts[0].iteration == 0


class TestLoadCachedReport:
    @pytest.mark.anyio
    async def test_no_report(self, tmp_path):
        loader = TaskLoader(task=make_mock_task(), root_dir=tmp_path)
        assert await loader.load_cached_report() is None

    @pytest.mark.anyio
    async def test_valid_report(self, tmp_path):
        (tmp_path / "report.json").write_bytes(orjson.dumps({"accuracy": 0.95}))
        loader = TaskLoader(task=make_mock_task(), root_dir=tmp_path)
        assert await loader.load_cached_report() == {"accuracy": 0.95}

    @pytest.mark.anyio
    async def test_corrupt_report(self, tmp_path):
        (tmp_path / "report.json").write_text("not json{{{")
        loader = TaskLoader(task=make_mock_task(), root_dir=tmp_path)
        assert await loader.load_cached_report() is None
