"""
Tests for TaskSaver → TaskLoader integration: write-then-read roundtrips,
idx offset correctness, and compensation (manifest rebuild from shards).

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import json
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from sieval.core.tasks.context import TaskAction, TaskContext, TaskStage
from sieval.core.tasks.loader import TaskLoader

from .conftest import make_ctx, make_mock_task, write_contexts, write_snapshot


def _write_manual_preprocessed_record(
    root, shard_id: int, preprocess_value: str, sample_id: int = 0, iteration: int = 0
) -> None:
    stage_dir = root / str(iteration) / TaskStage.PREPROCESSED.value
    stage_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "sample_id": sample_id,
        "iteration": iteration,
        "stage": TaskStage.PREPROCESSED.value,
        "preprocess_result": preprocess_value,
    }
    line = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")

    shard_path = stage_dir / f"{shard_id}.jsonl"
    idx_path = stage_dir / f"{shard_id}.idx"
    shard_path.write_bytes(line)
    idx_path.write_text(
        f"{sample_id}\t{iteration}\t{TaskStage.PREPROCESSED.value}\t0\t{len(line)}\t\t\t0\n",
        encoding="utf-8",
    )


class TestSaverLoaderIntegration:
    @pytest.mark.anyio
    async def test_write_then_load_manifest(self, tmp_path):
        root = tmp_path / "integration"
        contexts = [
            make_ctx(0, TaskStage.FINAL),
            make_ctx(1, TaskStage.FINAL),
            make_ctx(
                2,
                TaskStage.FAILED,
                error_action=TaskAction.INFER,
                error_reason="timeout",
            ),
        ]
        await write_contexts(root, contexts)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert len(loaded) == 3
        assert loaded[0].stage == TaskStage.FINAL
        assert loaded[2].stage == TaskStage.FAILED

    @pytest.mark.anyio
    async def test_write_then_hydrate(self, tmp_path):
        root = tmp_path / "integration"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"question": "What is 2+2?"},
            stage=TaskStage.FINAL,
            preprocess_result="formatted",
            infer_result="4",
            postprocess_result="4",
            feedback_result={"correct": True},
        )
        await write_contexts(root, [ctx])

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        assert 0 in hydrated_ids
        h = loaded[0]
        assert h.postprocess_result == "4"
        assert h.feedback_result == {"correct": True}

    @pytest.mark.anyio
    async def test_idx_offsets_roundtrip(self, tmp_path):
        """Idx offsets written by saver can be used by loader to hydrate correctly."""
        root = tmp_path / "integration"
        contexts = [make_ctx(i, TaskStage.FINAL) for i in range(5)]
        await write_contexts(root, contexts, shard_samples=1024)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        assert len(hydrated_ids) == 5

    @pytest.mark.anyio
    async def test_load_offsets_prefers_latest_offset_same_stage(self, tmp_path):
        """
        Trusted-manifest path should pick the latest append offset when
        sample/stage/iteration are identical.
        """
        root = tmp_path / "latest_offset_idx"
        ctx_old = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            iteration=0,
            preprocess_result="old",
        ).make_snapshot()
        await write_snapshot(root, ctx_old)

        ctx_new = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            iteration=0,
            preprocess_result="new",
        ).make_snapshot()
        await write_snapshot(root, ctx_new)

        # Force trusted-manifest path (_load_offsets_from_idx).
        (root / "report.json").write_text("{}", encoding="utf-8")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded, hydrated_ids, include_stages={TaskStage.PREPROCESSED}
        )
        assert loaded[0].preprocess_result == "new"

    @pytest.mark.anyio
    async def test_load_offsets_cross_shard_tiebreak_is_deterministic(self, tmp_path):
        """
        Trusted-manifest path should deterministically pick one offset when
        sample/stage/iteration/offset are identical across different shards.
        """
        root = tmp_path / "cross_shard_idx_tiebreak"
        _write_manual_preprocessed_record(root, shard_id=10, preprocess_value="old")
        _write_manual_preprocessed_record(root, shard_id=20, preprocess_value="new")

        manifest = [
            {
                "sample_id": 0,
                "iteration": 0,
                "stage": TaskStage.PREPROCESSED.value,
                "final": False,
                "failed": False,
            }
        ]
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "report.json").write_text("{}", encoding="utf-8")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded, hydrated_ids, include_stages={TaskStage.PREPROCESSED}
        )
        assert loaded[0].preprocess_result == "new"


class TestCompensationMechanism:
    @pytest.mark.anyio
    async def test_rebuild_manifest_when_manifest_missing(self, tmp_path):
        """When manifest.json is deleted, _compensate_scan rebuilds from idx files."""
        root = tmp_path / "compensate"
        contexts = [make_ctx(i, TaskStage.FINAL) for i in range(3)]
        await write_contexts(root, contexts)
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert len(loaded) == 3
        for sid in (0, 1, 2):
            assert loaded[sid].stage == TaskStage.FINAL

    @pytest.mark.anyio
    async def test_rebuild_manifest_preserves_failed_samples(self, tmp_path):
        """Compensation scan correctly restores FAILED samples with error info."""
        root = tmp_path / "compensate_fail"
        contexts = [
            make_ctx(0, TaskStage.FINAL),
            make_ctx(
                1,
                TaskStage.FAILED,
                error_action=TaskAction.INFER,
                error_reason="exception::TimeoutError",
                retry_count=2,
            ),
        ]
        await write_contexts(root, contexts)
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert loaded[0].stage == TaskStage.FINAL
        assert loaded[1].stage == TaskStage.FAILED
        assert loaded[1].error_reason == "exception::TimeoutError"
        assert loaded[1].retry_count == 2

    @pytest.mark.anyio
    async def test_compensate_picks_latest_stage(self, tmp_path):
        """
        When a sample has records at multiple stages, compensation picks the latest.
        """
        root = tmp_path / "compensate_latest"

        ctx_pre = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            preprocess_result="pre_result",
        ).make_snapshot()
        await write_contexts(root, [ctx_pre])

        ctx_final = make_ctx(0, TaskStage.FINAL)
        await write_snapshot(root, ctx_final)
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert loaded[0].stage == TaskStage.FINAL

    @pytest.mark.anyio
    async def test_compensate_picks_latest_iteration(self, tmp_path):
        """When a sample has records at different iterations, picks the latest."""
        root = tmp_path / "compensate_iter"

        ctx_iter0 = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.FEEDBACK,
            iteration=0,
            feedback_result={"correct": False},
        )
        await write_contexts(root, [ctx_iter0])

        ctx_iter1 = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.FINAL,
            iteration=1,
            feedback_result={"correct": True},
        )
        await write_snapshot(root, ctx_iter1)
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        assert loaded[0].iteration == 1
        assert loaded[0].stage == TaskStage.FINAL

    @pytest.mark.anyio
    async def test_compensate_with_empty_directory(self, tmp_path):
        """Compensation with empty directory returns empty state."""
        root = tmp_path / "compensate_empty"
        root.mkdir(parents=True)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()
        assert loaded == {}

    @pytest.mark.anyio
    async def test_compensate_prefers_latest_offset_same_stage(self, tmp_path):
        """
        Compensation path should pick the latest append offset when
        sample/stage/iteration are identical.
        """
        root = tmp_path / "latest_offset_compensate"
        ctx_old = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            iteration=0,
            preprocess_result="old",
        ).make_snapshot()
        await write_snapshot(root, ctx_old)

        ctx_new = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.PREPROCESSED,
            iteration=0,
            preprocess_result="new",
        ).make_snapshot()
        await write_snapshot(root, ctx_new)

        # Force compensation path (_compensate_scan).
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded, hydrated_ids, include_stages={TaskStage.PREPROCESSED}
        )
        assert loaded[0].preprocess_result == "new"

    @pytest.mark.anyio
    async def test_compensate_cross_shard_tiebreak_is_deterministic(self, tmp_path):
        """
        Compensation path should use the same deterministic cross-shard tie-break.
        """
        root = tmp_path / "cross_shard_comp_tiebreak"
        _write_manual_preprocessed_record(root, shard_id=10, preprocess_value="old")
        _write_manual_preprocessed_record(root, shard_id=20, preprocess_value="new")

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded, hydrated_ids, include_stages={TaskStage.PREPROCESSED}
        )
        assert loaded[0].preprocess_result == "new"

    @pytest.mark.anyio
    async def test_compensate_rebuilds_hydrateable_offsets(self, tmp_path):
        """After compensation, hydration should work using rebuilt offsets."""
        root = tmp_path / "compensate_hydrate"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"question": "What is 2+2?"},
            stage=TaskStage.FINAL,
            preprocess_result="formatted",
            infer_result="4",
            postprocess_result="4",
            feedback_result={"correct": True},
        )
        await write_contexts(root, [ctx])
        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(loaded, hydrated_ids, include_stages={TaskStage.FINAL})

        assert 0 in hydrated_ids
        assert loaded[0].postprocess_result == "4"
        assert loaded[0].feedback_result == {"correct": True}


class TestLoaderWarningPaths:
    """Exception paths in _parse_and_hydrate and _read_dependency_shard
    are non-fatal."""

    def test_parse_and_hydrate_field_error_continues(self, tmp_path):
        """If dict_to_obj raises for one field, other fields are still loaded."""
        import orjson

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)

        ctx = TaskContext(sample_id=0, raw_sample={})
        contexts = {0: ctx}
        hydrated_ids: set = set()

        record = {
            "sample_id": 0,
            "iteration": 0,
            "stage": "final",
            "preprocess_result": {"some": "data"},
            "postprocess_result": "safe_value",
        }
        blob = orjson.dumps(record)

        original_dict_to_obj = __import__(
            "sieval.core.tasks.loader", fromlist=["dict_to_obj"]
        ).dict_to_obj
        call_count = {"n": 0}

        def _fail_first(val, registry):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("forced deser error")
            return original_dict_to_obj(val, registry)

        with patch("sieval.core.tasks.loader.dict_to_obj", side_effect=_fail_first):
            # Should not raise
            loader._parse_and_hydrate(ctx, blob, contexts, hydrated_ids)

        assert 0 in hydrated_ids
        assert contexts[0].postprocess_result == "safe_value"

    @pytest.mark.anyio
    async def test_dependency_shard_io_error_is_non_fatal(self, tmp_path):
        """If a dependency shard fails to open, the error is caught."""
        root = tmp_path / "dep_io_err"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            preprocess_result="pre",
            infer_result="inf",
            postprocess_result="post",
            feedback_result="fb",
        )
        await write_contexts(root, [ctx])

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        import anyio as _anyio

        real_open_file = _anyio.open_file
        call_count = {"n": 0}

        async def _patched_open_file(path, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise OSError("simulated I/O error")
            return await real_open_file(path, *args, **kwargs)

        hydrated_ids: set = set()
        with patch("sieval.core.tasks.loader.anyio.open_file", _patched_open_file):
            # Should not raise even when dependency shard open fails
            await loader.hydrate(
                contexts,
                hydrated_ids,
                include_stages={TaskStage.FINAL, TaskStage.PREPROCESSED},
                record_each_stage=True,
            )

        # Main stage was hydrated (first open succeeded)
        assert 0 in hydrated_ids

    @pytest.mark.anyio
    async def test_primary_shard_io_error_is_non_fatal(self, tmp_path):
        """
        If the primary shard fails to open, the error is caught and hydration skips it.
        """
        root = tmp_path / "primary_io_err"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            preprocess_result="pre",
        )
        await write_contexts(root, [ctx])

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        async def _always_fail(_path, *_args, **_kwargs):
            raise OSError("simulated primary shard I/O error")

        hydrated_ids: set = set()
        with patch("sieval.core.tasks.loader.anyio.open_file", _always_fail):
            # Should not raise even when primary shard open fails
            await loader.hydrate(
                contexts,
                hydrated_ids,
                include_stages={TaskStage.FINAL},
            )

        # Nothing was hydrated because the shard couldn't be opened
        assert 0 not in hydrated_ids

    @pytest.mark.anyio
    async def test_hydrate_no_targets_still_prepares_retries(self, tmp_path):
        """When no targets are selected, prepare_retries path should still run."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "no_targets")
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
        )
        contexts = {0: ctx}

        with patch.object(loader, "_prepare_failed_retries") as prep_mock:
            await loader.hydrate(
                contexts,
                hydrated_ids=set(),
                include_stages={TaskStage.FINAL},
                prepare_retries=True,
                record_each_stage=False,
            )

        prep_mock.assert_called_once_with(contexts, False)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("prepare_retries", "expected_stage", "expected_retry_count"),
        [
            (False, TaskStage.FAILED, 0),
            (True, TaskStage.PREPROCESSED, 1),
        ],
        ids=["default_no_retry_prep", "retry_prep_uses_record_each_stage_default"],
    )
    async def test_hydrate_retry_preparation_defaults(
        self,
        tmp_path,
        prepare_retries,
        expected_stage,
        expected_retry_count,
    ):
        """
        Covers both defaults:
        - prepare_retries defaults to False.
        - when prepare_retries=True and record_each_stage is omitted,
          default True applies.
        """
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "hydrate_retry_defaults")
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
        )
        contexts = {0: ctx}

        await loader.hydrate(
            contexts,
            hydrated_ids=set(),
            include_stages={TaskStage.FINAL},
            prepare_retries=prepare_retries,
        )

        updated = contexts[0]
        assert updated.stage == expected_stage
        assert updated.retry_count == expected_retry_count

    @pytest.mark.anyio
    async def test_hydrate_stage_filter_requires_included_stage(self, tmp_path):
        """
        hydrate() should not hydrate samples whose stage is excluded even if
        they are not yet in hydrated_ids.
        """
        task = make_mock_task()
        root = tmp_path / "hydrate_stage_filter"
        await write_contexts(root, [make_ctx(0, TaskStage.FINAL)])
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        read_shard = AsyncMock()

        hydrated_ids: set = set()
        with patch.object(loader, "_read_shard_items", read_shard):
            await loader.hydrate(
                contexts,
                hydrated_ids,
                include_stages={TaskStage.FAILED},
            )

        assert hydrated_ids == set()
        read_shard.assert_not_awaited()

    @pytest.mark.anyio
    async def test_hydrate_missing_offset_does_not_block_other_targets(self, tmp_path):
        """
        A target without offset should be skipped, not stop processing later targets.
        """
        task = make_mock_task()
        root = tmp_path / "missing_offset_continue"
        await write_contexts(root, [make_ctx(1, TaskStage.FINAL)])
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()
        contexts = {
            0: TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.FINAL),
            1: loaded[1],
        }
        read_shard = AsyncMock()

        with patch.object(loader, "_read_shard_items", read_shard):
            await loader.hydrate(
                contexts,
                hydrated_ids=set(),
                include_stages={TaskStage.FINAL},
            )

        read_shard.assert_awaited_once()
        await_args = read_shard.await_args
        assert await_args is not None
        items = await_args.args[1]
        assert [ctx.sample_id for ctx, _, _ in items] == [1]

    @pytest.mark.anyio
    async def test_hydrate_missing_stage_offsets_entry_is_non_fatal(self, tmp_path):
        """
        Missing dependency-stage offsets should not break final-stage hydration.
        """
        task = make_mock_task()
        root = tmp_path / "missing_stage_offsets"
        await write_contexts(root, [make_ctx(0, TaskStage.FINAL)])
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        read_main = AsyncMock()
        read_dep = AsyncMock()

        with (
            patch.object(loader, "_read_shard_items", read_main),
            patch.object(loader, "_read_dependency_shard_items", read_dep),
        ):
            await loader.hydrate(
                contexts,
                hydrated_ids=set(),
                include_stages={TaskStage.FINAL},
                record_each_stage=True,
            )

        read_main.assert_awaited_once()
        read_dep.assert_not_awaited()

    @pytest.mark.anyio
    async def test_hydrate_dependency_selection_skips_current_stage_and_continues(
        self, tmp_path
    ):
        """
        Dependency hydration should:
        1) continue past missing earlier dependency stages, and
        2) never include the current stage itself.
        """
        task = make_mock_task()
        root = tmp_path / "dependency_selection"
        await write_contexts(
            root,
            [
                make_ctx(
                    0,
                    TaskStage.INFERRED,
                    iteration=2,
                    infer_result="inferred",
                ),
                make_ctx(
                    0,
                    TaskStage.POSTPROCESSED,
                    iteration=2,
                    postprocess_result="postprocessed",
                ),
            ],
        )
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        read_main = AsyncMock()
        read_dep = AsyncMock()
        with (
            patch.object(loader, "_read_shard_items", read_main),
            patch.object(loader, "_read_dependency_shard_items", read_dep),
        ):
            await loader.hydrate(
                contexts,
                hydrated_ids=set(),
                include_stages={TaskStage.POSTPROCESSED},
                record_each_stage=True,
            )

        read_dep.assert_awaited_once()
        dep_await_args = read_dep.await_args
        assert dep_await_args is not None
        dep_items = dep_await_args.args[1]
        assert [dep_stage for _, dep_stage, _, _ in dep_items] == [TaskStage.INFERRED]

    def test_parse_and_hydrate_invalid_json_is_non_fatal(self, tmp_path):
        """Corrupted JSON blobs should be ignored without mutating state."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "invalid_json")

        ctx = TaskContext(sample_id=0, raw_sample={})
        contexts = {0: ctx}
        hydrated_ids: set = set()

        loader._parse_and_hydrate(ctx, b"not valid json", contexts, hydrated_ids)

        assert hydrated_ids == set()
        assert contexts[0] is ctx

    @pytest.mark.parametrize(
        ("meta_payload", "expected_stage_meta"),
        [
            (
                {"stage_meta": {"inferred": [{"timing_s": 0.123}]}},
                {"inferred": [{"timing_s": 0.123}]},
            ),
            (
                {
                    "stage": TaskStage.INFERRED.value,
                    "meta_last": {"timing_s": 0.456},
                },
                {"inferred": [{"timing_s": 0.456}]},
            ),
        ],
        ids=["explicit_stage_meta", "meta_last_fallback"],
    )
    def test_parse_and_hydrate_restores_resume_fields(
        self, tmp_path, meta_payload, expected_stage_meta
    ):
        """Hydration should restore key resume fields, not silently drop them."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "hydrate_resume_fields")

        ctx = TaskContext(sample_id=0, raw_sample={"old": "value"})
        contexts = {0: ctx}
        hydrated_ids: set = set()

        record = {
            "sample_id": 0,
            "iteration": 3,
            "raw_sample": {"q": "current"},
            "error_action": TaskAction.INFER.value,
            "error_reason": "exception::TimeoutError",
            "error_msg": "timeout",
            "retry_count": 2,
            **meta_payload,
        }

        loader._parse_and_hydrate(
            ctx,
            orjson.dumps(record),
            contexts,
            hydrated_ids,
        )

        updated = contexts[0]
        assert 0 in hydrated_ids
        assert updated.raw_sample == {"q": "current"}
        assert updated.error_action == TaskAction.INFER
        assert updated.error_reason == "exception::TimeoutError"
        assert updated.error_msg == "timeout"
        assert updated.retry_count == 2
        assert updated.stage_meta == expected_stage_meta

    def test_parse_and_merge_dependency_adds_meta_last_for_missing_stage(
        self, tmp_path
    ):
        """Dependency hydration should add stage_meta when the stage key is missing."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "dep_meta")

        ctx = TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.POSTPROCESSED)
        contexts = {0: ctx}
        blob = orjson.dumps(
            {
                "preprocess_result": "pre_result",
                "meta_last": {"timing_s": 0.123},
            }
        )

        loader._parse_and_merge_dependency(ctx, TaskStage.PREPROCESSED, blob, contexts)

        updated = contexts[0]
        assert updated.preprocess_result == "pre_result"
        assert updated.stage_meta["preprocessed"] == [{"timing_s": 0.123}]

    def test_collect_idx_files_returns_empty_for_missing_root(self, tmp_path):
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path / "does_not_exist")

        assert loader._collect_idx_files() == []


class TestReadShardItemsErrorPaths:
    """_read_shard_items: already-hydrated skip, per-item error."""

    @pytest.mark.anyio
    async def test_skips_already_hydrated_samples(self, tmp_path):
        """Pre-populated hydrated_ids → sample skipped without re-read."""
        root = tmp_path / "skip_hydrated"
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "q"},
            stage=TaskStage.FINAL,
            preprocess_result="pre",
            feedback_result={"ok": True},
        )
        await write_contexts(root, [ctx])

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        hydrated_ids: set = {0}  # already hydrated
        original_ctx = contexts[0]
        await loader.hydrate(contexts, hydrated_ids, include_stages={TaskStage.FINAL})

        # Context should remain unchanged (not re-hydrated)
        assert contexts[0] is original_ctx

    @pytest.mark.anyio
    async def test_per_item_seek_error_continues(self, tmp_path):
        """Seek failure on one item doesn't stop hydration of others."""
        root = tmp_path / "per_item_err"
        contexts_to_write = [
            TaskContext(
                sample_id=i,
                raw_sample={"q": f"q{i}"},
                stage=TaskStage.FINAL,
                feedback_result={"i": i},
            )
            for i in range(3)
        ]
        await write_contexts(root, contexts_to_write)

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()

        import anyio as _anyio

        real_open = _anyio.open_file
        call_count = {"n": 0}

        async def _failing_open(path, *a, **kw):
            wrapper = await real_open(path, *a, **kw)
            real_seek = wrapper.seek

            async def _seek(off):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise OSError("simulated seek failure")
                return await real_seek(off)

            wrapper.seek = _seek
            return wrapper

        hydrated_ids: set = set()
        with patch("sieval.core.tasks.loader.anyio.open_file", _failing_open):
            await loader.hydrate(
                contexts, hydrated_ids, include_stages={TaskStage.FINAL}
            )

        # At least some samples hydrated despite one failure
        assert len(hydrated_ids) >= 1
        assert len(hydrated_ids) < 3  # not all, since one failed


class TestParseMergeDependencyErrorPaths:
    """Covers _parse_and_merge_dependency lines 371-406."""

    def test_invalid_json_returns_early(self, tmp_path):
        """Malformed dependency blob → early return, no crash."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)
        ctx = TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.POSTPROCESSED)
        contexts = {0: ctx}

        loader._parse_and_merge_dependency(
            ctx, TaskStage.PREPROCESSED, b"{malformed", contexts
        )
        assert contexts[0] is ctx  # unchanged

    def test_unmapped_stage_returns_early(self, tmp_path):
        """Stage not in STAGE_TO_RESULT_FIELD → early return."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)
        ctx = TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.FINAL)
        contexts = {0: ctx}

        blob = orjson.dumps({"some": "data"})
        # FINAL and INITIAL are not in STAGE_TO_RESULT_FIELD
        loader._parse_and_merge_dependency(ctx, TaskStage.FINAL, blob, contexts)
        assert contexts[0] is ctx

    def test_missing_context_returns_early(self, tmp_path):
        """Context removed from dict between offset load and merge → early return."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)
        ctx = TaskContext(sample_id=99, raw_sample={}, stage=TaskStage.POSTPROCESSED)
        contexts: dict = {}  # sample_id 99 not present

        blob = orjson.dumps({"preprocess_result": "value"})
        # Should not raise
        loader._parse_and_merge_dependency(ctx, TaskStage.PREPROCESSED, blob, contexts)
        assert len(contexts) == 0

    def test_deser_error_still_merges_meta_last(self, tmp_path):
        """dict_to_obj failure on result field → meta_last still merged."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)
        ctx = TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.POSTPROCESSED)
        contexts = {0: ctx}

        blob = orjson.dumps(
            {
                "preprocess_result": {"bad": "type"},
                "meta_last": {"timing_s": 0.5},
            }
        )

        with patch(
            "sieval.core.tasks.loader.dict_to_obj",
            side_effect=ValueError("deser fail"),
        ):
            loader._parse_and_merge_dependency(
                ctx, TaskStage.PREPROCESSED, blob, contexts
            )

        updated = contexts[0]
        assert updated.preprocess_result is None  # deser failed
        assert updated.stage_meta["preprocessed"] == [{"timing_s": 0.5}]

    def test_empty_blob_no_updates(self, tmp_path):
        """Blob with neither result field nor meta_last → context unchanged."""
        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=tmp_path)
        ctx = TaskContext(sample_id=0, raw_sample={}, stage=TaskStage.POSTPROCESSED)
        contexts = {0: ctx}

        blob = orjson.dumps({"sample_id": 0, "iteration": 0})
        loader._parse_and_merge_dependency(ctx, TaskStage.PREPROCESSED, blob, contexts)
        assert contexts[0] is ctx  # no replace() called
