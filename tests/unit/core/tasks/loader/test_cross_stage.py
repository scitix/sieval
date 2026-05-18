"""
Tests for cross-stage dependency loading during hydration.

When record_each_stage=True, resuming at a later stage must automatically
load earlier-stage results from their per-stage snapshot shards.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import orjson
import pytest

from sieval.core.tasks.context import TaskContext, TaskStage
from sieval.core.tasks.loader import TaskLoader

from .conftest import make_mock_task, write_snapshot


class TestCrossStageDependencyLoading:
    @pytest.mark.anyio
    async def test_dependency_loading_merges_previous_stages(self, tmp_path):
        """With record_each_stage=True, hydration merges previous stage snapshots.

        Key insight: snapshots only contain the current stage's result field.
        When resuming at FEEDBACK, the main hydrate reads only feedback_result
        from the FEEDBACK snapshot. The dependency loader must separately read
        PREPROCESSED, INFERRED, and POSTPROCESSED snapshots to fill in the
        earlier results. This test verifies that mechanism.
        """
        root = tmp_path / "dep_load"

        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_value"),
            (TaskStage.INFERRED, "infer_result", "inf_value"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_value"),
            (TaskStage.FEEDBACK, "feedback_result", {"correct": True}),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()
        assert loaded[0].stage == TaskStage.FEEDBACK

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded,
            hydrated_ids,
            include_stages={TaskStage.FEEDBACK},
            record_each_stage=True,
        )

        h = loaded[0]
        assert h.feedback_result == {"correct": True}
        # Dependency loader must fill in earlier stages
        assert h.preprocess_result == "pre_value"
        assert h.infer_result == "inf_value"
        assert h.postprocess_result == "post_value"

    @pytest.mark.anyio
    async def test_dependency_loading_for_intermediate_stage(self, tmp_path):
        """Resuming at POSTPROCESSED loads PREPROCESSED and INFERRED dependencies."""
        root = tmp_path / "dep_intermediate"

        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_value"),
            (TaskStage.INFERRED, "infer_result", "inf_value"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_value"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded,
            hydrated_ids,
            include_stages={TaskStage.POSTPROCESSED},
            record_each_stage=True,
        )

        h = loaded[0]
        assert h.stage == TaskStage.POSTPROCESSED
        assert h.preprocess_result == "pre_value"
        assert h.infer_result == "inf_value"
        assert h.postprocess_result == "post_value"

    @pytest.mark.anyio
    async def test_dependency_loading_prefers_latest_iteration_from_idx(
        self, tmp_path, monkeypatch
    ):
        """Trusted-manifest path must pick latest-iteration dependency offsets."""
        root = tmp_path / "dep_idx_latest_iter"

        # iteration 0 snapshots
        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_iter0"),
            (TaskStage.INFERRED, "infer_result", "inf_iter0"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_iter0"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                iteration=0,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        # iteration 1 snapshots (latest)
        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_iter1"),
            (TaskStage.INFERRED, "infer_result", "inf_iter1"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_iter1"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                iteration=1,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        # report.json present + manifest exists -> _load_offsets_from_idx path
        (root / "report.json").write_bytes(orjson.dumps({"ok": True}))

        # Force idx scan order where iteration 0 appears later.
        idx_files = [
            root / "1" / "preprocessed" / "0.idx",
            root / "1" / "inferred" / "0.idx",
            root / "1" / "postprocessed" / "0.idx",
            root / "0" / "preprocessed" / "0.idx",
            root / "0" / "inferred" / "0.idx",
            root / "0" / "postprocessed" / "0.idx",
        ]
        ordered_pairs = [(p, p.with_suffix(".jsonl")) for p in idx_files]

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        monkeypatch.setattr(loader, "_collect_idx_files", lambda: ordered_pairs)

        loaded = await loader.load_initial_state()
        hydrated_ids: set = set()
        await loader.hydrate(
            loaded,
            hydrated_ids,
            include_stages={TaskStage.POSTPROCESSED},
            record_each_stage=True,
        )

        h = loaded[0]
        assert h.iteration == 1
        assert h.preprocess_result == "pre_iter1"
        assert h.infer_result == "inf_iter1"
        assert h.postprocess_result == "post_iter1"

    @pytest.mark.anyio
    async def test_dependency_loading_prefers_latest_iteration_during_compensate(
        self, tmp_path, monkeypatch
    ):
        """Compensation path must pick latest-iteration dependency offsets."""
        root = tmp_path / "dep_comp_latest_iter"

        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_iter0"),
            (TaskStage.INFERRED, "infer_result", "inf_iter0"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_iter0"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                iteration=0,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_iter1"),
            (TaskStage.INFERRED, "infer_result", "inf_iter1"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_iter1"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                iteration=1,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        # manifest missing -> _compensate_scan path
        (root / "manifest.json").unlink()

        idx_files = [
            root / "1" / "preprocessed" / "0.idx",
            root / "1" / "inferred" / "0.idx",
            root / "1" / "postprocessed" / "0.idx",
            root / "0" / "preprocessed" / "0.idx",
            root / "0" / "inferred" / "0.idx",
            root / "0" / "postprocessed" / "0.idx",
        ]
        ordered_pairs = [(p, p.with_suffix(".jsonl")) for p in idx_files]

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        monkeypatch.setattr(loader, "_collect_idx_files", lambda: ordered_pairs)

        loaded = await loader.load_initial_state()
        hydrated_ids: set = set()
        await loader.hydrate(
            loaded,
            hydrated_ids,
            include_stages={TaskStage.POSTPROCESSED},
            record_each_stage=True,
        )

        h = loaded[0]
        assert h.iteration == 1
        assert h.preprocess_result == "pre_iter1"
        assert h.infer_result == "inf_iter1"
        assert h.postprocess_result == "post_iter1"

    @pytest.mark.anyio
    async def test_no_dependency_loading_when_record_each_stage_false(self, tmp_path):
        """With record_each_stage=False, dependency loading is skipped.

        The earlier stage snapshots should NOT be merged, so preprocess_result
        and infer_result remain None after hydrating POSTPROCESSED.
        """
        root = tmp_path / "dep_no_load"

        for stage, field, value in [
            (TaskStage.PREPROCESSED, "preprocess_result", "pre_value"),
            (TaskStage.INFERRED, "infer_result", "inf_value"),
            (TaskStage.POSTPROCESSED, "postprocess_result", "post_value"),
        ]:
            snap = TaskContext(
                sample_id=0,
                raw_sample={"q": "test"},
                stage=stage,
                **{field: value},  # type: ignore
            ).make_snapshot()
            await write_snapshot(root, snap)

        (root / "manifest.json").unlink()

        task = make_mock_task()
        loader = TaskLoader(task=task, root_dir=root)
        loaded = await loader.load_initial_state()

        hydrated_ids: set = set()
        await loader.hydrate(
            loaded,
            hydrated_ids,
            include_stages={TaskStage.POSTPROCESSED},
            record_each_stage=False,
        )

        h = loaded[0]
        assert h.postprocess_result == "post_value"
        # Dependency loading is disabled → earlier stages must be None
        assert h.preprocess_result is None
        assert h.infer_result is None
