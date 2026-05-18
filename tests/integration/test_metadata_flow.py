"""
Pattern: Metadata flow (implicit and explicit).

Covers "Metadata Patterns" (Pattern 1 & Pattern 2).

All tests verify metadata via disk round-trip (TaskLoader), not in-memory
runner._contexts, so they fail if record_meta or persistence is broken.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.models import ModelOutput
from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks import TaskStageOutput
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.task import Task
from sieval.core.utils.meta import build_stage_meta
from tests.conftest import MockChatModel, MockDataset, make_config

# ===================================================================
# Samples
# ===================================================================
SAMPLES = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+3?", "answer": "5"},
]


class ImplicitMetaTask(Task):
    """Task using Pattern 1: Implicit Metadata (return ModelOutput directly)."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        # Return ModelOutput directly → runner auto-captures metadata
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0}


class ExplicitMetaTask(Task):
    """Task using Pattern 2: Explicit Metadata (TaskStageOutput + build_stage_meta)."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        # Multiple model calls with explicit metadata aggregation
        output1 = await self.model.agenerate(pre)
        output2 = await self.model.agenerate(pre)

        combined_text = output1.texts[0] + "|" + output2.texts[0]
        meta = build_stage_meta(output1, output2, extra={"call_count": 2})
        return TaskStageOutput(
            value=ModelOutput(
                model=output1.model,
                texts=[combined_text],
                finish_reasons=["stop"],
            ),
            meta=meta,
        )

    async def postprocess(self, inf, ctx):
        # inf is a TaskStageOutput wrapping a ModelOutput
        return inf.value.texts[0].split("|")[0].strip()

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0}


class TestMetadataPersistence:
    """These tests confirm metadata discrimination power:
    they fail if record_meta=False or if stage_meta capture is broken,
    even when runner._contexts happens to hold the data in memory.
    """

    @pytest.mark.anyio
    async def test_implicit_metadata_persisted_to_disk(self, tmp_path):
        """Implicit metadata written to disk can be read back by a fresh TaskLoader."""

        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"What is 1+1?": "2", "What is 2+3?": "5"})
        task = ImplicitMetaTask(dataset=dataset, model=model, name="persist_implicit")
        config = make_config(tmp_path, record_meta=True, record_each_stage=True)

        runner = TaskRunner(task, config)
        report = await runner.arun()
        assert report["accuracy"] == 1.0

        root = runner.root_dir

        # Load from disk with a brand-new loader (no memory state)
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=True,
        )

        # Every FINAL sample must have stage_meta["inferred"] populated from disk
        for ctx in contexts.values():
            if ctx.stage == TaskStage.FINAL:
                inferred_meta = ctx.stage_meta.get("inferred", [])
                assert len(inferred_meta) > 0, (
                    "stage_meta['inferred'] must be loaded from disk — "
                    "if this fails, metadata was not persisted"
                )
                meta = inferred_meta[0]
                assert "timestamp" in meta, "timestamp must be in persisted meta"
                assert "model_calls" in meta, "model_calls must be in persisted meta"
                assert len(meta["model_calls"]) == 1
                assert meta["model_calls"][0]["usage"]["total_tokens"] == 12

    @pytest.mark.anyio
    async def test_explicit_metadata_aggregation_persisted_to_disk(self, tmp_path):
        """Explicit aggregated metadata (2 model calls) survives a disk round-trip."""

        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"What is 1+1?": "2", "What is 2+3?": "5"})
        task = ExplicitMetaTask(dataset=dataset, model=model, name="persist_explicit")
        config = make_config(tmp_path, record_meta=True, record_each_stage=True)

        runner = TaskRunner(task, config)
        await runner.arun()

        root = runner.root_dir
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=True,
        )

        for ctx in contexts.values():
            if ctx.stage == TaskStage.FINAL:
                inferred_meta = ctx.stage_meta.get("inferred", [])
                assert len(inferred_meta) > 0
                meta = inferred_meta[0]
                # Must have 2 model_calls from explicit aggregation
                assert len(meta["model_calls"]) == 2, (
                    "Expected 2 aggregated model calls from disk — "
                    "if this fails, explicit metadata aggregation is broken"
                )
                # User-provided extra must be preserved on disk
                assert meta.get("extra", {}).get("call_count") == 2, (
                    "extra.call_count must be 2 — "
                    "if this fails, extra metadata was dropped during serialization"
                )

    @pytest.mark.anyio
    async def test_metadata_disabled_not_persisted_to_disk(self, tmp_path):
        """With record_meta=False, stage_meta must be absent from disk records."""

        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"What is 1+1?": "2", "What is 2+3?": "5"})
        task = ImplicitMetaTask(dataset=dataset, model=model, name="no_meta")
        # Explicitly disable metadata recording
        config = make_config(tmp_path, record_meta=False, record_each_stage=True)

        runner = TaskRunner(task, config)
        await runner.arun()

        root = runner.root_dir
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=True,
        )

        # When record_meta=False, stage_meta must be empty on every loaded context
        for ctx in contexts.values():
            if ctx.stage == TaskStage.FINAL:
                assert ctx.stage_meta == {}, (
                    f"stage_meta should be empty dict when record_meta=False, "
                    f"got {ctx.stage_meta!r}"
                )

    @pytest.mark.anyio
    async def test_implicit_metadata_persisted_record_each_stage_false(self, tmp_path):
        """With record_each_stage=False, metadata in the FINAL record survives
        a disk round-trip.

        When record_each_stage=False, per-stage snapshots are not written.
        Only the complete FINAL context is saved, so metadata is stored in
        stage_meta (full dict) rather than per-stage snapshot meta_last fields.
        This test verifies that stage_meta["inferred"] is present and correct
        after loading the FINAL record from disk.
        """

        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"What is 1+1?": "2", "What is 2+3?": "5"})
        task = ImplicitMetaTask(
            dataset=dataset, model=model, name="persist_implicit_no_stage"
        )
        config = make_config(tmp_path, record_meta=True, record_each_stage=False)

        runner = TaskRunner(task, config)
        report = await runner.arun()
        assert report["accuracy"] == 1.0

        root = runner.root_dir
        loader = TaskLoader(task=task, root_dir=root)
        contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=False,
        )

        # With record_each_stage=False the FINAL record contains stage_meta
        # for all stages recorded during the run.
        for ctx in contexts.values():
            if ctx.stage == TaskStage.FINAL:
                inferred_meta = ctx.stage_meta.get("inferred", [])
                assert len(inferred_meta) > 0, (
                    "stage_meta['inferred'] must be present in the FINAL record "
                    "even when record_each_stage=False — "
                    "if this fails, metadata was lost during serialization."
                )
                meta = inferred_meta[0]
                assert "timestamp" in meta, "timestamp must be in persisted meta"
                assert "model_calls" in meta, "model_calls must be in persisted meta"
                assert len(meta["model_calls"]) == 1
                assert meta["model_calls"][0]["usage"]["total_tokens"] == 12
