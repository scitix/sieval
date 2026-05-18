"""
Shared helpers for tests/unit/core/tasks/loader/ sub-package.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from unittest.mock import MagicMock

from sieval.core.tasks.context import TaskContext, TaskStage
from sieval.core.tasks.saver import TaskSaver


def make_mock_task(name="test_task"):
    task = MagicMock()
    task.name = name
    task.__class__.__module__ = "sieval.core.tasks.task"
    task.__class__.__name__ = "MockTask"
    task.make_context.side_effect = lambda sid, raw: TaskContext(
        sample_id=sid, raw_sample=raw
    )
    task.dataset = MagicMock()
    task.dataset.test_set = None
    return task


def make_ctx(sid, stage=TaskStage.FINAL, iteration=0, **kwargs):
    return TaskContext(
        sample_id=sid,
        raw_sample={"q": f"question_{sid}"},
        stage=stage,
        iteration=iteration,
        **kwargs,
    )


async def write_contexts(root_dir, contexts, shard_samples=1024):
    saver = TaskSaver(
        root_dir=root_dir,
        shard_samples=shard_samples,
        record_type_metadata=True,
        record_meta=True,
    )
    for ctx in contexts:
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)
    await saver.flush()
    return saver


async def write_snapshot(root_dir, ctx):
    saver = TaskSaver(
        root_dir=root_dir,
        shard_samples=1024,
        record_type_metadata=True,
        record_meta=True,
    )
    saver._update_manifest_entry(ctx)
    saver._stage_queue.append(ctx)
    await saver.flush()
