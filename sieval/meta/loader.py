"""
Parse `sieval/meta/index.json` into DatasetMeta / TaskMeta objects.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import json
from functools import lru_cache
from importlib.resources import files

from sieval.core.datasets.meta import DatasetMeta, dataset_meta_from_dict
from sieval.core.tasks.meta import TaskMeta, task_meta_from_dict

_INDEX_RESOURCE = "index.json"


@lru_cache(maxsize=1)
def load_index() -> tuple[list[DatasetMeta], list[TaskMeta]]:
    """Parse ``sieval/meta/index.json``; return (datasets, tasks) in index order.

    Cached because the file is release-static and the returned dataclasses
    are frozen; tests mocking the file must call ``load_index.cache_clear()``.
    Raises ``RuntimeError`` on unsupported ``schema_version``.
    """
    text = files(__package__).joinpath(_INDEX_RESOURCE).read_text(encoding="utf-8")
    payload = json.loads(text)
    version = payload.get("schema_version")
    if version != 1:
        raise RuntimeError(
            f"sieval/meta/index.json schema_version={version!r} is not supported "
            f"by this sieval install (expected 1). Upgrade sieval or regenerate "
            f"the index with the matching version."
        )
    datasets = [dataset_meta_from_dict(d) for d in payload["datasets"]]
    tasks = [task_meta_from_dict(t) for t in payload["tasks"]]
    return datasets, tasks
