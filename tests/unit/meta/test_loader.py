"""Unit tests for `sieval.meta.loader`.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from sieval.core.datasets.meta import DatasetMeta, Level1Category
from sieval.core.tasks.meta import EvalMode, TaskMeta
from sieval.meta import load_index


def test_load_index_returns_datasets_and_tasks():
    datasets, tasks = load_index()
    # Pilot coverage is 11 datasets + 11 tasks at v0.1.
    assert len(datasets) >= 11
    assert len(tasks) >= 11
    assert all(isinstance(d, DatasetMeta) for d in datasets)
    assert all(isinstance(t, TaskMeta) for t in tasks)


def test_loader_does_not_walk_dataset_modules():
    """Pilot extras (chembench → torch, bioprobench → sentence-transformers)
    must NOT be imported as a side effect of `load_index()` — otherwise the
    "list without extras" story is dead."""
    import sys

    # sentence_transformers is a heavy transitive dep from a non-pilot module
    # (bioprobench); if the loader walked the package tree it would land in
    # sys.modules. Any heavy optional dep would do; pick one unlikely to be
    # dragged in by core imports.
    for probe in ("sentence_transformers", "rdkit", "e3fp"):
        sys.modules.pop(probe, None)
    load_index()
    for probe in ("sentence_transformers", "rdkit", "e3fp"):
        assert probe not in sys.modules, (
            f"{probe} got imported during load_index() — loader is walking "
            "the dataset package tree, defeating the purpose of meta/"
        )


def test_aime_2024_dataset_round_trips():
    datasets, _ = load_index()
    aime = next((d for d in datasets if d.name == "aime_2024"), None)
    assert aime is not None
    assert aime.display_name == "AIME 2024"
    assert aime.license == "Apache-2.0"
    # Single-source on wire = 1-element list → reconstructed as 1-tuple.
    assert aime.source == (
        "hf:HuggingFaceH4/aime_2024@2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    )
    assert any(c.level1 is Level1Category.MATHEMATICS for c in aime.categories)


def test_drop_multi_source_round_trips():
    datasets, _ = load_index()
    drop = next((d for d in datasets if d.name == "drop"), None)
    assert drop is not None
    assert len(drop.source) == 2
    assert all(s.startswith("url:") for s in drop.source)


def test_aime_2024_task_round_trips():
    _, tasks = load_index()
    t = next((x for x in tasks if x.name == "aime_2024_0shot_gen"), None)
    assert t is not None
    assert t.dataset == "aime_2024"
    assert t.eval_mode is EvalMode.GEN
    assert t.n_shot == 0
    assert t.status == "stable"


def test_load_index_rejects_wrong_schema_version():
    """A future schema_version=2 index must fail loudly on an older CLI
    install, not be silently mis-parsed."""
    import json
    from importlib.resources import files as _files
    from unittest.mock import patch

    import pytest

    from sieval.meta.loader import load_index

    real_text = _files("sieval.meta").joinpath("index.json").read_text("utf-8")
    payload = json.loads(real_text)
    payload["schema_version"] = 2
    fake_text = json.dumps(payload)

    class FakeTraversable:
        def read_text(self, encoding="utf-8"):
            return fake_text

    class FakeAnchor:
        def joinpath(self, name):
            return FakeTraversable()

    # load_index is lru_cached — earlier tests populated the cache with the
    # real index, and the mocked `files` below would otherwise never be hit.
    load_index.cache_clear()
    try:
        with (
            patch("sieval.meta.loader.files", return_value=FakeAnchor()),
            pytest.raises(RuntimeError, match="schema_version=2"),
        ):
            load_index()
    finally:
        # Don't leak the aborted-read state into later tests.
        load_index.cache_clear()


def test_load_index_accepts_schema_version_1():
    """Smoke: the real shipped index (schema_version=1) loads fine."""
    from sieval.meta.loader import load_index

    datasets, tasks = load_index()
    assert len(datasets) > 0
    assert len(tasks) > 0
