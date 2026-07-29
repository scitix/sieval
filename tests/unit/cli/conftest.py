"""Shared fixtures for sieval.cli.{dataset,task} unit tests.

Restores the dataset/task registries and the `sieval.datasets.*` /
`sieval.tasks.*` module cache after each test, so a test that triggers
`import_all_datasets/tasks()` cannot leave the next one with a half-moved pair.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.datasets.meta import DATASET_REGISTRY, SAMPLE_TO_DATASET
from sieval.core.tasks.meta import _TASK_CLASSES, TASK_REGISTRY
from tests.conftest import ModuleIsolation

# `sieval.datasets.downloaders.*` holds no `@sieval_dataset` decorators — it's the
# handler registry, and it has to stay out of scope entirely. `downloaders.base`
# caches handler *instances* in a module-level `_HANDLERS`, and
# `sieval.cli.dataset.{commands,render}` bind `resolve` at their own import time,
# which is outside this scope and so never re-binds. Recreate the subtree and the
# CLI keeps dispatching through the first `_HANDLERS` while
# `patch("...url.URLHandler.download")` — resolved by attribute traversal — lands
# on the new class, silently bypassing the mock and hitting the real network.
# Restoring the originals cannot repair that: the pinned reference predates us.
_DOWNLOADERS = "sieval.datasets.downloaders"


@pytest.fixture(autouse=True)
def _cleanup_registries_and_modules():
    """Save registry state + module cache; restore both after each test.

    Restore-only by design — unlike the isolation fixtures in
    `tests/unit/core/{tasks,datasets}/test_meta.py`, nothing here is cleared up
    front: these tests exercise the CLI against the real registries. See
    `ModuleIsolation` for why the module cache has to be restored alongside them.
    """
    saved_ds = dict(DATASET_REGISTRY)
    saved_map = dict(SAMPLE_TO_DATASET)
    saved_tasks = dict(TASK_REGISTRY)
    saved_task_classes = dict(_TASK_CLASSES)
    modules = ModuleIsolation(
        ("sieval.datasets.", "sieval.tasks."),
        lazy_packages=("sieval.datasets", "sieval.tasks"),
        exclude=(_DOWNLOADERS, f"{_DOWNLOADERS}."),
    )
    modules.snapshot()
    try:
        yield
    finally:
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(saved_ds)
        SAMPLE_TO_DATASET.clear()
        SAMPLE_TO_DATASET.update(saved_map)
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(saved_tasks)
        _TASK_CLASSES.clear()
        _TASK_CLASSES.update(saved_task_classes)
        modules.restore()


@pytest.fixture(autouse=True)
def _deterministic_help_output(monkeypatch):
    """Make CLI ``--help`` output deterministic across environments.

    Plain-substring assertions on help text (e.g. ``"--model" in result.output``)
    are environment-dependent for two reasons:

    * **Color** — under GitHub Actions (``GITHUB_ACTIONS=true``), typer sets its
      module-level ``FORCE_TERMINAL=True`` / ``COLOR_SYSTEM`` at import time, so
      Rich colorizes help and interleaves ANSI escapes *inside* option names —
      ``"--model"`` is no longer a contiguous substring. ``force_terminal=True``
      also overrides ``NO_COLOR``, so an env var can't fix it. Locally there is
      no TTY and no such forcing, so color is off and the asserts pass — hence
      the CI-only failures.
    * **Width** — Rich truncates long option names (e.g. ``--deterministic``)
      at narrow widths.

    ``typer.rich_utils._get_rich_console`` reads these constants on every call,
    so patching them makes help plain and wide for each test, deterministically.
    """
    import typer.rich_utils as _ru

    monkeypatch.setattr(_ru, "FORCE_TERMINAL", None, raising=False)
    monkeypatch.setattr(_ru, "COLOR_SYSTEM", None, raising=False)
    monkeypatch.setattr(_ru, "MAX_WIDTH", 200, raising=False)
