"""Shared fixtures for sieval.cli.{dataset,task} unit tests.

Tracks and cleans up `sieval.datasets.*` and `sieval.tasks.*` modules loaded
during the test session, and restores the dataset/task registries after each
test. This prevents cross-test pollution where pre-loaded modules prevent
`import_all_datasets/tasks()` from re-registering in downstream tests.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import sys
from contextlib import suppress

import pytest

from sieval.core.datasets.meta import DATASET_REGISTRY, SAMPLE_TO_DATASET
from sieval.core.tasks.meta import TASK_REGISTRY

_LAZY_DATASET_SUFFIXES = ("Dataset", "DatasetSample", "CSVSample")
_LAZY_TASK_SUFFIXES = ("Task",)

# `sieval.datasets.downloaders.*` holds no `@sieval_dataset` decorators — it's
# the handler registry. Evicting it desyncs the `URLHandler`/`HFHandler` class
# identities from the instances cached in `downloaders.base._HANDLERS` (held
# alive by the `from sieval.datasets.downloaders import resolve` closure in
# `sieval.cli.dataset.{commands,render}`). Downstream tests that
# `patch("...url.URLHandler.download")` then patch the *re-imported* class
# while `_HANDLERS` still dispatches through the pre-eviction one, so the
# mock is silently bypassed. Exclude the downloader subtree from eviction.
_DOWNLOADER_PREFIX = "sieval.datasets.downloaders"


def _clear_lazy_cache(pkg_name: str, suffixes: tuple[str, ...]) -> None:
    """Remove lazy-resolved names from a package's global dict."""

    try:
        pkg = sys.modules.get(pkg_name)
        if pkg is None:
            return
        for key in list(vars(pkg)):
            if not key.startswith("_") and key.endswith(suffixes):
                with suppress(AttributeError):
                    delattr(pkg, key)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _cleanup_registries_and_modules():
    """Save registry state + pre-loaded modules; restore all after each test."""
    saved_ds = dict(DATASET_REGISTRY)
    saved_map = dict(SAMPLE_TO_DATASET)
    saved_tasks = dict(TASK_REGISTRY)
    preloaded = {
        name
        for name in sys.modules
        if name.startswith("sieval.datasets.") or name.startswith("sieval.tasks.")
    }
    yield
    # Restore registries
    DATASET_REGISTRY.clear()
    DATASET_REGISTRY.update(saved_ds)
    SAMPLE_TO_DATASET.clear()
    SAMPLE_TO_DATASET.update(saved_map)
    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(saved_tasks)
    # Unload any newly loaded sieval.datasets.* / sieval.tasks.* submodules
    # (excluding the downloader subtree — see _DOWNLOADER_PREFIX note above).
    for name in list(sys.modules):
        if (
            (name.startswith("sieval.datasets.") or name.startswith("sieval.tasks."))
            and name not in preloaded
            and not name.startswith(_DOWNLOADER_PREFIX)
        ):
            del sys.modules[name]
    # Clear lazy-resolved globals in the package __init__ modules so that
    # subsequent tests that call import_all_datasets/tasks() can re-trigger
    # the @register_dataset/@register_task decorators properly.
    _clear_lazy_cache("sieval.datasets", _LAZY_DATASET_SUFFIXES)
    _clear_lazy_cache("sieval.tasks", _LAZY_TASK_SUFFIXES)
