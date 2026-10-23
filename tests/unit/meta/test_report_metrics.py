"""The built-in report metric table, against the registry and the identities.

Each task test's report is checked against its entry as it runs
(``tests/unit/tasks/conftest.py``); this file holds what no single report shows:
every built-in task is declared, no entry is dead, and declaring moves no
stage identity.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import pkgutil

import pytest

import sieval.tasks
from sieval.core.tasks.meta import _TASK_CLASSES
from sieval.core.tasks.report_metrics import RUNNER_GEN_METRICS, class_path
from sieval.core.tasks.run_meta import GRADER_PACKAGES
from sieval.core.tasks.task import Task
from sieval.meta.report_metrics import BUILTIN_REPORT_METRICS, task_report_metrics

type Registry = dict[str, type[Task]]


@pytest.fixture(scope="module")
def registry() -> Registry:
    for module in pkgutil.walk_packages(sieval.tasks.__path__, "sieval.tasks."):
        importlib.import_module(module.name)
    return dict(_TASK_CLASSES)


def test_every_registered_task_is_declared(registry: Registry) -> None:
    unknown = sorted(
        name for name, cls in registry.items() if task_report_metrics(cls) is None
    )
    assert unknown == []


def test_every_entry_names_a_class_that_owns_a_report(
    registry: dict[str, type],
) -> None:
    used = {
        class_path(klass)
        for cls in registry.values()
        for klass in cls.__mro__
        if class_path(klass) in BUILTIN_REPORT_METRICS
    }
    assert sorted(set(BUILTIN_REPORT_METRICS) - used) == []


def test_every_task_declares_its_headline(registry: Registry) -> None:
    """``score`` is what a smoke criterion names first; only T-Eval has none."""
    headless = sorted(
        name
        for name, cls in registry.items()
        if (declared := task_report_metrics(cls)) is not None
        and "score" not in declared.names
    )
    assert headless == ["t_eval_before_calling_0shot_gen"]


def test_a_generative_task_gains_the_runner_keys(
    registry: Registry,
) -> None:
    for name, cls in registry.items():
        declared = task_report_metrics(cls)
        assert declared is not None
        assert (declared.names >= RUNNER_GEN_METRICS) == ("gen" in cls.tags), name


def test_the_table_is_outside_every_stage_identity(
    registry: Registry,
) -> None:
    """A task module's source is in its stage identities; this module is not.

    Declaring on each class would rerun every resumed directory from
    ``postprocess``. The grader walk follows only :data:`GRADER_PACKAGES`.
    """
    assert not any(
        root == "sieval.meta" or "sieval.meta".startswith(root + ".")
        for root in GRADER_PACKAGES
    )
    # And no built-in declares on its class, which the walk would digest.
    declaring = sorted(
        class_path(klass)
        for cls in registry.values()
        for klass in cls.__mro__
        if "report_metrics" in vars(klass)
    )
    assert declaring == []
