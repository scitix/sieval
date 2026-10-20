"""Every task that posts to the code-eval service declares it.

A task reading ``SIEVAL_CODE_EVAL_API`` without declaring ``code-eval`` would
run against nothing when the variable is unset, and its grader identity would
not say which copy graded it.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import inspect
from pathlib import Path

import sieval.tasks as tasks_pkg
from sieval.core.tasks import Task
from sieval.core.tasks.services import CODE_EVAL, SERVICE_ENDPOINT_ENV

_ROOT = Path(tasks_pkg.__file__).parent


def _modules_reading_the_endpoint() -> list[str]:
    variable = SERVICE_ENDPOINT_ENV[CODE_EVAL]
    return sorted(
        "sieval.tasks." + ".".join(p.relative_to(_ROOT).with_suffix("").parts)
        for p in _ROOT.rglob("*.py")
        if variable in p.read_text()
    )


def test_every_task_reading_the_endpoint_declares_code_eval() -> None:
    modules = _modules_reading_the_endpoint()
    assert len(modules) >= 10, modules
    undeclared = []
    for name in modules:
        module = importlib.import_module(name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, Task)
                and cls.__module__ == name
                and CODE_EVAL not in cls.services
            ):
                undeclared.append(f"{name}.{cls.__qualname__}")
    assert undeclared == []


def test_a_task_that_calls_no_service_declares_none() -> None:
    from sieval.tasks import GSM8KZeroShotGenTask

    assert GSM8KZeroShotGenTask.services == frozenset()
