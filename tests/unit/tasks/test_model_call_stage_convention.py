"""Every task calls its model under test in ``infer`` and its grader in ``feedback``.

``sieval run`` plans each engine's lifecycle against that convention
(``_ROLE_STAGE`` in ``sieval.cli.lifecycle``): it refuses, before launch, a
lifecycle that leaves the model under test down in ``inference`` or a grader
down in ``feedback``. A task that called its grader in ``postprocess`` would make
that check refuse a lifecycle that works and pass one that fails every sample
with a connection error, and nothing in the task would say so.

This survey reads the AST of every module under ``sieval/tasks/``, so it covers
tasks nobody added to a list and calls that live in a shared base. An
``extractor`` is not surveyed: the lifecycle check does not rely on its stage.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import ast
import pathlib
from typing import NamedTuple

import sieval

#: Resolved from the imported package, so a worktree surveys its own tree.
TASKS_DIR = pathlib.Path(sieval.__file__).parent / "tasks"

#: The attribute a role's model is held under, and the method it may be called in.
ROLE_METHOD = {"model": "infer", "_grader": "feedback"}

#: Floors, not counts: they only need lowering when call sites are removed, and
#: they are what stops a broken scan from passing as "no offenders found".
MIN_EXPECTED = {"model": 40, "_grader": 8}


class _Call(NamedTuple):
    file: str
    lineno: int
    holder: str
    method: str

    def __str__(self) -> str:
        return f"{self.file}:{self.lineno} calls self.{self.holder} in {self.method}"


def _calls() -> list[_Call]:
    """Each ``self.<holder>.agenerate(...)`` for a surveyed holder, with the
    class method it is written in."""
    found = []
    for path in sorted(TASKS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for node in ast.walk(fn):
                    if not (
                        isinstance(node, ast.Attribute)
                        and node.attr == "agenerate"
                        and isinstance(holder := node.value, ast.Attribute)
                        and isinstance(holder.value, ast.Name)
                        and holder.value.id == "self"
                        and holder.attr in ROLE_METHOD
                    ):
                        continue
                    found.append(
                        _Call(
                            str(path.relative_to(TASKS_DIR)),
                            node.lineno,
                            holder.attr,
                            fn.name,
                        )
                    )
    return found


def test_the_survey_still_finds_the_call_sites():
    calls = _calls()
    for holder, floor in MIN_EXPECTED.items():
        n = sum(c.holder == holder for c in calls)
        assert n >= floor, (
            f"only {n} self.{holder}.agenerate call sites under {TASKS_DIR}; the "
            "AST scan has probably stopped matching rather than the tree shrinking"
        )


def test_each_role_is_called_in_the_stage_lifecycles_plan_for():
    offenders = [c for c in _calls() if c.method != ROLE_METHOD[c.holder]]
    assert not offenders, (
        "`sieval run` keeps a model under test up in inference and a grader up "
        "in feedback (`_ROLE_STAGE` in sieval/cli/lifecycle.py); a call in "
        "another stage can reach no engine:\n  "
        + "\n  ".join(str(c) for c in offenders)
        + "\nMove the call, or teach `check_lifecycles_cover_calls` that stage."
    )
