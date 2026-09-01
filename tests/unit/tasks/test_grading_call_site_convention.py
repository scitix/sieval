"""Every grading call site in `sieval/tasks/` catches `TimeoutError`, and nothing else.

The convention itself is argued in `.claude/rules/tasks.md`: a grade that could not
be computed *in time* is a wrong answer, while a grader that is **broken rather than
slow** must not be indistinguishable from a model that answered wrongly. What that
rule could not do on its own is stay true. It was written down beside a family test
that enumerates its members by hand, so the next task to copy an older template back
into the tree would have re-drifted with nothing to catch it — the survey proving the
tree uniform was a one-off run by a human, leaving no artifact.

This is that survey, kept. It reads the AST rather than the registry, so it covers
every member of every family at once, including tasks whose `feedback` lives in a
shared base and tasks nobody thought to add to a list.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import ast
import pathlib
from typing import NamedTuple

import sieval

#: Resolved from the imported package rather than from this file's parents, so a
#: worktree cannot survey the primary checkout's `sieval/` by accident.
TASKS_DIR = pathlib.Path(sieval.__file__).parent / "tasks"

#: A floor, not a count: it only needs raising when grading sites are *removed*,
#: and it is what stops a broken scan from passing as "no offenders found".
MIN_EXPECTED_SITES = 20


class _Site(NamedTuple):
    file: str
    lineno: int
    caught: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.file}:{self.lineno} catches {', '.join(self.caught) or 'BARE'}"


def _calls_run_cpu_bound(body: list[ast.stmt]) -> bool:
    """Whether *body* grades through `run_cpu_bound`.

    Scoped to the `try` BODY: a call in `else`/`finally` is not protected by the
    handlers, so it is not the thing this rule is about.
    """
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "run_cpu_bound":
                return True
    return False


def _caught(handler: ast.ExceptHandler) -> tuple[str, ...]:
    if handler.type is None:
        return ()
    if isinstance(handler.type, ast.Tuple):
        return tuple(ast.unparse(e) for e in handler.type.elts)
    return (ast.unparse(handler.type),)


def _grading_sites() -> list[_Site]:
    sites = []
    for path in sorted(TASKS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and _calls_run_cpu_bound(node.body):
                sites.extend(
                    _Site(path.name, h.lineno, _caught(h)) for h in node.handlers
                )
    return sites


def test_the_survey_still_finds_the_grading_sites():
    # Without this, a scan that silently stopped matching would report a clean
    # tree — the one way this file could pass while asserting nothing.
    sites = _grading_sites()
    assert len(sites) >= MIN_EXPECTED_SITES, (
        f"only {len(sites)} grading sites found under {TASKS_DIR}; the AST scan "
        "has probably stopped matching rather than the tree having shrunk"
    )


def test_a_grading_call_site_catches_timeout_and_nothing_else():
    """The rule, enforced over the whole tree instead of a hand-kept list.

    `except Exception` here records every way a grader can fail — a dead worker, an
    optional dependency missing from the environment, a defect in a vendored grader
    — as the model having answered wrongly, finishing the run with `fails = 0` and a
    depressed score whose only trace is a log line.
    """
    offenders = [s for s in _grading_sites() if s.caught != ("TimeoutError",)]
    assert not offenders, (
        "a grading call site must catch `TimeoutError` only, so that a grader "
        "which is broken rather than slow lands in `fails` as "
        "`exception::<class>` instead of scoring the sample wrong:\n  "
        + "\n  ".join(str(s) for s in offenders)
        + "\nSee the grading-call-site rule in `.claude/rules/tasks.md`."
    )
